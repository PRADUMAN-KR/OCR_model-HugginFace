"""
Integration tests for the OCR benchmark pipeline.
Run with: pytest tests/ -v
"""

import io
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from main import app
from app.core.config import settings
from app.core.model_registry import ModelRegistry


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_test_image(text: str = "Hello World", size=(400, 100)) -> bytes:
    """Generate a minimal white image with rendered text for testing."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
# httpx.ASGITransport does not run FastAPI lifespan — initialize the registry like production startup.

@pytest_asyncio.fixture(scope="module")
async def _test_registry():
    registry = ModelRegistry()
    await registry.initialize(settings.ENABLED_MODELS)
    app.state.model_registry = registry
    yield registry
    await registry.shutdown()


@pytest_asyncio.fixture
async def client(_test_registry):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(scope="module")
async def require_paddle(_test_registry):
    """Skip OCR/benchmark calls when Paddle did not load (install paddlepaddle separately)."""
    if "paddleocr_v4" not in _test_registry.loaded_models:
        pytest.skip(
            "paddleocr_v4 not loaded; install the Paddle framework (paddlepaddle / paddlepaddle-gpu). "
            "See https://www.paddlepaddle.org.cn/install/quick — "
            f"load error: {_test_registry.failed_models.get('paddleocr_v4', 'unknown')}"
        )
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "loaded_models" in data


@pytest.mark.asyncio
async def test_gpu_health(client):
    r = await client.get("/health/gpu")
    assert r.status_code == 200
    assert "gpu_available" in r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Model listing
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models(client):
    r = await client.get("/ocr/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert data["total_loaded"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# OCR /run
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ocr_run_english(client, require_paddle):
    image_bytes = make_test_image("Hello World 123")
    r = await client.post(
        "/ocr/run",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"language": "en"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["language"] == "en"
    assert data["models_run"] >= 0
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_ocr_run_arabic(client, require_paddle):
    image_bytes = make_test_image("Arabic Test")  # Placeholder — use real Arabic image in prod
    r = await client.post(
        "/ocr/run",
        files={"file": ("arabic.png", image_bytes, "image/png")},
        data={"language": "ar"},
    )
    assert r.status_code in (200, 404)  # 404 if no Arabic-capable model loaded


@pytest.mark.asyncio
async def test_ocr_run_hindi(client, require_paddle):
    image_bytes = make_test_image("Hindi Test")
    r = await client.post(
        "/ocr/run",
        files={"file": ("hindi.png", image_bytes, "image/png")},
        data={"language": "hi"},
    )
    assert r.status_code in (200, 404)


@pytest.mark.asyncio
async def test_ocr_run_unsupported_filetype(client):
    r = await client.post(
        "/ocr/run",
        files={"file": ("test.txt", b"some text", "text/plain")},
        data={"language": "en"},
    )
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_ocr_run_specific_model(client, require_paddle):
    """Test routing to a single specific model."""
    image_bytes = make_test_image("Selective model test")
    r = await client.post(
        "/ocr/run",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"language": "en", "models": "paddleocr_v4"},
    )
    assert r.status_code in (200, 404)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark /evaluate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_benchmark_evaluate(client, require_paddle):
    image_bytes = make_test_image("Hello World")
    r = await client.post(
        "/benchmark/evaluate",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"language": "en", "ground_truth": "Hello World"},
    )
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert "results" in data
        assert "best_model_cer" in data
        for result in data["results"]:
            assert "metrics" in result
            assert 0.0 <= result["metrics"]["cer"] <= 1.0
            assert 0.0 <= result["metrics"]["wer"] <= 1.0


@pytest.mark.asyncio
async def test_benchmark_empty_ground_truth(client):
    image_bytes = make_test_image("Test")
    r = await client.post(
        "/benchmark/evaluate",
        files={"file": ("test.png", image_bytes, "image/png")},
        data={"language": "en", "ground_truth": ""},
    )
    assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Metrics unit tests
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics_perfect_match():
    from app.core.metrics import compute_metrics
    m = compute_metrics("hello world", "hello world")
    assert m.cer == 0.0
    assert m.wer == 0.0
    assert m.exact_match is True


def test_metrics_total_mismatch():
    from app.core.metrics import compute_metrics
    m = compute_metrics("aaaa", "bbbb")
    assert m.cer == 1.0
    assert m.exact_match is False


def test_metrics_partial():
    from app.core.metrics import compute_metrics
    m = compute_metrics("hello wrold", "hello world")
    assert 0.0 < m.cer < 1.0
    assert m.exact_match is False


def test_metrics_arabic():
    from app.core.metrics import compute_metrics
    m = compute_metrics("مرحبا", "مرحبا")
    assert m.cer == 0.0
    assert m.exact_match is True


def test_metrics_hindi():
    from app.core.metrics import compute_metrics
    m = compute_metrics("नमस्ते दुनिया", "नमस्ते दुनिया")
    assert m.cer == 0.0


def test_metrics_empty_prediction():
    from app.core.metrics import compute_metrics
    m = compute_metrics("", "hello world")
    assert m.cer == 1.0
    assert m.word_f1 == 0.0
