# OCR Accuracy Benchmark Pipeline

Production-grade FastAPI backend for testing and comparing OCR model accuracy across **Arabic, Hindi, Punjabi, and English**.

---

## Architecture

```
ocr-pipeline/
├── main.py                        # FastAPI app + lifespan
├── app/
│   ├── core/
│   │   ├── config.py              # Pydantic-settings env config
│   │   ├── model_registry.py      # Startup loader, factory, registry
│   │   └── metrics.py             # CER, WER, NED, F1 computation
│   ├── models/
│   │   ├── base.py                # Abstract BaseOCRModel + result types
│   │   ├── paddleocr_v4.py        # Tier 2 — PaddleOCR PP-OCRv4
│   │   ├── easyocr_model.py       # Tier 2 — EasyOCR
│   │   ├── tesseract_model.py     # Tier 2 — Tesseract 5 LSTM
│   │   ├── paddleocr_vl.py        # Tier 1 — PaddleOCR-VL (0.9B)
│   │   ├── got_ocr2.py            # Tier 1 — GOT-OCR 2.0 (8B)
│   │   ├── qwen25_vl.py           # Tier 1 — Qwen2.5-VL (3B/7B)
│   │   └── olmocr2.py             # Tier 1 — olmOCR-2 (7B)
│   ├── routers/
│   │   ├── ocr.py                 # POST /ocr/run, GET /ocr/models
│   │   ├── benchmark.py           # POST /benchmark/evaluate, /benchmark/batch
│   │   └── health.py              # GET /health/, /health/gpu
│   └── schemas.py                 # All Pydantic request/response models
├── tests/
│   └── test_pipeline.py           # Integration + unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone & configure

```bash
git clone <your-repo>
cd ocr-pipeline
cp .env.example .env
```

Edit `.env` to set which models to load:

```env
ENABLED_MODELS=["paddleocr_v4","easyocr","tesseract"]
USE_GPU=false
```

### 2. Install system dependencies

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-hin tesseract-ocr-pan
```

### 3. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run

```bash
uvicorn main:app --reload --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

---

## API Reference

### `GET /health/`
Returns system status and loaded models.

```json
{
  "status": "ok",
  "loaded_models": ["paddleocr_v4", "easyocr", "tesseract"],
  "system": { "cpu_percent": 12.4, "ram_used_gb": 3.1 }
}
```

### `GET /health/gpu`
Returns GPU memory usage (requires torch).

---

### `GET /ocr/models`
Lists all loaded models and their capabilities.

```json
{
  "models": [
    { "name": "paddleocr_v4", "tier": 2, "supported_languages": ["en","ar","hi","pa"], "loaded": true }
  ],
  "total_loaded": 3
}
```

---

### `POST /ocr/run`
Run OCR on an image using all (or selected) models.

**Form fields:**
| Field    | Type   | Required | Description |
|----------|--------|----------|-------------|
| file     | File   | ✅       | PNG/JPG/TIFF/WebP image |
| language | string | ✅       | `en`, `ar`, `hi`, `pa` |
| models   | string | ❌       | Comma-separated model names (blank = all) |

**Response:**
```json
{
  "filename": "document.png",
  "language": "en",
  "results": [
    {
      "model_name": "paddleocr_v4",
      "raw_text": "Hello World",
      "avg_confidence": 0.9823,
      "inference_time_ms": 142.5,
      "words": [
        { "text": "Hello", "confidence": 0.99, "bbox": [10, 20, 80, 50] }
      ]
    }
  ],
  "models_run": 3,
  "total_time_ms": 560.2
}
```

---

### `POST /benchmark/evaluate`
Run OCR + compute accuracy metrics against a ground truth string.

**Form fields:**
| Field        | Type   | Required | Description |
|--------------|--------|----------|-------------|
| file         | File   | ✅       | Image to OCR |
| ground_truth | string | ✅       | Correct text for comparison |
| language     | string | ✅       | `en`, `ar`, `hi`, `pa` |
| models       | string | ❌       | Comma-separated model names |

**Response:**
```json
{
  "filename": "doc.png",
  "language": "hi",
  "ground_truth": "नमस्ते दुनिया",
  "results": [
    {
      "model_name": "paddleocr_v4",
      "raw_text": "नमस्ते दुनिया",
      "metrics": {
        "cer": 0.0,
        "wer": 0.0,
        "ned": 0.0,
        "char_f1": 1.0,
        "word_f1": 1.0,
        "exact_match": true
      }
    }
  ],
  "best_model_cer": "paddleocr_v4",
  "best_model_wer": "paddleocr_v4",
  "best_model_f1": "paddleocr_v4"
}
```

---

### `POST /benchmark/batch`
Benchmark multiple images at once. Returns per-image results + aggregate averages.

**Form fields:**
| Field         | Type        | Required | Description |
|---------------|-------------|----------|-------------|
| files         | File[]      | ✅       | Multiple images |
| ground_truths | string      | ✅       | Newline-separated, one per image |
| language      | string      | ✅       | `en`, `ar`, `hi`, `pa` |

**Response includes aggregate summary:**
```json
{
  "aggregate_summary": {
    "paddleocr_v4": {
      "avg_cer": 0.04,
      "avg_wer": 0.07,
      "avg_word_f1": 0.94,
      "avg_inference_ms": 138.2,
      "images_evaluated": 10
    }
  }
}
```

---

## Enabling Tier 1 VLMs (GPU)

Edit `.env`:

```env
USE_GPU=true
ENABLED_MODELS=["paddleocr_v4","easyocr","tesseract","got_ocr2","qwen25_vl","olmocr2"]
```

Install GPU deps:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate qwen-vl-utils
```

VRAM requirements:
| Model          | VRAM  |
|----------------|-------|
| paddleocr_vl   | ~4GB  |
| got_ocr2       | ~8GB  |
| qwen25_vl (3B) | ~8GB  |
| qwen25_vl (7B) | ~16GB |
| olmocr2 (7B)   | ~16GB |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Accuracy Metrics Guide

| Metric | Range | Meaning |
|--------|-------|---------|
| CER    | 0–1   | Character Error Rate — 0 is perfect |
| WER    | 0–1   | Word Error Rate — 0 is perfect |
| NED    | 0–1   | Normalized Edit Distance |
| char_f1| 0–1   | Character-level F1 — 1 is perfect |
| word_f1| 0–1   | Word-level F1 — 1 is perfect |

**Rule of thumb for production thresholds:**
- CER < 0.05 → production-grade accuracy
- CER 0.05–0.15 → acceptable for post-processing pipelines
- CER > 0.15 → needs fine-tuning or a stronger model
