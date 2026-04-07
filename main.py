"""
OCR Model Accuracy Testing Pipeline
Production-grade FastAPI backend for benchmarking OCR models
across Arabic, Hindi, Punjabi, and English scripts.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import ocr, benchmark, health
from app.core.config import settings
from app.core.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model registry. Shutdown: release GPU memory."""
    logger.info("Initializing OCR Model Registry...")
    registry = ModelRegistry()
    await registry.initialize(settings.ENABLED_MODELS)
    app.state.model_registry = registry
    logger.info(f"Loaded models: {list(registry.loaded_models.keys())}")
    yield
    logger.info("Shutting down — releasing model resources...")
    await registry.shutdown()


app = FastAPI(
    title="OCR Accuracy Benchmark API",
    description="Production pipeline to test and compare OCR model accuracy across Arabic, Hindi, Punjabi, and English.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(benchmark.router, prefix="/benchmark", tags=["Benchmark"])
