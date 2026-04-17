# PaddleOCR Pipeline API

FastAPI-based OCR API for multilingual document text extraction using PaddleOCR, with a stronger Arabic OCR pipeline, GPU inference support, structured JSON output, and built-in benchmarking endpoints.

This project is designed for teams building document OCR services, Arabic OCR workflows, scanned PDF text extraction, and OCR model evaluation pipelines.

## Why This Project

Many OCR repos stop at a notebook or a single inference script. This project packages OCR as a reusable API with:

- PaddleOCR-based document OCR for images and PDFs
- Arabic, English, Hindi, and best-effort multilingual OCR
- OCR benchmarking against ground truth text
- Word-level bounding boxes and confidence scores
- GPU-aware deployment controls for VRAM tuning
- Arabic-oriented preprocessing and post-processing for harder documents

If you are searching for a Python OCR API, FastAPI OCR server, PaddleOCR REST API, Arabic OCR pipeline, or document OCR benchmarking tool, this repository is built for that use case.

## Features

- PaddleOCR API built with FastAPI
- OCR for `jpg`, `png`, `webp`, `tiff`, and `pdf`
- Structured OCR responses with raw text, words, confidence, and bounding boxes
- Arabic OCR enhancements for RTL documents and noisy scans
- Health endpoints for service and GPU availability checks
- Benchmark endpoints for CER, WER, NED, precision, recall, and F1
- Configurable GPU memory behavior for Paddle inference
- Lazy model loading for practical server startup behavior
- Support for multilingual OCR requests via `language=all`

## Supported Languages

- `ar` — Arabic
- `en` — English
- `hi` — Hindi
- `all` — best-effort multilingual mode

The base model enum also includes Punjabi support, but the currently configured PaddleOCR model registry in this repo is centered on the languages above.

## Tech Stack

- Python
- FastAPI
- PaddleOCR
- PaddlePaddle
- OpenCV
- NumPy
- Pillow
- Pydantic Settings

## Project Structure

```text
.
├── app
│   ├── core
│   │   ├── config.py
│   │   ├── document.py
│   │   ├── metrics.py
│   │   ├── model_registry.py
│   │   └── model_selection.py
│   ├── models
│   │   ├── base.py
│   │   ├── paddleocr_v4.py
│   │   └── paddleocr_vl.py
│   └── routers
│       ├── benchmark.py
│       ├── health.py
│       └── ocr.py
├── main.py
├── requirements.txt
└── .env
```

## API Endpoints

### OCR

- `GET /ocr/models` — list currently loaded OCR models
- `GET /ocr/options` — list loaded models, languages, and runtime options
- `POST /ocr/run` — run OCR on an uploaded file

### Benchmarking

- `POST /benchmark/evaluate` — compare OCR output with ground truth text
- `POST /benchmark/batch` — benchmark multiple files in one request

### Health

- `GET /health/` — service health, loaded models, CPU, RAM
- `GET /health/gpu` — basic Paddle GPU availability check

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ocr-pipeline.git
cd ocr-pipeline
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PaddlePaddle

Install the correct CPU or GPU build for your machine from the official PaddlePaddle install guide:

https://www.paddlepaddle.org.cn/install/quick

For NVIDIA GPU servers, make sure the PaddlePaddle build matches your CUDA environment.

### 5. Configure environment variables

Create or edit `.env` in the project root.

Example:

```env
DEBUG=false
ENABLED_MODELS=["paddleocr_v4"]

PADDLE_USE_GPU=true
GPU_DEVICE_ID=0
PADDLE_PRECISION=fp32
PADDLE_FP16=false
PADDLE_TENSORRT=false

PADDLE_FLAGS_ALLOCATOR_STRATEGY=auto_growth
PADDLE_FLAGS_FRACTION_OF_GPU_MEMORY_TO_USE=0.5
PADDLE_EMPTY_CACHE_BETWEEN_PAGES=true
PADDLE_DET_LIMIT_SIDE_LEN=1280
PADDLE_MAX_ACCURACY=true
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true

MAX_FILE_SIZE_MB=20
MODEL_TIMEOUT=60
BENCHMARK_TIMEOUT=300
CORS_ORIGINS=["*"]
```

### 6. Run the API

```bash
uvicorn main:app 
```

Open:

- `http://localhost:4018/docs` for Swagger UI
- `http://localhost:4018/redoc` for ReDoc

## Example OCR Request

```bash
curl -X POST "http://localhost:4018/ocr/run" \
  -F "file=@sample.jpg" \
  -F "language=ar"
```

Example response shape:

```json
{
  "filename": "sample.jpg",
  "language": "ar",
  "results": [
    {
      "model_name": "paddleocr_v4",
      "language": "ar",
      "raw_text": "example extracted text",
      "words": [
        {
          "text": "example",
          "confidence": 0.98,
          "bbox": [10, 20, 80, 50]
        }
      ],
      "inference_time_ms": 842.14,
      "avg_confidence": 0.94,
      "error": null,
      "metadata": {}
    }
  ],
  "models_run": 1,
  "total_time_ms": 842.14
}
```

## Example Benchmark Request

```bash
curl -X POST "http://localhost:4018/benchmark/evaluate" \
  -F "file=@sample.jpg" \
  -F "language=ar" \
  -F "ground_truth=your expected text here"
```

This returns OCR output plus evaluation metrics like:

- CER
- WER
- NED
- character precision, recall, F1
- word precision, recall, F1
- overall accuracy
- exact match

## Arabic OCR Pipeline Notes

The main `paddleocr_v4` wrapper includes extra logic intended to improve difficult Arabic OCR cases, especially for scanned or noisy documents:

- multi-pass full-page variant generation
- CLAHE contrast enhancement
- adaptive thresholding for diacritics
- ink bleed cleanup
- skew correction
- RTL-aware line ordering
- Arabic punctuation normalization
- optional Alef normalization
- isolated letter noise filtering

These additions are particularly useful when working on Arabic OCR API services, document digitization, archive OCR, and multilingual OCR evaluation.

## Performance and VRAM Tuning

The most important knobs for GPU memory behavior are:

- `PADDLE_PRECISION`
- `PADDLE_FP16`
- `PADDLE_TENSORRT`
- `PADDLE_FLAGS_FRACTION_OF_GPU_MEMORY_TO_USE`
- `PADDLE_EMPTY_CACHE_BETWEEN_PAGES`
- `PADDLE_DET_LIMIT_SIDE_LEN`
- `PADDLE_MAX_ACCURACY`

Practical guidance:

- `PADDLE_DET_LIMIT_SIDE_LEN` is often the biggest VRAM lever
- `fp16` may reduce memory, but the gain depends on your model/runtime mix
- TensorRT is optional and requires a compatible `tensorrt` install
- `fp32` is the safest choice when accuracy stability matters most

## Configuration Reference

Common environment variables:

- `ENABLED_MODELS` — models to load at startup
- `PADDLE_USE_GPU` — enable GPU inference
- `GPU_DEVICE_ID` — choose GPU device
- `PADDLE_PRECISION` — `fp32` or `fp16` in your current setup
- `PADDLE_FP16` — compatibility flag for fp16 mode
- `PADDLE_TENSORRT` — enable TensorRT when available
- `PADDLE_FLAGS_ALLOCATOR_STRATEGY` — Paddle allocator strategy
- `PADDLE_FLAGS_FRACTION_OF_GPU_MEMORY_TO_USE` — initial GPU allocation fraction
- `PADDLE_EMPTY_CACHE_BETWEEN_PAGES` — clear cache between pages
- `PADDLE_DET_LIMIT_SIDE_LEN` — cap detection-side image resolution
- `PADDLE_MAX_ACCURACY` — keep the higher-cost multi-pass OCR flow enabled
- `PADDLE_DEBUG_OUTPUT_DIR` — save debug visualizations

## Development

Install test dependencies from `requirements.txt`, then run:

```bash
pytest
```

Start the service locally:

```bash
uvicorn main:app --reload --port 4018
```

## Use Cases

- Arabic OCR API for business documents
- OCR for scanned PDFs and images
- GPU-accelerated OCR microservice
- OCR benchmarking against ground truth
- Document digitization pipelines
- Multilingual OCR experimentation with PaddleOCR

## Roadmap Ideas

- Docker deployment examples
- `.env.example` template
- request/response examples for every endpoint
- model comparison dashboards
- async batch processing
- object storage integration for large OCR workloads

## License

Add your preferred license here, for example MIT or Apache-2.0.

