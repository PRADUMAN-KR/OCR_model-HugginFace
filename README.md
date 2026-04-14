# 🚀 Hugginface OCR Pipeline using Deep Learning & Computer Vision

 **Optical Character Recognition (OCR) system** built using state-of-the-art deep learning models for multilingual text detection and recognition, including **Arabic and English scripts**.

This project focuses on improving **OCR accuracy, bounding box detection, text normalization, benchmarking, and post-processing optimization** for real-world scanned documents and images.

---

## 📌 Overview

This repository provides a modular and scalable OCR pipeline designed for:

- Multilingual text extraction (Arabic & English)
- Deep learning-based text recognition
- Bounding box detection and polygon conversion
- OCR accuracy benchmarking against ground truth
- GPU-accelerated inference


The system is designed to handle **low-resolution images, noisy scans, skewed documents, and right-to-left (RTL) languages like Arabic.**

---

## 🧠 Key Features

- ✅ Multi-model OCR benchmarking (Tesseract, EasyOCR, Transformer-based OCR)
- ✅ Arabic OCR optimization for RTL text
- ✅ Polygon → Bounding Box (XYXY) conversion utilities
- ✅ Character-level and word-level accuracy evaluation
- ✅ Ground truth comparison pipeline
- ✅ Preprocessing (denoising, thresholding, resizing, normalization)
- ✅ Post-processing correction layer
- ✅ Modular architecture for experimentation
- ✅ Linux + GPU support

---

## 🛠 Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **Tesseract OCR**
- **EasyOCR**
- **HuggingFace Transformers (TrOCR / VisionEncoderDecoder)**
- **NumPy**
- **Docker**
- **Linux (GPU-enabled environment)**

---

## 📂 Project Architecture
