# 🏪 Micro-Merchant Informal Credit Assessment Engine
> **Vision + Voice Micro-Auditing for Low-Resource Informal Underwriting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Problem Statement
Micro-finance institutions (MFIs) and rural lenders struggle to underwrite informal micro-merchants (kirana stores, street vendors) due to a complete lack of formal financial documentation (tax returns, GST, POS logs). Current manual field officer visits are subjective, slow, and prone to human bias.

## 💡 Solution
An on-device, privacy-preserving AI micro-auditing application that:
1. **Analyzes Video Inventory**: Uses a lightweight YOLOv8-Nano model to detect items and estimate shelf stock density from a 15-second walkthrough.
2. **Transcribes Spoken Interviews**: Converts multilingual spoken disclosures (revenue, supplier payments) into text using Whisper.
3. **Generates Explainable Credit Reports**: An on-device Small Language Model (SLM) cross-verifies physical shelf volume against declared cash flow to output a structured credit score (300–850) and recommended loan limit in JSON format.

## 🛠️ Architecture & Tech Stack
- **Vision**: YOLOv8-Nano / ONNX Runtime (CPU quantized)
- **Audio**: Whisper.cpp (Tiny)
- **SLM**: Qwen-2.5-1.5B / Llama-3.2 via Ollama
- **Frontend**: Gradio Dashboard
- **Language**: Python 3.10+

## 🚀 Repository Structure
├── data/ # Raw and processed datasets
├── models/ # Quantized ONNX & model weights
├── notebooks/ # Exploratory data analysis (EDA)
├── src/
│ ├── vision/ # Detection & shelf density calculations
│ ├── voice/ # Audio preprocessing & STT
│ ├── slm/ # Reasoning prompt & JSON validation
│ └── pipeline/ # Orchestration engine
└── tests/ # Automated test suites5