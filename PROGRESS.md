# 📈 Project Sprint Tracker (7-Day Plan)

This document tracks the daily engineering milestones and academic concepts mastered during the development of the **Micro-Merchant Informal Credit Assessment Engine**.

---

## 📅 Day 1: System Scaffolding, Image Physics & Visual Foundations
- **Milestone Completed**: Established a clean project directory architecture, configured an isolated Python virtual environment, and built the single-box coordinate visualizer in VS Code.
- **Key Files Created**:
  - `requirements.txt`: Specified package versions (`ultralytics`, `opencv-python-headless`, `Pillow`, `matplotlib`, `pydantic`).
  - `.gitignore`: Implemented strict exclusion rules for virtual environments (`venv/`), system caches, and heavy deep learning weights (`models/*.pt`, `models/*.onnx`).
  - `README.md`: Documented project architecture, problem statement, and repository layout.
  - `notebooks/01_eda_sku110k.ipynb`: Interactive notebook for coordinate math and visual rendering.
- **Concepts Mastered**:
  - **Process Isolation**: How virtual environments prevent system-wide dependency collisions.
  - **Memory Layouts**: Digital images represented as 3D NumPy matrices of `uint8` values ($0 \dots 255$); swapping OpenCV's BGR order to RGB.
  - **Coordinate Translation**: Mathematical conversion between normalized YOLO percentages ($c_x, c_y, w, h$) and discrete pixel corners ($x_1, y_1, x_2, y_2$).
  - **Alpha Blending**: Mathematical color blending equation for semi-transparent bounding box overlays.

---

## 📅 Day 2: Dense Shelf Detection & YOLOv8-Nano Model Training
- **Milestone Completed**: Fine-tuned YOLOv8-Nano on dense retail shelf items using Google Colab T4 GPU; successfully saved trained model weights to `models/best.pt`.
- **Key Files Created**:
  - `data.yaml`: Dataset configuration mapping retail item classes.
  - `src/vision/train_yolo.py`: Python training reference pipeline.
  - `models/best.pt`: Trained neural network weights (ignored by Git for lightweight repository).
- **Concepts Mastered**:
  - **YOLOv8 Anatomy**: Backbone (CSPDarknet feature extraction), Neck (PAN-FPN multi-scale feature fusion), and Anchor-Free Head.
  - **Loss Mathematics**: Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$) measuring overlap, center point distance, and aspect ratio consistency, coupled with Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$) for foreground-background separation.
  - **Academic Evaluation Metrics**: Mean Average Precision (**mAP@50**), Precision-Recall trade-offs, and confusion matrix interpretation.

---

## 📅 Day 3: Edge Optimization — ONNX Export & CPU Inference Benchmarking
- **Milestone Completed**: Successfully exported fine-tuned YOLOv8 model to hardware-neutral ONNX format; built lightweight CPU inference engine achieving sub-200ms latency; verified detections on dense retail shelf image with Non-Maximum Suppression (NMS).
- **Key Files Created**:
  - `src/vision/export_onnx.py`: Static graph export pipeline.
  - `src/vision/inference.py`: CPU-optimized inventory evaluation engine.
  - `src/vision/test_onnx_shelf.py`: NMS-filtered local shelf verification script.
  - `create_shelf_image.py`: Synthetic dense shelf test fixture generator.
  - `models/best.onnx`: Exported ONNX graph model (Git-ignored).
- **Concepts Mastered**:
  - **Static Graph Compilation**: How ONNX decouples models from Python runtime, executing graph nodes directly via C++ instruction sets.
  - **Quantization Mathematics**: Mapping continuous 32-bit floats ($\text{FP32}$) to discrete 8-bit integers ($\text{INT8}$) via affine linear scaling: $q = \text{round}(r/S) + Z$.
  - **Hardware Vectorization (SIMD)**: Why 8-bit integers allow 64-bit CPU registers to perform 8 simultaneous operations per clock cycle instead of 2.
  - **Non-Maximum Suppression (NMS)**: Post-processing filtering that eliminates redundant overlapping candidate boxes using IoU thresholding.

---

## 📅 Day 4: Voice Interview Processing — The Whisper.cpp Pipeline
- **Milestone Completed**: Built and verified local, quantized Whisper speech-to-text pipeline; implemented audio preprocessing and 16kHz resampling with peak normalization; achieved an RTF of 0.16x on CPU.
- **Key Files Created**:
  - `src/voice/transcriber.py`: Core `VoiceInterviewTranscriber` class utilizing `faster-whisper` INT8 CPU inference.
  - `src/voice/create_sample_audio.py`: Audio test fixture generator for realistic merchant disclosures.
  - `src/voice/test_transcription.py`: Benchmarking test script measuring Real-Time Factor (RTF) and latency.
  - `models/whisper/`: Cached quantized Whisper weights (Git-ignored).
- **Concepts Mastered**:
  - **Acoustic Physics & Sampling**: Nyquist-Shannon theorem ($f_s \ge 2 f_{\max}$) and why 16 kHz mono audio is optimal for phonetic speech representations.
  - **Time-to-Frequency Transforms**: Short-Time Fourier Transforms (STFT), Mel-scale non-linear auditory filtering, and Log-Mel spectrogram generation ($80 \times 3000$ tensors).
  - **Transformer Encoder-Decoder for Speech**: Self-attention over acoustic spectrograms, cross-attention in the decoder, and language token conditioning.
  - **Real-Time Factor (RTF)**: Benchmarked transcription velocity ($\text{RTF} = 0.16\times$), proving adherence to sub-5s processing latency.

---

## 📅 Next Sprint (Day 5): Generative Underwriting — Local SLM & Pydantic JSON
- **Goal**: Connect an on-device Small Language Model (Qwen-2.5-1.5B via Ollama) to cross-verify visual shelf inventory against the audio transcript using financial stress tests, outputting an explainable credit score (300–850) in strict Pydantic JSON format.
