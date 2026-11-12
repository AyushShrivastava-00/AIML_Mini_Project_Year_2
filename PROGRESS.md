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

## 📅 Next Sprint (Day 3): Edge Optimization — ONNX Export & INT8 Quantization
- **Goal**: Convert `best.pt` to hardware-neutral ONNX format and apply post-training INT8 quantization to achieve sub-200ms CPU inference on edge devices.
