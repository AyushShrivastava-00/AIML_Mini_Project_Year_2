# 📈 Project Progress Tracker

This document serves as an active log of the engineering milestones and learning outcomes achieved throughout the 8-week development cycle of the **Micro-Merchant Informal Credit Assessment Engine**.

---

## 📅 Week 1: Project Scaffolding, Environment Design & Image Physics

### 🚀 Week 1 — Day 1: Project Scaffolding & Environment Setup
- **Milestone Completed**: Established a clean, modular directory structure and configured the virtual environment and local version control.
- **Key Files Created**:
  - `requirements.txt`: Specified production package versions (`ultralytics`, `opencv-python-headless`, `onnxruntime`, `Pillow`, `gradio`, `pytest`).
  - `.gitignore`: Implemented strict exclusion rules for virtual environment files (`venv/`), operating system caches, and heavy deep learning model weights (`.pt`, `.onnx`).
  - `README.md`: Created the landing page with the project scope, problem statement, architecture, and tech stack details.
- **Concepts Mastered**:
  - **Virtual Environment (`venv`)**: Understanding why project libraries must be isolated to prevent dependency version collisions.
  - **Git vs. GitHub**: Understanding that Git is an offline, local version control database (`.git` folder) while GitHub is simply a remote, collaborative hosting mirror.
  - **Branch Semantics**: Learned why modern repositories use `main` as the primary branch timeline instead of the outdated `master` label.

---

### 👁️ Week 1 — Day 2: Bounding Box Mathematics & Single-Box Visualization
- **Milestone Completed**: Configured VS Code Jupyter Kernel, loaded a raw local asset, and wrote the coordinate denormalization and alpha-channel drawing pipeline.
- **Key Files Created**:
  - `notebooks/01_eda_sku110k.ipynb`: The interactive notebook containing path verification and custom drawing algorithms.
- **Concepts Mastered**:
  - **Computer Graphics Coordinate Space**: Learned that digital images have their origin `(0,0)` in the top-left corner (increasing X moves right, increasing Y moves down).
  - **YOLO Label format**: Dissected the normalized coordinates (`class_id`, `center_x`, `center_y`, `width`, `height`), understanding why values are scaled to decimals between `0.0` and `1.0` to maintain resolution independence.
  - **Coordinate Translation Math**: Mastered the formulas to convert center percentages back to pixel integers:
    - $C_x = c_x \times W$, $C_y = c_y \times H$, $P_w = w \times W$, $P_h = h \times H$
    - $x_1 = C_x - (P_w / 2)$, $y_1 = C_y - (P_h / 2)$
    - $x_2 = C_x + (P_w / 2)$, $y_2 = C_y + (P_h / 2)$
  - **Memory Representation & Color Conversion**: Studied how raw image arrays are structured as 3D NumPy matrices of `uint8` values in memory. Swapped color dimensions from OpenCV's BGR order to standard RGB to prevent color inversion.
  - **Alpha Blending**: Explored how the CPU uses mathematical weight equations to mix new color layers with existing pixel backgrounds, creating the illusion of semi-transparent bounding box overlays.
  - **Zero-Copy Hand-off**: Integrated OpenCV (preprocessing entry point) with Pillow (RGBA rendering presentation layer) without duplicating the pixel matrices in memory, saving system RAM.

---

## 📊 Next Session (Week 1 — Day 3)
- **Goal**: Multi-line label parsing and dense layout exploration.
- **Focus**: We will take the single-box logic we wrote today and write a loop-driven file stream parser to read and draw dozens of overlapping packages over a busy retail shelf image from the SKU-110K dataset.
