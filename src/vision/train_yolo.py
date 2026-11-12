"""
YOLOv8-Nano Training Reference Script for Retail Item Detection.
Executed on Google Colab NVIDIA Tesla T4 GPU Runtime.
"""
import os
from ultralytics import YOLO

def train_model():
    # 1. Load the lightweight Nano architecture pre-trained on general visual features
    model = YOLO('yolov8n.pt')
    
    # 2. Fine-tune on retail shelf dataset configuration
    data_config = os.path.join('dataset', 'data.yaml')
    
    results = model.train(
        data=data_config,
        epochs=15,
        imgsz=640,
        batch=8,
        name='retail_shelf_yolov8n',
        device=0  # Uses NVIDIA CUDA GPU
    )
    return results

if __name__ == '__main__':
    print("This script is designed to run in a GPU-accelerated environment (e.g. Google Colab).")
