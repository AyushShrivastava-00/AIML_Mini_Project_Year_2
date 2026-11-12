"""
ONNX Export and Post-Training Optimization Pipeline.
Converts trained PyTorch weights (.pt) to hardware-neutral ONNX format.
"""
import os
import sys
from ultralytics import YOLO

def export_to_onnx(model_path="models/best.pt", output_dir="models"):
    """
    Exports a trained YOLOv8 PyTorch model to ONNX format.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please ensure models/best.pt exists before exporting.")
        return False

    print(f"Loading trained weights from: {model_path}")
    model = YOLO(model_path)

    print("Exporting to ONNX format with input resolution (640, 640)...")
    # Export options:
    # format='onnx': Target format
    # imgsz=640: Standard input resolution
    # dynamic=False: Static graph shape for maximum edge CPU optimization
    # opset=12: Stable Open Neural Network Exchange operator set
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        opset=12
    )

    print(f"\n[SUCCESS] Model successfully exported to ONNX: {exported_path}")
    
    # Calculate file size reduction
    pt_size = os.path.getsize(model_path) / (1024 * 1024)
    onnx_path = os.path.splitext(model_path)[0] + ".onnx"
    
    if os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"PyTorch (.pt) Size: {pt_size:.2f} MB")
        print(f"ONNX (.onnx) Size:   {onnx_size:.2f} MB")
    
    return True

if __name__ == "__main__":
    export_to_onnx()
