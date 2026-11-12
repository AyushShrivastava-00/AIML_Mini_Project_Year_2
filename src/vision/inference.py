"""
Edge CPU Inference Engine for Micro-Merchant Shelf Inventory Auditing.
Executes purely on standard CPU using ONNX Runtime.
"""
import time
import os
import cv2
import numpy as np

class ShelfInventoryAuditor:
    def __init__(self, onnx_model_path="models/best.onnx", conf_threshold=0.25):
        """
        Initializes the ONNX Runtime CPU inference session.
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
        
        import onnxruntime as ort
        
        # Configure purely CPU-based execution providers
        self.session = ort.InferenceSession(
            onnx_model_path, 
            providers=['CPUExecutionProvider']
        )
        self.conf_threshold = conf_threshold
        
        # Get model input layer metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess(self, bgr_image):
        """
        Prepares a raw image matrix for the neural network:
        Resizes to 640x640, swaps BGR to RGB, normalizes [0, 255] to [0.0, 1.0],
        and transposes from [H, W, C] to [1, C, H, W].
        """
        h_orig, w_orig = bgr_image.shape[:2]
        
        # Resize to fixed network input dimension
        resized = cv2.resize(bgr_image, (self.input_width, self.input_height))
        
        # Swap BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixels from uint8 [0, 255] to float32 [0.0, 1.0]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose [H, W, C] -> [C, H, W]
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension: [1, C, H, W]
        tensor = np.expand_dims(transposed, axis=0)
        return tensor, (h_orig, w_orig)

    def audit_shelf(self, image_path):
        """
        Performs full inventory audit on a store shelf image:
        Returns item count, estimated shelf fill percentage, and inference latency.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        tensor, (orig_h, orig_w) = self.preprocess(img)

        # Benchmark inference latency
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Post-process YOLO outputs: output shape is [1, 5, 8400]
        # (cx, cy, w, h, confidence) across 8400 candidate anchor locations
        predictions = np.squeeze(outputs[0]).T  # Shape: [8400, 5]
        
        scores = predictions[:, 4]
        valid_mask = scores >= self.conf_threshold
        valid_predictions = predictions[valid_mask]
        
        item_count = len(valid_predictions)
        
        # Calculate Shelf Density / Fill Percentage
        # High retail density benchmark: ~50 items per primary shelf view = 100% full
        BENCHMARK_CAPACITY = 50.0
        shelf_fill_pct = min(100.0, (item_count / BENCHMARK_CAPACITY) * 100.0)

        return {
            "item_count": int(item_count),
            "shelf_fill_pct": round(float(shelf_fill_pct), 1),
            "latency_ms": round(float(latency_ms), 1),
            "edge_compliant": bool(latency_ms < 200.0)
        }

if __name__ == "__main__":
    print("=" * 60)
    print("  EDGE CPU INFERENCE BENCHMARK - MICRO-MERCHANT ENGINE")
    print("=" * 60)
    
    # Test on our local raw image sample
    sample_img = "data/raw/image_001.jpg"
    model_file = "models/best.onnx"
    
    if os.path.exists(sample_img) and os.path.exists(model_file):
        auditor = ShelfInventoryAuditor(onnx_model_path=model_file)
        results = auditor.audit_shelf(sample_img)
        
        print(f"Input Image:             {sample_img}")
        print(f"Detected Retail Items:   {results['item_count']}")
        print(f"Estimated Shelf Fill:    {results['shelf_fill_pct']}%")
        print(f"CPU Inference Latency:   {results['latency_ms']} ms")
        print(f"Sub-200ms SLO Met:       {'✅ YES' if results['edge_compliant'] else '❌ NO'}")
        print("=" * 60)
    else:
        print("Missing sample image or ONNX model file. Verify paths.")
