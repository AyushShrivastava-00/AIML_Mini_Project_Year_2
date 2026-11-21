"""
Edge CPU Inference Engine for Micro-Merchant Shelf Inventory Auditing.
Executes purely on standard CPU using ONNX Runtime with NMS suppression.
"""
import time
import os
import cv2
import numpy as np


class ShelfInventoryAuditor:
    def __init__(self, onnx_model_path="models/best.onnx", conf_threshold=0.25, iou_threshold=0.45):
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
        self.iou_threshold = iou_threshold

        # Get model input layer metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
        self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640

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
        Returns NMS-filtered item count, shelf fill %, and inference latency.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image file at {image_path}")

        tensor, (orig_h, orig_w) = self.preprocess(img)

        # Benchmark inference latency
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Post-process YOLO outputs: output shape is [1, 5, 8400]
        preds = np.squeeze(outputs[0]).T  # Shape: [8400, 5]

        boxes = []
        scores = []
        scale_x = orig_w / float(self.input_width)
        scale_y = orig_h / float(self.input_height)

        for pred in preds:
            conf = float(pred[4])
            if conf >= self.conf_threshold:
                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                x1 = int((cx - (w / 2.0)) * scale_x)
                y1 = int((cy - (h / 2.0)) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)
                boxes.append([x1, y1, box_w, box_h])
                scores.append(conf)

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.conf_threshold, nms_threshold=self.iou_threshold)
        item_count = len(indices) if len(indices) > 0 else 0

        # High retail density benchmark: 30 items per primary shelf view = 100% full
        BENCHMARK_CAPACITY = 30.0
        shelf_fill_pct = min(100.0, (item_count / BENCHMARK_CAPACITY) * 100.0)

        return {
            "item_count": int(item_count),
            "shelf_fill_pct": round(float(shelf_fill_pct), 1),
            "latency_ms": round(float(latency_ms), 1),
            "edge_compliant": bool(latency_ms < 200.0)
        }
