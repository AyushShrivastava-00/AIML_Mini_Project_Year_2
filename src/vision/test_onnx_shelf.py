"""
Verifies the exported best.onnx model on local shelf images.
Draws detected boxes and prints accuracy metrics.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def test_onnx_detection(
    model_path="models/best.onnx",
    image_path="data/raw/dense_shelf.jpg",
    conf_thresh=0.25,
    iou_thresh=0.45
):
    import onnxruntime as ort

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load ONNX model on CPU
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 2. Load and preprocess image
    orig_bgr = cv2.imread(image_path)
    h_orig, w_orig = orig_bgr.shape[:2]

    # Resize to 640x640, convert BGR to RGB, normalize to [0.0, 1.0]
    resized = cv2.resize(orig_bgr, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))   # [H, W, C] -> [C, H, W]
    tensor = np.expand_dims(tensor, axis=0)    # Add batch: [1, C, H, W]

    # 3. Run ONNX Inference
    outputs = session.run(None, {input_name: tensor})
    
    # YOLOv8 output shape is [1, 5, 8400] -> squeeze to [5, 8400] -> transpose to [8400, 5]
    preds = np.squeeze(outputs[0]).T  
    
    # 4. Filter by confidence threshold
    boxes = []
    scores = []
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    for pred in preds:
        conf = float(pred[4])  # Index 4 is the confidence score
        if conf >= conf_thresh:
            cx = float(pred[0])
            cy = float(pred[1])
            w = float(pred[2])
            h = float(pred[3])

            # Convert (cx, cy, w, h) in 640x640 space back to original image dimensions
            x1 = int((cx - (w / 2.0)) * scale_x)
            y1 = int((cy - (h / 2.0)) * scale_y)
            box_w = int(w * scale_x)
            box_h = int(h * scale_y)

            boxes.append([x1, y1, box_w, box_h])
            scores.append(conf)

    # 5. Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thresh, nms_threshold=iou_thresh)

    detected_count = len(indices) if len(indices) > 0 else 0
    print("=" * 60)
    print("  ONNX LOCAL SHELF TEST RESULTS")
    print("=" * 60)
    print(f"Model Evaluated:        {model_path}")
    print(f"Test Image:             {image_path} ({w_orig}x{h_orig})")
    print(f"Raw Box Candidates:     {len(boxes)}")
    print(f"Filtered Detections:    {detected_count} retail items")
    print(f"Estimated Shelf Fill:   {min(100.0, (detected_count / 30.0) * 100):.1f}%")
    print("=" * 60)

    # 6. Draw visual boxes with Pillow
    pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    if len(indices) > 0:
        # Flatten handles different OpenCV versions (1D vs 2D array returns)
        flat_indices = np.array(indices).flatten()
        for idx in flat_indices:
            bx, by, bw, bh = boxes[idx]
            draw.rectangle(
                [bx, by, bx + bw, by + bh],
                fill=(0, 255, 0, 70),       # semi-transparent green
                outline=(0, 255, 0, 255),   # solid green border
                width=3
            )

    output_preview = "data/raw/onnx_test_output.jpg"
    pil_img.save(output_preview)
    print(f"Visual validation saved to: {output_preview}")

if __name__ == "__main__":
    test_onnx_detection()
