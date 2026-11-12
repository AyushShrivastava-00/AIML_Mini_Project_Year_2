"""
Central Pipeline Orchestrator for Micro-Merchant Credit Assessment Engine.
Integrates Vision (ONNX YOLOv8), Voice (Whisper.cpp), and SLM Underwriting (Qwen-2.5 via Ollama).
"""
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.inference import ShelfInventoryAuditor
from src.voice.transcriber import VoiceInterviewTranscriber
from src.slm.reasoner import CreditUnderwritingEngine
from src.slm.schema import CreditUnderwritingDecision

class CreditAuditPipeline:
    def __init__(self, 
                 onnx_model_path="models/best.onnx",
                 whisper_size="tiny",
                 slm_model="qwen2.5:1.5b"):
        print("Initializing Multi-Modal Credit Audit Pipeline...")
        self.vision_engine = ShelfInventoryAuditor(onnx_model_path=onnx_model_path)
        self.voice_engine = VoiceInterviewTranscriber(model_size=whisper_size, compute_type="int8")
        self.slm_engine = CreditUnderwritingEngine(model_name=slm_model)
        print("✅ Pipeline fully loaded and operational on edge CPU!")

    def sample_video_frames(self, video_path, num_samples=3):
        """
        Extracts key representative frames from a shelf video walkthrough.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid or corrupted video file: {video_path}")

        frame_indices = np.linspace(int(total_frames * 0.15), int(total_frames * 0.85), num_samples, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def annotate_frame(self, bgr_image, conf_threshold=0.25):
        """
        Draws green semi-transparent bounding boxes around detected shelf items using NMS.
        """
        tensor, (orig_h, orig_w) = self.vision_engine.preprocess(bgr_image)
        outputs = self.vision_engine.session.run(None, {self.vision_engine.input_name: tensor})
        preds = np.squeeze(outputs[0]).T

        boxes, scores = [], []
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        for pred in preds:
            conf = float(pred[4])  # Corrected: Index 4 is confidence
            if conf >= conf_threshold:
                cx = float(pred[0])  # Index 0 is center X
                cy = float(pred[1])  # Index 1 is center Y
                w = float(pred[2])   # Index 2 is width
                h = float(pred[3])   # Index 3 is height

                x1 = int((cx - (w / 2.0)) * scale_x)
                y1 = int((cy - (h / 2.0)) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)

                boxes.append([x1, y1, box_w, box_h])
                scores.append(conf)

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_threshold, nms_threshold=0.45)
        
        pil_img = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        if len(indices) > 0:
            flat_indices = np.array(indices).flatten()
            for idx in flat_indices:
                bx, by, bw, bh = boxes[idx]
                draw.rectangle([bx, by, bx + bw, by + bh], fill=(0, 255, 0, 70), outline=(0, 255, 0, 255), width=3)

        return pil_img, len(indices)

    def run_full_audit(self, media_path, audio_path):
        """
        Executes end-to-end multi-modal audit:
        Supports either a video file (.mp4) or a single image (.jpg) for visual inspection.
        """
        pipeline_start = time.perf_counter()

        # 1. Vision Processing
        is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        if is_video:
            frames = self.sample_video_frames(media_path, num_samples=3)
            key_frame = frames[len(frames) // 2]
        else:
            key_frame = cv2.imread(media_path)

        annotated_image, detected_count = self.annotate_frame(key_frame)
        shelf_fill_pct = min(100.0, (detected_count / 30.0) * 100.0)

        # 2. Voice Processing
        voice_result = self.voice_engine.transcribe(audio_path)
        transcript = voice_result["transcript"]

        # 3. SLM Financial Reasoning
        slm_decision = self.slm_engine.evaluate_merchant(
            visual_items_count=detected_count,
            shelf_fill_pct=shelf_fill_pct,
            interview_transcript=transcript
        )

        total_latency = time.perf_counter() - pipeline_start

        return {
            "annotated_image": annotated_image,
            "detected_items": detected_count,
            "shelf_fill_pct": shelf_fill_pct,
            "transcript": transcript,
            "audio_duration_sec": voice_result["duration_sec"],
            "decision": slm_decision,
            "total_latency_sec": round(total_latency, 2)
        }

if __name__ == "__main__":
    print("CreditAuditPipeline compiled successfully!")
