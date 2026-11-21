"""
Central Multi-Modal Pipeline Orchestrator.
Hardened with multi-frame max-pooling to eliminate camera transit and occlusion blindness.
"""
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vision.inference import ShelfInventoryAuditor
from src.voice.transcriber import VoiceInterviewTranscriber
from src.slm.reasoner import CreditUnderwritingEngine
from src.slm.schema import CreditUnderwritingDecision


class CreditAuditPipeline:
    def __init__(self,
                 onnx_model_path=None,
                 whisper_size="tiny",
                 local_slm="qwen2.5:1.5b",
                 cloud_slm="llama-3.1-8b-instant"):
        if onnx_model_path is None:
            onnx_model_path = os.path.join(ROOT_DIR, "models", "best.onnx")

        self.vision_engine = ShelfInventoryAuditor(onnx_model_path=onnx_model_path)
        self.voice_engine = VoiceInterviewTranscriber(model_size=whisper_size, compute_type="int8")
        self.slm_engine = CreditUnderwritingEngine(local_model=local_slm, cloud_model=cloud_slm)

    def sample_video_frames(self, video_path, num_samples=5):
        """
        Samples distributed frames across the video to capture different store aisles.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid or unreadable video file: {video_path}")

        indices = np.linspace(int(total_frames * 0.10), int(total_frames * 0.90), num_samples, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        if not frames:
            raise ValueError("Could not extract any valid video frames from file.")
        return frames

    def annotate_frame(self, bgr_image, conf_threshold=0.25):
        """
        Draws green semi-transparent bounding boxes using NMS.
        """
        if bgr_image is None or bgr_image.size == 0:
            bgr_image = np.zeros((640, 640, 3), dtype=np.uint8)

        tensor, (orig_h, orig_w) = self.vision_engine.preprocess(bgr_image)
        outputs = self.vision_engine.session.run(None, {self.vision_engine.input_name: tensor})
        preds = np.squeeze(outputs[0]).T

        boxes, scores = [], []
        scale_x = orig_w / float(self.vision_engine.input_width)
        scale_y = orig_h / float(self.vision_engine.input_height)

        for pred in preds:
            conf = float(pred[4])
            if conf >= conf_threshold:
                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                x1 = int((cx - (w / 2.0)) * scale_x)
                y1 = int((cy - (h / 2.0)) * scale_y)
                box_w = int(w * scale_x)
                box_h = int(h * scale_y)
                boxes.append([x1, y1, box_w, box_h])
                scores.append(conf)

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_threshold, nms_threshold=0.45)

        pil_img = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        detected_count = 0
        if len(indices) > 0:
            flat_indices = np.array(indices).flatten()
            detected_count = len(flat_indices)
            for idx in flat_indices:
                bx, by, bw, bh = boxes[idx]
                draw.rectangle([bx, by, bx + bw, by + bh], fill=(0, 255, 0, 70), outline=(0, 255, 0, 255), width=3)

        return pil_img, detected_count

    def run_full_audit(self, media_path, audio_path, store_category="Grocery / Kirana (FMCG)"):
        """
        Runs full multi-modal audit with multi-frame max-pooling on video.
        """
        pipeline_start = time.perf_counter()

        # 1. Computer Vision Processing with Multi-Frame Max-Pooling
        is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        best_count = -1
        best_annotated = None

        if is_video:
            frames = self.sample_video_frames(media_path, num_samples=5)
            for frame in frames:
                annotated_img, count = self.annotate_frame(frame)
                if count > best_count:
                    best_count = count
                    best_annotated = annotated_img
        else:
            key_frame = cv2.imread(media_path)
            if key_frame is None:
                key_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            best_annotated, best_count = self.annotate_frame(key_frame)

        detected_count = max(0, best_count)
        shelf_fill_pct = min(100.0, (detected_count / 30.0) * 100.0)

        # 2. Voice Transcription
        voice_result = self.voice_engine.transcribe(audio_path)
        transcript = voice_result["transcript"]

        # 3. SLM Financial Underwriting
        slm_decision = self.slm_engine.evaluate_merchant(
            visual_items_count=detected_count,
            shelf_fill_pct=shelf_fill_pct,
            interview_transcript=transcript,
            store_category=store_category
        )

        total_latency = time.perf_counter() - pipeline_start

        return {
            "annotated_image": best_annotated,
            "detected_items": detected_count,
            "shelf_fill_pct": shelf_fill_pct,
            "transcript": transcript,
            "audio_duration_sec": voice_result["duration_sec"],
            "decision": slm_decision,
            "total_latency_sec": round(total_latency, 2)
        }
