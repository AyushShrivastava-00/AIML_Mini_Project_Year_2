"""
Multi-Modal Credit Assessment Pipeline Orchestrator.
Ingests media (image or video walkthrough with 5-frame max-pooling) and audio interview,
then calls the vision, voice, and SLM underwriting engines to produce a credit decision.
"""

import os
import cv2
import time
from src.vision.inference import ShelfInventoryAuditor
from src.voice.transcriber import VoiceInterviewTranscriber
from src.slm.reasoner import CreditUnderwritingEngine

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class CreditAuditPipeline:
    def __init__(self,
                 onnx_model_path=None,
                 whisper_size="tiny",
                 local_slm="qwen2.5:1.5b",
                 cloud_slm="openai/gpt-oss-20b",
                 groq_api_key=None):
        if onnx_model_path is None:
            onnx_model_path = os.path.join(ROOT_DIR, "models", "best.onnx")

        self.vision_engine = ShelfInventoryAuditor(onnx_model_path=onnx_model_path)
        self.voice_engine = VoiceInterviewTranscriber(model_size=whisper_size, compute_type="int8")
        self.slm_engine = CreditUnderwritingEngine(
            local_model=local_slm,
            cloud_model=cloud_slm,
            groq_api_key=groq_api_key
        )

    def extract_best_video_frame(self, video_path, num_samples=5):
        """Samples distributed frames across video and selects the frame with highest detected count."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None, 0, 0.0

        sample_indices = [int(total_frames * (i + 1) / (num_samples + 1)) for i in range(num_samples)]

        best_frame = None
        max_items = -1
        best_fill_pct = 0.0

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            annotated, count, fill_pct = self.vision_engine.audit_shelf_image(frame)
            if count > max_items:
                max_items = count
                best_frame = annotated
                best_fill_pct = fill_pct

        cap.release()

        if best_frame is None:
            # Fallback to reading first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                best_frame, max_items, best_fill_pct = self.vision_engine.audit_shelf_image(frame)

        return best_frame, max(0, max_items), best_fill_pct

    def run_full_audit(self, media_path, audio_path, store_category="Grocery / Kirana (FMCG)"):
        start_time = time.time()

        # 1. Vision Module Execution (Handles video or static image)
        ext = os.path.splitext(media_path)[-1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            annotated_img, detected_count, shelf_fill_pct = self.extract_best_video_frame(media_path)
        else:
            raw_img = cv2.imread(media_path)
            annotated_img, detected_count, shelf_fill_pct = self.vision_engine.audit_shelf_image(raw_img)

        # Convert BGR (OpenCV) to RGB (for Streamlit & PDF display)
        if annotated_img is not None:
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        else:
            annotated_rgb = None

        # 2. Voice Transcription Execution
        transcript = self.voice_engine.transcribe_interview(audio_path)

        # 3. SLM Financial Underwriting Execution
        slm_decision = self.slm_engine.evaluate_merchant(
            visual_items_count=detected_count,
            shelf_fill_pct=shelf_fill_pct,
            interview_transcript=transcript,
            store_category=store_category
        )

        total_latency = round(time.time() - start_time, 2)

        return {
            "annotated_image": annotated_rgb,
            "detected_items": detected_count,
            "shelf_fill_pct": shelf_fill_pct,
            "transcript": transcript,
            "decision": slm_decision,
            "total_latency_sec": total_latency
        }
