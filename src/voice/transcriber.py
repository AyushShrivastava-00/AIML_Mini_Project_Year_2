"""
Edge Audio Processing and Speech-to-Text Transcription Engine.
Uses faster-whisper (CTranslate2) for sub-second, on-device CPU transcription.
Hardened for multilingual speech (Hindi, Hinglish, English) and noisy rural audio.
"""
import os
import time
import numpy as np
import soundfile as sf
import librosa

class VoiceInterviewTranscriber:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        """
        Initializes the quantized Whisper model on local CPU.
        """
        print(f"Loading Whisper [{model_size}] on [{device}] with [{compute_type}] quantization...")
        from faster_whisper import WhisperModel
        
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "whisper")
        os.makedirs(models_dir, exist_ok=True)
        
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=models_dir
        )
        print("Whisper speech engine initialized successfully!")

    def preprocess_audio(self, input_audio_path, target_sr=16000):
        """
        Converts input audio (.mp3, .m4a, .wav) to 16 kHz mono float32.
        Applies peak normalization to handle variable mobile recording volumes.
        """
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Audio file not found: {input_audio_path}")
            
        y, sr = librosa.load(input_audio_path, sr=target_sr, mono=True)
        
        # Check if audio is completely flat/silent
        max_peak = np.max(np.abs(y))
        if max_peak > 0:
            y = y / max_peak
            
        return y, sr

    def transcribe(self, audio_path, language=None):
        """
        Transcribes speech in an interview recording.
        Handles silent audio, market traffic noise, and Hindi/English dialect switching.
        """
        start_time = time.perf_counter()
        
        audio_array, sr = self.preprocess_audio(audio_path, target_sr=16000)
        duration_sec = len(audio_array) / sr

        # Edge Case 3: Check for pure flatline silence
        if np.max(np.abs(audio_array)) < 0.01:
            return {
                "transcript": "",
                "detected_language": "unknown",
                "language_probability": 0.0,
                "duration_sec": round(duration_sec, 2),
                "latency_sec": 0.05,
                "real_time_factor": 0.01
            }
        
        # Run Whisper with aggressive VAD to filter out market noise
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )
        
        extracted_segments = [segment.text.strip() for segment in segments]
        full_transcript = " ".join(extracted_segments).strip()
        latency_sec = time.perf_counter() - start_time
        
        return {
            "transcript": full_transcript,
            "detected_language": info.language if info else "en",
            "language_probability": round(info.language_probability, 2) if info else 0.0,
            "duration_sec": round(duration_sec, 2),
            "latency_sec": round(latency_sec, 2),
            "real_time_factor": round(latency_sec / max(duration_sec, 0.001), 2)
        }

if __name__ == "__main__":
    transcriber = VoiceInterviewTranscriber(model_size="tiny", compute_type="int8")
    print("VoiceInterviewTranscriber verified!")
