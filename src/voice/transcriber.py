"""
Edge Audio Processing and Speech-to-Text Transcription Engine.
Uses faster-whisper (CTranslate2) for sub-second, on-device CPU transcription.
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
        Model sizes: 'tiny' (~39M params, fastest on edge CPU), 'base' (~74M params).
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
            
        # librosa automatically converts stereo to mono and resamples to target_sr
        y, sr = librosa.load(input_audio_path, sr=target_sr, mono=True)
        
        # Peak normalization to prevent audio clipping
        max_peak = np.max(np.abs(y))
        if max_peak > 0:
            y = y / max_peak
            
        return y, sr

    def transcribe(self, audio_path, language=None):
        """
        Transcribes speech in an interview recording.
        Returns:
            dict containing transcript, detected language, duration, and latency.
        """
        start_time = time.perf_counter()
        
        # 1. Preprocess audio to 16kHz float32 array
        audio_array, sr = self.preprocess_audio(audio_path, target_sr=16000)
        duration_sec = len(audio_array) / sr
        
        # 2. Run Whisper beam-search decoding with voice activity detection (VAD) filter
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        full_transcript = " ".join([segment.text.strip() for segment in segments])
        latency_sec = time.perf_counter() - start_time
        
        return {
            "transcript": full_transcript,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration_sec": round(duration_sec, 2),
            "latency_sec": round(latency_sec, 2),
            "real_time_factor": round(latency_sec / max(duration_sec, 0.001), 2)
        }

if __name__ == "__main__":
    transcriber = VoiceInterviewTranscriber(model_size="tiny", compute_type="int8")
    print("VoiceInterviewTranscriber class verified and ready!")
