"""
Verifies the end-to-end voice transcription pipeline and latency.
"""
import os
import sys

# Ensure project root is always in sys.path regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.voice.transcriber import VoiceInterviewTranscriber

def test_interview_transcription():
    audio_file = "data/raw/merchant_interview.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found. Run create_sample_audio.py first.")
        return
        
    print("=" * 65)
    print(f"  PROCESSING INTERVIEW AUDIO: {audio_file}")
    print("=" * 65)
    
    # 1. Initialize engine
    transcriber = VoiceInterviewTranscriber(model_size="tiny", compute_type="int8")
    
    # 2. Run transcription
    result = transcriber.transcribe(audio_file)
    
    print(f"\nDuration:           {result['duration_sec']} seconds")
    print(f"Transcription Time: {result['latency_sec']} seconds")
    print(f"Real-Time Factor:   {result['real_time_factor']}x ( < 1.0 means faster than real-time!)")
    print(f"Detected Language:  {result['detected_language']} (Confidence: {result['language_probability']})")
    print(f"\nTranscribed Text:\n\"{result['transcript']}\"")
    print("=" * 65)

if __name__ == "__main__":
    test_interview_transcription()
