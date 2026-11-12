"""
Generates a realistic spoken merchant interview audio file using TTS.
Saved to: data/raw/merchant_interview.wav
"""
import os

def generate_sample_interview(output_wav="data/raw/merchant_interview.wav"):
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    
    # Realistic spoken disclosure by a Kirana store merchant
    interview_text = (
        "Namaste sir. My name is Suresh Patel. I run this grocery store. "
        "My daily sales are around 2500 rupees. Every Monday, I pay my wholesale supplier "
        "10000 rupees to restock rice, pulses, and oil. My shop rent is 3000 rupees per month."
    )
    
    print("Synthesizing realistic spoken interview audio...")
    from gtts import gTTS
    import librosa
    import soundfile as sf
    
    temp_mp3 = "data/raw/temp_interview.mp3"
    tts = gTTS(text=interview_text, lang='en', tld='co.in')  # Indian English accent
    tts.save(temp_mp3)
    
    # Convert to standard 16kHz mono PCM .wav
    y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
    sf.write(output_wav, y, 16000)
    
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
        
    print(f"✅ Generated spoken interview clip: {output_wav} (16kHz Mono PCM)")
    return output_wav

if __name__ == "__main__":
    generate_sample_interview()
