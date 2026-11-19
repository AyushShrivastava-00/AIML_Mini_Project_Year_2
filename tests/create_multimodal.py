"""
Multi-Modal Evaluation Fixture Generator:
Generates three 15-second MP4 video walkthroughs of store shelves (dynamic moving frames)
paired with multilingual spoken audio interviews using gTTS and OpenCV.
"""
import os
import cv2
import numpy as np
from gtts import gTTS
import librosa
import soundfile as sf

def generate_dynamic_shelf_video(filename, item_count, duration_sec=15, fps=24):
    """
    Simulates a 15-second smartphone camera pan across grocery shelves.
    Generates a real, playable .mp4 video with 360 frames of moving retail boxes.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    width, height = 640, 640
    total_frames = duration_sec * fps
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    np.random.seed(item_count)
    
    # Pre-generate 30 distinct retail boxes with vibrant packaging
    boxes = []
    for _ in range(item_count):
        box_w = np.random.randint(40, 50)
        box_h = np.random.randint(70, 95)
        color = (int(np.random.randint(80, 240)), int(np.random.randint(50, 210)), int(np.random.randint(50, 210)))
        boxes.append((box_w, box_h, color))
        
    # Generate each frame simulating a smooth pan (panning offset moves x horizontally)
    for frame_idx in range(total_frames):
        # Calculate horizontal camera panning offset (smooth sinus wave pan)
        pan_offset = int(120 * np.sin(2 * np.pi * 0.5 * (frame_idx / total_frames)))
        
        # Draw background shelf
        frame = np.ones((height, width, 3), dtype=np.uint8) * 55  # dark shop ambiance
        
        # Panned horizontal shelves
        cv2.line(frame, (0, 210), (640, 210), (75, 65, 55), 12)
        cv2.line(frame, (0, 430), (640, 430), (75, 65, 55), 12)
        
        items_drawn = 0
        for tier_y in [140, 350, 560]:
            # Apply pan offset directly to the items' horizontal positions
            for base_x in range(50, 600, 55):
                if items_drawn >= item_count:
                    break
                
                # Apply panning shift
                x_center = base_x + pan_offset
                
                box_w, box_h, color = boxes[items_drawn]
                x1, y1 = x_center - box_w // 2, tier_y - box_h // 2
                x2, y2 = x_center + box_w // 2, tier_y + box_h // 2
                
                # Keep bounding drawing within safe canvas boundaries
                if -50 < x_center < width + 50:
                    cv2.rectangle(frame, (max(0, x1), y1), (min(width, x2), y2), color, -1)
                    cv2.rectangle(frame, (max(0, x1), y1), (min(width, x2), y2), (20, 20, 20), 2)
                    cv2.line(frame, (max(0, x1) + 3, y1 + 15), (min(width, x2) - 3, y1 + 15), (240, 240, 240), 2)
                    
                items_drawn += 1
                
        out.write(frame)
        
    out.release()
    print(f"✅ Generated 15s Walkthrough Video ({item_count} items): {filename}")

def generate_voice_clip(filename, speech_text, lang='en'):
    """
    Generates a clean 16kHz mono .wav PCM speech file from text.
    Resilient against 'translate.google.None' domain errors.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    temp_mp3 = "data/raw/temp_voice.mp3"
    
    # Ensure tld defaults cleanly to 'com' to prevent domain parsing failures
    tld = 'co.in' if lang == 'en' else 'com'
    
    from gtts import gTTS
    tts = gTTS(text=speech_text, lang=lang, tld=tld)
    tts.save(temp_mp3)
    
    # Convert and resample to standard 16,000 Hz Mono PCM Wav for Whisper
    y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
    sf.write(filename, y, 16000)
    
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    print(f"✅ Generated 16kHz Mono Voice Clip ({lang}): {filename}")


def build_multimodal_fixtures():
    print("=" * 70)
    print("  BUILDING DYNAMIC MULTI-MODAL 15s VIDEO & MULTILINGUAL AUDIO FIXTURES  ")
    print("=" * 70)
    
    # ── CASE 1: Manoj Gupta (Hindi / Approved / Low Risk) ────────────────────
    # Fully stocked dynamically panning video (28 items) paired with Hindi interview
    generate_dynamic_shelf_video("data/raw/case4_hindi_walkthrough.mp4", item_count=28)
    case4_hindi_speech = (
        "Namaste sir, mera naam Manoj Gupta hai. Meri kirana ki dukana hai. "
        "Mera roz ka dhandha lagbhag char hazar rupees hai. Har somwar ko hum "
        "pandrah hazar rupees ka naya mal mangate hain, aur dukana ka kiraya teen hazar rupees hai."
    )
    generate_voice_clip("data/raw/case4_hindi_interview.wav", case4_hindi_speech, lang='hi')
    
    # ── CASE 5: Rajesh Sharma (Dead Stock / Stagnant Capital / Moderate Risk) ─
    # Well-stocked panning video (30 items) but stagnant sales (only ₹300/day!)
    generate_dynamic_shelf_video("data/raw/case5_deadstock_walkthrough.mp4", item_count=30)
    case5_speech = (
        "Hello sir, my name is Rajesh Sharma. I run this grocery shop. "
        "The market is extremely slow right now. My daily sales are just 300 rupees. "
        "All the stock on my shelves is sitting idle for months with no buyers. I need help."
    )
    generate_voice_clip("data/raw/case5_deadstock_interview.wav", case5_speech, lang='en')

    # ── CASE 7: Sunita Devi (Dynamic Range Claim / Approved / Low Risk) ──────
    # Moderately stocked panning video (22 items) paired with sales range claim
    generate_dynamic_shelf_video("data/raw/case7_range_walkthrough.mp4", item_count=22)
    case7_speech = (
        "Hello sir, my name is Sunita Devi. I operate this stall. "
        "My daily turnover fluctuates, ranging from 2000 to 4000 rupees depending on customer footfall. "
        "I restock twice a week for 8000 rupees."
    )
    generate_voice_clip("data/raw/case7_range_interview.wav", case7_speech, lang='en')
    
    print("=" * 70)
    print("ALL MULTI-MODAL 15s VIDEO & MULTILINGUAL AUDIO FIXTURES READY!")
    print("Files located in: data/raw/")
    print("=" * 70)

if __name__ == "__main__":
    build_multimodal_fixtures()
