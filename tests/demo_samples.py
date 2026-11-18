"""
Generates 3 realistic evaluation demo test fixtures:
- Case 1: Ramesh Kumar (Flourishing Kirana - Low Risk / Approved)
- Case 2: Anita Verma (Small Tea/Snack Kiosk - Moderate Risk / Manual Review)
- Case 3: Vikram Singh (Severe Fraud / Empty Shelves vs Claimed ₹20k/day)
"""
import os
import cv2
import numpy as np
from gtts import gTTS
import librosa
import soundfile as sf

def make_shelf_image(filename, num_items, shelf_color=(70, 60, 50)):
    os.makedirs("data/raw", exist_ok=True)
    img = np.ones((640, 640, 3), dtype=np.uint8) * 55  # Dark store ambient
    
    # Planks
    cv2.line(img, (0, 210), (640, 210), shelf_color, 12)
    cv2.line(img, (0, 430), (640, 430), shelf_color, 12)
    
    np.random.seed(num_items)
    items_drawn = 0
    
    # Draw items across 3 tiers
    for shelf_y in [140, 350, 560]:
        for x in range(50, 600, 55):
            if items_drawn >= num_items:
                break
            box_w = np.random.randint(40, 50)
            box_h = np.random.randint(70, 95)
            color = (int(np.random.randint(70, 240)), int(np.random.randint(50, 210)), int(np.random.randint(50, 220)))
            
            x1, y1 = x - box_w // 2, shelf_y - box_h // 2
            x2, y2 = x + box_w // 2, shelf_y + box_h // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
            cv2.line(img, (x1 + 3, y1 + 15), (x2 - 3, y1 + 15), (240, 240, 240), 2)
            items_drawn += 1
            
    cv2.imwrite(filename, img)
    print(f"✅ Generated shelf image ({num_items} items): {filename}")

def make_audio_clip(filename, speech_text):
    os.makedirs("data/raw", exist_ok=True)
    temp_mp3 = "data/raw/temp.mp3"
    tts = gTTS(text=speech_text, lang='en', tld='co.in')
    tts.save(temp_mp3)
    
    y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
    sf.write(filename, y, 16000)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    print(f"✅ Generated 16kHz audio: {filename}")

def build_all_demo_samples():
    print("=" * 60)
    print("  GENERATING MULTI-MODAL DEMO SAMPLES FOR EVALUATION")
    print("=" * 60)
    
    # ── CASE 1: Ramesh Kumar (Flourishing Kirana) ────────────────
    # Dense shelf (30 visible packages) + ₹3,500 daily turnover
    make_shelf_image("data/raw/case1_ramesh_shelf.jpg", num_items=30)
    case1_text = (
        "Namaste sir. My name is Ramesh Kumar. I run this grocery store for five years. "
        "My daily sales are around 3500 rupees. Every Wednesday I pay my supplier 15000 rupees "
        "to restock flour, grains, and spices. My shop rent is 4000 rupees."
    )
    make_audio_clip("data/raw/case1_ramesh_interview.wav", case1_text)

    # ── CASE 2: Anita Verma (Small Kiosk / Moderate Risk) ────────
    # Moderate shelf (14 visible items) + ₹1,500 daily turnover
    make_shelf_image("data/raw/case2_anita_shelf.jpg", num_items=14)
    case2_text = (
        "Hello sir, my name is Anita Verma. I run a small tea and snack stall. "
        "My daily sales are around 1500 rupees. I buy milk and biscuits every two days "
        "for 1800 rupees. My kiosk space fee is 1200 rupees per month."
    )
    make_audio_clip("data/raw/case2_anita_interview.wav", case2_text)

    # ── CASE 3: Vikram Singh (Severe Fraud / Empty Shelves) ──────
    # Bare shelf (only 4 items!) but claiming inflated ₹20,000/day turnover!
    make_shelf_image("data/raw/case3_vikram_shelf.jpg", num_items=4)
    case3_text = (
        "Sir, my name is Vikram Singh. I do massive business here. "
        "My daily sales are 20000 rupees every single day. I need a loan of 500000 rupees immediately."
    )
    make_audio_clip("data/raw/case3_vikram_interview.wav", case3_text)
    print("=" * 60)
    print("All demo test fixtures generated successfully in data/raw/!")

if __name__ == "__main__":
    build_all_demo_samples()
