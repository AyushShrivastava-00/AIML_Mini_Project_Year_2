"""
Edge-Case Stress Test Generator:
- Case 4: Manoj Gupta (Pure Hindi Voice Input)
- Case 5: Rajesh Sharma (Dead Stock / Stagnant Inventory - Full shelf, ₹300 sales)
- Case 6: Silent / Noisy Audio (Market traffic with inaudible speech)
- Case 7: Sunita Devi (Range Disclosure: '₹2,000 to ₹4,000 daily')
- Case 8: Empty Shop (0 items detected on bare shelves)
"""
import os
import cv2
import numpy as np
from gtts import gTTS
import librosa
import soundfile as sf

def make_shelf(filename, num_items):
    os.makedirs("data/raw", exist_ok=True)
    img = np.ones((640, 640, 3), dtype=np.uint8) * 50
    cv2.line(img, (0, 210), (640, 210), (75, 65, 55), 12)
    cv2.line(img, (0, 430), (640, 430), (75, 65, 55), 12)
    
    np.random.seed(num_items + 7)
    items_drawn = 0
    for shelf_y in [140, 350, 560]:
        for x in range(50, 600, 55):
            if items_drawn >= num_items:
                break
            w = np.random.randint(40, 50)
            h = np.random.randint(70, 95)
            color = (int(np.random.randint(80, 240)), int(np.random.randint(50, 210)), int(np.random.randint(50, 210)))
            cv2.rectangle(img, (x - w//2, shelf_y - h//2), (x + w//2, shelf_y + h//2), color, -1)
            cv2.rectangle(img, (x - w//2, shelf_y - h//2), (x + w//2, shelf_y + h//2), (20, 20, 20), 2)
            items_drawn += 1
            
    cv2.imwrite(filename, img)
    print(f"✅ Generated shelf: {filename} ({num_items} items)")

def make_audio(filename, text, lang='en'):
    os.makedirs("data/raw", exist_ok=True)
    temp_mp3 = "data/raw/temp_edge.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_mp3)
    y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
    sf.write(filename, y, 16000)
    if os.path.exists(temp_mp3): os.remove(temp_mp3)
    print(f"✅ Generated audio ({lang}): {filename}")

def build_all_edge_cases():
    print("=" * 65)
    print("  GENERATING COMPREHENSIVE EDGE-CASE TEST FIXTURES")
    print("=" * 65)

    # ── CASE 4: Pure Hindi Input (Manoj Gupta) ─────────────────────────
    make_shelf("data/raw/case4_hindi_shelf.jpg", num_items=28)
    hindi_text = "नमस्ते सर, मेरा नाम मनोज गुप्ता है। मेरी किराना की दुकान है। रोज़ की बिक्री लगभग चार हज़ार रुपये है। हर सोमवार को हम पंद्रह हज़ार रुपये का माल मंगाते हैं।"
    make_audio("data/raw/case4_hindi_audio.wav", hindi_text, lang='hi')

    # ── CASE 5: Dead Stock (Stagnant Inventory) ─────────────────────────
    make_shelf("data/raw/case5_deadstock_shelf.jpg", num_items=30)
    deadstock_text = "Hello sir, my name is Rajesh Sharma. Market is very slow, daily sales are only 300 rupees. Items are sitting on shelves for months."
    make_audio("data/raw/case5_deadstock_audio.wav", deadstock_text, lang='en')

    # ── CASE 6: Ambient Noise / No Speech ──────────────────────────────
    make_shelf("data/raw/case6_noise_shelf.jpg", num_items=18)
    ambient_noise = np.random.normal(0, 0.02, 16000 * 4).astype(np.float32)
    sf.write("data/raw/case6_noise_audio.wav", ambient_noise, 16000)
    print("✅ Generated noise/no-speech audio: data/raw/case6_noise_audio.wav")

    # ── CASE 7: Range Claim ('₹2,000 to ₹4,000 daily') ─────────────────
    make_shelf("data/raw/case7_range_shelf.jpg", num_items=22)
    range_text = "Hello sir, my name is Sunita Devi. My daily sales vary between 2000 to 4000 rupees depending on customer footfall."
    make_audio("data/raw/case7_range_audio.wav", range_text, lang='en')

    # ── CASE 8: Completely Empty Shop (0 items detected) ───────────────
    make_shelf("data/raw/case8_empty_shelf.jpg", num_items=0)
    empty_text = "Sir, my name is Imran Khan. I make 5000 rupees daily, but my shop is completely empty right now."
    make_audio("data/raw/case8_empty_audio.wav", empty_text, lang='en')

    print("=" * 65)
    print("All 5 edge-case test fixtures generated successfully in data/raw/!")

if __name__ == "__main__":
    build_all_edge_cases()
