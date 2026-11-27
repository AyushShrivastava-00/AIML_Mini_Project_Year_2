"""
Complete Multi-Modal Test Suite Generator (100% Feature Coverage)
Generates:
1. Dense, Bare, Kiosk, and Empty Shelf Images (.jpg)
2. Synthetic Store Video Walkthrough (.mp4) for 5-Frame Max-Pooling
3. Multilingual Speech Audio (Hinglish numbers, English, and Silent Safeguard)
All saved to: data/audit_eval_suite/
"""
import os
import cv2
import numpy as np

BASE_DIR = os.path.join("data", "audit_eval_suite")


def draw_shelf_background():
    img = np.ones((640, 640, 3), dtype=np.uint8) * 55
    cv2.line(img, (0, 210), (640, 210), (90, 80, 65), 12)
    cv2.line(img, (0, 430), (640, 430), (90, 80, 65), 12)
    cv2.line(img, (0, 630), (640, 630), (90, 80, 65), 14)
    return img


def draw_product_box(img, x_center, shelf_y, w, h, color):
    x1, y1 = int(x_center - w // 2), int(shelf_y - h // 2)
    x2, y2 = int(x_center + w // 2), int(shelf_y + h // 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
    if h > 30 and w > 20:
        cv2.line(img, (x1 + 4, y1 + 16), (x2 - 4, y1 + 16), (240, 240, 240), 2)


def generate_dense_shelf(output_path):
    img = draw_shelf_background()
    np.random.seed(42)
    for shelf_y in [145, 360, 560]:
        for x_center in range(45, 610, 55):
            w = np.random.randint(42, 50)
            h = np.random.randint(75, 95)
            color = (int(np.random.randint(80, 240)), int(np.random.randint(60, 220)), int(np.random.randint(60, 220)))
            draw_product_box(img, x_center, shelf_y, w, h, color)
    cv2.imwrite(output_path, img)


def generate_bare_shelf(output_path):
    img = draw_shelf_background()
    np.random.seed(101)
    placements = [(100, 145), (450, 145), (200, 360), (320, 360), (520, 360), (150, 560), (400, 560)]
    for x_center, shelf_y in placements:
        w, h = np.random.randint(42, 48), np.random.randint(70, 85)
        color = (int(np.random.randint(80, 220)), int(np.random.randint(70, 200)), int(np.random.randint(70, 200)))
        draw_product_box(img, x_center, shelf_y, w, h, color)
    cv2.imwrite(output_path, img)


def generate_empty_shelf(output_path):
    img = draw_shelf_background()
    draw_product_box(img, 80, 570, 44, 75, (120, 110, 100))
    cv2.imwrite(output_path, img)


def generate_kiosk_shelf(output_path):
    img = draw_shelf_background()
    np.random.seed(77)
    for shelf_y in [145, 360, 560]:
        for x_center in range(70, 590, 90):
            w, h = np.random.randint(38, 46), np.random.randint(65, 80)
            color = (int(np.random.randint(90, 230)), int(np.random.randint(80, 210)), int(np.random.randint(80, 210)))
            draw_product_box(img, x_center, shelf_y, w, h, color)
    cv2.imwrite(output_path, img)


def generate_store_walkthrough_video(output_mp4, duration_sec=3, fps=15):
    """Generates a realistic MP4 walkthrough video simulating camera motion across aisles."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (640, 640))

    total_frames = duration_sec * fps
    dense_frame = draw_shelf_background()
    np.random.seed(42)
    for shelf_y in [145, 360, 560]:
        for x_center in range(45, 610, 55):
            w, h = np.random.randint(42, 50), np.random.randint(75, 95)
            color = (int(np.random.randint(80, 240)), int(np.random.randint(60, 220)), int(np.random.randint(60, 220)))
            draw_product_box(dense_frame, x_center, shelf_y, w, h, color)

    # Frame 1 to 10: Blurry aisle transition
    blur_frame = cv2.GaussianBlur(dense_frame, (35, 35), 0)

    for i in range(total_frames):
        if i < 8 or i > (total_frames - 8):
            out.write(blur_frame)  # Camera entering / leaving aisle
        else:
            # Clear shelf view with slight camera pan
            shift_x = int(np.sin(i / 5.0) * 8)
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            panned = cv2.warpAffine(dense_frame, M, (640, 640))
            out.write(panned)

    out.release()


def synthesize_voice_audio(text, output_wav):
    from gtts import gTTS
    import soundfile as sf
    import librosa

    temp_mp3 = output_wav.replace(".wav", "_temp.mp3")
    tts = gTTS(text=text, lang="en", tld="co.in")
    tts.save(temp_mp3)

    y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
    sf.write(output_wav, y, 16000)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)


def create_silent_audio(output_wav, duration_sec=3):
    import soundfile as sf
    # Low-level ambient noise floor (below 0.01 amplitude)
    noise = np.random.normal(0, 0.002, 16000 * duration_sec).astype(np.float32)
    sf.write(output_wav, noise, 16000)


def build_full_suite():
    print("=" * 75)
    print("⚡ GENERATING 100% COMPLETE MULTI-MODAL TEST SUITE (VIDEO + SPEECH + VISION)")
    print("=" * 75)

    cases = [
        {
            "folder": "case01_suresh_patel_hinglish",
            "media_type": "image",
            "shelf_type": "dense",
            "media_name": "case01_dense_shelf.jpg",
            "audio_name": "case01_hinglish_audio.wav",
            "audio_type": "speech",
            "title": "Case 1: Prime Kirana (Hinglish: char hazar, pandra hazar)",
            "speech": (
                "Namaste sir, my name is Suresh Patel. Mera dhandha lagbhag char hazar rupees daily hai. "
                "Har somvar ko pandra hazar rupees ka restock karte hain aur dukaan ka kiraya teen hazar rupees hai."
            )
        },
        {
            "folder": "case02_ramesh_kumar_deadstock",
            "media_type": "image",
            "shelf_type": "dense",
            "media_name": "case02_deadstock_shelf.jpg",
            "audio_name": "case02_deadstock_audio.wav",
            "audio_type": "speech",
            "title": "Case 2: Dead Stock (Ramesh Kumar: daily sales 300)",
            "speech": (
                "Hello sir, my name is Ramesh Kumar. I run this grocery shop. "
                "The market is extremely slow right now. My daily sales are just 300 rupees. "
                "All the stock on my shelves is sitting idle for months with no buyers."
            )
        },
        {
            "folder": "case03_video_walkthrough_test",
            "media_type": "video",
            "shelf_type": "dense",
            "media_name": "case03_aisle_walkthrough.mp4",
            "audio_name": "case03_walkthrough_audio.wav",
            "audio_type": "speech",
            "title": "Case 3: Video Walkthrough (.mp4) Multi-Frame Max-Pooling Test",
            "speech": (
                "Namaste sir, my name is Suresh Patel. I run this grocery store. "
                "My daily sales are around 4000 rupees. Every Monday I restock 15000 rupees."
            )
        },
        {
            "folder": "case04_anita_verma_fraud",
            "media_type": "image",
            "shelf_type": "bare",
            "media_name": "case04_bare_shelf.jpg",
            "audio_name": "case04_fraud_audio.wav",
            "audio_type": "speech",
            "title": "Case 4: Turnover Inflation Fraud (Bare Shelves + 10k Claim)",
            "speech": (
                "Namaste sir, my name is Anita Verma. My daily sales are 10000 rupees every single day. "
                "My shop is very famous and making huge turnover. I need a big loan."
            )
        },
        {
            "folder": "case05_mobile_accessories_kiosk",
            "media_type": "image",
            "shelf_type": "kiosk",
            "media_name": "case05_mobile_shelf.jpg",
            "audio_name": "case05_mobile_audio.wav",
            "audio_type": "speech",
            "title": "Case 5: Mobile / Electronics Kiosk Category Profile",
            "speech": (
                "Namaste sir, my name is Imran Khan. I run this mobile accessories kiosk. "
                "My daily sales are 2000 rupees. My kiosk rent is 1000 rupees per month."
            )
        },
        {
            "folder": "case06_sunita_devi_empty_store",
            "media_type": "image",
            "shelf_type": "empty",
            "media_name": "case06_empty_shelf.jpg",
            "audio_name": "case06_empty_audio.wav",
            "audio_type": "speech",
            "title": "Case 6: Empty Store (Sunita Devi: 0 stock)",
            "speech": (
                "Namaste sir, my name is Sunita Devi. I run this retail store. "
                "My daily sales are 2000 rupees. I need working capital to run the business."
            )
        },
        {
            "folder": "case07_silent_audio_guard",
            "media_type": "image",
            "shelf_type": "dense",
            "media_name": "case07_dense_shelf.jpg",
            "audio_name": "case07_silent_background_noise.wav",
            "audio_type": "silent",
            "title": "Case 7: Silent / Market Noise Safeguard Test",
            "speech": ""
        }
    ]

    for c in cases:
        case_dir = os.path.join(BASE_DIR, c["folder"])
        os.makedirs(case_dir, exist_ok=True)

        media_path = os.path.join(case_dir, c["media_name"])
        audio_path = os.path.join(case_dir, c["audio_name"])

        print(f"\n📦 Building [{c['title']}]...")

        # 1. Media Generation (Video or Image)
        if c["media_type"] == "video":
            generate_store_walkthrough_video(media_path)
            print(f"  🎬  Walkthrough Video -> {media_path} (.mp4 Video)")
        else:
            if c["shelf_type"] == "dense": generate_dense_shelf(media_path)
            elif c["shelf_type"] == "bare": generate_bare_shelf(media_path)
            elif c["shelf_type"] == "empty": generate_empty_shelf(media_path)
            elif c["shelf_type"] == "kiosk": generate_kiosk_shelf(media_path)
            print(f"  🖼️  Shelf Image -> {media_path}")

        # 2. Audio Generation
        if c["audio_type"] == "silent":
            create_silent_audio(audio_path)
            print(f"  🔇  Silent Audio Safeguard -> {audio_path}")
        else:
            synthesize_voice_audio(c["speech"], audio_path)
            print(f"  🎙️  Spoken Audio -> {audio_path}")

    print("\n" + "=" * 75)
    print(f"✅ ALL 7 TEST CASES READY IN: {BASE_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    build_full_suite()
