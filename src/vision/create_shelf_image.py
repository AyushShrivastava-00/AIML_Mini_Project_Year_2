"""
Generates a realistic dense retail shelf test image with packaged products.
Saved to: data/raw/dense_shelf.jpg
"""
import os
import cv2
import numpy as np

def create_dense_shelf():
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Create wooden background shelf (640x640)
    img = np.ones((640, 640, 3), dtype=np.uint8) * 60  # dark shop ambiance
    
    # Draw horizontal wooden shelf planks
    cv2.line(img, (0, 210), (640, 210), (90, 80, 65), 10)
    cv2.line(img, (0, 430), (640, 430), (90, 80, 65), 10)
    
    # 2. Populate shelves with tightly packed FMCG products
    np.random.seed(42)  # reproducible test
    
    for shelf_y in [140, 350, 560]:  # 3 shelf tiers
        for x_center in range(45, 610, 55):  # 10 items per shelf tier = 30 items
            box_w = np.random.randint(42, 50)
            box_h = np.random.randint(75, 95)
            
            # Vibrant retail packaging colors (soap boxes, biscuit packs, tea cartons)
            color = (
                int(np.random.randint(80, 240)),
                int(np.random.randint(60, 220)),
                int(np.random.randint(60, 220))
            )
            
            x1 = x_center - box_w // 2
            y1 = shelf_y - box_h // 2
            x2 = x_center + box_w // 2
            y2 = shelf_y + box_h // 2
            
            # Draw package body and sharp border
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
            
            # Add packaging brand line stripes
            cv2.line(img, (x1 + 4, y1 + 18), (x2 - 4, y1 + 18), (240, 240, 240), 2)

    output_path = "data/raw/dense_shelf.jpg"
    cv2.imwrite(output_path, img)
    print(f"✅ Successfully created dense retail shelf test image: {output_path}")

if __name__ == "__main__":
    create_dense_shelf()
