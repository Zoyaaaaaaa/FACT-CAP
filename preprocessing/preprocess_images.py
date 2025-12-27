import os
from PIL import Image

INPUT_ROOT = "data/val"
OUTPUT_ROOT = "preprocessed/val"
IMG_SIZE = (128, 128)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for cls in ["real", "fake"]:
    in_dir = os.path.join(INPUT_ROOT, cls)
    out_dir = os.path.join(OUTPUT_ROOT, cls)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        try:
            img = Image.open(in_path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            img.save(out_path, "JPEG", quality=90)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
