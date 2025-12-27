#!/usr/bin/env python3
# preprocessing/crop_faces.py
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import os

def clamp(v, low, high):
    return max(low, min(high, v))

def process_image(img, box, size):
    """
    Crop and resize a face region from `img` using `box` (x, y, w, h).
    Accepts ints/floats/numpy scalars, clamps to image bounds and validates area.
    Raises ValueError for invalid boxes (zero/negative width or height).
    """
    if box is None:
        raise ValueError("box is None")
    try:
        x, y, w, h = box
    except Exception:
        raise ValueError("box must be iterable of four values (x, y, w, h)")

    # Normalize types and ensure integer coordinates
    x = int(round(float(x)))
    y = int(round(float(y)))
    w = int(round(float(w)))
    h = int(round(float(h)))

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid box width/height: w={w}, h={h}")

    pad = int(max(w, h) * 0.25)
    x1 = clamp(x - pad, 0, img.width)
    y1 = clamp(y - pad, 0, img.height)
    x2 = clamp(x + w + pad, 0, img.width)
    y2 = clamp(y + h + pad, 0, img.height)

    # Ensure integer and non-zero area
    x1, y1 = int(x1), int(y1)
    x2 = int(max(x2, x1 + 1))
    y2 = int(max(y2, y1 + 1))

    return img.crop((x1, y1, x2, y2)).resize(size, Image.BILINEAR)

def run(src="data", dst="preprocessed", size=224, save_all_faces=False, no_fallback=False, verbose=False):
    detector = MTCNN()
    size = (size, size)
    processed = 0
    skipped = 0

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            in_dir = os.path.join(src, split, cls)
            out_dir = os.path.join(dst, split, cls)
            if not os.path.isdir(in_dir):
                if verbose:
                    print(f"Skipping missing directory: {in_dir}")
                continue
            os.makedirs(out_dir, exist_ok=True)

            for fname in os.listdir(in_dir):
                in_path = os.path.join(in_dir, fname)
                try:
                    img = Image.open(in_path).convert("RGB")
                    arr = np.array(img)
                    faces = detector.detect_faces(arr)
                    root = os.path.splitext(fname)[0]

                    if not faces:
                        if no_fallback:
                            skipped += 1
                            if verbose:
                                print(f"No face: skip {fname}")
                            continue
                        # fallback: save resized full image
                        out_name = f"{root}.jpg"
                        img.resize(size, Image.BILINEAR).save(os.path.join(out_dir, out_name), "JPEG", quality=92)
                        processed += 1
                        if verbose:
                            print(f"No face: saved full-image fallback {out_name}")
                        continue

                    # if faces found
                    if save_all_faces:
                        for i, f in enumerate(faces, 1):
                            crop = process_image(img, f["box"], size)
                            out_name = f"{root}_face{i}.jpg"
                            crop.save(os.path.join(out_dir, out_name), "JPEG", quality=92)
                            processed += 1
                    else:
                        # save largest face
                        f = max(faces, key=lambda ff: ff["box"][2] * ff["box"][3])
                        crop = process_image(img, f["box"], size)
                        out_name = f"{root}.jpg"
                        crop.save(os.path.join(out_dir, out_name), "JPEG", quality=92)
                        processed += 1

                    if verbose and processed % 100 == 0:
                        print(f"Processed {processed} images...")

                except Exception as e:
                    skipped += 1
                    if verbose:
                        print(f"Error processing {fname}: {e}")

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


def main():
    run()

if __name__ == "__main__":

    main()