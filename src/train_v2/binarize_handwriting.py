#!/usr/bin/env python3
"""Binarize handwriting images."""
import os
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

hw_dir = Path("data/handwriting_raw/resizing")
out_dir = Path("data/handwriting_binary")
out_dir.mkdir(parents=True, exist_ok=True)

writers = sorted([d for d in hw_dir.iterdir() if d.is_dir()])
print(f"[INFO] Found {len(writers)} writers")

total = 0
processed = 0
lock = threading.Lock()

def process_image(args):
    global processed
    img_path, writer_out_dir = args
    try:
        img = Image.open(img_path).convert('L').resize((256, 256))
        arr = np.array(img)
        threshold = max(np.mean(arr) - 20, 128)
        binary = ((arr > threshold) * 255).astype(np.uint8)
        out_path = writer_out_dir / f"{img_path.stem}.png"
        Image.fromarray(binary).save(out_path)
        with lock:
            global processed
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed} images...")
        return True
    except:
        return False

for writer in writers:
    writer_out_dir = out_dir / writer.name
    writer_out_dir.mkdir(exist_ok=True)

    images = list(writer.glob("*.jpg")) + list(writer.glob("*.png"))
    total += len(images)

    tasks = [(img, writer_out_dir) for img in images]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_image, tasks))

    print(f"[{writer.name}] {len(images)} images")

print(f"\n✅ Done! Processed {processed}/{total} images")
print(f"   Saved to: {out_dir}")
