#!/usr/bin/env python
"""
MINIMAL TEST: Just verify pretrained decoder works by itself.
Pass content latent directly to decoder - NO style, NO modification.
This MUST produce the same results as test_content_autoenc.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras


def load_image(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--content_latents", type=Path, required=True)
    parser.add_argument("--pretrained_decoder", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    entries = json.loads(args.index.read_text())
    content_latents = np.load(args.content_latents)

    # Load images and content for first 8 samples
    images = []
    latents = []
    for e in entries[:8]:
        text_id = int(e["text_id"])
        img_rel = e["image_path"]
        if "/" in img_rel:
            parts = img_rel.split("/")
            img_path = args.root / parts[-2] / parts[-1]
        else:
            img_path = args.root / str(e["writer_id"]) / img_rel

        if img_path.exists() and text_id < len(content_latents):
            images.append(load_image(img_path))
            latents.append(content_latents[text_id])

    images = np.stack(images)
    latents = np.stack(latents)
    print(f"Loaded {len(images)} images")

    # Load pretrained decoder
    decoder = keras.models.load_model(args.pretrained_decoder, compile=False, safe_mode=False)
    print(f"Loaded decoder from {args.pretrained_decoder}")

    # Direct forward pass - NO training, NO modification
    preds = decoder(latents, training=False).numpy()
    preds = np.clip(preds, 0, 1)

    # Visualize
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(min(8, len(images))):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].set_title("Original", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].set_title("Decoder Output", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
