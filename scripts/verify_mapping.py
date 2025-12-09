#!/usr/bin/env python
"""
Verify correct character mapping: pass correct content latent to decoder.
NO style, NO training - just verify mapping is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras


def build_char_to_index(png_dir: Path) -> dict:
    """Build character-to-content_latent_index mapping from PNG files."""
    png_files = sorted(png_dir.glob("*.png"))
    mapping = {}
    for idx, f in enumerate(png_files):
        name = f.stem
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                char = chr(codepoint)
                mapping[char] = idx
            except ValueError:
                continue
    return mapping


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
    parser.add_argument("--png_dir", type=Path, required=True)
    parser.add_argument("--pretrained_decoder", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build mapping
    char_to_index = build_char_to_index(args.png_dir)
    print(f"Char-to-index mapping: {len(char_to_index)} chars")

    # Load data
    entries = json.loads(args.index.read_text())
    content_latents = np.load(args.content_latents)

    images = []
    latents = []
    chars = []

    for e in entries[:8]:
        char_text = e.get("text", "")
        if not char_text:
            print(f"  Entry missing text: {e}")
            continue
        if char_text not in char_to_index:
            print(f"  Char '{char_text}' not in mapping")
            continue

        latent_idx = char_to_index[char_text]
        print(f"  '{char_text}' (U+{ord(char_text):04X}) -> latent index {latent_idx}")

        img_rel = e["image_path"]
        if "/" in img_rel:
            parts = img_rel.split("/")
            img_path = args.root / parts[-2] / parts[-1]
        else:
            img_path = args.root / str(e["writer_id"]) / img_rel

        if img_path.exists() and latent_idx < len(content_latents):
            images.append(load_image(img_path))
            latents.append(content_latents[latent_idx])
            chars.append(char_text)

    if not images:
        print("No images loaded!")
        return

    images = np.stack(images)
    latents = np.stack(latents)
    print(f"\nLoaded {len(images)} images")

    # Load decoder
    decoder = keras.models.load_model(args.pretrained_decoder, compile=False, safe_mode=False)

    # Direct decode - NO modification
    preds = decoder(latents, training=False).numpy()
    preds = np.clip(preds, 0, 1)

    # Visualize
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        # Use Unicode hex instead of actual char for font compatibility
        axes[0, i].set_title(f"Orig: U+{ord(chars[i]):04X}", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"Dec: U+{ord(chars[i]):04X}", fontsize=7)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
