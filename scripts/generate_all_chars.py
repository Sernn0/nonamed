#!/usr/bin/env python
"""
Generate all 2356 Korean characters using fine-tuned decoder.
Outputs individual PNG files for each character.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf


def build_char_to_index(png_dir: Path) -> dict:
    png_files = sorted(png_dir.glob("*.png"))
    mapping = {}
    for idx, f in enumerate(png_files):
        name = f.stem
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                mapping[chr(codepoint)] = idx
            except ValueError:
                continue
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_decoder", type=Path, required=True)
    parser.add_argument("--content_latents", type=Path, required=True)
    parser.add_argument("--png_dir", type=Path, required=True, help="Original PNG dir for mapping")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping
    char_to_index = build_char_to_index(args.png_dir)
    print(f"Character mapping: {len(char_to_index)} chars")

    # Load decoder and latents
    decoder = keras.models.load_model(args.finetuned_decoder, compile=False, safe_mode=False)
    content_latents = np.load(args.content_latents)
    print(f"Loaded decoder and {len(content_latents)} content latents")

    # Get all Korean syllables (가-힣: U+AC00 to U+D7A3)
    korean_chars = []
    korean_indices = []
    for codepoint in range(0xAC00, 0xD7A4):  # 가 to 힣
        char = chr(codepoint)
        if char in char_to_index:
            idx = char_to_index[char]
            if idx < len(content_latents):
                korean_chars.append(char)
                korean_indices.append(idx)

    print(f"Generating {len(korean_chars)} Korean characters...")

    # Generate in batches
    all_latents = content_latents[korean_indices]
    n = len(korean_chars)

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_latents = tf.constant(all_latents[start:end])
        batch_preds = decoder(batch_latents, training=False).numpy()
        batch_preds = np.clip(batch_preds, 0, 1)

        for i, pred in enumerate(batch_preds):
            char_idx = start + i
            char = korean_chars[char_idx]
            codepoint = ord(char)

            # Save as PNG
            img = (pred.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode="L")
            pil_img.save(args.output_dir / f"U+{codepoint:04X}.png")

        if (end % 500 == 0) or (end == n):
            print(f"Generated {end}/{n} characters")

    print(f"Saved all {n} characters to {args.output_dir}")


if __name__ == "__main__":
    main()
