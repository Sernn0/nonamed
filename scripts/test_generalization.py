#!/usr/bin/env python
"""
Test generalization: Generate characters NOT in training set.
Uses fine-tuned decoder to generate unseen characters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


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
    parser.add_argument("--png_dir", type=Path, required=True)
    parser.add_argument("--test_chars", type=str, default="가각간갈감갑강개객갱거건걸검겁게겨견결경계고곡곤골공과곽관광괘괴교구국군굴궁권궐귀규균귤그극근글금급긍기긴길김",
                        help="Characters to test (should NOT be in training set)")
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build mapping
    char_to_index = build_char_to_index(args.png_dir)
    print(f"Mapping: {len(char_to_index)} chars")

    # Load decoder and content latents
    decoder = keras.models.load_model(args.finetuned_decoder, compile=False, safe_mode=False)
    content_latents = np.load(args.content_latents)
    print(f"Loaded decoder and {len(content_latents)} content latents")

    # Get latents for test characters
    test_chars = list(args.test_chars)
    valid_chars = []
    valid_latents = []

    for c in test_chars:
        if c in char_to_index:
            idx = char_to_index[c]
            if idx < len(content_latents):
                valid_chars.append(c)
                valid_latents.append(content_latents[idx])

    if not valid_latents:
        print("No valid test characters found!")
        return

    latents = np.stack(valid_latents)
    print(f"Testing {len(valid_chars)} characters: {''.join(valid_chars[:20])}...")

    # Generate
    preds = decoder(latents, training=False).numpy()
    preds = np.clip(preds, 0, 1)

    # Visualize (grid)
    n = len(valid_chars)
    cols = 10
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(preds[i].squeeze(), cmap="gray")
            ax.set_title(f"U+{ord(valid_chars[i]):04X}", fontsize=6)
        ax.axis("off")

    plt.suptitle("Generalization Test: Unseen Characters", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
