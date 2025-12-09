#!/usr/bin/env python
"""
Build character-to-content_latent_index mapping.
content_latents.npy is indexed by sorted(glob("*.png")) order.
This script creates a mapping from character text to correct index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png_dir",
        type=Path,
        required=True,
        help="Directory containing U+xxxx.png files",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output JSON mapping file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Get sorted PNG files (same order as content_latents)
    png_files = sorted(args.png_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")

    # Build mapping: character -> index
    char_to_index = {}
    for idx, f in enumerate(png_files):
        # Parse U+XXXX.png -> unicode codepoint -> character
        name = f.stem  # "U+AC00"
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                char = chr(codepoint)
                char_to_index[char] = idx
            except ValueError:
                print(f"Warning: Could not parse {name}")
                continue

    print(f"Built mapping for {len(char_to_index)} characters")

    # Show some examples
    examples = ["가", "각", "간", "떰", "훌", "다", "1", "A"]
    print("\nExample mappings:")
    for c in examples:
        if c in char_to_index:
            print(f"  '{c}' (U+{ord(c):04X}) -> index {char_to_index[c]}")
        else:
            print(f"  '{c}' -> NOT FOUND")

    # Save mapping
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(char_to_index, f, ensure_ascii=False, indent=2)
    print(f"\nSaved mapping to {args.output_path}")


if __name__ == "__main__":
    main()
