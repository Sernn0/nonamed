#!/usr/bin/env python
"""
Convert generated PNG glyphs to TTF font file.
Uses potrace for vectorization and fontforge for font creation.

Requirements (install on Colab):
    !apt-get install -y potrace fontforge python3-fontforge
    !pip install potrace
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with U+XXXX.png files")
    parser.add_argument("--output_ttf", type=Path, required=True, help="Output TTF file path")
    parser.add_argument("--font_name", type=str, default="MyHandwriting")
    parser.add_argument("--em_size", type=int, default=1024)
    return parser.parse_args()


def png_to_svg(png_path: Path, svg_path: Path, threshold: int = 220) -> bool:
    """Convert PNG to SVG using potrace."""
    try:
        # Load and binarize
        img = Image.open(png_path).convert("L")
        arr = np.array(img)

        # Our glyphs are dark (low values) on white background (high values)
        # Potrace traces black (1) areas in PBM
        # So we mark pixels BELOW threshold as the glyph to trace
        binary = (arr < threshold).astype(np.uint8)  # glyph=1, background=0

        # Save as PBM (portable bitmap)
        with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as tmp:
            pbm_path = tmp.name
            # PBM format
            h, w = binary.shape
            with open(pbm_path, "wb") as f:
                f.write(f"P4\n{w} {h}\n".encode())
                # Pack bits
                packed = np.packbits(binary > 0, axis=1)
                f.write(packed.tobytes())

        # Run potrace
        result = subprocess.run(
            ["potrace", "-s", "-o", str(svg_path), pbm_path],
            capture_output=True
        )

        Path(pbm_path).unlink()  # Clean up
        return result.returncode == 0

    except Exception as e:
        print(f"Error converting {png_path}: {e}")
        return False


def create_font_with_fontforge(svg_dir: Path, output_ttf: Path, font_name: str, em_size: int):
    """Create TTF font using fontforge."""
    # Create fontforge script
    script = f'''
import fontforge

font = fontforge.font()
font.fontname = "{font_name}"
font.familyname = "{font_name}"
font.fullname = "{font_name}"
font.em = {em_size}

import os
svg_dir = "{svg_dir}"

for filename in os.listdir(svg_dir):
    if not filename.endswith(".svg"):
        continue

    # Parse codepoint from filename (U+XXXX.svg)
    name = filename[:-4]  # Remove .svg
    if not name.startswith("U+"):
        continue

    try:
        codepoint = int(name[2:], 16)
    except ValueError:
        continue

    # Create glyph
    glyph = font.createChar(codepoint)
    glyph.importOutlines(os.path.join(svg_dir, filename))
    glyph.width = {em_size}

    # Auto-hint
    try:
        glyph.autoHint()
    except:
        pass

font.generate("{output_ttf}")
print(f"Generated font with {{len(font)}} glyphs")
'''

    # Save and run script
    script_path = svg_dir / "create_font.py"
    script_path.write_text(script)

    result = subprocess.run(
        ["fontforge", "-script", str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"FontForge error: {result.stderr}")
        return False

    print(result.stdout)
    return True


def main() -> None:
    args = parse_args()

    # Get all PNG files
    png_files = sorted(args.input_dir.glob("U+*.png"))
    print(f"Found {len(png_files)} PNG files")

    # Create SVG directory
    svg_dir = args.input_dir / "svg"
    svg_dir.mkdir(exist_ok=True)

    # Convert PNGs to SVGs
    print("Converting PNGs to SVGs...")
    success = 0
    for i, png_path in enumerate(png_files):
        svg_path = svg_dir / f"{png_path.stem}.svg"
        if png_to_svg(png_path, svg_path):
            success += 1

        if (i + 1) % 500 == 0:
            print(f"Converted {i + 1}/{len(png_files)}")

    print(f"Converted {success}/{len(png_files)} to SVG")

    # Create font
    print("Creating TTF font...")
    args.output_ttf.parent.mkdir(parents=True, exist_ok=True)

    if create_font_with_fontforge(svg_dir, args.output_ttf, args.font_name, args.em_size):
        print(f"Font saved to {args.output_ttf}")
    else:
        print("Font creation failed")


if __name__ == "__main__":
    main()
