from __future__ import annotations

"""
Convert bitmap glyph PNGs to SVG using potrace (test pipeline).
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image


def convert_png_to_svg(
    input_path: Path, output_path: Path, threshold: int, turdsize: int
) -> bool:
    """Convert a single PNG to SVG via potrace."""
    tmp_pbm = output_path.with_suffix(".pbm")
    try:
        img = Image.open(input_path).convert("L")
    except Exception as exc:
        print(f"[WARN] load error: {input_path} ({exc})")
        return False

    # Binarize
    img = img.point(lambda p: 255 if p >= threshold else 0, "L")
    try:
        tmp_pbm.parent.mkdir(parents=True, exist_ok=True)
        img.save(tmp_pbm, format="PPM")
    except Exception as exc:
        print(f"[WARN] save temp error: {tmp_pbm} ({exc})")
        return False

    cmd = [
        "potrace",
        str(tmp_pbm),
        "-s",
        "-o",
        str(output_path),
        "--turdsize",
        str(turdsize),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stderr:
            print(result.stderr.strip())
    except FileNotFoundError:
        print("[ERROR] potrace not found. Install potrace and ensure it's in PATH.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] potrace failed for {input_path}: {exc.stderr}")
        return False
    finally:
        try:
            if tmp_pbm.exists():
                tmp_pbm.unlink()
        except OSError:
            pass

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PNG glyphs to SVG using potrace.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/content_font/png_2350"),
        help="Directory containing PNG files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/svg_test"),
        help="Directory to save SVG files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarization threshold (0-255).",
    )
    parser.add_argument(
        "--turdsize",
        type=int,
        default=2,
        help="Potrace turdsize parameter to remove small specks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"[WARN] Input directory not found: {input_dir}")

    png_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG"))
    total = len(png_files)
    success = 0

    for png_path in png_files:
        svg_path = output_dir / (png_path.stem + ".svg")
        if convert_png_to_svg(png_path, svg_path, args.threshold, args.turdsize):
            success += 1
        else:
            print(f"[WARN] Failed: {png_path}")

    print(f"Conversion complete: {success}/{total} succeeded.")


if __name__ == "__main__":
    main()
