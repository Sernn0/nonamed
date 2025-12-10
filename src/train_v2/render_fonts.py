#!/usr/bin/env python3
"""
Font Rendering Script - Generate 2350 Korean characters from font files.

Each font will create a subfolder under data/font_rendered/ with 2350 PNG images.

Usage:
    python src/train_v2/render_fonts.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import os

# Paths
ROOT = Path(__file__).resolve().parents[2]
FONTS_DIR = ROOT / "fonts"
OUTPUT_DIR = ROOT / "data" / "font_rendered"
CHARSET_FILE = ROOT / "charset_2350.txt"

# Image settings
IMAGE_SIZE = 256
FONT_SIZE = 200  # Will adjust based on bounding box
BACKGROUND_COLOR = 255  # White
TEXT_COLOR = 0  # Black


def load_charset() -> list:
    """Load 2350 Korean characters from charset file."""
    with open(CHARSET_FILE, 'r', encoding='utf-8') as f:
        chars = f.read().strip()
    # Remove any whitespace/newlines
    chars = [c for c in chars if c.strip()]
    print(f"[INFO] Loaded {len(chars)} characters")
    return chars


def render_char(char: str, font: ImageFont.FreeTypeFont, size: int = IMAGE_SIZE) -> Image.Image:
    """Render a single character to an image, centered."""
    img = Image.new('L', (size, size), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # Get bounding box
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center the character
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=TEXT_COLOR)

    return img


def process_font(font_path: Path, chars: list) -> int:
    """Process a single font file and generate all character images."""
    font_name = font_path.stem
    output_dir = OUTPUT_DIR / font_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(str(font_path), FONT_SIZE)
    except Exception as e:
        print(f"[ERROR] Failed to load font {font_path}: {e}")
        return 0

    count = 0
    errors = 0
    for char in chars:
        codepoint = ord(char)
        try:
            img = render_char(char, font)
            out_path = output_dir / f"U+{codepoint:04X}.png"
            img.save(out_path)
            count += 1
        except Exception as e:
            errors += 1
            if errors == 1:  # Only print first error
                print(f"  [WARN] Skipped char error: {e}")

    if errors > 0:
        print(f"  [WARN] Skipped {errors} characters with errors")

    return count


def main():
    print("=" * 50)
    print("Font Rendering Script")
    print("=" * 50)

    # Load charset
    chars = load_charset()

    # Find all fonts
    font_extensions = ['*.ttf', '*.otf', '*.TTF', '*.OTF']
    font_files = []
    for ext in font_extensions:
        font_files.extend(FONTS_DIR.glob(ext))
        # Also check subdirectories
        font_files.extend(FONTS_DIR.glob(f"**/{ext}"))

    font_files = list(set(font_files))  # Remove duplicates
    print(f"[INFO] Found {len(font_files)} font files in {FONTS_DIR}")

    if not font_files:
        print("[ERROR] No font files found! Add .ttf or .otf files to fonts/ directory.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each font
    total_images = 0
    for i, font_path in enumerate(font_files, 1):
        print(f"\n[{i}/{len(font_files)}] Processing: {font_path.name}")
        count = process_font(font_path, chars)
        print(f"  → Generated {count} images")
        total_images += count

    print("\n" + "=" * 50)
    print(f"✅ DONE! Generated {total_images} total images")
    print(f"   Location: {OUTPUT_DIR}")
    print("=" * 50)

    # Create index JSON
    index = {
        "fonts": [f.stem for f in font_files],
        "chars": len(chars),
        "total_images": total_images
    }
    with open(OUTPUT_DIR / "index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"   Index saved to: {OUTPUT_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
