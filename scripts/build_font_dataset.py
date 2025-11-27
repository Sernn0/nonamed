from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

DEFAULT_CHARSET: List[str] = ["가", "나", "다", "A", "B", "C", "1", "2", "3"]
BACKGROUND_COLOR = 255
FOREGROUND_COLOR = 0


def load_charset(charset_file: Optional[str]) -> List[str]:
    """Load characters from a file or return the default charset."""
    if charset_file is None:
        return DEFAULT_CHARSET

    path = Path(charset_file)
    if not path.is_file():
        print(f"Error: charset file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with path.open("r", encoding="utf-8") as handle:
            chars = [line.rstrip("\n") for line in handle if line.strip()]
    except OSError as exc:
        print(f"Error reading charset file: {exc}", file=sys.stderr)
        sys.exit(1)

    return chars


def render_character_image(
    char: str, font: ImageFont.FreeTypeFont, image_size: int
) -> Image.Image:
    """Render a single character to a centered grayscale image."""
    image = Image.new("L", (image_size, image_size), color=BACKGROUND_COLOR)
    drawer = ImageDraw.Draw(image)
    bbox = drawer.textbbox((0, 0), char, font=font)

    if bbox is None:
        return image

    left, top, right, bottom = bbox
    text_width = right - left
    text_height = bottom - top
    x = (image_size - text_width) / 2 - left
    y = (image_size - text_height) / 2 - top
    drawer.text((x, y), char, fill=FOREGROUND_COLOR, font=font)
    return image


def save_glyph_dataset(
    chars: List[str], font_path: str, output_dir: str, image_size: int, font_size: int
) -> None:
    """Render and save glyphs for each character in chars."""
    font_path_obj = Path(font_path)
    if not font_path_obj.is_file():
        print(f"Error: font file not found: {font_path_obj}", file=sys.stderr)
        sys.exit(1)

    try:
        font = ImageFont.truetype(str(font_path_obj), font_size)
    except OSError as exc:
        print(f"Error loading font: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(output_dir) / font_path_obj.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_chars = [char for char in chars if char]
    print(f"Rendering {len(valid_chars)} characters...")

    for char in valid_chars:
        if len(char) != 1:
            print(
                f"Skipping invalid entry (expected one character): {repr(char)}",
                file=sys.stderr,
            )
            continue

        image = render_character_image(char, font, image_size)
        codepoint = f"U+{ord(char):04X}"
        filename = out_dir / f"{codepoint}.png"
        image.save(filename, "PNG")
        print(f"Saved: {filename.name}")


def train_style_model() -> None:
    # TODO: implement style model training
    pass


def transfer_style_to_font() -> None:
    # TODO: implement style transfer to target font
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render glyphs from a font into individual PNG images."
    )
    parser.add_argument(
        "--font-path",
        required=True,
        help="Path to the TTF/OTF font file used for rendering.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the generated glyph images.",
    )
    parser.add_argument(
        "--charset-file",
        help="Optional path to a text file with one character per line.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square image size in pixels (default: 256).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=220,
        help="Font size used for rendering (default: 220).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    charset = load_charset(args.charset_file)
    save_glyph_dataset(
        chars=charset,
        font_path=args.font_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
