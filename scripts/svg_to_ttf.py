from __future__ import annotations

"""
Skeleton script to build a TTF font from SVG glyph files using fontTools/fontmake.
SVG path parsing is left as a TODO; currently uses dummy outlines.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont


UNITS_PER_EM = 1000
DEFAULT_ADVANCE = 600


def load_svg_files(svg_dir: Path) -> List[Path]:
    """Return list of SVG paths (non-recursive)."""
    return sorted([p for p in svg_dir.glob("*.svg") if p.is_file()])


def parse_codepoint_from_filename(path: Path) -> Optional[int]:
    """
    Parse codepoint from filename.
    Accepts:
      - "ac00.svg" -> U+AC00
      - "uniAC00.svg" -> U+AC00
    Returns None if parsing fails.
    """
    stem = path.stem.lower()
    if stem.startswith("uni"):
        stem = stem[3:]
    try:
        return int(stem, 16)
    except ValueError:
        return None


def build_dummy_glyph(width: int = DEFAULT_ADVANCE, height: int = UNITS_PER_EM) -> TTGlyphPen:
    """Build a simple rectangle glyph as placeholder."""
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, height - 50))
    pen.lineTo((width - 50, height - 50))
    pen.lineTo((width - 50, 0))
    pen.closePath()
    return pen


def build_font_from_svgs(
    svg_paths: List[Path],
    output_ttf: Path,
    family_name: str,
    style_name: str,
    units_per_em: int = UNITS_PER_EM,
    advance_width: int = DEFAULT_ADVANCE,
) -> None:
    """Create a basic TTF from SVG placeholders."""
    # Base glyphs
    glyph_order = [".notdef", "space"]
    glyphs: Dict[str, any] = {}
    cmap: Dict[int, str] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}

    # .notdef
    pen = build_dummy_glyph(width=advance_width, height=units_per_em)
    glyphs[".notdef"] = pen.glyph()
    hmtx[".notdef"] = (advance_width, 0)

    # space (empty)
    pen_space = TTGlyphPen(None)
    glyphs["space"] = pen_space.glyph()
    hmtx["space"] = (advance_width, 0)
    cmap[0x0020] = "space"

    # Add glyphs from SVGs
    for svg_path in svg_paths:
        cp = parse_codepoint_from_filename(svg_path)
        if cp is None:
            print(f"[WARN] Skip (cannot parse codepoint): {svg_path.name}")
            continue
        glyph_name = f"uni{cp:04X}"
        glyph_order.append(glyph_name)
        # TODO: Parse SVG path and draw with TTGlyphPen
        pen = build_dummy_glyph(width=advance_width, height=units_per_em)
        glyphs[glyph_name] = pen.glyph()
        hmtx[glyph_name] = (advance_width, 0)
        cmap[cp] = glyph_name

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(hmtx)
    fb.setupHorizontalHeader(ascent=int(units_per_em * 0.8), descent=-int(units_per_em * 0.2))
    fb.setupOS2(
        sTypoAscender=int(units_per_em * 0.8),
        sTypoDescender=-int(units_per_em * 0.2),
        usWinAscent=int(units_per_em * 0.8),
        usWinDescent=int(units_per_em * 0.2),
    )
    fb.setupNameTable(
        {
            "familyName": family_name,
            "styleName": style_name,
            "uniqueFontIdentifier": f"{family_name} {style_name}",
            "fullName": f"{family_name} {style_name}",
        }
    )
    fb.setupPost()
    fb.setupMaxp()
    fb.setupHead()

    output_ttf.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(output_ttf))
    print(f"[INFO] Saved TTF: {output_ttf}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TTF font from SVG glyphs (skeleton).")
    parser.add_argument(
        "--svg_dir",
        type=Path,
        default=Path("runs/generated_svg"),
        help="Directory containing SVG files.",
    )
    parser.add_argument(
        "--output_ttf",
        type=Path,
        default=Path("runs/fonts/FontByMe_test.ttf"),
        help="Path to output TTF file.",
    )
    parser.add_argument(
        "--family_name",
        type=str,
        default="FontByMeTest",
        help="Font family name.",
    )
    parser.add_argument(
        "--style_name",
        type=str,
        default="Regular",
        help="Font style name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svg_dir: Path = args.svg_dir
    if not svg_dir.is_dir():
        print(f"[WARN] SVG directory not found: {svg_dir}")
    svg_paths = load_svg_files(svg_dir)
    if not svg_paths:
        print(f"[WARN] No SVG files found in {svg_dir}. Creating font with .notdef and space only.")
    build_font_from_svgs(
        svg_paths=svg_paths,
        output_ttf=args.output_ttf,
        family_name=args.family_name,
        style_name=args.style_name,
    )


if __name__ == "__main__":
    main()
