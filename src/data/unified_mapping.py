"""
Unified character mapping for FontByMe.

Creates consistent character -> index mapping based on actual characters,
not file ordering or text_id.

This resolves the mismatch between:
- JSON text_id (starts with 가=0)
- PNG file index (starts with 1=0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Project root
ROOT = Path(__file__).resolve().parents[1]


def load_json_text_mapping(json_path: Path) -> Dict[str, int]:
    """Load text -> text_id mapping from handwriting JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mapping = {}
    for entry in data:
        text = entry.get("text", "")
        text_id = entry.get("text_id", 0) # Default to 0 if missing
        if text:
            mapping[text] = text_id
    return mapping


def load_png_file_mapping(png_dir: Path) -> Dict[str, int]:
    """Load char -> file_index mapping from PNG filenames."""
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


def build_unified_mapping(
    json_path: Path,
    png_dir: Path,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build unified character -> index mapping.

    Returns:
        (char_to_idx, idx_to_char): Character to unified index, and reverse list
    """
    # Get all characters from both sources
    json_mapping = load_json_text_mapping(json_path)
    png_mapping = load_png_file_mapping(png_dir)

    # Use intersection of both (characters that exist in both)
    common_chars = set(json_mapping.keys()) & set(png_mapping.keys())

    # Sort by Unicode codepoint for consistent ordering
    sorted_chars = sorted(common_chars, key=ord)

    # Create unified mapping
    char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}

    return char_to_idx, sorted_chars


def get_original_indices(
    char: str,
    json_mapping: Dict[str, int],
    png_mapping: Dict[str, int],
) -> Tuple[int, int]:
    """Get original indices from both sources for a character."""
    json_idx = json_mapping.get(char, -1)
    png_idx = png_mapping.get(char, -1)
    return json_idx, png_idx


def main():
    """Test unified mapping."""
    json_path = ROOT / "data/handwriting_processed/handwriting_index_train_shared.json"
    png_dir = ROOT / "data/content_font/NotoSansKR-Regular"

    char_to_idx, sorted_chars = build_unified_mapping(json_path, png_dir)

    print(f"Unified mapping: {len(char_to_idx)} characters")
    print(f"\nFirst 10 characters:")
    for i in range(10):
        char = sorted_chars[i]
        print(f"  unified_idx {i}: '{char}' (U+{ord(char):04X})")

    # Verify with original mappings
    json_mapping = load_json_text_mapping(json_path)
    png_mapping = load_png_file_mapping(png_dir)

    print(f"\nVerification - '가' indices:")
    j, p = get_original_indices('가', json_mapping, png_mapping)
    print(f"  JSON text_id: {j}, PNG file_idx: {p}, Unified: {char_to_idx.get('가', -1)}")

    print(f"\nVerification - '긔' indices:")
    j, p = get_original_indices('긔', json_mapping, png_mapping)
    print(f"  JSON text_id: {j}, PNG file_idx: {p}, Unified: {char_to_idx.get('긔', -1)}")


if __name__ == "__main__":
    main()
