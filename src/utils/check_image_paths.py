from __future__ import annotations

"""
Check existence of image paths in split index files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_FILES = [
    Path("data/handwriting_processed/handwriting_index_train.json"),
    Path("data/handwriting_processed/handwriting_index_val.json"),
    Path("data/handwriting_processed/handwriting_index_test.json"),
]
REPORT_PATH = Path("data/handwriting_processed/image_check_report.json")
DEFAULT_ROOT = Path("data/handwriting_raw/resizing")


def load_index(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_str: str, root: Optional[Path]) -> str:
    """
    Resolve image path against optional root.
    Tries, in order:
      1) as-is
      2) root / (path without leading '/')
      3) if path starts with root prefix, drop it and join remainder to root
      4) root / filename
    """
    candidates = [Path(path_str)]
    if root:
        stripped = Path(path_str.lstrip("/"))
        candidates.append(root / stripped)

        root_parts = tuple(root.parts)
        stripped_parts = tuple(stripped.parts)
        if stripped_parts[: len(root_parts)] == root_parts:
            remaining = Path(*stripped_parts[len(root_parts) :])
            candidates.append(root / remaining)

        filename = stripped.name
        candidates.append(root / filename)

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[-1])


def check_paths(files: List[Path], root: Optional[Path]) -> Dict:
    report = {"missing": {}, "checked_files": []}
    for file_path in files:
        if not file_path.is_file():
            report["missing"][str(file_path)] = ["index_file_missing"]
            continue
        entries = load_index(file_path)
        missing_images = []
        for entry in entries:
            img_path = entry.get("image_path")
            if not isinstance(img_path, str):
                missing_images.append(img_path)
                continue
            # try resolving with root
            resolved = resolve_path(img_path, root)
            entry["resolved_image_path"] = resolved
            if not os.path.exists(resolved):
                missing_images.append(resolved)
        if missing_images:
            report["missing"][str(file_path)] = missing_images
        report["checked_files"].append(str(file_path))
    if not report["missing"]:
        return {"status": "ok", "checked_files": report["checked_files"]}
    report["status"] = "fail"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image paths in split index files.")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=DEFAULT_FILES,
        help="List of index JSON files to check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help="Path to save image check report.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to resolve image paths (default: data/handwriting_raw/resizing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_paths(args.files, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Image path check status: {report.get('status')}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
