from __future__ import annotations

"""
Check existence of image paths in split index files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

DEFAULT_FILES = [
    Path("data/handwriting_processed/handwriting_index_train.json"),
    Path("data/handwriting_processed/handwriting_index_val.json"),
    Path("data/handwriting_processed/handwriting_index_test.json"),
]
REPORT_PATH = Path("data/handwriting_processed/image_check_report.json")


def load_index(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_paths(files: List[Path]) -> Dict:
    report = {"missing": {}, "checked_files": []}
    for file_path in files:
        if not file_path.is_file():
            report["missing"][str(file_path)] = ["index_file_missing"]
            continue
        entries = load_index(file_path)
        missing_images = []
        for entry in entries:
            img_path = entry.get("image_path")
            if not isinstance(img_path, str) or not os.path.exists(img_path):
                missing_images.append(img_path)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_paths(args.files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Image path check status: {report.get('status')}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
