from __future__ import annotations

"""
Validate master handwriting index JSON for required fields and paths.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS = ["image_path", "text", "text_id", "writer_id"]
DEFAULT_INPUT = Path("data/handwriting_processed/master_index.json")
DEFAULT_REPORT = Path("data/handwriting_processed/schema_check_report.json")


def validate_entry(entry: Dict) -> List[str]:
    """Return a list of problems for a single entry."""
    problems = []
    for key in REQUIRED_KEYS:
        if key not in entry:
            problems.append(f"missing_key:{key}")
    if "image_path" in entry:
        path = entry["image_path"]
        if not isinstance(path, str):
            problems.append("image_path_not_str")
        elif not os.path.exists(path):
            problems.append("image_missing")
    if "text" in entry and not isinstance(entry.get("text"), str):
        problems.append("text_not_str")
    if "text_id" in entry and not isinstance(entry.get("text_id"), int):
        problems.append("text_id_not_int")
    if "writer_id" in entry and not isinstance(entry.get("writer_id"), int):
        problems.append("writer_id_not_int")
    return problems


def validate_index(input_path: Path) -> Dict:
    """Validate the master index and return a report dictionary."""
    if not input_path.is_file():
        return {"status": "error", "message": f"file not found: {input_path}"}

    try:
        entries = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "message": f"failed to read json: {exc}"}

    issues = []
    for idx, entry in enumerate(entries):
        probs = validate_entry(entry)
        if probs:
            issues.append({"index": idx, "problems": probs, "entry": entry})

    if issues:
        return {"status": "fail", "issues": issues}
    return {"status": "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate master handwriting index JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to master index JSON (default: data/handwriting_processed/master_index.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to write validation report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_index(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation status: {report.get('status')}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
