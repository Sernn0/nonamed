from __future__ import annotations

"""
Split handwriting index by writer into train/val/test (80/10/10).
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

DEFAULT_INPUT = Path("data/handwriting_processed/master_index.json")
OUTPUT_DIR = Path("data/handwriting_processed")


def load_index(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_writer(entries: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in entries:
        writer_id = entry.get("writer_id")
        writer_key = str(writer_id)
        grouped.setdefault(writer_key, []).append(entry)
    return grouped


def split_writers(writer_ids: List[str], seed: int = 42) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    rng.shuffle(writer_ids)
    n = len(writer_ids)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    return {
        "train": writer_ids[:train_end],
        "val": writer_ids[train_end:val_end],
        "test": writer_ids[val_end:],
    }


def save_split(entries: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(splits: Dict[str, List[Dict]]) -> None:
    for name, data in splits.items():
        writers = {str(e.get("writer_id")) for e in data}
        print(f"{name}: writers={len(writers)}, samples={len(data)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split handwriting index by writer (80/10/10).")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to master index JSON (default: data/handwriting_processed/master_index.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_index(args.input)
    grouped = group_by_writer(entries)
    writer_ids = list(grouped.keys())
    splits_ids = split_writers(writer_ids, seed=args.seed)

    splits_entries = {
        name: [e for wid in wid_list for e in grouped[wid]]
        for name, wid_list in splits_ids.items()
    }

    train_path = OUTPUT_DIR / "handwriting_index_train.json"
    val_path = OUTPUT_DIR / "handwriting_index_val.json"
    test_path = OUTPUT_DIR / "handwriting_index_test.json"

    save_split(splits_entries.get("train", []), train_path)
    save_split(splits_entries.get("val", []), val_path)
    save_split(splits_entries.get("test", []), test_path)

    summarize(splits_entries)
    print(f"Saved splits to: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
