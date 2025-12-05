from __future__ import annotations

"""
tf.data pipeline for style training dataset.
Outputs: image (256,256,1 float32), text_id (int32), writer_id (int32).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import tensorflow as tf

TARGET_SIZE = (256, 256)
DEFAULT_ROOT = Path("data/handwriting_raw/resizing")


def load_index(path: Path) -> Iterable[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(image_path: str, root: Path) -> str:
    """
    Normalize an image path so it points under the given root.
    If the path already contains data/handwriting_raw/resizing or the root prefix,
    strip that part and reattach root to avoid duplicate segments.
    """
    root_parts = tuple(root.parts)
    p = Path(str(image_path).strip().lstrip("/"))
    parts = tuple(p.parts)

    # If absolute, strip known prefixes if present
    if p.is_absolute():
        for i in range(len(parts) - 2):
            if parts[i : i + 3] == ("data", "handwriting_raw", "resizing"):
                return str(root / Path(*parts[i + 3 :]))
        for i in range(len(parts) - len(root_parts) + 1):
            if parts[i : i + len(root_parts)] == root_parts:
                return str(root / Path(*parts[i + len(root_parts) :]))
        return str(p)

    # Relative path
    for i in range(len(parts) - 2):
        if parts[i : i + 3] == ("data", "handwriting_raw", "resizing"):
            return str(root / Path(*parts[i + 3 :]))
    if parts[: len(root_parts)] == root_parts:
        return str(root / Path(*parts[len(root_parts) :]))
    return str(root / p)


def build_style_dataset(index_json_path: Path, batch_size: int = 32, shuffle: bool = True, root: Path = DEFAULT_ROOT) -> tf.data.Dataset:
    """Create tf.data.Dataset from index JSON."""
    entries = list(load_index(index_json_path))

    def generator():
        for e in entries:
            resolved = resolve_path(e["image_path"], root)
            yield resolved, int(e["text_id"]), int(e["writer_id"])

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    def _map_fn(image_path, text_id, writer_id):
        img_bytes = tf.io.read_file(image_path)
        img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
        img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        return img, tf.cast(text_id, tf.int32), tf.cast(writer_id, tf.int32)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build style dataset from index JSON.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_train.json"),
        help="Path to index JSON.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to resolve image paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = build_style_dataset(args.index, batch_size=args.batch_size, root=args.root)
    for batch in ds.take(2):
        imgs, text_ids, writer_ids = batch
        print(f"Image batch shape: {imgs.shape}")
        print(f"Writer IDs shape: {writer_ids.shape}")
        print(f"Text IDs shape: {text_ids.shape}")
        print(f"Pixel min/max: {tf.reduce_min(imgs).numpy():.3f} ~ {tf.reduce_max(imgs).numpy():.3f}")


if __name__ == "__main__":
    main()
