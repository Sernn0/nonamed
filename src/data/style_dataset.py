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


def load_index(path: Path) -> Iterable[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sample(image_path: tf.Tensor, text_id: tf.Tensor, writer_id: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Read image, convert to grayscale float32 [0,1]."""
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_bytes, channels=1)
    img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img, tf.cast(text_id, tf.int32), tf.cast(writer_id, tf.int32)


def build_style_dataset(index_json_path: Path, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """Create tf.data.Dataset from index JSON."""
    entries = list(load_index(index_json_path))

    def generator():
        for e in entries:
            yield e["image_path"], e["text_id"], e["writer_id"]

    output_types = (tf.string, tf.int32, tf.int32)
    ds = tf.data.Dataset.from_generator(generator, output_types=output_types)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    ds = ds.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)
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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = build_style_dataset(args.index, batch_size=args.batch_size)
    for batch in ds.take(1):
        imgs, text_ids, writer_ids = batch
        print("Image batch shape:", imgs.shape)
        print("Text IDs shape:", text_ids.shape)
        print("Writer IDs shape:", writer_ids.shape)


if __name__ == "__main__":
    main()
