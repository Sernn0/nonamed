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
    Resolve image_path using the given root.
    Tries as-is, root/stripped, and if the path already has the root prefix, strip it.
    """
    p = Path(image_path.lstrip("/"))
    if p.is_absolute():
        return str(p)
    # If path already starts with root parts, strip them
    root_parts = tuple(root.parts)
    p_parts = tuple(p.parts)
    if p_parts[: len(root_parts)] == root_parts:
        remaining = Path(*p_parts[len(root_parts) :])
        return str(root / remaining)
    return str(root / p)


def parse_sample(image_path: tf.Tensor, text_id: tf.Tensor, writer_id: tf.Tensor, root: Path) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Read image, convert to grayscale float32 [0,1]."""
    resolved = resolve_path(image_path.numpy().decode("utf-8"), root)
    img_bytes = tf.io.read_file(resolved)
    img = tf.io.decode_png(img_bytes, channels=1)
    img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img, tf.cast(text_id, tf.int32), tf.cast(writer_id, tf.int32)


def tf_parse_sample(root: Path):
    """Wrapper to use parse_sample inside tf.data map."""

    def _fn(image_path, text_id, writer_id):
        def py_parse(ip, tid, wid):
            # ip may be tf.Tensor/bytes/str; normalize to str
            if hasattr(ip, "numpy"):
                ip = ip.numpy()
            if isinstance(ip, bytes):
                ip_str = ip.decode("utf-8")
            else:
                ip_str = str(ip)
            resolved = resolve_path(ip_str, root)
            img_bytes = tf.io.read_file(resolved)
            img = tf.io.decode_png(img_bytes, channels=1)
            img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            return img.numpy(), int(tid), int(wid)

        img, tid, wid = tf.py_function(
            func=py_parse,
            inp=[image_path, text_id, writer_id],
            Tout=[tf.float32, tf.int32, tf.int32],
        )
        img.set_shape((TARGET_SIZE[0], TARGET_SIZE[1], 1))
        tid.set_shape(())
        wid.set_shape(())
        return img, tid, wid

    return _fn


def build_style_dataset(index_json_path: Path, batch_size: int = 32, shuffle: bool = True, root: Path = DEFAULT_ROOT) -> tf.data.Dataset:
    """Create tf.data.Dataset from index JSON."""
    entries = list(load_index(index_json_path))

    def generator():
        for e in entries:
            yield e["image_path"], int(e["text_id"]), int(e["writer_id"])

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(entries))
    ds = ds.map(tf_parse_sample(root), num_parallel_calls=tf.data.AUTOTUNE)
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
