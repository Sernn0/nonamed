from __future__ import annotations

"""
Evaluate style encoder classifier and extract style embeddings on validation set.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure project root on sys.path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.style_dataset import build_style_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate style encoder and extract embeddings.")
    parser.add_argument(
        "--val_index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_val.json"),
        help="Path to validation index JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/handwriting_raw/resizing"),
        help="Root directory for resolving image paths.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("runs/style_encoder/encoder_style_full.h5"),
        help="Path to classifier model (full).",
    )
    parser.add_argument(
        "--backbone_path",
        type=Path,
        default=Path("runs/style_encoder/encoder_style_backbone.h5"),
        help="Path to backbone model.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/style_encoder/eval"),
        help="Output directory for embeddings.",
    )
    return parser.parse_args()


def evaluate_classifier(model_path: Path, val_ds: tf.data.Dataset) -> None:
    clf = keras.models.load_model(model_path)
    # Optional: print summary
    # clf.summary()
    results = clf.evaluate(val_ds, verbose=1)
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        loss, acc = results[0], results[1]
    else:
        loss, acc = results, None
    if acc is not None:
        print(f"Val loss: {loss:.4f}, Val accuracy: {acc:.4f}")
    else:
        print(f"Val loss: {loss}")


def extract_embeddings(backbone_path: Path, val_ds: tf.data.Dataset, out_dir: Path) -> None:
    backbone = keras.models.load_model(backbone_path)
    style_vecs = []
    writer_ids = []
    for batch in val_ds:
        images = batch["image"]
        wids = batch["writer_id"]
        vecs = backbone(images, training=False).numpy()
        style_vecs.append(vecs)
        writer_ids.append(wids.numpy())

    if style_vecs:
        style_vecs_arr = np.concatenate(style_vecs, axis=0)
        writer_ids_arr = np.concatenate(writer_ids, axis=0)
    else:
        style_vecs_arr = np.zeros((0, backbone.output_shape[-1]), dtype=np.float32)
        writer_ids_arr = np.zeros((0,), dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "val_style_embeddings.npy", style_vecs_arr)
    np.save(out_dir / "val_writer_ids.npy", writer_ids_arr)

    print("Embedding shape:", style_vecs_arr.shape)
    print("Writer IDs shape:", writer_ids_arr.shape)
    if writer_ids_arr.size > 0:
        uniq = np.unique(writer_ids_arr)
        print("Unique writers:", uniq.size, "min:", uniq.min(), "max:", uniq.max())
    print("Embedding mean:", style_vecs_arr.mean(), "std:", style_vecs_arr.std())


def main() -> None:
    args = parse_args()
    val_ds = build_style_dataset(
        index_json_path=args.val_index,
        root=args.root,
        batch_size=args.batch_size,
        shuffle=False,
    )

    evaluate_classifier(args.model_path, val_ds)
    extract_embeddings(args.backbone_path, val_ds, args.out_dir)


if __name__ == "__main__":
    main()
