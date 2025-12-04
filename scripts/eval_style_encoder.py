from __future__ import annotations

"""
Evaluate style encoder classifier and extract style embeddings on validation set.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure project root on sys.path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.style_dataset import build_style_dataset
from src.models.style_encoder import build_style_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate style encoder and extract embeddings.")
    parser.add_argument(
        "--train_index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_train.json"),
        help="Path to train index JSON (for writer mapping).",
    )
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
        help="Path to classifier model weights.",
    )
    parser.add_argument(
        "--backbone_path",
        type=Path,
        default=Path("runs/style_encoder/encoder_style_backbone.h5"),
        help="Path to backbone model weights.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/style_encoder/eval"),
        help="Output directory for embeddings.",
    )
    parser.add_argument("--style_dim", type=int, default=32, help="Style vector dimension.")
    return parser.parse_args()


def load_writer_mapping(index_paths: List[Path]) -> Dict[int, int]:
    writer_ids: Set[int] = set()
    for path in index_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for e in data:
            if "writer_id" in e:
                try:
                    writer_ids.add(int(e["writer_id"]))
                except Exception:
                    continue
    sorted_ids = sorted(writer_ids)
    return {wid: idx for idx, wid in enumerate(sorted_ids)}


def remap_dataset(ds: tf.data.Dataset, writer_to_idx: Dict[int, int]) -> tf.data.Dataset:
    keys = tf.constant(list(writer_to_idx.keys()), dtype=tf.int32)
    vals = tf.constant(list(writer_to_idx.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1,
    )

    def _map_fn(image, text_id, writer_id):
        mapped = table.lookup(tf.cast(writer_id, tf.int32))
        return image, mapped

    return ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def build_classifier(style_dim: int, num_writers: int) -> keras.Model:
    base_encoder = build_style_encoder(style_dim=style_dim, input_shape=(256, 256, 1))
    x = base_encoder.output
    logits = keras.layers.Dense(num_writers, activation="softmax", name="writer_logits")(x)
    model = keras.Model(inputs=base_encoder.input, outputs=logits, name="style_encoder_classifier")
    return model, base_encoder


def evaluate_classifier(model_path: Path, style_dim: int, num_writers: int, val_ds: tf.data.Dataset) -> None:
    model, _ = build_classifier(style_dim=style_dim, num_writers=num_writers)
    model.load_weights(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    results = model.evaluate(val_ds, verbose=1)
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        loss, acc = results[0], results[1]
    else:
        loss, acc = results, None
    if acc is not None:
        print(f"Val loss: {loss:.4f}, Val accuracy: {acc:.4f}")
    else:
        print(f"Val loss: {loss}")


def extract_embeddings(backbone_path: Path, style_dim: int, val_ds: tf.data.Dataset, out_dir: Path) -> None:
    backbone = build_style_encoder(style_dim=style_dim, input_shape=(256, 256, 1))
    backbone.load_weights(backbone_path)
    style_vecs = []
    writer_ids = []
    for batch in val_ds:
        # Support both tuple (image, text_id, writer_id) and dict batches
        if isinstance(batch, dict):
            images = batch["image"]
            wids = batch["writer_id"]
        else:
            images, _, wids = batch
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    writer_to_idx = load_writer_mapping([args.train_index, args.val_index])
    num_writers = len(writer_to_idx)
    if num_writers == 0:
        raise ValueError("No writer_ids found in index files.")
    print(f"[INFO] Num writers: {num_writers}")

    val_ds_raw = build_style_dataset(
        index_json_path=args.val_index,
        root=args.root,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_ds_mapped = remap_dataset(val_ds_raw, writer_to_idx)

    evaluate_classifier(args.model_path, args.style_dim, num_writers, val_ds_mapped)
    extract_embeddings(args.backbone_path, args.style_dim, val_ds_raw, args.out_dir)


if __name__ == "__main__":
    main()
