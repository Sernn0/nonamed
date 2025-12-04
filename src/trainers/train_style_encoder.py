from __future__ import annotations

"""
Train style encoder backbone with writer classification.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks

# Ensure project root on sys.path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.style_dataset import build_style_dataset
from src.models.style_encoder import build_style_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train style encoder via writer classification.")
    parser.add_argument(
        "--train_index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_train.json"),
        help="Path to train index JSON.",
    )
    parser.add_argument(
        "--val_index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_val.json"),
        help="Path to val index JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/handwriting_raw/resizing"),
        help="Root directory for resolving image paths.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--style_dim", type=int, default=32, help="Style vector dimension.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/style_encoder"),
        help="Output directory for models and logs.",
    )
    return parser.parse_args()


def load_writer_mapping(index_paths: List[Path]) -> Dict[int, int]:
    """Create writer_id -> class index mapping from given index files."""
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
    """Map raw writer_id to contiguous class indices and drop text_id."""
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


def build_datasets(args: argparse.Namespace, writer_to_idx: Dict[int, int]):
    train_ds_raw = build_style_dataset(
        index_json_path=args.train_index,
        batch_size=args.batch_size,
        shuffle=True,
        root=args.root,
    )
    val_ds_raw = build_style_dataset(
        index_json_path=args.val_index,
        batch_size=args.batch_size,
        shuffle=False,
        root=args.root,
    )
    train_ds = remap_dataset(train_ds_raw, writer_to_idx)
    val_ds = remap_dataset(val_ds_raw, writer_to_idx)
    return train_ds, val_ds


def build_classifier(style_dim: int, num_writers: int) -> keras.Model:
    base_encoder = build_style_encoder(style_dim=style_dim, input_shape=(256, 256, 1))
    x = base_encoder.output
    logits = layers.Dense(num_writers, activation="softmax", name="writer_logits")(x)
    model = keras.Model(inputs=base_encoder.input, outputs=logits, name="style_encoder_classifier")
    return model, base_encoder


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    writer_to_idx = load_writer_mapping([args.train_index, args.val_index])
    num_writers = len(writer_to_idx)
    if num_writers == 0:
        raise ValueError("No writer_ids found in index files.")
    print(f"[INFO] Num writers: {num_writers}")

    train_ds, val_ds = build_datasets(args, writer_to_idx)

    model, base_encoder = build_classifier(style_dim=args.style_dim, num_writers=num_writers)
    optimizer = optimizers.Adam(learning_rate=1e-4)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "encoder_style_full.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        callbacks.TensorBoard(log_dir=str(logs_dir)),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list,
    )

    # Save final classifier and backbone
    model.save(args.out_dir / "encoder_style_full.h5")
    backbone = keras.Model(inputs=base_encoder.input, outputs=base_encoder.output, name="style_encoder_backbone")
    backbone.save(args.out_dir / "encoder_style_backbone.h5")
    print(f"[INFO] Saved models to {args.out_dir}")


if __name__ == "__main__":
    main()
