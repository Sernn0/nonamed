from __future__ import annotations

"""
Joint training script: style encoder + decoder with fixed content latents.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses

# Ensure project root on sys.path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.style_dataset import build_style_dataset
from src.models.style_encoder import build_style_encoder
from src.models.decoder import build_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint training: style encoder + decoder."
    )
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
    parser.add_argument(
        "--content_latents",
        type=Path,
        default=Path("runs/autoenc/content_latents.npy"),
        help="Path to content_latents.npy (shape: [num_chars, content_dim]).",
    )
    parser.add_argument(
        "--style_encoder_path",
        type=Path,
        default=Path("runs/style_encoder/encoder_style_backbone.h5"),
        help="Path to pretrained style encoder (optional).",
    )
    parser.add_argument(
        "--decoder_path",
        type=Path,
        default=None,
        help="Path to decoder weights (optional).",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--style_dim", type=int, default=32, help="Style latent dimension."
    )
    parser.add_argument(
        "--content_dim", type=int, default=64, help="Content latent dimension."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--mae_weight",
        type=float,
        default=0.0,
        help="Optional MAE weight to mix with MSE.",
    )
    return parser.parse_args()


def load_content_latents(path: Path) -> tf.Tensor:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D content latents, got shape {arr.shape}")
    return tf.constant(arr, dtype=tf.float32)


def load_style_encoder(path: Path, style_dim: int) -> keras.Model:
    model = build_style_encoder(style_dim=style_dim, input_shape=(256, 256, 1))
    if path.is_file():
        try:
            print(f"[INFO] Loading style encoder weights from {path}")
            model.load_weights(path)
            return model
        except Exception as exc:
            print(
                f"[WARN] Failed to load style encoder weights: {exc}. Using fresh initialization."
            )
    return model


def load_decoder(path: Path, content_dim: int, style_dim: int) -> keras.Model:
    model = build_decoder(
        content_dim=content_dim, style_dim=style_dim, img_shape=(256, 256, 1)
    )
    if path and Path(path).is_file():
        try:
            print(f"[INFO] Loading decoder weights from {path}")
            model.load_weights(path)
        except Exception as exc:
            print(f"[WARN] Failed to load decoder weights: {exc}. Using fresh weights.")
    return model


def compute_loss(
    images: tf.Tensor,
    preds: tf.Tensor,
    mse_fn: losses.Loss,
    mae_fn: losses.Loss,
    mae_weight: float,
) -> tf.Tensor:
    mse = mse_fn(images, preds)
    if mae_weight > 0.0:
        mae = mae_fn(images, preds)
        return mse + mae_weight * mae
    return mse


class Trainer:
    def __init__(
        self,
        content_latents: tf.Tensor,
        style_encoder: keras.Model,
        decoder: keras.Model,
        learning_rate: float,
        mae_weight: float,
    ):
        self.content_latents = content_latents  # (num_chars, content_dim)
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.mse_fn = losses.MeanSquaredError()
        self.mae_fn = losses.MeanAbsoluteError()
        self.mae_weight = mae_weight
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")

    def _forward(
        self, batch: Dict[str, tf.Tensor], training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Support dict batches and tuple batches
        if isinstance(batch, dict):
            images = batch["image"]
            text_ids = tf.cast(batch["text_id"], tf.int32)
        else:
            images, text_ids, _ = batch
            text_ids = tf.cast(text_ids, tf.int32)
        content_vec = tf.gather(self.content_latents, text_ids)
        style_vec = self.style_encoder(images, training=training)
        preds = self.decoder([content_vec, style_vec], training=training)
        loss = compute_loss(images, preds, self.mse_fn, self.mae_fn, self.mae_weight)
        return preds, loss, images

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            preds, loss, _ = self._forward(batch, training=True)
        vars_to_train = (
            self.style_encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, vars_to_train)
        self.optimizer.apply_gradients(zip(grads, vars_to_train))
        self.train_loss.update_state(loss)
        return loss

    @tf.function
    def val_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        preds, loss, _ = self._forward(batch, training=False)
        self.val_loss.update_state(loss)
        return loss


def run_training(args: argparse.Namespace) -> None:
    # Load content latents
    content_latents = load_content_latents(args.content_latents)

    # Datasets
    train_ds = build_style_dataset(
        index_json_path=args.train_index,
        batch_size=args.batch_size,
        shuffle=True,
        root=args.root,
    )
    val_ds = build_style_dataset(
        index_json_path=args.val_index,
        batch_size=args.batch_size,
        shuffle=False,
        root=args.root,
    )

    # Models
    style_encoder = load_style_encoder(
        args.style_encoder_path, style_dim=args.style_dim
    )
    decoder = load_decoder(
        args.decoder_path, content_dim=args.content_dim, style_dim=args.style_dim
    )

    trainer = Trainer(
        content_latents=content_latents,
        style_encoder=style_encoder,
        decoder=decoder,
        learning_rate=args.learning_rate,
        mae_weight=args.mae_weight,
    )

    output_dir = Path("runs/joint")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        trainer.train_loss.reset_state()
        trainer.val_loss.reset_state()

        for _, batch in enumerate(train_ds):
            loss = trainer.train_step(batch)
            trainer.train_loss.update_state(loss)

        for _, batch in enumerate(val_ds):
            val_loss = trainer.val_step(batch)
            trainer.val_loss.update_state(val_loss)

        train_loss_val = trainer.train_loss.result().numpy()
        val_loss_val = trainer.val_loss.result().numpy()
        print(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {train_loss_val:.4f} - val_loss: {val_loss_val:.4f}"
        )

        # Save per-epoch checkpoints
        style_encoder.save(output_dir / f"style_encoder_epoch{epoch+1:02d}.h5")
        decoder.save(output_dir / f"decoder_epoch{epoch+1:02d}.h5")

    # Final save
    style_encoder.save(output_dir / "style_encoder_final.h5")
    decoder.save(output_dir / "decoder_final.h5")
    print(f"[INFO] Training complete. Models saved to {output_dir}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
