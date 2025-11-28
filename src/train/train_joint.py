from __future__ import annotations

"""
Skeleton for joint training of content encoder (frozen), style encoder, and decoder.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root on sys.path for src imports when run as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from tensorflow.keras.models import load_model

from src.models.style_encoder import build_style_encoder
from src.models.decoder import build_decoder


# Default paths (adjust if needed)
CONTENT_ENCODER_PATH = Path("runs/autoenc/encoder.h5")
CONTENT_LATENTS_PATH = Path("data/latents/content_latents.npy")


def load_content_encoder(path: Path) -> tf.keras.Model:
    """Load and freeze the pretrained content encoder."""
    encoder = load_model(path)
    encoder.trainable = False
    return encoder


def lookup_content_vec(char_id: tf.Tensor) -> tf.Tensor:
    """
    Placeholder for content latent lookup using char_id.
    TODO: load CONTENT_LATENTS_PATH and index by char_id.
    """
    # Dummy zero vector; replace with real lookup.
    return tf.zeros((tf.shape(char_id)[0], 64), dtype=tf.float32)


class JointTrainer:
    """Encapsulate the joint training step."""

    def __init__(
        self,
        content_encoder: tf.keras.Model,
        style_encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        learning_rate: float = 1e-4,
    ):
        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self, handwriting_img: tf.Tensor, char_id: tf.Tensor, writer_id: tf.Tensor) -> Dict[str, tf.Tensor]:
        """One training step (reconstruction-based)."""
        del writer_id  # Placeholder; could be used for style supervision later.
        with tf.GradientTape() as tape:
            content_vec = lookup_content_vec(char_id)
            style_vec = self.style_encoder(handwriting_img, training=True)
            pred_img = self.decoder([content_vec, style_vec], training=True)
            loss = self.loss_fn(handwriting_img, pred_img)

        trainable_vars = self.style_encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {"loss": loss}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint training skeleton.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load/freeze content encoder.
    content_encoder = load_content_encoder(CONTENT_ENCODER_PATH)

    # Build trainable style encoder and decoder.
    style_encoder = build_style_encoder(style_dim=32)
    decoder = build_decoder(content_dim=64, style_dim=32)

    trainer = JointTrainer(
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        decoder=decoder,
        learning_rate=args.learning_rate,
    )

    # Print model summaries.
    print("Content encoder (frozen):")
    content_encoder.summary()
    print("Style encoder:")
    style_encoder.summary()
    print("Decoder:")
    decoder.summary()

    # TODO: Replace with actual tf.data pipeline.
    # Example placeholder: train_dataset = build_style_dataset("train", batch_size=args.batch_size)
    train_dataset = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        if train_dataset is None:
            print("TODO: train_dataset not implemented; skipping training loop.")
            break
        for step, (handwriting_img, char_id, writer_id) in enumerate(train_dataset):
            metrics = trainer.train_step(handwriting_img, char_id, writer_id)
            if step % 10 == 0:
                print(f"Step {step}: loss={metrics['loss'].numpy():.4f}")


if __name__ == "__main__":
    main()
