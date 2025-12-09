#!/usr/bin/env python
"""
Joint training with CORRECT character-to-index mapping.
Uses character text to find the correct content latent index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.style_encoder import build_style_encoder
from src.models.decoder import build_decoder


def build_char_to_index(png_dir: Path) -> dict:
    """Build character-to-content_latent_index mapping from PNG files."""
    png_files = sorted(png_dir.glob("*.png"))
    mapping = {}
    for idx, f in enumerate(png_files):
        name = f.stem  # "U+AC00"
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                char = chr(codepoint)
                mapping[char] = idx
            except ValueError:
                continue
    return mapping


def load_image(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--content_latents", type=Path, required=True)
    parser.add_argument("--png_dir", type=Path, required=True, help="Directory with U+xxxx.png files")
    parser.add_argument("--pretrained_decoder", type=Path, required=True)
    parser.add_argument("--style_encoder_path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--style_dim", type=int, default=32)
    parser.add_argument("--content_dim", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_data(args, char_to_index: dict):
    """Load data with CORRECT content latent indexing."""
    entries = json.loads(args.index.read_text())
    content_latents = np.load(args.content_latents)

    images = []
    latents = []
    chars = []

    for e in entries:
        char_text = e.get("text", "")
        if not char_text or char_text not in char_to_index:
            continue

        # CORRECT: use character text to get index
        latent_idx = char_to_index[char_text]

        img_rel = e["image_path"]
        if "/" in img_rel:
            parts = img_rel.split("/")
            img_path = args.root / parts[-2] / parts[-1]
        else:
            img_path = args.root / str(e["writer_id"]) / img_rel

        if img_path.exists() and latent_idx < len(content_latents):
            images.append(load_image(img_path))
            latents.append(content_latents[latent_idx])
            chars.append(char_text)

    if not images:
        raise ValueError("No images loaded!")

    print(f"Loaded {len(images)} images with correct mapping")
    print(f"First 5 chars: {chars[:5]}")
    return np.stack(images), np.stack(latents), chars


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build correct mapping
    print("Building character-to-index mapping...")
    char_to_index = build_char_to_index(args.png_dir)
    print(f"Mapping has {len(char_to_index)} characters")

    # Load data with correct mapping
    images, content_latents, chars = load_data(args, char_to_index)

    # Build models
    style_encoder = build_style_encoder(style_dim=args.style_dim)
    if args.style_encoder_path and args.style_encoder_path.exists():
        print(f"Loading style encoder from {args.style_encoder_path}")
        style_encoder.load_weights(args.style_encoder_path)

    # Load pretrained decoder - UNFREEZE for fine-tuning to learn handwriting style
    decoder = keras.models.load_model(args.pretrained_decoder, compile=False, safe_mode=False)
    decoder.trainable = True  # Fine-tune with lower learning rate
    print(f"Loaded pretrained decoder (trainable for fine-tuning)")

    # Build small fusion MLP: style -> delta to add to content
    # CRITICAL: Initialize final layer to zeros so we start from content (standard font)
    style_to_delta = keras.Sequential([
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(args.content_dim, kernel_initializer="zeros", bias_initializer="zeros"),
    ], name="style_to_delta")

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    mse_loss = keras.losses.MeanSquaredError()

    n_samples = len(images)
    steps_per_epoch = max(1, n_samples // args.batch_size)

    for epoch in range(args.epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_images = images[batch_idx]
            batch_content = tf.constant(content_latents[batch_idx])

            with tf.GradientTape() as tape:
                # Style encoder -> style delta -> add to content
                style_vec = style_encoder(batch_images, training=True)
                style_delta = style_to_delta(style_vec, training=True)
                fused_latent = batch_content + style_delta

                # Decoder (now trainable)
                preds = decoder(fused_latent, training=True)

                loss = mse_loss(batch_images, preds)

            # Include decoder in training
            trainable_vars = style_encoder.trainable_variables + style_to_delta.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

    # Save
    style_encoder.save(args.output_dir / "style_encoder_final.h5")
    style_to_delta.save(args.output_dir / "style_to_delta_final.h5")

    # Visualize
    print("Generating visualization...")
    n_vis = min(8, len(images))
    style_vecs = style_encoder(images[:n_vis], training=False)
    style_deltas = style_to_delta(style_vecs, training=False)
    fused = tf.constant(content_latents[:n_vis]) + style_deltas
    preds = decoder(fused, training=False).numpy()

    fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 2, 4))
    for i in range(n_vis):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"Orig: {chars[i]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].set_title("Generated", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_dir / "reconstruction.png", dpi=150)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
