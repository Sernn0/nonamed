
"""
Joint Training Script for FontByMe.
Trains the Decoder to generate images from (Content Latent + Style Vector).
Also trains the Style Encoder if needed (or fine-tunes it).
Uses Unified Mapping and standard dataset splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

import sys
# Project root setup
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.unified_mapping import load_json_text_mapping
from src.models.decoder import build_decoder
from src.models.style_encoder import build_style_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Joint Decoder (Content + Style -> Image)")

    # Data arguments
    parser.add_argument("--train_index", type=Path, default=ROOT / "data/handwriting_processed/handwriting_index_train_shared.json")
    parser.add_argument("--val_index", type=Path, default=ROOT / "data/handwriting_processed/handwriting_index_val_shared.json")
    parser.add_argument("--content_latents", type=Path, default=ROOT / "runs/autoenc/content_latents_unified.npy")
    parser.add_argument("--png_dir", type=Path, default=ROOT / "data/content_font/NotoSansKR-Regular")

    # Model arguments
    parser.add_argument("--style_dim", type=int, default=32)
    parser.add_argument("--content_dim", type=int, default=64)
    # Using the separate Style Encoder approach
    # If starting fresh, we build new. If continuing, we load.
    parser.add_argument("--style_encoder_path", type=Path, default=None, help="Path to pretrained style encoder (optional)")
    parser.add_argument("--content_encoder_path", type=Path, default=ROOT / "runs/autoenc/encoder.h5", help="Path to content encoder to keep fixed (if needed)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4) # Moderate LR
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs/joint")

    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    """Load image as (256, 256, 1) float32 [0, 1]."""
    img = Image.open(path).convert("L")
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def create_dataset(index_path: Path, content_latents_path: Path, unified_mapping: dict, batch_size: int, shuffle: bool = True):
    """
    Create a tf.data.Dataset for efficient parallel data loading.

    Returns:
        (dataset, num_samples): TF Dataset and total sample count
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content_latents = np.load(content_latents_path)

    # Filter valid entries and prepare paths/indices
    image_paths = []
    content_indices = []

    for e in data:
        char = e.get('text')
        idx = unified_mapping.get(char)
        if idx is not None and idx < len(content_latents):
            p = ROOT / e['image_path']
            if p.exists():
                image_paths.append(str(p))
                content_indices.append(idx)

    num_samples = len(image_paths)
    print(f"[DATA] Valid samples: {num_samples}")

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    content_indices = np.array(content_indices, dtype=np.int32)
    content_latents_tensor = tf.constant(content_latents, dtype=tf.float32)

    # TF image loading function
    def load_and_preprocess(path, c_idx):
        # Read file
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32) / 255.0

        # Get content latent
        c_vec = tf.gather(content_latents_tensor, c_idx)

        return img, c_vec

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, content_indices))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, num_samples))

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_samples


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Unified Mapping (using scripts/unified_mapping.py logic which we moved to src/data)
    # But wait, we need the mapping dict char -> idx.
    # We can rebuild it or load it? unified_mapping.py builds it on fly usually.
    # Let's import build_unified_mapping
    from src.data.unified_mapping import build_unified_mapping
    # We need a json path and png dir to build it validly
    print("[INFO] Building Unified Mapping...")
    unified_mapping, _ = build_unified_mapping(args.train_index, args.png_dir)
    print(f"[INFO] Mapping size: {len(unified_mapping)}")

    # 2. Models
    print("[INFO] Building Models...")
    # Style Encoder
    style_encoder = build_style_encoder(style_dim=args.style_dim)
    if args.style_encoder_path and args.style_encoder_path.exists():
        style_encoder.load_weights(str(args.style_encoder_path))
        print(f"[INFO] Loaded Style Encoder from {args.style_encoder_path}")

    # Decoder
    # Inputs: [Content(64), Style(32)] -> Image(256,256,1)
    decoder = build_decoder(content_dim=args.content_dim, style_dim=args.style_dim)
    # If we had a pretrained decoder (content-only), we can't load it directly because architecture changed (2 inputs vs 1).
    # Content Autoencoder Decoder: Input(64) -> Dense -> Reshape -> Conv2DTranspose...
    # Joint Decoder: Input([64,32]) -> Dense -> Reshape -> Conv2DTranspose...
    # The layers AFTER the first Dense are identical.
    # We *could* transfer weights for the convolutional layers if we want.
    # For now, let's train from scratch or handle weight transfer later if convergence is slow.

    # 3. Training Loop Setup
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    # Loss weights
    MSE_WEIGHT = 0.4
    L1_WEIGHT = 0.4
    SSIM_WEIGHT = 0.2
    GLYPH_WEIGHT = 20.0  # Extra weight for dark pixels (glyphs)

    # 4. Custom Training Step with Combined Loss
    @tf.function
    def compute_loss(images, preds):
        """Combined Weighted MSE + L1 + SSIM loss."""
        # Pixel weights: emphasize glyph (dark) pixels
        weights = 1.0 + (1.0 - images) * GLYPH_WEIGHT

        # Weighted MSE
        mse = tf.reduce_mean(weights * tf.square(images - preds))

        # Weighted L1 (MAE) for sharper edges
        l1 = tf.reduce_mean(weights * tf.abs(images - preds))

        # SSIM (Structural Similarity) for perceptual quality
        # SSIM returns values in [-1, 1], we want loss so use 1 - ssim
        ssim_val = tf.reduce_mean(tf.image.ssim(images, preds, max_val=1.0))
        ssim_loss = 1.0 - ssim_val

        # Combined loss
        loss = MSE_WEIGHT * mse + L1_WEIGHT * l1 + SSIM_WEIGHT * ssim_loss
        return loss, mse, l1, ssim_loss

    @tf.function
    def train_step(images, content_vecs):
        with tf.GradientTape() as tape:
            # 1. Encode Style
            style_vecs = style_encoder(images, training=True)

            # 2. Decode (Content + Style)
            preds = decoder([content_vecs, style_vecs], training=True)

            # 3. Combined Loss
            loss, _, _, _ = compute_loss(images, preds)

        # Gradients
        trainable_vars = style_encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    @tf.function
    def val_step(images, content_vecs):
        style_vecs = style_encoder(images, training=False)
        preds = decoder([content_vecs, style_vecs], training=False)

        loss, _, _, _ = compute_loss(images, preds)
        return loss

    # 5. Run Training
    print("[INFO] Starting Training...")

    # Create tf.data.Dataset (parallel loading)
    train_dataset, n_train = create_dataset(args.train_index, args.content_latents, unified_mapping, args.batch_size, shuffle=True)
    val_dataset, n_val = create_dataset(args.val_index, args.content_latents, unified_mapping, args.batch_size, shuffle=False)

    steps_per_epoch = n_train // args.batch_size
    val_steps = n_val // args.batch_size

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss_sum = 0
        step = 0
        for images, content_vecs in train_dataset:
            loss = train_step(images, content_vecs)
            train_loss_sum += loss
            step += 1
            if step % 100 == 0:
                print(f"  Step {step}/{steps_per_epoch} Loss: {loss:.4f}", end='\r')

        avg_train_loss = train_loss_sum / max(step, 1)
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # Val
        val_loss_sum = 0
        val_step_count = 0
        for images, content_vecs in val_dataset:
            loss = val_step(images, content_vecs)
            val_loss_sum += loss
            val_step_count += 1

        avg_val_loss = val_loss_sum / max(val_step_count, 1)
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  [Save] New best validation loss!")
            style_encoder.save(args.output_dir / "style_encoder_best.h5")
            decoder.save(args.output_dir / "decoder_best.h5")

        # Periodic Save
        if (epoch + 1) % 5 == 0:
             # Visualization
             # (Reuse same batch from val info if possible or take new)
             pass

    print("[INFO] Training Complete.")

if __name__ == "__main__":
    main()
