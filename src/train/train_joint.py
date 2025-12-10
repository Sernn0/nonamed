
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


def data_generator(index_path: Path, content_latents_path: Path, unified_mapping: dict, batch_size: int):
    """
    Generator that yields ( [content_vec, image], target_image ) pairs?
    No, for Joint Training:
    Input: [Content Latent, Style Vector] (Style Vector comes from Style Encoder(Reference Image))

    Training Loop Strategy:
    1. Sample a batch of (Image, Character) pairs.
    2. Get Content Latent for Character (from lookup).
    3. Pass Image to Style Encoder -> Style Vector.
    4. Pass [Content Latent, Style Vector] to Decoder -> Pred Image.
    5. Loss = MSE(Image, Pred Image).

    Wait, if we train Style Encoder simultaneously, this works.
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content_latents = np.load(content_latents_path)

    # Filter valid entries
    valid_data = []
    for e in data:
        char = e.get('text')
        idx = unified_mapping.get(char)
        if idx is not None and idx < len(content_latents):
            e['_unified_idx'] = idx
            valid_data.append(e)

    num_samples = len(valid_data)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_entries = [valid_data[k] for k in batch_idx]

            # Prepare batch
            images = []
            c_latents = []

            for entry in batch_entries:
                # Load image
                # Logic to find absolute path
                # index json usually has relative path from project root or data root
                # "image_path": "data/handwriting_raw/..."
                p = ROOT / entry['image_path']
                if not p.exists():
                    # Fallback or skip?
                    # For generator safety, maybe output zero/skip?
                    # Ideally data cleaning removed these.
                    img = np.zeros((256, 256, 1), dtype=np.float32)
                else:
                    img = load_image(p)

                images.append(img)
                c_latents.append(content_latents[entry['_unified_idx']])

            batch_images = np.array(images)      # (B, 256, 256, 1)
            batch_content = np.array(c_latents)  # (B, 64)

            yield batch_images, batch_content


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
    mse_loss = keras.losses.MeanSquaredError()

    # 4. Custom Training Step with Weighted Loss
    @tf.function
    def train_step(images, content_vecs):
        with tf.GradientTape() as tape:
            # 1. Encode Style
            style_vecs = style_encoder(images, training=True)

            # 2. Decode (Content + Style)
            preds = decoder([content_vecs, style_vecs], training=True)

            # 3. Weighted Loss
            # images: 0=black(glyph), 1=white(ckpt)
            # We want to penalize errors on black pixels (glyph) more.
            # Weight = 1 + (1 - target) * 20  -> White pixel weight=1, Black pixel weight=21
            weights = 1.0 + (1.0 - images) * 20.0

            mse = tf.square(images - preds)
            loss = tf.reduce_mean(weights * mse)

        # Gradients
        trainable_vars = style_encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    @tf.function
    def val_step(images, content_vecs):
        style_vecs = style_encoder(images, training=False)
        preds = decoder([content_vecs, style_vecs], training=False)

        # Consistent weighted loss for validation
        weights = 1.0 + (1.0 - images) * 20.0
        mse = tf.square(images - preds)
        loss = tf.reduce_mean(weights * mse)
        return loss

    # 5. Run Training
    print("[INFO] Starting Training...")

    # Generators
    train_gen = data_generator(args.train_index, args.content_latents, unified_mapping, args.batch_size)
    val_gen = data_generator(args.val_index, args.content_latents, unified_mapping, args.batch_size)

    # Quick count for steps
    with open(args.train_index) as f: n_train = len(json.load(f))
    with open(args.val_index) as f: n_val = len(json.load(f))
    steps_per_epoch = n_train // args.batch_size
    val_steps = n_val // args.batch_size

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss_sum = 0
        for step in range(steps_per_epoch):
            images, content_vecs = next(train_gen)
            loss = train_step(images, content_vecs)
            train_loss_sum += loss
            if step % 100 == 0:
                print(f"  Step {step}/{steps_per_epoch} Loss: {loss:.4f}", end='\r')

        avg_train_loss = train_loss_sum / steps_per_epoch
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # Val
        val_loss_sum = 0
        for step in range(val_steps):
            images, content_vecs = next(val_gen)
            loss = val_step(images, content_vecs)
            val_loss_sum += loss

        avg_val_loss = val_loss_sum / val_steps
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
