#!/usr/bin/env python
"""
MINIMAL TEST: Fine-tune decoder on handwriting WITHOUT style.
Just verify decoder can learn to output handwriting from correct content latent.
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


def build_char_to_index(png_dir: Path) -> dict:
    png_files = sorted(png_dir.glob("*.png"))
    mapping = {}
    for idx, f in enumerate(png_files):
        name = f.stem
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                mapping[chr(codepoint)] = idx
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
    parser.add_argument("--png_dir", type=Path, required=True)
    parser.add_argument("--pretrained_decoder", type=Path, required=True)
    parser.add_argument("--writer_id", type=str, default=None, help="Filter by specific writer ID")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    char_to_index = build_char_to_index(args.png_dir)
    print(f"Mapping: {len(char_to_index)} chars")

    entries = json.loads(args.index.read_text())
    content_latents_all = np.load(args.content_latents)

    # Filter by writer_id if specified
    if args.writer_id:
        entries = [e for e in entries if str(e.get("writer_id")) == args.writer_id]
        print(f"Filtered to writer {args.writer_id}: {len(entries)} entries")

    images = []
    latents = []
    chars = []

    for e in entries:
        # Check max_samples limit
        if args.max_samples and len(images) >= args.max_samples:
            break

        char_text = e.get("text", "")
        if not char_text or char_text not in char_to_index:
            continue
        latent_idx = char_to_index[char_text]

        img_rel = e["image_path"]
        if "/" in img_rel:
            parts = img_rel.split("/")
            img_path = args.root / parts[-2] / parts[-1]
        else:
            img_path = args.root / str(e["writer_id"]) / img_rel

        if img_path.exists() and latent_idx < len(content_latents_all):
            images.append(load_image(img_path))
            latents.append(content_latents_all[latent_idx])
            chars.append(char_text)

    images = np.stack(images)
    latents = np.stack(latents)
    print(f"Loaded {len(images)} samples")

    # Load decoder
    decoder = keras.models.load_model(args.pretrained_decoder, compile=False, safe_mode=False)

    # Use Adam with lower lr for fine-tuning
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Training loop - direct latent -> image
    n = len(images)
    steps = max(1, n // args.batch_size)

    for epoch in range(args.epochs):
        idx = np.random.permutation(n)
        loss_sum = 0.0

        for step in range(steps):
            bi = idx[step * args.batch_size : (step + 1) * args.batch_size]
            x = tf.constant(latents[bi])
            y = tf.constant(images[bi])

            with tf.GradientTape() as tape:
                pred = decoder(x, training=True)
                loss = tf.reduce_mean(tf.square(y - pred))

            grads = tape.gradient(loss, decoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, decoder.trainable_variables))
            loss_sum += loss.numpy()

        print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss_sum/steps:.4f}")

    # Visualize
    n_vis = min(8, len(images))
    preds = decoder(tf.constant(latents[:n_vis]), training=False).numpy()
    preds = np.clip(preds, 0, 1)

    fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 2, 4))
    for i in range(n_vis):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"Target", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"Output", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_dir / "finetune_result.png", dpi=150)
    decoder.save(args.output_dir / "decoder_finetuned.h5")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
