from __future__ import annotations

"""
Extract latent vectors for content glyphs using a pretrained content encoder.
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def load_images(input_dir: str | Path, image_size: int) -> np.ndarray:
    """
    Load all PNG images from a directory, resize, normalize to [0, 1],
    and return array of shape (N, H, W, 1).
    """
    input_path = Path(input_dir)
    files: List[Path] = sorted(input_path.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {input_path}")

    imgs: List[np.ndarray] = []
    for f in files:
        img = Image.open(f).convert("L")
        img = img.resize((image_size, image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
        imgs.append(arr)

    images = np.stack(imgs, axis=0)
    print(f"Loaded {len(files)} images from {input_path}")
    print(f"Images shape: {images.shape}")
    return images


def extract_latents(images: np.ndarray, encoder: keras.Model, batch_size: int) -> np.ndarray:
    """Encode images to latent vectors in batches."""
    ds = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)
    latents: List[np.ndarray] = []
    for batch in ds:
        z = encoder(batch, training=False)
        latents.append(z.numpy())
    if latents:
        return np.concatenate(latents, axis=0)
    return np.zeros((0, encoder.output_shape[-1]), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode content glyph set into latent vectors.")
    parser.add_argument("--input_dir", type=str, default="data/generated_png", help="Directory containing PNG glyphs.")
    parser.add_argument("--model_path", type=str, default="runs/autoenc/encoder.h5", help="Path to content encoder .h5.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="runs/autoenc/content_latents.npy",
        help="Output path for latent numpy file.",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = load_images(args.input_dir, args.image_size)

    encoder = keras.models.load_model(args.model_path, compile=False, safe_mode=False)
    try:
        encoder.summary()
    except Exception:
        pass

    latents = extract_latents(images, encoder, args.batch_size)
    print(f"Latents shape: {latents.shape}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, latents)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved latents to {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
