#!/usr/bin/env python3
"""
Pipeline v2 - Binary Font Generation

Uses the v2 models trained on binary font data.
No domain adaptation needed since user PDF input matches training domain.
"""
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import json

# TensorFlow import (suppress warnings)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "runs" / "font_v2"
CONTENT_LATENTS_PATH = ROOT / "runs" / "autoenc" / "content_latents_unified.npy"
CHAR_VOCAB_PATH = ROOT / "src" / "data" / "char_vocab.json"

IMAGE_SIZE = 256


class ContentScaleLayer(keras.layers.Layer):
    """Custom layer for scaling content latent."""
    def __init__(self, scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def load_image(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess image - no domain adaptation needed for v2."""
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def generate_all_glyphs_v2(
    style_images: List[Path],
    output_dir: Path,
) -> List[Path]:
    """Generate all 2350 Korean glyphs using v2 models.

    Args:
        style_images: List of user's handwriting sample images (from PDF)
        output_dir: Directory to save generated glyph PNGs

    Returns:
        List of paths to generated PNG files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    style_encoder_path = MODEL_DIR / "style_encoder_best.h5"
    decoder_path = MODEL_DIR / "decoder_best.h5"

    if not style_encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(f"V2 models not found in {MODEL_DIR}")

    print(f"[Pipeline v2] Loading Style Encoder: {style_encoder_path}")
    style_encoder = keras.models.load_model(str(style_encoder_path), compile=False)

    print(f"[Pipeline v2] Loading Decoder: {decoder_path}")
    decoder = keras.models.load_model(
        str(decoder_path),
        custom_objects={'ContentScaleLayer': ContentScaleLayer},
        compile=False
    )

    # Load content latents and vocab
    content_latents = np.load(CONTENT_LATENTS_PATH)
    with open(CHAR_VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # Extract style from user samples (first 5)
    max_samples = 5
    samples_to_use = style_images[:max_samples]
    print(f"[Pipeline v2] Extracting style from {len(samples_to_use)} samples (of {len(style_images)} total)...")

    style_vecs = []
    for img_path in samples_to_use:
        if not img_path.exists():
            continue
        img = load_image(img_path)
        img_batch = np.expand_dims(img, 0)
        style_vec = style_encoder(img_batch, training=False).numpy()
        style_vecs.append(style_vec)

    if not style_vecs:
        raise ValueError("No valid style images provided")

    avg_style = np.mean(np.concatenate(style_vecs, axis=0), axis=0, keepdims=True)
    avg_style_tensor = tf.constant(avg_style, dtype=tf.float32)
    print(f"[Pipeline v2] Style vector shape: {avg_style.shape}")

    # Get all Korean syllables from vocab
    korean_chars = []
    korean_indices = []
    for char, idx in vocab.items():
        codepoint = ord(char)
        if 0xAC00 <= codepoint <= 0xD7A3:
            korean_chars.append(char)
            korean_indices.append(idx)

    print(f"[Pipeline v2] Generating {len(korean_chars)} characters...")

    # Generate in batches
    batch_size = 64
    out_paths = []
    n = len(korean_chars)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = korean_indices[start:end]

        c_vecs = tf.constant(content_latents[batch_indices], dtype=tf.float32)
        s_vecs = tf.tile(avg_style_tensor, [len(batch_indices), 1])

        batch_preds = decoder([c_vecs, s_vecs], training=False).numpy()
        batch_preds = np.clip(batch_preds, 0, 1)

        for i, pred in enumerate(batch_preds):
            char_idx = start + i
            char = korean_chars[char_idx]
            codepoint = ord(char)

            img = (pred.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode="L")
            out_path = output_dir / f"U+{codepoint:04X}.png"
            pil_img.save(out_path)
            out_paths.append(out_path)

        if (end % 500 == 0) or (end == n):
            print(f"[Pipeline v2] Generated {end}/{n} characters")

    return out_paths
