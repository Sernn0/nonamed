#!/usr/bin/env python3
"""
Test Generation Script v2 - Binary Font Data

Tests the trained v2 model by generating sample characters.

Usage:
    python src/train_v2/test_generate_v2.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "runs" / "font_v2"
DATA_DIR = ROOT / "data" / "font_rendered"
CONTENT_LATENTS = ROOT / "runs" / "autoenc" / "content_latents_unified.npy"
CHAR_VOCAB = ROOT / "src" / "data" / "char_vocab.json"


def main():
    # Check if models exist
    style_encoder_path = MODEL_DIR / "style_encoder_best.h5"
    decoder_path = MODEL_DIR / "decoder_best.h5"

    if not style_encoder_path.exists() or not decoder_path.exists():
        print("[ERROR] Models not found. Train first with train_joint_v2.py")
        return

    # Load models
    print("[INFO] Loading models...")
    from src.models.decoder import ContentScaleLayer

    style_encoder = keras.models.load_model(style_encoder_path, compile=False)
    decoder = keras.models.load_model(
        decoder_path,
        custom_objects={'ContentScaleLayer': ContentScaleLayer},
        compile=False
    )

    # Load content latents and vocab
    content_latents = np.load(CONTENT_LATENTS)
    with open(CHAR_VOCAB, 'r') as f:
        vocab = json.load(f)

    # Load font index
    with open(DATA_DIR / "index.json", 'r') as f:
        index = json.load(f)
    fonts = index["fonts"]

    # Select test fonts (first 3)
    test_fonts = fonts[:3]
    test_chars = ['가', '나', '다', '라', '마', '한', '글', '자']

    print(f"[INFO] Testing with fonts: {test_fonts}")
    print(f"[INFO] Testing chars: {test_chars}")

    # Create visualization
    fig, axes = plt.subplots(len(test_fonts), len(test_chars) + 1, figsize=(18, 6))

    for row, font_name in enumerate(test_fonts):
        font_dir = DATA_DIR / font_name

        # Get a reference image from this font for style extraction
        ref_path = font_dir / "U+AC00.png"  # 가
        if not ref_path.exists():
            print(f"[WARN] Reference not found for {font_name}")
            continue

        ref_img = Image.open(ref_path).convert('L').resize((256, 256))
        ref_arr = np.array(ref_img, dtype=np.float32) / 255.0
        ref_batch = np.expand_dims(ref_arr, [0, -1])

        # Extract style vector
        style_vec = style_encoder(ref_batch, training=False)

        # Show reference
        axes[row, 0].imshow(ref_arr, cmap='gray')
        axes[row, 0].set_title(f'{font_name[:10]}...')
        axes[row, 0].axis('off')

        # Generate each test character
        for col, char in enumerate(test_chars):
            if char not in vocab:
                continue

            char_idx = vocab[char]
            c_vec = tf.constant(content_latents[char_idx:char_idx+1], dtype=tf.float32)

            pred = decoder([c_vec, style_vec], training=False)
            pred_arr = pred.numpy().squeeze()

            axes[row, col + 1].imshow(pred_arr, cmap='gray', vmin=0, vmax=1)
            axes[row, col + 1].set_title(char)
            axes[row, col + 1].axis('off')

    plt.tight_layout()

    # Save
    out_path = MODEL_DIR / "test_result_v2.png"
    plt.savefig(out_path)
    print(f"[INFO] Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
