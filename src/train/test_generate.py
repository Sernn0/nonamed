
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

# Path setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Use centralized config
from src.config import (
    CONTENT_LATENTS, CHAR_VOCAB, HANDWRITING_RAW,
    STYLE_ENCODER_BEST, DECODER_BEST, RUNS_DIR
)

def test_generate():
    print("Loading resources...")

    # Detect environment (Colab vs Local)
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        # In Colab, check Drive first, then local runs/
        DRIVE_RUNS = Path("/content/drive/MyDrive/FontByMe/runs")
        if DRIVE_RUNS.exists():
            style_enc_path = DRIVE_RUNS / "style_encoder_best.h5"
            decoder_path = DRIVE_RUNS / "decoder_best.h5"
            output_dir = DRIVE_RUNS
        else:
            style_enc_path = STYLE_ENCODER_BEST
            decoder_path = DECODER_BEST
            output_dir = RUNS_DIR
    else:
        # Local environment
        style_enc_path = STYLE_ENCODER_BEST
        decoder_path = DECODER_BEST
        output_dir = RUNS_DIR

    print(f"Style Encoder: {style_enc_path}")
    print(f"Decoder: {decoder_path}")

    if not style_enc_path.exists() or not decoder_path.exists():
        print(f"❌ Model files not found!")
        print(f"   Expected: {style_enc_path}")
        print(f"   Expected: {decoder_path}")
        return

    # Import custom layers for deserialization
    from src.models.decoder import ContentScaleLayer

    style_encoder = keras.models.load_model(style_enc_path, compile=False)
    decoder = keras.models.load_model(
        decoder_path,
        custom_objects={'ContentScaleLayer': ContentScaleLayer},
        compile=False
    )

    # Load Latents & Vocab using config paths
    content_latents = np.load(CONTENT_LATENTS)
    with open(CHAR_VOCAB, "r") as f:
        vocab = json.load(f) # char -> idx

    # Select test chars (must be in vocab - 2350 Korean syllables only)
    test_chars = ["가", "나", "다", "라", "마", "한", "글", "자"]

    # Select a random reference image from data
    from PIL import Image
    import glob

    # Use config path for handwriting data
    style_pattern = str(HANDWRITING_RAW / "resizing" / "*" / "*.jpg")
    style_imgs = glob.glob(style_pattern)
    if not style_imgs:
        style_pattern = str(HANDWRITING_RAW / "resizing" / "*" / "*.png")
        style_imgs = glob.glob(style_pattern)

    if not style_imgs:
        print("❌ No style images found to test with.")
        print(f"   Searched in: {HANDWRITING_RAW / 'resizing'}")
        return

    ref_img_path = style_imgs[0]
    print(f"Using Style Reference: {ref_img_path}")

    # Preprocess Style Image
    img = Image.open(ref_img_path).convert("L")
    img = img.resize((256, 256))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_arr, axis=[0, -1]) # (1, 256, 256, 1)

    # Encode Style
    style_vec = style_encoder(img_tensor)

    # Generate
    fig, axes = plt.subplots(1, len(test_chars) + 1, figsize=(15, 3))

    # Show Reference
    axes[0].imshow(img_arr, cmap="gray")
    axes[0].set_title("Ref Style")
    axes[0].axis("off")

    for i, char in enumerate(test_chars):
        if char not in vocab:
            print(f"Skipping {char} (not in vocab)")
            continue

        c_idx = vocab[char]
        if c_idx >= len(content_latents):
            continue

        c_vec = content_latents[c_idx:c_idx+1] # (1, 64)
        c_vec = tf.convert_to_tensor(c_vec, dtype=tf.float32)

        # DEBUG: Print stats
        if i == 0:
            print(f"Content Latent Stats: Min={np.min(c_vec):.4f}, Max={np.max(c_vec):.4f}, Mean={np.mean(c_vec):.4f}")
            print(f"Style Vector Stats: Min={np.min(style_vec):.4f}, Max={np.max(style_vec):.4f}, Mean={np.mean(style_vec):.4f}")

        # Decode
        pred = decoder([c_vec, style_vec], training=False)
        pred_img = pred.numpy().squeeze()

        if i == 0:
            print(f"Pred Image Stats: Min={np.min(pred_img):.4f}, Max={np.max(pred_img):.4f}, Mean={np.mean(pred_img):.4f}")


        axes[i+1].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
        axes[i+1].set_title(char)
        axes[i+1].axis("off")

    plt.tight_layout()

    # Save locally first
    local_out = Path("test_result.png")
    plt.savefig(local_out)
    print(f"✅ Saved visualization to {local_out}")

    # Also save to output_dir (Drive in Colab, runs/ locally)
    if output_dir.exists():
        drive_out = output_dir / "test_result.png"
        plt.savefig(drive_out)
        print(f"✅ Saved visualization to {drive_out}")

    plt.show()

if __name__ == "__main__":
    test_generate()
