
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

# Path setup
ROOT = Path(".").resolve()
sys.path.append(str(ROOT))

def test_generate():
    print("Loading resources...")

    # Paths (Colab environment assumption)
    RUNS_DIR = Path("/content/drive/MyDrive/FontByMe/runs")
    DATA_DIR = ROOT / "data/content_font/NotoSansKR-Regular"
    LATENTS_PATH = ROOT / "runs/autoenc/content_latents_unified.npy"
    VOCAB_PATH = ROOT / "src/data/char_vocab.json"

    # Load Models
    style_enc_path = RUNS_DIR / "style_encoder_best.h5"
    decoder_path = RUNS_DIR / "decoder_best.h5"

    if not style_enc_path.exists() or not decoder_path.exists():
        print("❌ Model files not found in Drive!")
        return

    style_encoder = keras.models.load_model(style_enc_path, compile=False)
    decoder = keras.models.load_model(decoder_path, compile=False)

    # Load Latents & Vocab
    content_latents = np.load(LATENTS_PATH)
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f) # char -> idx

    # Select test chars
    test_chars = ["가", "나", "다", "빎", "조", "훍", "A", "B"]

    # Select a random reference image from data
    # Just grab any png from resizing dir as 'style'
    from PIL import Image
    import glob
    style_imgs = glob.glob("data/handwriting_raw/resizing/*/*.jpg")
    # Try png if jpg not found (depends on user data unzip)
    if not style_imgs:
        style_imgs = glob.glob("data/handwriting_raw/resizing/*/*.png")

    if not style_imgs:
        print("❌ No style images found to test with.")
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

        # Decode
        pred = decoder([c_vec, style_vec], training=False)
        pred_img = pred.numpy().squeeze()

        axes[i+1].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
        axes[i+1].set_title(char)
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig("test_result.png")
    print("✅ Saved visualization to test_result.png")
    plt.show()

if __name__ == "__main__":
    test_generate()
