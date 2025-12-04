from __future__ import annotations

"""
Visualize joint model reconstructions: original handwriting vs generated output.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Add project root for local imports
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.style_encoder import build_style_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize joint reconstructions.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/handwriting_processed/handwriting_index_val.json"),
        help="Path to index JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/handwriting_raw/resizing"),
        help="Root directory to resolve image paths.",
    )
    parser.add_argument(
        "--content_latents",
        type=Path,
        default=Path("runs/autoenc/content_latents.npy"),
        help="Path to content_latents.npy",
    )
    parser.add_argument(
        "--style_encoder_path",
        type=Path,
        default=Path("runs/style_encoder/encoder_style_backbone.h5"),
        help="Path to style encoder backbone weights.",
    )
    parser.add_argument(
        "--decoder_path",
        type=Path,
        default=Path("runs/joint/decoder_final.h5"),
        help="Path to decoder model.",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--output_path", type=Path, default=Path("runs/joint/vis_joint_recon.png"), help="Output image.")
    parser.add_argument("--image_size", type=int, default=256, help="Resize target for images.")
    parser.add_argument("--style_dim", type=int, default=32, help="Style latent dimension.")
    return parser.parse_args()


def load_index(index_path: Path) -> List[dict]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Index JSON must be a list.")
    return data


def resolve_path(image_path: str, root: Path) -> Path:
    p = Path(image_path)
    # If absolute and exists, use it.
    if p.is_absolute() and p.exists():
        return p
    # If absolute but missing, try to remap assuming prefix like "/data/handwriting_raw/resizing/..."
    if p.is_absolute() and "resizing" in p.parts:
        try:
            idx = p.parts.index("resizing")
            rel = Path(*p.parts[idx + 1 :])
            candidate = root / rel
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    # Fallback: treat as relative to root
    return root / p


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr


def select_samples(entries: List[dict], num_samples: int) -> List[dict]:
    if num_samples >= len(entries):
        return entries
    return random.sample(entries, num_samples)


def main() -> None:
    args = parse_args()

    entries = load_index(args.index)
    samples = select_samples(entries, args.num_samples)

    # Load content latents
    content_latents = np.load(args.content_latents)
    content_dim = content_latents.shape[1]

    # Load style encoder via fresh build + weights to avoid Lambda deserialization issues
    style_encoder = build_style_encoder(style_dim=args.style_dim, input_shape=(args.image_size, args.image_size, 1))
    if args.style_encoder_path.is_file():
        style_encoder.load_weights(args.style_encoder_path)
    else:
        print(f"[WARN] Style encoder weights not found at {args.style_encoder_path}, using fresh weights.")

    decoder = keras.models.load_model(args.decoder_path, compile=False, safe_mode=False)

    # Prepare batch images and content vectors
    imgs = []
    content_vecs = []
    for e in samples:
        img_path = resolve_path(e["image_path"], args.root)
        img = load_image(img_path, args.image_size)
        imgs.append(img)
        text_id = int(e["text_id"])
        assert 0 <= text_id < content_latents.shape[0], f"text_id {text_id} out of range"
        content_vecs.append(content_latents[text_id])

    batch_images = np.stack(imgs, axis=0)
    content_vecs = np.stack(content_vecs, axis=0)

    # Style vectors
    style_vecs = style_encoder(batch_images, training=False).numpy()

    # Decode
    preds = []
    for c_vec, s_vec in zip(content_vecs, style_vecs):
        c_in = np.expand_dims(c_vec, axis=0)
        s_in = np.expand_dims(s_vec, axis=0)
        out = decoder([c_in, s_in], training=False).numpy()
        out = np.squeeze(out, axis=0)
        out = np.clip(out, 0.0, 1.0)
        preds.append(out)
    preds = np.stack(preds, axis=0)

    # Plot grid
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
    for i in range(cols):
        axes[0, i].imshow(batch_images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original", fontsize=8)

        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Generated", fontsize=8)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
