from __future__ import annotations

"""
Visualize decoder reconstructions given content latents and synthetic style vectors.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # type: ignore

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.models.decoder import build_decoder
except Exception:
    build_decoder = None  # type: ignore

# Optional: style encoder loading placeholder
try:
    from src.models.style_encoder import build_style_encoder
except Exception:
    build_style_encoder = None  # type: ignore


def load_content_latents(path: Path) -> np.ndarray:
    latents = np.load(path)
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D content latents, got shape {latents.shape}")
    return latents


def load_decoder_model(decoder_path: Path, content_dim: int, style_dim: int) -> tf.keras.Model:
    if decoder_path.is_file():
        try:
            model = tf.keras.models.load_model(decoder_path, compile=False)
            return model
        except Exception as exc:
            print(f"Warning: failed to load decoder from {decoder_path}: {exc}")
    if not build_decoder:
        raise RuntimeError("build_decoder not available and decoder_path could not be loaded.")
    print("Building decoder with random init (weights not loaded).")
    return build_decoder(content_dim=content_dim, style_dim=style_dim)


def prepare_style_vector(mode: str, style_dim: int, batch: int = 1, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "random":
        return rng.normal(loc=0.0, scale=1.0, size=(batch, style_dim)).astype(np.float32)
    if mode == "zero":
        return np.zeros((batch, style_dim), dtype=np.float32)
    raise ValueError(f"Unsupported style mode: {mode}")


def rescale_image(img: np.ndarray) -> np.ndarray:
    """Rescale image to [0,1] for visualization."""
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def visualize(args: argparse.Namespace) -> None:
    content_latents = load_content_latents(args.content_latents)
    content_dim = content_latents.shape[1]

    # Determine style_dim
    style_dim = args.style_dim
    decoder = load_decoder_model(args.decoder_path, content_dim=content_dim, style_dim=style_dim)
    if style_dim is None and len(decoder.inputs) > 1 and decoder.inputs[1].shape.rank == 2:
        style_dim = int(decoder.inputs[1].shape[1])
    if style_dim is None:
        style_dim = 32

    num_samples = min(args.num_samples, content_latents.shape[0])
    sample_ids = np.linspace(0, content_latents.shape[0] - 1, num_samples, dtype=int)

    content_batch = content_latents[sample_ids]
    style_batch = prepare_style_vector(args.style_mode, style_dim=style_dim, batch=num_samples, seed=args.seed)

    preds = decoder([content_batch, style_batch], training=False).numpy()
    preds = np.squeeze(preds, axis=-1) if preds.ndim == 4 else preds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    # Top row: placeholders with content ids (real image loading optional)
    for i, cid in enumerate(sample_ids):
        ax = axes[0, i]
        ax.imshow(np.ones((256, 256)), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"content {cid}")
        ax.axis("off")

    # Bottom row: decoder outputs
    for i, img in enumerate(preds[:num_samples]):
        ax = axes[1, i]
        ax.imshow(rescale_image(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"gen {sample_ids[i]}")
        ax.axis("off")

    plt.tight_layout()
    out_path = args.output_dir / "recon_test.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize decoder reconstructions from content/style latents.")
    parser.add_argument(
        "--content_latents",
        type=Path,
        default=Path("runs/content/content_latents.npy"),
        help="Path to content_latents.npy (shape: [N, content_dim]).",
    )
    parser.add_argument(
        "--decoder_path",
        type=Path,
        default=Path("runs/autoenc/decoder.h5"),
        help="Path to decoder model file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/vis"),
        help="Directory to save visualization.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--style_mode",
        choices=["random", "zero"],
        default="random",
        help="How to generate style vectors.",
    )
    parser.add_argument(
        "--style_dim",
        type=int,
        default=32,
        help="Style vector dimension (used if decoder cannot be loaded to infer).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for style vector generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
