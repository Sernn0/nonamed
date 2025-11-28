from __future__ import annotations

"""Template script to encode a glyph dataset using a content encoder."""

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

try:
    from src.trainers.train_content_encoder import build_dataset  # type: ignore
except Exception:  # pragma: no cover
    build_dataset = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore


class DummyContentEncoder:
    """Fallback encoder producing zeros to avoid import errors."""

    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim

    def predict(self, batch):
        batch_size = batch.shape[0]
        return np.zeros((batch_size, self.latent_dim), dtype=np.float32)


def load_png(path: Path, image_size: int = 256) -> np.ndarray:
    """Load and preprocess a single PNG file."""
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
    return arr


def encode_dataset(
    data_dir: Path, output_path: Path, latent_dim: int = 64, batch_size: int = 32
):
    """Encode all PNGs in data_dir and save as .npy latent array."""
    paths: List[Path] = sorted(data_dir.glob("*.png"))
    if not paths:
        print(f"Warning: no PNG files found in {data_dir}. Nothing to encode.")
        return

    # TODO: replace DummyContentEncoder with actual content encoder loading.
    encoder = DummyContentEncoder(latent_dim=latent_dim)

    latents: List[np.ndarray] = []
    batch: List[np.ndarray] = []
    for path in paths:
        arr = load_png(path)
        batch.append(arr)
        if len(batch) >= batch_size:
            batch_arr = np.stack(batch, axis=0)
            latents.append(encoder.predict(batch_arr))
            batch = []

    if batch:
        batch_arr = np.stack(batch, axis=0)
        latents.append(encoder.predict(batch_arr))

    all_latents = np.concatenate(latents, axis=0) if latents else np.zeros((0, latent_dim))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_latents)
    print(f"Saved latent vectors to {output_path} shape={all_latents.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode glyph dataset into latent vectors using the content encoder."
    )
    parser.add_argument("--data-dir", required=True, help="Directory with glyph PNG files.")
    parser.add_argument("--output", default="content_latents.npy", help="Path to save latents.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output = Path(args.output)
    encode_dataset(
        data_dir=data_dir,
        output_path=output,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
