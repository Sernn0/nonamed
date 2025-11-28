from __future__ import annotations

"""
Template for joint training of content encoder, style encoder, and decoder.
All training logic is left as TODO.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore

try:
    from src.trainers.train_content_encoder import build_autoencoder  # type: ignore
except Exception:  # pragma: no cover
    build_autoencoder = None  # type: ignore

try:
    from src.models.style_encoder.style_encoder import build_style_encoder
except Exception:  # pragma: no cover
    build_style_encoder = None  # type: ignore

try:
    from src.models.decoder.decoder import build_decoder
except Exception:  # pragma: no cover
    build_decoder = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint training template.")
    parser.add_argument("--output-dir", default="runs/joint", help="Output directory.")
    parser.add_argument("--content-dim", type=int, default=64, help="Content vector size.")
    parser.add_argument("--style-dim", type=int, default=32, help="Style vector size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tf:
        print("TensorFlow not available. Skipping joint training template.", file=sys.stderr)
        return

    # Load/build components.
    content_encoder = None
    if build_autoencoder:
        autoencoder, encoder, _ = build_autoencoder((256, 256, 1), args.content_dim)
        content_encoder = encoder
    style_encoder = build_style_encoder(style_dim=args.style_dim) if build_style_encoder else None
    decoder = build_decoder(content_dim=args.content_dim, style_dim=args.style_dim) if build_decoder else None

    if not all([content_encoder, style_encoder, decoder]):
        print("One or more components missing; add implementations before training.", file=sys.stderr)
        return

    # TODO: implement dataset loading, loss functions, optimizers, and training loop.
    # Example placeholders:
    # content_loss = ...
    # style_loss = ...
    # recon_loss = ...
    # optimizer = tf.keras.optimizers.Adam(...)
    # for batch in dataset: ...

    # Save placeholders to verify wiring.
    content_encoder.save(output_dir / "content_encoder.h5")
    style_encoder.save(output_dir / "style_encoder.h5")
    decoder.save(output_dir / "decoder.h5")
    print(f"Saved placeholder joint components to {output_dir}")


if __name__ == "__main__":
    main()
