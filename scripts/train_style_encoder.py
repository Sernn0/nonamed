from __future__ import annotations

"""
Template for training the style encoder.
Uses dummy data to validate shapes; actual training TODO.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore

try:
    from models.style_encoder import build_style_encoder
except Exception:  # pragma: no cover
    build_style_encoder = None  # type: ignore

try:
    from models.decoder import build_decoder
except Exception:  # pragma: no cover
    build_decoder = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train style encoder (template).")
    parser.add_argument("--output-dir", default="runs/style_encoder", help="Output directory.")
    parser.add_argument("--style-dim", type=int, default=32, help="Style vector size.")
    parser.add_argument("--content-dim", type=int, default=64, help="Content vector size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tf:
        print("TensorFlow not available. Skipping training template.", file=sys.stderr)
        return

    encoder = (
        build_style_encoder(style_dim=args.style_dim)
        if build_style_encoder
        else None
    )
    decoder = (
        build_decoder(content_dim=args.content_dim, style_dim=args.style_dim)
        if build_decoder
        else None
    )

    if encoder is None:
        print("Style encoder not available. Add implementation.", file=sys.stderr)
        return

    # Dummy data to test forward pass shape.
    dummy_images = tf.zeros((2, 256, 256, 1))
    style_vec = encoder(dummy_images)
    print("Style vector shape:", style_vec.shape)

    if decoder is not None:
        dummy_content = tf.zeros((2, args.content_dim))
        decoded = decoder([dummy_content, style_vec])
        print("Decoder output shape:", decoded.shape)

    # TODO: implement real dataset loading and training loop.
    # encoder.compile(...)
    # encoder.fit(...)

    encoder.save(output_dir / "style_encoder.h5")
    if decoder is not None:
        decoder.save(output_dir / "decoder.h5")
    print(f"Saved placeholder models to {output_dir}")


if __name__ == "__main__":
    main()
