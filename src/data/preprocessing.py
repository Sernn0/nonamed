from __future__ import annotations

"""Basic preprocessing utilities for glyph images."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_and_preprocess_image(path: Path, image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load an image, convert to grayscale, resize, and normalize to [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != image_size:
            img = img.resize(image_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
    return arr


def save_image(array: np.ndarray, path: Path) -> None:
    """Save a normalized image array back to disk as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_uint8 = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    if arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 1:
        arr_uint8 = arr_uint8[..., 0]
    Image.fromarray(arr_uint8, mode="L").save(path)


if __name__ == "__main__":
    print("Preprocess utilities loaded.")
