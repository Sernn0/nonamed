from __future__ import annotations

"""
Train a convolutional autoencoder on 256x256 grayscale glyph images.
The model learns a content embedding and reconstructs the input glyphs.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path: bytes, image_size: int) -> np.ndarray:
    """Load a single image path (bytes) into a normalized float32 array."""
    path_str = path.decode("utf-8")
    with Image.open(path_str) as img:
        img = img.convert("L")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
    return arr


def build_dataset(
    data_dir: str, batch_size: int, image_size: int = 256
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Create train/validation datasets from PNG files in data_dir."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: data directory not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    file_paths: List[str] = sorted(str(p) for p in data_path.glob("*.png"))
    if not file_paths:
        print(f"Error: no PNG files found in {data_path}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(42)
    rng.shuffle(file_paths)

    split_index = max(1, int(len(file_paths) * 0.9))
    train_files = file_paths[:split_index]
    val_files = file_paths[split_index:] or train_files[-1:]
    if not train_files:
        print("Error: training set is empty after split.", file=sys.stderr)
        sys.exit(1)

    def _make_ds(paths: List[str], shuffle: bool) -> tf.data.Dataset:
        def _map_fn(path: tf.Tensor) -> tf.Tensor:
            img = tf.numpy_function(
                func=load_image, inp=[path, image_size], Tout=tf.float32
            )
            img.set_shape((image_size, image_size, 1))
            return img

        ds = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = _make_ds(train_files, shuffle=True)
    val_ds = _make_ds(val_files, shuffle=False)

    print(f"Found {len(file_paths)} images. Train: {len(train_files)}, Val: {len(val_files)}")
    return train_ds, val_ds, len(train_files), len(val_files)


def build_autoencoder(
    input_shape: Tuple[int, int, int], latent_dim: int
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build encoder, decoder, and full autoencoder models."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(latent_dim, name="latent")(x)
    encoder = tf.keras.Model(inputs, latent, name="encoder")

    decoder_input = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(np.prod(shape_before_flatten))(decoder_input)
    x = tf.keras.layers.Reshape(shape_before_flatten)(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    decoder = tf.keras.Model(decoder_input, outputs, name="decoder")

    autoencoder_output = decoder(encoder(inputs))
    autoencoder = tf.keras.Model(inputs, autoencoder_output, name="autoencoder")
    return autoencoder, encoder, decoder


def save_reconstructions(
    model: tf.keras.Model, dataset: tf.data.Dataset, output_path: Path, max_images: int = 8
) -> None:
    """Save a grid of original (top row) and reconstructed (bottom row) images."""
    for batch in dataset.take(1):
        originals = batch[:max_images]
        recon = model.predict(originals, verbose=0)
        count = originals.shape[0]
        images = np.concatenate([originals, recon], axis=0)
        h, w = images.shape[1], images.shape[2]
        grid = Image.new("L", (count * w, 2 * h))
        for idx, img_arr in enumerate(images):
            img = Image.fromarray(
                np.clip(img_arr[..., 0] * 255.0, 0, 255).astype(np.uint8), mode="L"
            )
            row = 0 if idx < count else 1
            col = idx if idx < count else idx - count
            grid.paste(img, (col * w, row * h))
        grid.save(output_path)
        return
    print("Warning: validation dataset is empty; skipping reconstructions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder on glyph PNG images."
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing glyph PNG files."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save models and outputs."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent vector size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for Adam optimizer."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_count, val_count = build_dataset(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    input_shape = (256, 256, 1)
    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=input_shape, latent_dim=args.latent_dim
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mse"],
    )

    autoencoder.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    autoencoder.save(output_dir / "autoencoder.h5")
    encoder.save(output_dir / "encoder.h5")
    decoder.save(output_dir / "decoder.h5")

    save_reconstructions(autoencoder, val_ds, output_dir / "reconstructions.png")


if __name__ == "__main__":
    main()
