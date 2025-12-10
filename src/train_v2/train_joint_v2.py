#!/usr/bin/env python3
"""
Joint Training Script v2 - Binary Font Data

Trains Style Encoder + Joint Decoder on rendered font data.
Each font is treated as a "style" (like a writer in v1).

Usage:
    python src/train_v2/train_joint_v2.py
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "font_rendered"
CONTENT_LATENTS = ROOT / "runs" / "autoenc" / "content_latents_unified.npy"
CHAR_VOCAB = ROOT / "src" / "data" / "char_vocab.json"
OUTPUT_DIR = ROOT / "runs" / "font_v2"

# Hyperparameters
LATENT_DIM = 64
STYLE_DIM = 32
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 2e-4

# Style loss weight
STYLE_WEIGHT = 0.3


def build_style_encoder():
    """CNN encoder that extracts style vector from font image."""
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="input_image")

    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(STYLE_DIM)(x)

    return keras.Model(inputs, outputs, name="style_encoder")


def build_decoder():
    """Decoder that combines content latent and style to generate image."""
    # Custom layer for scaling content latent
    from src.models.decoder import ContentScaleLayer

    content_input = keras.Input(shape=(LATENT_DIM,), name="content_latent")
    style_input = keras.Input(shape=(STYLE_DIM,), name="style_vector")

    # Scale content to prevent saturation
    content_scaled = ContentScaleLayer(scale=0.1)(content_input)

    # Concatenate content + style
    combined = keras.layers.Concatenate()([content_scaled, style_input])

    # Initial projection
    x = keras.layers.Dense(16 * 16 * 256)(combined)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Reshape((16, 16, 256))(x)

    # Upsample to 256x256
    for filters in [128, 64, 32, 16]:
        x = keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    # Output layer (sigmoid for binary-like output)
    outputs = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    return keras.Model([content_input, style_input], outputs, name="decoder")


def load_data():
    """Load rendered font data and create train/val splits."""
    # Load index
    with open(DATA_DIR / "index.json", 'r') as f:
        index = json.load(f)

    fonts = index["fonts"]
    print(f"[DATA] Found {len(fonts)} fonts")

    # Load content latents and vocab
    content_latents = np.load(CONTENT_LATENTS)
    with open(CHAR_VOCAB, 'r') as f:
        vocab = json.load(f)

    # Reverse vocab: codepoint -> index
    char_to_idx = {char: idx for char, idx in vocab.items()}

    # Collect all image paths with font_id and char_idx
    all_data = []
    for font_id, font_name in enumerate(fonts):
        font_dir = DATA_DIR / font_name
        for png in font_dir.glob("U+*.png"):
            # Parse codepoint from filename
            codepoint = int(png.stem[2:], 16)
            char = chr(codepoint)
            if char in char_to_idx:
                char_idx = char_to_idx[char]
                all_data.append({
                    "path": str(png),
                    "font_id": font_id,
                    "char_idx": char_idx
                })

    print(f"[DATA] Total samples: {len(all_data)}")

    # Shuffle and split
    np.random.shuffle(all_data)
    n = len(all_data)
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)

    train_data = all_data[:train_n]
    val_data = all_data[train_n:train_n+val_n]

    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")

    return train_data, val_data, content_latents, len(fonts)


def create_dataset(data_list, content_latents, batch_size, shuffle=True):
    """Create tf.data.Dataset from data list."""
    paths = [d["path"] for d in data_list]
    font_ids = [d["font_id"] for d in data_list]
    char_idxs = [d["char_idx"] for d in data_list]

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def generator():
        for i in range(len(paths)):
            yield paths[i], char_idxs[i], font_ids[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    def process(path, char_idx, font_id):
        img = load_image(path)
        c_vec = tf.gather(content_latents, char_idx)
        return img, c_vec, font_id

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def compute_style_loss(style_vecs, font_ids, num_fonts):
    """Style consistency loss: same font should have similar style vectors."""
    batch_size = tf.shape(style_vecs)[0]

    # Create font ID matrix for comparison
    font_ids = tf.cast(font_ids, tf.int32)
    font_ids_row = tf.expand_dims(font_ids, 1)
    font_ids_col = tf.expand_dims(font_ids, 0)
    same_font = tf.cast(tf.equal(font_ids_row, font_ids_col), tf.float32)

    # Compute pairwise distances
    style_row = tf.expand_dims(style_vecs, 1)
    style_col = tf.expand_dims(style_vecs, 0)
    distances = tf.reduce_sum(tf.square(style_row - style_col), axis=-1)

    # Same font: minimize distance, different font: maximize distance
    same_loss = tf.reduce_sum(same_font * distances) / (tf.reduce_sum(same_font) + 1e-6)
    diff_loss = tf.reduce_sum((1 - same_font) * tf.maximum(1.0 - distances, 0)) / (tf.reduce_sum(1 - same_font) + 1e-6)

    return same_loss + diff_loss


@tf.function
def train_step(images, c_vecs, font_ids, style_encoder, decoder, optimizer, num_fonts):
    """One training step."""
    with tf.GradientTape() as tape:
        # Forward pass
        style_vecs = style_encoder(images, training=True)
        preds = decoder([c_vecs, style_vecs], training=True)

        # Reconstruction loss (BCE for binary-like data)
        recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(images, preds))

        # Style consistency loss
        style_loss = compute_style_loss(style_vecs, font_ids, num_fonts)

        # Total loss
        total_loss = recon_loss + STYLE_WEIGHT * style_loss

    # Compute gradients
    trainable_vars = style_encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, trainable_vars))

    return total_loss, recon_loss, style_loss


@tf.function
def val_step(images, c_vecs, font_ids, style_encoder, decoder, num_fonts):
    """Validation step."""
    style_vecs = style_encoder(images, training=False)
    preds = decoder([c_vecs, style_vecs], training=False)

    recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(images, preds))
    style_loss = compute_style_loss(style_vecs, font_ids, num_fonts)

    return recon_loss + STYLE_WEIGHT * style_loss


def main():
    print("=" * 60)
    print("Joint Training v2 - Binary Font Data")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data, val_data, content_latents, num_fonts = load_data()
    content_latents = tf.constant(content_latents, dtype=tf.float32)

    # Create datasets
    train_ds = create_dataset(train_data, content_latents, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(val_data, content_latents, BATCH_SIZE, shuffle=False)

    steps_per_epoch = len(train_data) // BATCH_SIZE
    print(f"[TRAIN] Steps per epoch: {steps_per_epoch}")

    # Build models
    style_encoder = build_style_encoder()
    decoder = build_decoder()

    style_encoder.summary()
    decoder.summary()

    # Optimizer with learning rate decay
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, steps_per_epoch * EPOCHS
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Training
        train_losses = []
        for batch_idx, (images, c_vecs, font_ids) in enumerate(train_ds):
            loss, recon, style = train_step(
                images, c_vecs, font_ids,
                style_encoder, decoder, optimizer, num_fonts
            )
            train_losses.append(loss.numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"  Step {batch_idx+1}: loss={loss:.4f} (recon={recon:.4f}, style={style:.4f})")

        avg_train_loss = np.mean(train_losses)

        # Validation
        val_losses = []
        for images, c_vecs, font_ids in val_ds:
            loss = val_step(images, c_vecs, font_ids, style_encoder, decoder, num_fonts)
            val_losses.append(loss.numpy())

        avg_val_loss = np.mean(val_losses)

        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            style_encoder.save(OUTPUT_DIR / "style_encoder_best.h5")
            decoder.save(OUTPUT_DIR / "decoder_best.h5")
            print(f"  ✅ Saved best model (val_loss={best_val_loss:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
