#!/usr/bin/env python3
"""
Joint Training Script v3 - Binarized Handwriting Data

Trains Style Encoder + Joint Decoder on binarized handwriting data.
Each writer is treated as a "style".
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
DATA_DIR = ROOT / "data" / "handwriting_binary"
CONTENT_LATENTS = ROOT / "runs" / "autoenc" / "content_latents_unified.npy"
CHAR_VOCAB = ROOT / "src" / "data" / "char_vocab.json"
OUTPUT_DIR = ROOT / "runs" / "handwriting_v3"

# Hyperparameters
LATENT_DIM = 64
STYLE_DIM = 32
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 2e-4
STYLE_WEIGHT = 0.3


def build_style_encoder():
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
    class ContentScaleLayer(keras.layers.Layer):
        def __init__(self, scale=0.1, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale
        def call(self, inputs):
            return inputs * self.scale
        def get_config(self):
            config = super().get_config()
            config.update({"scale": self.scale})
            return config

    content_input = keras.Input(shape=(LATENT_DIM,), name="content_latent")
    style_input = keras.Input(shape=(STYLE_DIM,), name="style_vector")
    content_scaled = ContentScaleLayer(scale=0.1)(content_input)
    combined = keras.layers.Concatenate()([content_scaled, style_input])
    x = keras.layers.Dense(16 * 16 * 256)(combined)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Reshape((16, 16, 256))(x)
    for filters in [128, 64, 32, 16]:
        x = keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    outputs = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model([content_input, style_input], outputs, name="decoder")


def load_data():
    """Load binarized handwriting data."""
    # Find all writer folders
    writers = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"[DATA] Found {len(writers)} writers")

    # Load content latents and vocab
    content_latents = np.load(CONTENT_LATENTS)
    with open(CHAR_VOCAB, 'r') as f:
        vocab = json.load(f)

    # Build char index (Unicode codepoint -> index)
    char_to_idx = {char: idx for char, idx in vocab.items()}

    # Parse handwriting filename: 00130001001.png
    # Format: NNNNYYYYXXX where NNNN=?, YYYY=unicode, XXX=sample
    all_data = []
    for writer_id, writer_name in enumerate(writers):
        writer_dir = DATA_DIR / writer_name
        for png in writer_dir.glob("*.png"):
            # Extract unicode from filename (positions 4-8)
            fname = png.stem
            if len(fname) >= 8:
                try:
                    unicode_part = int(fname[4:8])
                    # Map to actual codepoint (1=AC00, 2=AC01, ...)
                    codepoint = 0xAC00 + unicode_part - 1
                    char = chr(codepoint)
                    if char in char_to_idx:
                        all_data.append({
                            "path": str(png),
                            "writer_id": writer_id,
                            "char_idx": char_to_idx[char]
                        })
                except:
                    continue

    print(f"[DATA] Total samples: {len(all_data)}")

    # Shuffle and split
    np.random.shuffle(all_data)
    n = len(all_data)
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)

    train_data = all_data[:train_n]
    val_data = all_data[train_n:train_n+val_n]

    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data, content_latents, len(writers)


def create_dataset(data_list, content_latents, batch_size, shuffle=True):
    paths = [d["path"] for d in data_list]
    writer_ids = [d["writer_id"] for d in data_list]
    char_idxs = [d["char_idx"] for d in data_list]

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def generator():
        for i in range(len(paths)):
            yield paths[i], char_idxs[i], writer_ids[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    def process(path, char_idx, writer_id):
        img = load_image(path)
        c_vec = tf.gather(content_latents, char_idx)
        return img, c_vec, writer_id

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_style_loss(style_vecs, writer_ids, num_writers):
    writer_ids = tf.cast(writer_ids, tf.int32)
    writer_ids_row = tf.expand_dims(writer_ids, 1)
    writer_ids_col = tf.expand_dims(writer_ids, 0)
    same_writer = tf.cast(tf.equal(writer_ids_row, writer_ids_col), tf.float32)

    style_row = tf.expand_dims(style_vecs, 1)
    style_col = tf.expand_dims(style_vecs, 0)
    distances = tf.reduce_sum(tf.square(style_row - style_col), axis=-1)

    same_loss = tf.reduce_sum(same_writer * distances) / (tf.reduce_sum(same_writer) + 1e-6)
    diff_loss = tf.reduce_sum((1 - same_writer) * tf.maximum(1.0 - distances, 0)) / (tf.reduce_sum(1 - same_writer) + 1e-6)
    return same_loss + diff_loss


@tf.function
def train_step(images, c_vecs, writer_ids, style_encoder, decoder, optimizer, num_writers):
    with tf.GradientTape() as tape:
        style_vecs = style_encoder(images, training=True)
        preds = decoder([c_vecs, style_vecs], training=True)
        recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(images, preds))
        style_loss = compute_style_loss(style_vecs, writer_ids, num_writers)
        total_loss = recon_loss + STYLE_WEIGHT * style_loss

    trainable_vars = style_encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return total_loss, recon_loss, style_loss


@tf.function
def val_step(images, c_vecs, writer_ids, style_encoder, decoder, num_writers):
    style_vecs = style_encoder(images, training=False)
    preds = decoder([c_vecs, style_vecs], training=False)
    recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(images, preds))
    style_loss = compute_style_loss(style_vecs, writer_ids, num_writers)
    return recon_loss + STYLE_WEIGHT * style_loss


def main():
    print("=" * 60)
    print("Joint Training v3 - Binarized Handwriting Data")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data, val_data, content_latents, num_writers = load_data()
    content_latents = tf.constant(content_latents, dtype=tf.float32)

    train_ds = create_dataset(train_data, content_latents, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(val_data, content_latents, BATCH_SIZE, shuffle=False)

    steps_per_epoch = len(train_data) // BATCH_SIZE
    print(f"[TRAIN] Steps per epoch: {steps_per_epoch}")

    style_encoder = build_style_encoder()
    decoder = build_decoder()
    style_encoder.summary()
    decoder.summary()

    lr_schedule = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, steps_per_epoch * EPOCHS)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_losses = []
        for batch_idx, (images, c_vecs, writer_ids) in enumerate(train_ds):
            loss, recon, style = train_step(images, c_vecs, writer_ids, style_encoder, decoder, optimizer, num_writers)
            train_losses.append(loss.numpy())
            if (batch_idx + 1) % 100 == 0:
                print(f"  Step {batch_idx+1}: loss={loss:.4f} (recon={recon:.4f}, style={style:.4f})")

        avg_train_loss = np.mean(train_losses)

        val_losses = []
        for images, c_vecs, writer_ids in val_ds:
            loss = val_step(images, c_vecs, writer_ids, style_encoder, decoder, num_writers)
            val_losses.append(loss.numpy())
        avg_val_loss = np.mean(val_losses)

        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
