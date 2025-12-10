from __future__ import annotations

"""
Decoder (generator) that maps content and style latents to a 256x256 grayscale image.
Uses the same architecture as the proven content autoencoder decoder.
"""

from typing import Tuple

import tensorflow as tf


def build_decoder(
    content_dim: int = 64,
    style_dim: int = 32,
    img_shape: Tuple[int, int, int] = (256, 256, 1),
) -> tf.keras.Model:
    """
    Build a decoder that generates an image from content and style vectors.
    Architecture mirrors the working content autoencoder decoder.
    """
    content_input = tf.keras.Input(shape=(content_dim,), name="content_vec")
    style_input = tf.keras.Input(shape=(style_dim,), name="style_vec")

    # Content Latents are large (-24 ~ 23), potentially causing saturation.
    # Scale them down to ~ (-1.2 ~ 1.2) range to match style vectors (-0.3 ~ 0.3).
    # Using 0.05 scaling factor.
    scaled_content = tf.keras.layers.Lambda(lambda t: t * 0.05, name="scale_content")(content_input)

    x = tf.keras.layers.Concatenate(name="concat_latent")([scaled_content, style_input])

    # Project to 16x16x256 feature map (same as content autoencoder decoder)
    x = tf.keras.layers.Dense(16 * 16 * 256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 256))(x)

    # Decoder blocks: Conv2DTranspose + BatchNorm + ReLU + UpSampling2D
    x = tf.keras.layers.Conv2DTranspose(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)  # 32x32

    x = tf.keras.layers.Conv2DTranspose(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)  # 64x64

    x = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)  # 128x128

    x = tf.keras.layers.Conv2DTranspose(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D(2)(x)  # 256x256

    output = tf.keras.layers.Conv2D(
        filters=img_shape[-1],
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name="output_image",
    )(x)

    model = tf.keras.Model(
        inputs=[content_input, style_input],
        outputs=output,
        name="decoder",
    )
    return model


def _dummy_forward() -> None:
    """Run a dummy forward pass for quick verification."""
    content_dim = 64
    style_dim = 32
    decoder = build_decoder(content_dim=content_dim, style_dim=style_dim)
    decoder.summary()
    content_vec = tf.random.uniform((4, content_dim))
    style_vec = tf.random.uniform((4, style_dim))
    output = decoder([content_vec, style_vec])
    print("Output shape:", output.shape)


if __name__ == "__main__":
    _dummy_forward()
