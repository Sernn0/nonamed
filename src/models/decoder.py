from __future__ import annotations

"""
Decoder (generator) that maps content and style latents to a 256x256 grayscale image.
"""

from typing import Tuple

import tensorflow as tf


def up_block(x: tf.Tensor, filters: int, kernel_size: int = 3, strides: int = 1) -> tf.Tensor:
    """Upsampling block: Conv2DTranspose -> BatchNorm -> ReLU."""
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_decoder(
    content_dim: int = 64,
    style_dim: int = 32,
    img_shape: Tuple[int, int, int] = (256, 256, 1),
) -> tf.keras.Model:
    """
    Build a decoder that generates an image from content and style vectors.
    """
    content_input = tf.keras.Input(shape=(content_dim,), name="content_vec")
    style_input = tf.keras.Input(shape=(style_dim,), name="style_vec")

    x = tf.keras.layers.Concatenate(name="concat_latent")([content_input, style_input])

    # Project to a low-res feature map (8x8x512 for richer features).
    x = tf.keras.layers.Dense(8 * 8 * 512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((8, 8, 512))(x)

    x = up_block(x, 512, strides=2)  # 16x16
    x = up_block(x, 256, strides=2)  # 32x32
    x = up_block(x, 256, strides=2)  # 64x64
    x = up_block(x, 128, strides=2)  # 128x128
    x = up_block(x, 64, strides=2)   # 256x256

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
    content_vec = tf.random.uniform((4, content_dim))
    style_vec = tf.random.uniform((4, style_dim))
    output = decoder([content_vec, style_vec])
    print("Output shape:", output.shape)


if __name__ == "__main__":
    _dummy_forward()
