from __future__ import annotations

"""
Style Encoder implemented with Keras Functional API.
Input:  (256, 256, 1) grayscale image
Output: style latent vector of size style_dim
"""

from typing import Tuple

import tensorflow as tf


class _UnitL2(tf.keras.layers.Layer):
    """
    L2 normalize along a given axis. Fallback when UnitNormalization is unavailable.
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, strides: int = 1) -> tf.Tensor:
    """Conv2D -> BatchNorm -> ReLU block."""
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_style_encoder(
    style_dim: int = 32,
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    l2_normalize: bool = True,
) -> tf.keras.Model:
    """
    Build a CNN-based style encoder.

    Args:
        style_dim: Dimension of the style latent vector.
        input_shape: Shape of the input image (H, W, C).
        l2_normalize: Whether to L2-normalize the output vector.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = conv_block(inputs, 32)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = conv_block(x, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = conv_block(x, 128)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = conv_block(x, 256)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = conv_block(x, 256)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    style_vec = tf.keras.layers.Dense(style_dim, name="style_dense")(x)
    if l2_normalize:
        # Prefer built-in UnitNormalization; fall back to custom layer for older TF.
        norm_layer = getattr(tf.keras.layers, "UnitNormalization", _UnitL2)
        style_vec = norm_layer(axis=-1, name="style_l2_norm")(style_vec)

    model = tf.keras.Model(inputs=inputs, outputs=style_vec, name="style_encoder")
    return model


def _dummy_forward() -> None:
    """Run a dummy forward pass for quick verification."""
    model = build_style_encoder()
    dummy_input = tf.random.uniform((4, 256, 256, 1))
    output = model(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    _dummy_forward()
