from __future__ import annotations

"""
Skeleton CNN-based style encoder.
Input: (256, 256, 1) grayscale glyph.
Output: style vector of size style_dim.
"""

from typing import Optional

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore


class StyleEncoder(tf.keras.Model if tf else object):
    """CNN style encoder producing a style vector."""

    def __init__(self, style_dim: int = 32, name: Optional[str] = None):
        if tf:
            super().__init__(name=name or "style_encoder")
            self.style_dim = style_dim
            # Template layers; adjust architecture during implementation.
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")
            self.pool1 = tf.keras.layers.MaxPool2D(2)
            self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")
            self.pool2 = tf.keras.layers.MaxPool2D(2)
            self.conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")
            self.pool3 = tf.keras.layers.MaxPool2D(2)
            self.flatten = tf.keras.layers.Flatten()
            self.fc = tf.keras.layers.Dense(style_dim, name="style_vector")
        else:  # pragma: no cover
            # Dummy attributes to avoid attribute errors when TensorFlow is missing.
            self.style_dim = style_dim

    def call(self, inputs, training: bool = False):
        """Forward pass returning the style vector."""
        if not tf:  # pragma: no cover
            raise RuntimeError("TensorFlow is required to run the style encoder.")
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        style_vec = self.fc(x)
        return style_vec


def build_style_encoder(input_shape=(256, 256, 1), style_dim: int = 32):
    """Helper to build and return a compiled style encoder model."""
    if not tf:  # pragma: no cover
        return StyleEncoder(style_dim)
    inputs = tf.keras.Input(shape=input_shape)
    encoder = StyleEncoder(style_dim=style_dim)
    outputs = encoder(inputs)
    model = tf.keras.Model(inputs, outputs, name="style_encoder_model")
    # TODO: add optimizer/compile when training is implemented.
    return model


if __name__ == "__main__":
    # Minimal smoke test to ensure the skeleton runs.
    if tf:
        model = build_style_encoder()
        dummy = tf.zeros((1, 256, 256, 1))
        out = model(dummy)
        print("Style vector shape:", out.shape)
    else:
        print("TensorFlow not available; style encoder skeleton loaded.")
