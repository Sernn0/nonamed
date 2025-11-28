from __future__ import annotations

"""
Skeleton decoder combining content and style vectors to generate a glyph.
"""

from typing import Optional

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None  # type: ignore


class Decoder(tf.keras.Model if tf else object):
    """Template decoder: concatenates content/style and upsamples to image."""

    def __init__(
        self,
        content_dim: int = 64,
        style_dim: int = 32,
        output_shape=(256, 256, 1),
        name: Optional[str] = None,
    ):
        if tf:
            super().__init__(name=name or "decoder")
            self.output_shape_target = output_shape
            latent_dim = content_dim + style_dim
            # Template latent projection size; adjust as needed.
            self.dense = tf.keras.layers.Dense(16 * 16 * 128, activation="relu")
            self.reshape_layer = tf.keras.layers.Reshape((16, 16, 128))
            self.deconv_blocks = [
                tf.keras.layers.Conv2DTranspose(128, 3, padding="same", activation="relu"),
                tf.keras.layers.UpSampling2D(2),
                tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu"),
                tf.keras.layers.UpSampling2D(2),
                tf.keras.layers.Conv2DTranspose(32, 3, padding="same", activation="relu"),
                tf.keras.layers.UpSampling2D(2),
                tf.keras.layers.Conv2DTranspose(16, 3, padding="same", activation="relu"),
                tf.keras.layers.UpSampling2D(2),
            ]
            self.output_conv = tf.keras.layers.Conv2D(
                filters=output_shape[-1],
                kernel_size=3,
                padding="same",
                activation="sigmoid",
            )
        else:  # pragma: no cover
            self.output_shape_target = output_shape

    def call(self, inputs, training: bool = False):
        """Forward pass returning generated glyph."""
        if not tf:  # pragma: no cover
            raise RuntimeError("TensorFlow is required to run the decoder.")
        content_vec, style_vec = inputs
        x = tf.concat([content_vec, style_vec], axis=-1)
        x = self.dense(x)
        x = self.reshape_layer(x)
        for layer in self.deconv_blocks:
            x = layer(x)
        x = self.output_conv(x)
        return x


def build_decoder(
    content_dim: int = 64, style_dim: int = 32, output_shape=(256, 256, 1)
):
    """Helper to build decoder model."""
    if not tf:  # pragma: no cover
        return Decoder(content_dim=content_dim, style_dim=style_dim, output_shape=output_shape)
    content_input = tf.keras.Input(shape=(content_dim,), name="content_vec")
    style_input = tf.keras.Input(shape=(style_dim,), name="style_vec")
    decoder = Decoder(content_dim=content_dim, style_dim=style_dim, output_shape=output_shape)
    outputs = decoder((content_input, style_input))
    model = tf.keras.Model([content_input, style_input], outputs, name="decoder_model")
    # TODO: compile when training logic is added.
    return model


if __name__ == "__main__":
    if tf:
        model = build_decoder()
        c = tf.zeros((1, 64))
        s = tf.zeros((1, 32))
        out = model([c, s])
        print("Decoder output shape:", out.shape)
    else:
        print("TensorFlow not available; decoder skeleton loaded.")
