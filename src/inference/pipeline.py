"""
Pipeline module for FontByMe UI integration.
Handles: User Samples -> Style Encoding -> Generate All Chars -> TTF/SVG

Updated to use Joint Decoder (Content + Style) architecture.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import json

# Use centralized config
from src.config import (
    ROOT, DATA_DIR, CONTENT_LATENTS, CHAR_VOCAB, OUTPUT_DIR,
    STYLE_ENCODER_BEST, DECODER_BEST, CONTENT_DIM, STYLE_DIM, IMAGE_SIZE
)

# Legacy paths for backward compatibility
PRETRAINED_DECODER = ROOT / "runs" / "autoenc" / "decoder.h5"
PNG_DIR = ROOT / "data" / "content_font" / "NotoSansKR-Regular"



def load_char_vocab() -> dict:
    """Load unified character mapping."""
    if not CHAR_VOCAB.exists():
        raise FileNotFoundError(f"Character vocab not found at {CHAR_VOCAB}")
    with open(CHAR_VOCAB, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path, size: int = 256) -> np.ndarray:
    """Load and preprocess image.

    Args:
        path: Image file path
        size: Target size
    """
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def finetune_decoder(
    sample_images: List[Path],
    chars: List[str],
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
) -> Path:
    """Fine-tune decoder on user's handwriting samples.

    Args:
        sample_images: List of paths to user's handwriting PNG images
        chars: Corresponding characters for each image
        output_dir: Directory to save fine-tuned decoder
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning

    Returns:
        Path to fine-tuned decoder.h5
    """
    # Lazy imports to avoid loading TF when needed
    import tensorflow as tf
    from tensorflow import keras
    from src.data.unified_mapping import load_json_text_mapping, build_unified_mapping

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping
    char_to_index = load_char_vocab()
    content_latents = np.load(CONTENT_LATENTS)

    # Prepare data
    images = []
    latents = []

    for img_path, char in zip(sample_images, chars):
        if char not in char_to_index:
            continue
        idx = char_to_index[char]
        if idx >= len(content_latents):
            continue

        images.append(load_image(img_path))
        latents.append(content_latents[idx])

    if not images:
        raise ValueError("No valid samples found")

    images = np.stack(images)
    latents = np.stack(latents)
    print(f"[Pipeline] Loaded {len(images)} samples for fine-tuning")

    # Load decoder
    decoder = keras.models.load_model(str(PRETRAINED_DECODER), compile=False, safe_mode=False)
    decoder.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Training loop
    n = len(images)
    steps = max(1, n // batch_size)

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        loss_sum = 0.0

        for step in range(steps):
            bi = idx[step * batch_size : (step + 1) * batch_size]
            x = tf.constant(latents[bi])
            y = tf.constant(images[bi])

            with tf.GradientTape() as tape:
                pred = decoder(x, training=True)
                # Weighted loss: emphasize glyph pixels (dark areas)
                # y is normalized 0-1, where dark glyph = low values
                glyph_weight = 10.0  # Weight for glyph pixels
                weights = 1.0 + (1.0 - y) * (glyph_weight - 1.0)  # Higher weight for darker pixels
                loss = tf.reduce_mean(weights * tf.square(y - pred))

            grads = tape.gradient(loss, decoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, decoder.trainable_variables))
            loss_sum += loss.numpy()

        print(f"[Pipeline] Epoch {epoch+1}/{epochs} - loss: {loss_sum/steps:.4f}")

    # Save
    decoder_path = output_dir / "decoder_finetuned.h5"
    decoder.save(str(decoder_path))
    print(f"[Pipeline] Saved fine-tuned decoder to {decoder_path}")

    return decoder_path


def generate_all_glyphs(
    style_images: List[Path],
    output_dir: Path,
    style_encoder_path: Optional[Path] = None,
    decoder_path: Optional[Path] = None,
    batch_size: int = 32,
) -> List[Path]:
    """Generate all 2356 Korean glyphs using Joint Decoder with user's style.

    Args:
        style_images: List of user's handwriting sample images for style extraction
        output_dir: Output directory for generated PNGs
        style_encoder_path: Path to style encoder model
        decoder_path: Path to joint decoder model
        batch_size: Batch size for generation

    Returns:
        List of paths to generated glyph PNGs
    """
    import tensorflow as tf
    from tensorflow import keras

    output_dir.mkdir(parents=True, exist_ok=True)

    char_to_index = load_char_vocab()
    content_latents = np.load(CONTENT_LATENTS)

    # Load models
    if style_encoder_path is None or not style_encoder_path.exists():
        style_encoder_path = STYLE_ENCODER_BEST
    if decoder_path is None or not decoder_path.exists():
        decoder_path = DECODER_BEST

    if not style_encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"Joint models not found. Train with train_joint.py first.\n"
            f"  Style Encoder: {style_encoder_path}\n"
            f"  Decoder: {decoder_path}"
        )

    # Import custom layers for deserialization
    from src.models.decoder import ContentScaleLayer

    print(f"[Pipeline] Loading Style Encoder from {style_encoder_path}")
    style_encoder = keras.models.load_model(str(style_encoder_path), compile=False)

    print(f"[Pipeline] Loading Decoder from {decoder_path}")
    decoder = keras.models.load_model(
        str(decoder_path),
        custom_objects={'ContentScaleLayer': ContentScaleLayer},
        compile=False
    )

    # Extract style vector from user samples (average of all samples)
    print(f"[Pipeline] Extracting style from {len(style_images)} samples...")
    style_vecs = []
    for img_path in style_images:
        if not img_path.exists():
            continue
        img = load_image(img_path)
        img_batch = np.expand_dims(img, 0)
        style_vec = style_encoder(img_batch, training=False).numpy()
        style_vecs.append(style_vec)

    if not style_vecs:
        raise ValueError("No valid style images provided")

    # Average style vector
    avg_style = np.mean(np.concatenate(style_vecs, axis=0), axis=0, keepdims=True)
    print(f"[Pipeline] Style vector shape: {avg_style.shape}")

    # Get all Korean syllables
    korean_chars = []
    korean_indices = []
    for codepoint in range(0xAC00, 0xD7A4):
        char = chr(codepoint)
        if char in char_to_index:
            idx = char_to_index[char]
            if idx < len(content_latents):
                korean_chars.append(char)
                korean_indices.append(idx)

    print(f"[Pipeline] Generating {len(korean_chars)} characters...")

    all_latents = content_latents[korean_indices]
    n = len(korean_chars)
    out_paths = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_latents = tf.constant(all_latents[start:end], dtype=tf.float32)

        # Repeat style vector for batch
        batch_size_actual = end - start
        batch_style = tf.constant(np.tile(avg_style, (batch_size_actual, 1)), dtype=tf.float32)

        # Generate with Joint Decoder
        batch_preds = decoder([batch_latents, batch_style], training=False).numpy()
        batch_preds = np.clip(batch_preds, 0, 1)

        for i, pred in enumerate(batch_preds):
            char_idx = start + i
            char = korean_chars[char_idx]
            codepoint = ord(char)

            # Contrast enhancement: model output is too faint (mean ~0.93)
            # Invert (so glyph is high value), multiply, clip, invert back
            arr = pred.squeeze()
            arr = 1.0 - arr  # Invert: background=0, glyph=high
            arr = arr * 4.0  # Amplify contrast 4x
            arr = np.clip(arr, 0, 1)
            arr = 1.0 - arr  # Invert back: background=white, glyph=dark

            img = (arr * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode="L")
            out_path = output_dir / f"U+{codepoint:04X}.png"
            pil_img.save(out_path)
            out_paths.append(out_path)

        if (end % 500 == 0) or (end == n):
            print(f"[Pipeline] Generated {end}/{n} characters")

    return out_paths


def png_to_svg(png_path: Path, svg_path: Path, threshold: int = 220) -> bool:
    """Convert PNG to SVG using potrace."""
    try:
        img = Image.open(png_path).convert("L")
        arr = np.array(img)

        # Glyph = dark pixels (below threshold)
        binary = (arr < threshold).astype(np.uint8)

        # Save as PBM
        with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as tmp:
            pbm_path = tmp.name

        h, w = binary.shape
        with open(pbm_path, "wb") as f:
            f.write(f"P4\n{w} {h}\n".encode())
            packed = np.packbits(binary, axis=1)
            f.write(packed.tobytes())

        # Run potrace
        result = subprocess.run(
            ["potrace", "-s", "-o", str(svg_path), pbm_path],
            capture_output=True
        )

        Path(pbm_path).unlink()
        return result.returncode == 0

    except Exception as e:
        print(f"[Pipeline] Error converting {png_path}: {e}")
        return False


def create_ttf_font(
    glyph_dir: Path,
    output_ttf: Path,
    font_name: str = "MyHandwriting",
    em_size: int = 1024,
) -> bool:
    """Create TTF font from glyph PNGs."""
    svg_dir = glyph_dir / "svg"
    svg_dir.mkdir(exist_ok=True)

    # Convert PNGs to SVGs
    png_files = sorted(glyph_dir.glob("U+*.png"))
    print(f"[Pipeline] Converting {len(png_files)} PNGs to SVG...")

    success = 0
    for i, png_path in enumerate(png_files):
        svg_path = svg_dir / f"{png_path.stem}.svg"
        if png_to_svg(png_path, svg_path):
            success += 1
        if (i + 1) % 500 == 0:
            print(f"[Pipeline] Converted {i+1}/{len(png_files)}")

    print(f"[Pipeline] Converted {success}/{len(png_files)} to SVG")

    # Create font using fontforge
    script = f'''
import fontforge

font = fontforge.font()
font.fontname = "{font_name}"
font.familyname = "{font_name}"
font.fullname = "{font_name}"
font.em = {em_size}

import os
svg_dir = "{svg_dir}"

for filename in os.listdir(svg_dir):
    if not filename.endswith(".svg"):
        continue

    name = filename[:-4]
    if not name.startswith("U+"):
        continue

    try:
        codepoint = int(name[2:], 16)
    except ValueError:
        continue

    glyph = font.createChar(codepoint)
    glyph.importOutlines(os.path.join(svg_dir, filename))
    glyph.width = {em_size}

    try:
        glyph.autoHint()
    except:
        pass

font.generate("{output_ttf}")
print(f"Generated font with {{len(font)}} glyphs")
'''

    script_path = svg_dir / "create_font.py"
    script_path.write_text(script)

    print("[Pipeline] Creating TTF font...")
    result = subprocess.run(
        ["fontforge", "-script", str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[Pipeline] FontForge error: {result.stderr}")
        return False

    print(result.stdout)
    return output_ttf.exists()


def run_full_pipeline(
    sample_images: List[Path],
    font_name: str = "MyHandwriting",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Run complete pipeline: Style Extract -> Generate All Glyphs -> TTF/SVG.

    Uses pre-trained Joint Decoder + Style Encoder. No fine-tuning needed.

    Args:
        sample_images: User's handwriting PNG files (for style extraction)
        font_name: Name for the generated font

    Returns:
        Tuple of (svg_path, ttf_path)
    """
    work_dir = OUTPUT_DIR / "pipeline_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate all glyphs using user's style
    print("[Pipeline] Step 1: Generating all characters with user style...")
    glyph_dir = work_dir / "glyphs"
    generate_all_glyphs(
        style_images=sample_images,
        output_dir=glyph_dir,
    )

    # Step 2: Create font
    print("[Pipeline] Step 2: Creating TTF font...")
    ttf_path = OUTPUT_DIR / f"{font_name}.ttf"
    svg_dir = glyph_dir / "svg"

    if create_ttf_font(glyph_dir, ttf_path, font_name):
        # Package first SVG as representative
        svg_files = list(svg_dir.glob("*.svg"))
        if svg_files:
            svg_path = OUTPUT_DIR / f"{font_name}_sample.svg"
            if not svg_path.exists():
                svg_files[0].rename(svg_path)
        else:
            svg_path = None

        print(f"[Pipeline] Complete! TTF: {ttf_path}")
        return svg_path, ttf_path
    else:
        return None, None

