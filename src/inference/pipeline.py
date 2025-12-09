"""
Pipeline module for FontByMe UI integration.
Handles: PDF -> Fine-tune -> Generate All Chars -> TTF/SVG
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# Paths relative to project root
ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DECODER = ROOT / "runs" / "autoenc" / "decoder.h5"
CONTENT_LATENTS = ROOT / "runs" / "autoenc" / "content_latents_clean.npy"
PNG_DIR = ROOT / "data" / "content_font" / "NotoSansKR-Regular"
OUTPUT_DIR = ROOT / "outputs"


def build_char_to_index(png_dir: Path) -> dict:
    """Build character -> content latent index mapping."""
    png_files = sorted(png_dir.glob("*.png"))
    mapping = {}
    for idx, f in enumerate(png_files):
        name = f.stem
        if name.startswith("U+"):
            try:
                codepoint = int(name[2:], 16)
                mapping[chr(codepoint)] = idx
            except ValueError:
                continue
    return mapping


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
    # Lazy imports to avoid loading TF when not needed
    import tensorflow as tf
    from tensorflow import keras

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping
    char_to_index = build_char_to_index(PNG_DIR)
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
    decoder_path: Optional[Path],
    output_dir: Path,
    batch_size: int = 32,
) -> List[Path]:
    """Generate all 2356 Korean glyphs using decoder.

    Args:
        decoder_path: Path to decoder. If None, uses pretrained decoder.
        output_dir: Output directory for generated PNGs
        batch_size: Batch size for generation
    """
    import tensorflow as tf
    from tensorflow import keras

    output_dir.mkdir(parents=True, exist_ok=True)

    char_to_index = build_char_to_index(PNG_DIR)
    content_latents = np.load(CONTENT_LATENTS)

    # Use pretrained decoder if no fine-tuned path provided
    if decoder_path is None or not decoder_path.exists():
        decoder_path = PRETRAINED_DECODER
        print("[Pipeline] Using pretrained decoder")

    decoder = keras.models.load_model(str(decoder_path), compile=False, safe_mode=False)

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
        batch_latents = tf.constant(all_latents[start:end])
        batch_preds = decoder(batch_latents, training=False).numpy()
        batch_preds = np.clip(batch_preds, 0, 1)

        for i, pred in enumerate(batch_preds):
            char_idx = start + i
            char = korean_chars[char_idx]
            codepoint = ord(char)

            # Decoder outputs black glyph on white bg (same as training data)
            img = (pred.squeeze() * 255).astype(np.uint8)
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
    chars: List[str],
    font_name: str = "MyHandwriting",
    epochs: int = 30,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Run complete pipeline: Fine-tune -> Generate -> TTF/SVG.

    Args:
        sample_images: User's handwriting PNG files
        chars: Corresponding characters
        font_name: Name for the generated font
        epochs: Training epochs

    Returns:
        Tuple of (svg_path, ttf_path)
    """
    work_dir = OUTPUT_DIR / "pipeline_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fine-tune decoder
    print("[Pipeline] Step 1: Fine-tuning decoder...")
    decoder_path = finetune_decoder(
        sample_images, chars,
        output_dir=work_dir,
        epochs=epochs
    )

    # Step 2: Generate all glyphs
    print("[Pipeline] Step 2: Generating all characters...")
    glyph_dir = work_dir / "glyphs"
    generate_all_glyphs(decoder_path, glyph_dir)

    # Step 3: Create font
    print("[Pipeline] Step 3: Creating TTF font...")
    ttf_path = OUTPUT_DIR / f"{font_name}.ttf"
    svg_dir = glyph_dir / "svg"

    if create_ttf_font(glyph_dir, ttf_path, font_name):
        # Package first SVG as representative
        svg_files = list(svg_dir.glob("*.svg"))
        if svg_files:
            svg_path = OUTPUT_DIR / f"{font_name}_sample.svg"
            svg_files[0].rename(svg_path) if not svg_path.exists() else None
        else:
            svg_path = None

        print(f"[Pipeline] Complete! TTF: {ttf_path}")
        return svg_path, ttf_path
    else:
        return None, None
