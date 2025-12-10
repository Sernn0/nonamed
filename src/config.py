"""
Centralized configuration for FontByMe project.
All paths and constants are defined here for easy maintenance.
"""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = ROOT / "data"
HANDWRITING_RAW = DATA_DIR / "handwriting_raw"
HANDWRITING_PROCESSED = DATA_DIR / "handwriting_processed"
CONTENT_FONT_DIR = DATA_DIR / "content_font" / "NotoSansKR-Regular"

# Dataset index files
TRAIN_INDEX = HANDWRITING_PROCESSED / "handwriting_index_train_shared.json"
VAL_INDEX = HANDWRITING_PROCESSED / "handwriting_index_val_shared.json"
TEST_INDEX = HANDWRITING_PROCESSED / "handwriting_index_test_shared.json"
MASTER_INDEX = HANDWRITING_PROCESSED / "master_index.json"

# Model directories
RUNS_DIR = ROOT / "runs"
AUTOENC_DIR = RUNS_DIR / "autoenc"
JOINT_DIR = RUNS_DIR / "joint"

# Pretrained model paths
CONTENT_ENCODER = AUTOENC_DIR / "encoder.h5"
CONTENT_DECODER = AUTOENC_DIR / "decoder.h5"
CONTENT_LATENTS = AUTOENC_DIR / "content_latents_unified.npy"

# Joint model paths (after training)
STYLE_ENCODER_BEST = JOINT_DIR / "style_encoder_best.h5"
DECODER_BEST = JOINT_DIR / "decoder_best.h5"

# Character vocab
CHAR_VOCAB = ROOT / "src" / "data" / "char_vocab.json"

# Output
OUTPUT_DIR = ROOT / "outputs"

# Charset files
CHARSET_50 = ROOT / "charset_50.txt"
CHARSET_220 = ROOT / "charset_220.txt"
CHARSET_2350 = ROOT / "charset_2350.txt"

# Model hyperparameters
CONTENT_DIM = 64
STYLE_DIM = 32
IMAGE_SIZE = 256

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_LR = 2e-4
