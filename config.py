"""Shared paths/constants for data/*.py and model/*.py."""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# Storage root for raw data, processed dataset, and checkpoints. Defaults to
# ROOT (local disk) so the local venv workflow is untouched; set DEEPFAKE_DATA_ROOT
# to a mounted Google Drive path in Colab (e.g. /content/drive/MyDrive/deepfake)
# to persist raw/processed data and checkpoints across runtime recycles.
DATA_ROOT = Path(os.environ.get("DEEPFAKE_DATA_ROOT", ROOT))

# --- raw download targets ---
RAW_DIR = DATA_ROOT / "data_raw"
REAL_SRC = RAW_DIR / "real_src"
DEEPFAKE_SRC = RAW_DIR / "deepfake_src"
EDITED_SRC = RAW_DIR / "edited_src"
EDITED_SRC_PSBATTLES = RAW_DIR / "edited_src_psbattles"

HF_DATASET = "NasrinImp/COCO_AI"
KAGGLE_DATASET = "divg07/casia-20-image-tampering-detection-dataset"
PS_BATTLES_DATASET = "timocasti/psbattles"

# --- face-filtered dataset ---
DATASET_DIR = DATA_ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
CLASSES = ["real", "edited", "deepfake"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGE_SIZE = 380
FACE_MARGIN = 40
SEED = 42
VAL_FRACTION = 0.15

# --- model ---
EMBED_DIM = 256
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- eval outputs (metrics JSON, Grad-CAM/explainability dump) ---
EVAL_DIR = DATA_ROOT / "eval"
