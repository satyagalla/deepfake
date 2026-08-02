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

# --- 0004: patch-based frozen-CLIP adaptation probe (v2) ---
# COCO_AI's real schema -- one real column + six generator columns, paired
# 1:1 by row. `0003` only ever read 2 of these 7; `0004` §5 uses all of them.
COCO_AI_COLUMNS = {
    "real": "coco_image",
    "sd21": "sd21_image",
    "sdxl": "sdxl_image",
    "sd3": "sd3_image",
    "sd35": "sd35_image",
    "dalle": "dalle_image",
    "midjourney": "midjourney_image",
}
# Train/val fake generators -- everything except the real column and the
# held-out honesty anchor (I7: midjourney never enters training at any N).
COCO_AI_TRAIN_GENERATORS = ["sd21", "sdxl", "sd3", "sd35", "dalle"]
HELDOUT_GENERATOR = "midjourney"

# Self-generated N-shot pool (S2) -- COCO-caption-prompted, off-domain-prompted,
# and web-UI-sourced images for the two live-test generators.
SELFGEN_GENERATORS = ["gemini", "gptimage"]

# Head B's classes (0004 §7.2) -- real + the 5 trained-on generators + the
# 2 self-generated ones. Midjourney is deliberately absent: it is 0-shot only.
GENERATOR_CLASSES = ["real"] + COCO_AI_TRAIN_GENERATORS + SELFGEN_GENERATORS

COCO_AI_RAW_DIR = RAW_DIR / "coco_ai"  # coco_ai/<column_name>/<row_id>.jpg, row_id shared across columns (I2/I3 join key)
SELFGEN_INTAKE_DIR = RAW_DIR / "selfgen_intake"  # drop zone: selfgen_intake/<generator>/<domain>/<container>/*, as generated
SELFGEN_RAW_DIR = RAW_DIR / "selfgen"  # organized output: selfgen/<generator>/<domain>/<container>/<name>.jpg, JPEG q95
SELFGEN_MANIFEST_PATH = SELFGEN_RAW_DIR / "manifest.csv"
SELFGEN_DOMAINS = ["indomain", "offdomain"]
SELFGEN_CONTAINERS = ["api", "web"]

JPEG_QUALITY = 95  # I4: uniform container across every class, re-saved regardless of source

# --- patch extraction (0004 §6) ---
N_PATCHES = 16  # I1: fixed per image regardless of source resolution -- the bag-size leak (0004 §6.3)
PATCH_SIZE = 224  # CLIP ViT-L/14's native input
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
# CLIP's own normalization (I5) -- NOT model/demo.py's IMAGENET_MEAN/STD.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PROBE_DATA_ROOT = DATA_ROOT / "probe"  # feature cache, splits, fitted heads for the v2 pipeline
FEATURES_DIR = PROBE_DATA_ROOT / "features"  # native-patches arm (default)
FEATURES_STANDARD_DIR = PROBE_DATA_ROOT / "features_standard"  # E7 control arm: standard resize+centrecrop
SPLITS_DIR = PROBE_DATA_ROOT / "splits"
PROBE_CHECKPOINT_DIR = PROBE_DATA_ROOT / "checkpoints"
PROBE_OUTPUTS_DIR = PROBE_DATA_ROOT / "outputs"  # experiment results (json) + plots (png)
EXTRACT_BATCH_SIZE = 384  # build plan S3: batch 256-512 on an A100

# --- E1 (0004 §9) ---
NSHOT_VALUES = [0, 5, 10, 20, 30, 50, 100]
NSHOT_TRIALS = 5  # random draws per N -- plot the spread, not just the mean

# --- evidence cards / fusion (0004 §8, verdicts per 0005 §6) ---
# Verdict cut points are fitted from a **false-positive budget** on the real
# val split (0005 §7), not guessed: the LIKELY_SYNTHETIC cut is the 99th
# percentile of real fused scores, so it means "one real photo in a hundred
# gets called synthetic." `probe/fit_production.py` fits and saves them;
# these are the fallbacks used when no fitted file is present. 0.85/0.5 are
# Plurall's [stated] numbers (notes.md:57) and remain the defaults so the
# demo degrades to their spec rather than to nothing.
VERDICT_FPR_BUDGETS = {"likely_synthetic": 0.01, "weak_evidence": 0.10}
VERDICT_LIKELY_SYNTHETIC_THRESHOLD = 0.85
VERDICT_WEAK_EVIDENCE_THRESHOLD = 0.5
VERDICT_THRESHOLDS_NAME = "verdict_thresholds"

# --- checkpoint names (probe/heads.py save/load), shared between the
# fitting notebook cells and probe/demo.py so they agree on where to look ---
DEFAULT_POOL_METHOD = "mean"
HEAD_A_NAME = "head_a"
HEAD_B_NAME = "head_b"
GATE_NAME = "mahalanobis_gate"
SPECTRAL_HEAD_NAME = "spectral_head"

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
