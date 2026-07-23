"""Download raw sources for the real / edited / deepfake pipeline.

Sources:
- deepfake + real: NasrinImp/COCO_AI on Hugging Face (coco_image = pristine
  original, dalle_image = DALL-E 3 generation), paired 1:1 by row.
- edited: CASIA v2.0 tampered images only, via Kaggle
  (divg07/casia-20-image-tampering-detection-dataset). The authentic set
  CASIA also ships is ignored -- real comes solely from COCO_AI.

Auth:
- Kaggle: set KAGGLE_API_TOKEN (new single-token auth) or the classic
  kaggle.json (KAGGLE_CONFIG_DIR) before running.
- Hugging Face: HF_TOKEN env var, only needed if you hit rate limits on an
  anonymous pull (COCO_AI is a public dataset).
"""
import argparse
import itertools
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEEPFAKE_SRC, EDITED_SRC, HF_DATASET, KAGGLE_DATASET, REAL_SRC, SEED


def download_coco_ai(n_pairs: int, seed: int = SEED) -> None:
    from datasets import load_dataset

    REAL_SRC.mkdir(parents=True, exist_ok=True)
    DEEPFAKE_SRC.mkdir(parents=True, exist_ok=True)

    buffer_size = min(n_pairs * 2, 2000)
    print(f"Streaming {HF_DATASET} (train split), sampling {n_pairs} pairs (shuffle buffer={buffer_size})...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.select_columns(["coco_image", "dalle_image"])
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    print("Filling shuffle buffer (downloads underlying shards -- no per-row progress until this completes)...")
    written = 0
    with tqdm(total=n_pairs, desc="COCO_AI pairs", unit="pair") as pbar:
        for i, row in enumerate(itertools.islice(ds, n_pairs)):
            real_img, fake_img = row["coco_image"], row["dalle_image"]
            if real_img is None or fake_img is None:
                continue
            name = f"{i:05d}.jpg"
            real_img.convert("RGB").save(REAL_SRC / name, quality=95)
            fake_img.convert("RGB").save(DEEPFAKE_SRC / name, quality=95)
            written += 1
            pbar.update(1)

    print(f"COCO_AI: wrote {written} real/deepfake pairs to {REAL_SRC} and {DEEPFAKE_SRC}")
    if written == 0:
        raise RuntimeError(
            "No COCO_AI pairs written -- the dataset schema (coco_image/dalle_image "
            "columns) may have changed upstream; check https://huggingface.co/datasets/NasrinImp/COCO_AI"
        )


def download_casia(force: bool = False) -> None:
    EDITED_SRC.mkdir(parents=True, exist_ok=True)
    if any(EDITED_SRC.iterdir()) and not force:
        print(f"{EDITED_SRC} already has files, skipping Kaggle download (use --force-casia to redo).")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "The 'kaggle' package is required for CASIA v2.0 download. Install it "
            "(`uv pip install kaggle` in this venv), set KAGGLE_API_TOKEN (or a "
            "kaggle.json via KAGGLE_CONFIG_DIR), then re-run."
        ) from e

    print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(EDITED_SRC), unzip=True, quiet=False)
    print(f"CASIA v2.0 downloaded and unzipped into {EDITED_SRC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=3000, help="real/deepfake pairs to pull from COCO_AI")
    parser.add_argument("--skip-coco-ai", action="store_true")
    parser.add_argument("--skip-casia", action="store_true")
    parser.add_argument("--force-casia", action="store_true", help="re-download CASIA even if edited_src is non-empty")
    args = parser.parse_args()

    if not args.skip_coco_ai:
        download_coco_ai(args.n_pairs)
    if not args.skip_casia:
        download_casia(force=args.force_casia)


if __name__ == "__main__":
    main()
