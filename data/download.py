"""Download raw sources for the real / edited / deepfake pipeline.

Sources:
- deepfake + real: NasrinImp/COCO_AI on Hugging Face (coco_image = pristine
  original, dalle_image = DALL-E 3 generation), paired 1:1 by row.
- edited: CASIA v2.0 tampered images, via Kaggle
  (divg07/casia-20-image-tampering-detection-dataset), plus PS-Battles
  derivative (Photoshopped) images, via Kaggle (timocasti/psbattles) for
  manipulation-technique diversity. Each dataset's own authentic/original
  set is ignored -- real comes solely from COCO_AI.

Auth:
- Kaggle: set KAGGLE_API_TOKEN (new single-token auth) or the classic
  kaggle.json (KAGGLE_CONFIG_DIR) before running.
- Hugging Face: HF_TOKEN env var, only needed if you hit rate limits on an
  anonymous pull (COCO_AI is a public dataset).

All three downloads are independent, I/O-bound network pulls -- main() runs
them concurrently (ThreadPoolExecutor) rather than one after another, so
total download wall-clock is roughly the slowest of the three instead of
the sum.
"""
import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEEPFAKE_SRC,
    EDITED_SRC,
    EDITED_SRC_PSBATTLES,
    HF_DATASET,
    KAGGLE_DATASET,
    PS_BATTLES_DATASET,
    REAL_SRC,
    SEED,
)

# COCO_AI pairs are sampled from all of COCO, most of which has no person/face
# in frame at all -- that (not the detector) is why face-filtering's survival
# rate was ~15-20% across every class. Pre-filter on the caption's wording so
# the pool MTCNN runs on is mostly person-containing to begin with.
PERSON_KEYWORDS = {
    "man", "men", "woman", "women", "person", "people", "boy", "boys", "girl", "girls",
    "child", "children", "kid", "kids", "baby", "babies", "guy", "guys", "lady", "ladies",
    "face", "faces", "toddler", "toddlers", "adult", "adults", "player", "players",
}


def caption_has_person(caption: str) -> bool:
    words = re.findall(r"[a-z']+", caption.lower())
    return any(w in PERSON_KEYWORDS for w in words)


def download_coco_ai(n_pairs: int, seed: int = SEED) -> None:
    from datasets import load_dataset

    REAL_SRC.mkdir(parents=True, exist_ok=True)
    DEEPFAKE_SRC.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {HF_DATASET} (train split), collecting {n_pairs} person-caption pairs...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.select_columns(["caption", "coco_image", "dalle_image"])
    ds = ds.shuffle(seed=seed, buffer_size=10_000)  # whole dataset is ~10k rows

    written = seen = 0
    with tqdm(total=n_pairs, desc="COCO_AI person pairs", unit="pair") as pbar:
        for row in ds:
            seen += 1
            if not caption_has_person(row.get("caption") or ""):
                continue
            real_img, fake_img = row["coco_image"], row["dalle_image"]
            if real_img is None or fake_img is None:
                continue
            name = f"{written:05d}.jpg"
            real_img.convert("RGB").save(REAL_SRC / name, quality=95)
            fake_img.convert("RGB").save(DEEPFAKE_SRC / name, quality=95)
            written += 1
            pbar.update(1)
            if written >= n_pairs:
                break

    print(
        f"COCO_AI: scanned {seen} rows, wrote {written} person-caption real/deepfake "
        f"pairs to {REAL_SRC} and {DEEPFAKE_SRC}"
    )
    if written == 0:
        raise RuntimeError(
            "No COCO_AI pairs written -- the dataset schema (coco_image/dalle_image/caption "
            "columns) may have changed upstream; check https://huggingface.co/datasets/NasrinImp/COCO_AI"
        )
    if written < n_pairs:
        print(
            f"WARNING: only {written}/{n_pairs} requested pairs found (dataset exhausted). "
            "Proceeding with what was collected -- check the deepfake survival floor in face_filter.py."
        )


def _download_kaggle(dataset_slug: str, dest: Path, force: bool, label: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()) and not force:
        print(f"{dest} already has files, skipping Kaggle download (use --force to redo).")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            f"The 'kaggle' package is required for {label} download. Install it "
            "(`uv pip install kaggle` in this venv), set KAGGLE_API_TOKEN (or a "
            "kaggle.json via KAGGLE_CONFIG_DIR), then re-run."
        ) from e

    print(f"Downloading {dataset_slug} via Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True, quiet=False)
    print(f"{label} downloaded and unzipped into {dest}")


def download_casia(force: bool = False) -> None:
    _download_kaggle(KAGGLE_DATASET, EDITED_SRC, force, "CASIA v2.0")


def download_ps_battles(force: bool = False) -> None:
    """PS-Battles derivative (Photoshopped) images -- second edited-class
    source alongside CASIA for manipulation-technique diversity. The exact
    on-disk layout of this Kaggle mirror is unverified; face_filter.py's
    find_ps_battles_derivatives() inspects it after download and raises
    loudly instead of guessing if it can't tell originals from derivatives.
    Run `python data/download.py --test` first to check the same heuristic
    against Kaggle's file listing before paying for the full download."""
    _download_kaggle(PS_BATTLES_DATASET, EDITED_SRC_PSBATTLES, force, "PS-Battles")


def download_all(
    n_pairs: int = 3000,
    skip_coco_ai: bool = False,
    skip_casia: bool = False,
    skip_ps_battles: bool = False,
    force_casia: bool = False,
    force_ps_battles: bool = False,
) -> None:
    """Runs the requested downloads concurrently -- independent I/O-bound
    network pulls (HF streaming, 2x Kaggle), so total wall-clock is roughly
    the slowest of the three instead of the sum. tqdm bars from each source
    interleave in the log (cosmetic only); each writes to its own directory."""
    tasks = {}
    if not skip_coco_ai:
        tasks["COCO_AI"] = lambda: download_coco_ai(n_pairs)
    if not skip_casia:
        tasks["CASIA"] = lambda: download_casia(force=force_casia)
    if not skip_ps_battles:
        tasks["PS-Battles"] = lambda: download_ps_battles(force=force_ps_battles)
    if not tasks:
        print("download_all: nothing to do -- all sources skipped.")
        return

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            fut.result()  # re-raise here so a failed download surfaces its actual traceback, not a silent thread death
            print(f"[{name}] done.")


def _test_coco_ai(n_sample: int = 200) -> None:
    from datasets import load_dataset

    print(f"[COCO_AI] streaming {n_sample} rows to check schema + person-caption hit rate (no files written)...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.select_columns(["caption", "coco_image", "dalle_image"])
    seen = hits = 0
    for row in ds:
        seen += 1
        if caption_has_person(row.get("caption") or "") and row.get("coco_image") is not None and row.get("dalle_image") is not None:
            hits += 1
        if seen >= n_sample:
            break
    pct = 100 * hits / max(seen, 1)
    print(f"[COCO_AI] schema OK -- {hits}/{seen} sampled rows are usable person-caption pairs (~{pct:.0f}% hit rate)")
    if hits == 0:
        raise RuntimeError(
            "[COCO_AI] 0 usable pairs in sample -- coco_image/dalle_image/caption schema may have "
            "changed upstream, or PERSON_KEYWORDS needs revisiting."
        )


def _test_kaggle_source(dataset_slug: str, label: str, classify, require_label: str | None = None) -> None:
    """Kaggle auth + file listing only, no download -- a broken token, wrong
    slug, or (for PS-Battles) an unmatched folder-layout heuristic fails
    here in seconds instead of after paying for the full transfer."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    names = [f.name for f in api.dataset_list_files(dataset_slug).files]
    if not names:
        raise RuntimeError(f"[{label}] Kaggle listed 0 files for {dataset_slug} -- check the dataset slug.")
    counts = Counter(classify(Path(n)) for n in names)
    print(f"[{label}] Kaggle auth OK, {len(names)} files listed for {dataset_slug} (sample: {names[:5]})")
    print(f"[{label}] path-heuristic counts across the full listing: {dict(counts)}")
    if require_label is not None and counts.get(require_label, 0) == 0:
        raise RuntimeError(
            f"[{label}] 0 files matched required label '{require_label}' in the Kaggle listing -- "
            "the folder-name heuristic won't survive a full download as-is. Inspect the sample "
            "filenames printed above and fix the heuristic before downloading."
        )


def test_sources(n_coco_sample: int = 200) -> None:
    """Cheap, no-full-download validation of every source -- run this once
    (`python data/download.py --test`) before committing to the full
    download. Catches a broken Kaggle token, a changed HF schema, or
    PS-Battles' unverified folder layout in seconds instead of after paying
    for gigabytes of transfer."""
    from data.face_filter import classify_ps_battles_path

    tasks = {
        "COCO_AI": lambda: _test_coco_ai(n_coco_sample),
        "CASIA": lambda: _test_kaggle_source(
            KAGGLE_DATASET,
            "CASIA",
            lambda p: "tampered" if p.name.lower().startswith("tp_") else "other",
            require_label="tampered",
        ),
        "PS-Battles": lambda: _test_kaggle_source(
            PS_BATTLES_DATASET, "PS-Battles", classify_ps_battles_path, require_label="derivative"
        ),
    }
    failures = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"[{name}] OK\n")
            except Exception as e:
                failures[name] = e
                print(f"[{name}] FAILED: {e}\n")

    if failures:
        raise SystemExit(
            f"test_sources: {len(failures)}/{len(tasks)} source(s) failed -- fix before running the "
            f"full download: {list(failures)}"
        )
    print("All sources OK -- safe to run the full download.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-pairs", type=int, default=3000, help="person-caption real/deepfake pairs to collect from COCO_AI"
    )
    parser.add_argument("--skip-coco-ai", action="store_true")
    parser.add_argument("--skip-casia", action="store_true")
    parser.add_argument("--skip-ps-battles", action="store_true")
    parser.add_argument("--force-casia", action="store_true", help="re-download CASIA even if edited_src is non-empty")
    parser.add_argument(
        "--force-ps-battles", action="store_true", help="re-download PS-Battles even if its raw dir is non-empty"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="validate all sources (auth, schema, PS-Battles heuristic) without downloading, then exit",
    )
    args = parser.parse_args()

    if args.test:
        test_sources()
        return

    download_all(
        n_pairs=args.n_pairs,
        skip_coco_ai=args.skip_coco_ai,
        skip_casia=args.skip_casia,
        skip_ps_battles=args.skip_ps_battles,
        force_casia=args.force_casia,
        force_ps_battles=args.force_ps_battles,
    )


if __name__ == "__main__":
    main()
