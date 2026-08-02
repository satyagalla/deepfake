"""Row-level split builder for the COCO_AI corpus + self-generated N-shot pool.

Enforces three of the build plan's invariants (`docs/notes/2026-08-02-build-plan.md`
§1) as code, not eyeballing:

- **I2** -- split at the COCO_AI *row* level. One row is 1 real + 6 fakes;
  splitting within a row would leak the pairing.
- **I3** -- no image's patches straddle train/val. Automatic here since
  patches are extracted per image and every image inherits its row's split.
- **I7** -- Midjourney never enters training at any N. It is excluded from
  the train/val partition entirely and returned as its own held-out set.

Also partitions the self-generated pool (gemini/gptimage) into an
N-shot **adaptation pool** (drawn from for E1's curve) and a fixed
**eval** slice that is never used for fitting, so E1's accuracy numbers
aren't measured on images the adapted head has seen.

Usage:
    python -m probe.split
"""
import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COCO_AI_COLUMNS,
    COCO_AI_RAW_DIR,
    COCO_AI_TRAIN_GENERATORS,
    HELDOUT_GENERATOR,
    SEED,
    SELFGEN_MANIFEST_PATH,
    SELFGEN_RAW_DIR,
    SPLITS_DIR,
    VAL_FRACTION,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class RowSplit:
    train_row_ids: list[str]
    val_row_ids: list[str]
    heldout_row_ids: list[str]  # midjourney -- same rows as train+val, never trained on
    heldout_generator: str = HELDOUT_GENERATOR


@dataclass
class SelfgenSplit:
    # generator -> domain -> {"pool": [...], "eval": [...]} (relative paths under SELFGEN_RAW_DIR)
    generators: dict = field(default_factory=dict)


def list_coco_ai_row_ids() -> list[str]:
    real_dir = COCO_AI_RAW_DIR / "real"
    if not real_dir.exists():
        raise SystemExit(f"{real_dir} does not exist -- run data/download.py first.")
    return sorted(p.stem for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def assert_pairing_intact(row_ids: list[str]) -> None:
    """I2/I3 as an assertion, not an eyeball check: every row_id from `real`
    must have a matching file in every one of the 7 COCO_AI columns."""
    missing: dict[str, list[str]] = {}
    for key in COCO_AI_COLUMNS:
        col_dir = COCO_AI_RAW_DIR / key
        have = {p.stem for p in col_dir.iterdir()} if col_dir.exists() else set()
        gap = [rid for rid in row_ids if rid not in have]
        if gap:
            missing[key] = gap
    if missing:
        raise RuntimeError(
            "I2/I3 violation: the following COCO_AI columns are missing files for row_ids "
            f"present in 'real' (pairing is broken): { {k: len(v) for k, v in missing.items()} }. "
            "This should not happen if download.py only wrote complete rows -- check for "
            "partial/interrupted downloads."
        )


def build_row_split(val_fraction: float = VAL_FRACTION, seed: int = SEED) -> RowSplit:
    row_ids = list_coco_ai_row_ids()
    assert_pairing_intact(row_ids)

    rng = random.Random(seed)
    shuffled = row_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return RowSplit(train_row_ids=train_ids, val_row_ids=val_ids, heldout_row_ids=row_ids)


def expand_row_split_to_images(split: RowSplit) -> dict[str, list[tuple[str, str, str]]]:
    """Returns {'train': [...], 'val': [...], 'heldout': [...]} of
    (image_path, row_id, generator) tuples, generator in
    {'real'} | COCO_AI_TRAIN_GENERATORS for train/val, {'midjourney'} for
    heldout. Labels are derived downstream from `generator` (real vs not)."""
    out: dict[str, list[tuple[str, str, str]]] = {"train": [], "val": [], "heldout": []}
    for split_name, row_ids, generators in (
        ("train", split.train_row_ids, ["real"] + COCO_AI_TRAIN_GENERATORS),
        ("val", split.val_row_ids, ["real"] + COCO_AI_TRAIN_GENERATORS),
        ("heldout", split.heldout_row_ids, [HELDOUT_GENERATOR]),
    ):
        for rid in row_ids:
            for gen in generators:
                col_dir = COCO_AI_RAW_DIR / gen
                candidates = [p for p in col_dir.glob(f"{rid}.*") if p.suffix.lower() in IMG_EXTS]
                if not candidates:
                    continue
                out[split_name].append((candidates[0].relative_to(COCO_AI_RAW_DIR).as_posix(), rid, gen))
    return out


def build_selfgen_split(eval_fraction: float = 0.5, seed: int = SEED) -> SelfgenSplit:
    """Image-level split (no row pairing for self-generated images -- they
    aren't COCO_AI rows). Per generator x domain: a fixed eval slice held
    out from every N-shot draw, and a pool E1 samples N images from."""
    if not SELFGEN_MANIFEST_PATH.exists():
        print(f"NOTE: {SELFGEN_MANIFEST_PATH} not found yet -- run data/selfgen_organize.py first. "
              "Returning an empty selfgen split.")
        return SelfgenSplit(generators={})

    import csv

    rows_by_key: dict[tuple[str, str], list[str]] = {}
    with open(SELFGEN_MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["generator"], row["domain"])
            rows_by_key.setdefault(key, []).append(row["path"])

    rng = random.Random(seed)
    generators: dict = {}
    for (generator, domain), paths in rows_by_key.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_fraction)) if len(shuffled) > 1 else 0
        eval_paths = sorted(shuffled[:n_eval])
        pool_paths = sorted(shuffled[n_eval:])
        generators.setdefault(generator, {})[domain] = {"pool": pool_paths, "eval": eval_paths}
    return SelfgenSplit(generators=generators)


def save_splits(row_split: RowSplit, selfgen_split: SelfgenSplit, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco_ai_row_split.json").write_text(json.dumps(asdict(row_split), indent=2))
    (out_dir / "selfgen_split.json").write_text(json.dumps(asdict(selfgen_split), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--selfgen-eval-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    row_split = build_row_split(val_fraction=args.val_fraction, seed=args.seed)
    images = expand_row_split_to_images(row_split)
    selfgen_split = build_selfgen_split(eval_fraction=args.selfgen_eval_fraction, seed=args.seed)
    save_splits(row_split, selfgen_split)

    print(f"COCO_AI rows: {len(row_split.train_row_ids)} train, {len(row_split.val_row_ids)} val, "
          f"{len(row_split.heldout_row_ids)} heldout ({HELDOUT_GENERATOR}, 0-shot only)")
    print(f"COCO_AI images: {len(images['train'])} train, {len(images['val'])} val, "
          f"{len(images['heldout'])} heldout")
    for generator, domains in selfgen_split.generators.items():
        for domain, parts in domains.items():
            print(f"selfgen[{generator}][{domain}]: pool={len(parts['pool'])} eval={len(parts['eval'])}")
    print(f"\nSplits written to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
