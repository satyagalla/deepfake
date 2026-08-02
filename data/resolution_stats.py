"""Native-resolution histogram per COCO_AI column (build-plan S1 step 4).

`bottlenecks.md` §9.1 flags the corpus's native resolution as
`[second-hand]` -- every downscale ratio in `0004` §6.1 inherits that
status, and the ratio (not just the absolute resolution) is what
determines whether N=16 patches (`0004` §6.3) is the right count: if the
corpus turns out nearer 270px than 480px, a 224 patch is most of the
image and N should drop. This is the first `[measured]` pass over that
question -- run after `data/download.py` has populated `coco_ai/`.

Usage:
    python data/resolution_stats.py
    python data/resolution_stats.py --sample 500  # cap per column, for a quick look
"""
import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COCO_AI_COLUMNS, COCO_AI_RAW_DIR, PATCH_SIZE

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def column_stats(col_dir: Path, sample: int | None) -> dict:
    paths = sorted(p for p in col_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if sample is not None:
        paths = paths[:sample]
    short_sides, non_overlap_patch_counts = [], []
    unreadable = 0
    for p in tqdm(paths, desc=col_dir.name, unit="img", leave=False):
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            unreadable += 1
            continue
        short = min(w, h)
        short_sides.append(short)
        # non-overlapping PATCH_SIZE tiles that fit -- a proxy for "how much
        # headroom N=16 native patches actually has" at this resolution.
        non_overlap_patch_counts.append((w // PATCH_SIZE) * (h // PATCH_SIZE))
    short_sides.sort()
    return {
        "n": len(short_sides),
        "unreadable": unreadable,
        "short_side_p10": _percentile(short_sides, 10),
        "short_side_p50": _percentile(short_sides, 50),
        "short_side_p90": _percentile(short_sides, 90),
        "min": min(short_sides, default=float("nan")),
        "max": max(short_sides, default=float("nan")),
        "non_overlap_patches_p50": _percentile(sorted(non_overlap_patch_counts), 50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="cap images inspected per column")
    args = parser.parse_args()

    if not COCO_AI_RAW_DIR.exists():
        raise SystemExit(f"{COCO_AI_RAW_DIR} does not exist -- run data/download.py first.")

    rows = []
    for key in COCO_AI_COLUMNS:
        col_dir = COCO_AI_RAW_DIR / key
        if not col_dir.exists() or not any(col_dir.iterdir()):
            print(f"[{key}] skipped -- {col_dir} empty or missing")
            continue
        rows.append((key, column_stats(col_dir, args.sample)))

    if not rows:
        raise SystemExit(f"No populated columns found under {COCO_AI_RAW_DIR}.")

    header = f"{'column':<12}{'n':>7}{'p10':>7}{'p50':>7}{'p90':>7}{'min':>7}{'max':>7}{'nonoverlap@224(p50)':>22}"
    print(header)
    print("-" * len(header))
    for key, s in rows:
        print(
            f"{key:<12}{s['n']:>7}{s['short_side_p10']:>7.0f}{s['short_side_p50']:>7.0f}"
            f"{s['short_side_p90']:>7.0f}{s['min']:>7.0f}{s['max']:>7.0f}{s['non_overlap_patches_p50']:>22.1f}"
        )
        if s["unreadable"]:
            print(f"  ({s['unreadable']} unreadable files skipped)")

    real_p50 = dict(rows).get("real", {}).get("short_side_p50")
    if real_p50 is not None:
        ratio = real_p50 / PATCH_SIZE
        print(
            f"\nreal median short side / {PATCH_SIZE} = {ratio:.2f}x -- "
            + ("near 1x: patching is close to a no-op at the low end, N=16 may be more than the corpus supports (0004 §6.4)."
               if ratio < 1.5 else
               "comfortably above 1x: N=16 native patches has room to sample non-degenerately.")
        )


if __name__ == "__main__":
    main()
