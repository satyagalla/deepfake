"""Resolution-swap counterfactual probe: push an image's *rgb* pixels toward
another class's measured typical MTCNN upscale factor via an extra bilinear
downsample/upsample round-trip, then check whether the trained model's
prediction follows. See docs/reference/resolution_swap_probe.md for the full
design and docs/investigations/2026-07-26-upscale-artifact.md for the
hypothesis this tests.

Unlike model/counterfactual_probe.py (which swaps fft_mag/srm_residual only,
never touching rgb), this probe manipulates rgb directly so the spatial branch
is actually exercised. Each achievable target class is paired with a same-size
(r=1) round-trip control, so a real shift can be told apart from
resampling-roundtrip noise alone.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, CLASSES, DEVICE, EVAL_DIR, IMAGE_SIZE
from model.branches import SRMFilter
from model.counterfactual_probe import _predict
from model.dataset import IMAGENET_MEAN, IMAGENET_STD, ForgeryDataset
from model.eval import load_model

# Measured median MTCNN upscale factors per class (see
# docs/investigations/2026-07-26-upscale-artifact.md). Small, noisy samples
# (n=5/8/19) -- an approximation, not a precise per-image factor.
MEDIAN_UPSCALE_FACTOR = {"real": 10.40, "deepfake": 11.05, "edited": 11.83}

# Only source->target pairs with source_median < target_median are achievable:
# the round-trip can only add blur, never remove it.
ACHIEVABLE_TARGETS = {"real": ["deepfake", "edited"], "deepfake": ["edited"]}

_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
_srm = SRMFilter()


def _class_indices(dataset: ForgeryDataset) -> dict:
    indices = {cls: [] for cls in CLASSES}
    for i in range(len(dataset)):
        indices[dataset.df.iloc[i]["class"]].append(i)
    return indices


def _load_raw01(path: str) -> torch.Tensor:
    """Reload the already-processed 380x380 image straight from disk as an
    unnormalized (3,H,W) tensor in [0,1] -- mirrors ForgeryDataset.__getitem__'s
    loading step, needed here since the dataset only exposes normalized rgb."""
    img = Image.open(path).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return _to_tensor(img)


def resize_roundtrip(raw01: torch.Tensor, ratio: float) -> torch.Tensor:
    """Downsample raw01 (3,H,W) in [0,1] to IMAGE_SIZE/ratio then back up to
    IMAGE_SIZE, both passes plain PIL.BILINEAR with no anti-aliasing -- matches
    MTCNN's own crop_resize exactly. ratio=1.0 is the no-op-size control: still
    a genuine resize round-trip, just without the deliberate size change."""
    intermediate = round(IMAGE_SIZE / ratio)
    img = transforms.functional.to_pil_image(raw01)
    down = img.resize((intermediate, intermediate), Image.BILINEAR)
    up = down.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return _to_tensor(up)


def _channels_from_raw01(raw01: torch.Tensor) -> dict:
    """Recompute all three model inputs from raw01 -- never inject a channel
    computed a different way than training/eval does."""
    return {
        "rgb": _normalize(raw01.clone()),
        "fft_mag": ForgeryDataset._fft_magnitude(raw01),
        "srm_residual": _srm(raw01 * 255.0),
    }


def run_probe(
    model,
    dataset: ForgeryDataset,
    indices_by_class: dict,
    probe_n_per_source_class: int,
    device: str = DEVICE,
) -> list[dict]:
    """For probe_n_per_source_class held-out samples per achievable source
    class, record the baseline prediction plus, for each achievable target
    class, both the r=1 control and the real factor-swap prediction."""
    results = []
    for source_cls, target_classes in ACHIEVABLE_TARGETS.items():
        idxs = indices_by_class[source_cls][:probe_n_per_source_class]
        for i in idxs:
            sample = dataset[i]
            raw01 = _load_raw01(sample["path"])
            record = {
                "path": sample["path"],
                "true_class": source_cls,
                "baseline": _predict(model, sample["rgb"], sample["fft_mag"], sample["srm_residual"], device),
                "resolution_swaps": {},
            }
            for target_cls in target_classes:
                ratio = MEDIAN_UPSCALE_FACTOR[target_cls] / MEDIAN_UPSCALE_FACTOR[source_cls]
                control_ch = _channels_from_raw01(resize_roundtrip(raw01, 1.0))
                swap_ch = _channels_from_raw01(resize_roundtrip(raw01, ratio))
                record["resolution_swaps"][target_cls] = {
                    "ratio": ratio,
                    "control": _predict(model, control_ch["rgb"], control_ch["fft_mag"], control_ch["srm_residual"], device),
                    "swap": _predict(model, swap_ch["rgb"], swap_ch["fft_mag"], swap_ch["srm_residual"], device),
                }
            results.append(record)
            _print_record(record)
    return results


def _print_record(record: dict) -> None:
    base = record["baseline"]
    print(f"\n[{record['true_class']}] {record['path']}")
    print(f"  baseline      : pred={base['pred']:>9s}  probs=" + _fmt_probs(base["probs"]))
    for target_cls, swap in record["resolution_swaps"].items():
        print(f"  -> {target_cls} (r={swap['ratio']:.3f})")
        print(f"       control (r=1): pred={swap['control']['pred']:>9s}  probs=" + _fmt_probs(swap["control"]["probs"]))
        print(f"       swap         : pred={swap['swap']['pred']:>9s}  probs=" + _fmt_probs(swap["swap"]["probs"]))


def _fmt_probs(probs: dict) -> str:
    return ", ".join(f"{c}={p:.3f}" for c, p in probs.items())


def summarize(results: list[dict]) -> dict:
    """Per achievable target class: mean delta in P(target_class), real swap
    vs its matched r=1 control (the correct reference point -- isolates the
    deliberate factor increase from resize-roundtrip-only noise). Also the
    fraction of images whose pred flips to target_class under the real swap
    but not under the control."""
    all_targets = sorted({t for targets in ACHIEVABLE_TARGETS.values() for t in targets})
    buckets = {t: {"deltas": [], "flips": 0, "n": 0} for t in all_targets}
    for r in results:
        for target_cls, swap in r["resolution_swaps"].items():
            control_p = swap["control"]["probs"][target_cls]
            swap_p = swap["swap"]["probs"][target_cls]
            bucket = buckets[target_cls]
            bucket["deltas"].append(swap_p - control_p)
            bucket["n"] += 1
            if swap["swap"]["pred"] == target_cls and swap["control"]["pred"] != target_cls:
                bucket["flips"] += 1
    return {
        t: {
            "mean_delta_p_target": (sum(b["deltas"]) / len(b["deltas"])) if b["deltas"] else float("nan"),
            "flip_fraction": (b["flips"] / b["n"]) if b["n"] else float("nan"),
            "n": b["n"],
        }
        for t, b in buckets.items()
    }


def print_summary(summary: dict) -> None:
    print("\n=== Summary: mean delta in P(target_class), real swap vs matched r=1 control ===")
    print(f"{'target':>10s} {'mean_delta_p_target':>20s} {'flip_fraction':>14s} {'n':>5s}")
    for target_cls, stats in summary.items():
        print(f"{target_cls:>10s} {stats['mean_delta_p_target']:>20.4f} {stats['flip_fraction']:>14.3f} {stats['n']:>5d}")


def save_probe_results(results: list[dict], summary: dict, out_dir: Path = EVAL_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"resolution_swap_probe_{stamp}.json"
    out_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2))
    print(f"\nSaved probe results -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--probe-samples", type=int, default=20, help="held-out samples per achievable source class")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataset = ForgeryDataset(args.split)
    indices_by_class = _class_indices(dataset)

    results = run_probe(model, dataset, indices_by_class, args.probe_samples)
    summary = summarize(results)
    print_summary(summary)
    save_probe_results(results, summary)


if __name__ == "__main__":
    main()
