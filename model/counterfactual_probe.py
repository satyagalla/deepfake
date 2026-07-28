"""Counterfactual shortcut probe: for a given image, keep its true `rgb`
content and swap its `fft_mag`/`srm_residual` for another class's averaged
template, then check whether the trained model's prediction follows content
or follows the swapped artifact channels. See
docs/investigations/2026-07-26-upscale-artifact.md for the hypothesis this
tests (tier 2 of the recommended next steps).

Each probed sample reports its baseline (own fft/srm, no swap) alongside
every swap result, so the shift can be read directly per image.

`summarize()` alone can't tell a targeted per-class shortcut apart from a
generic "any averaged template moves things" artifact -- `swap_target_spread()`
disambiguates by comparing the 3 swap-target predictions against each other.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, CLASSES, DEVICE, EVAL_DIR
from model.dataset import ForgeryDataset
from model.eval import BRANCH_NAMES, load_model

SWAP_VARIANTS = ["fft", "srm", "both"]


def _class_indices(dataset: ForgeryDataset) -> dict:
    indices = {cls: [] for cls in CLASSES}
    for i in range(len(dataset)):
        indices[dataset.df.iloc[i]["class"]].append(i)
    return indices


@torch.no_grad()
def extract_templates(dataset: ForgeryDataset, indices_by_class: dict, n_per_class: int) -> dict:
    """Average fft_mag/srm_residual over the first n_per_class samples of
    each class -- the per-class artifact 'fingerprint', with per-image
    content averaged out."""
    templates = {}
    for cls, idxs in indices_by_class.items():
        take = idxs[:n_per_class]
        if not take:
            raise ValueError(f"No samples for class {cls!r} to build a template from")
        fft_sum, srm_sum = None, None
        for i in take:
            sample = dataset[i]
            fft_sum = sample["fft_mag"].clone() if fft_sum is None else fft_sum + sample["fft_mag"]
            srm_sum = sample["srm_residual"].clone() if srm_sum is None else srm_sum + sample["srm_residual"]
        templates[cls] = {"fft_mag": fft_sum / len(take), "srm_residual": srm_sum / len(take)}
    return templates


@torch.no_grad()
def _predict(model, rgb: torch.Tensor, fft_mag: torch.Tensor, srm_residual: torch.Tensor, device: str) -> dict:
    logits, gate = model(rgb.unsqueeze(0).to(device), fft_mag.unsqueeze(0).to(device), srm_residual.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu()
    return {
        "pred": CLASSES[probs.argmax().item()],
        "probs": dict(zip(CLASSES, probs.tolist())),
        "gate": dict(zip(BRANCH_NAMES, gate[0].cpu().tolist())),
    }


def run_probe(
    model,
    dataset: ForgeryDataset,
    indices_by_class: dict,
    templates: dict,
    probe_n_per_class: int,
    template_n_per_class: int,
    device: str = DEVICE,
) -> list[dict]:
    """For probe_n_per_class held-out samples per class (disjoint from the
    samples used to build templates), record the baseline prediction plus
    every (swap_class x variant) prediction."""
    results = []
    for true_cls, idxs in indices_by_class.items():
        probe_idxs = idxs[template_n_per_class : template_n_per_class + probe_n_per_class]
        for i in probe_idxs:
            sample = dataset[i]
            rgb = sample["rgb"]
            record = {
                "path": sample["path"],
                "true_class": true_cls,
                "baseline": _predict(model, rgb, sample["fft_mag"], sample["srm_residual"], device),
                "swaps": {},
            }
            for swap_cls in CLASSES:
                tmpl = templates[swap_cls]
                variants = {
                    "fft": (tmpl["fft_mag"], sample["srm_residual"]),
                    "srm": (sample["fft_mag"], tmpl["srm_residual"]),
                    "both": (tmpl["fft_mag"], tmpl["srm_residual"]),
                }
                record["swaps"][swap_cls] = {
                    variant: _predict(model, rgb, fft_in, srm_in, device) for variant, (fft_in, srm_in) in variants.items()
                }
            results.append(record)
            _print_record(record)
    return results


def _print_record(record: dict) -> None:
    true_cls = record["true_class"]
    base = record["baseline"]
    print(f"\n[{true_cls}] {record['path']}")
    print(f"  baseline (own fft/srm) : pred={base['pred']:>9s}  probs=" + _fmt_probs(base["probs"]))
    for swap_cls in CLASSES:
        tag = "same-class (control)" if swap_cls == true_cls else "cross-class"
        for variant, pred in record["swaps"][swap_cls].items():
            print(f"  swap->{swap_cls:<9s}{variant:<5s}[{tag:<20s}]: pred={pred['pred']:>9s}  probs=" + _fmt_probs(pred["probs"]))


def _fmt_probs(probs: dict) -> str:
    return ", ".join(f"{c}={p:.3f}" for c, p in probs.items())


def summarize(results: list[dict]) -> dict:
    """Mean change in P(swap_class), relative to that sample's baseline
    P(swap_class), split same-class (control) vs cross-class, per variant.
    Cross-class delta minus same-class delta is the clean shortcut signal --
    same-class delta alone is just noise from using an averaged template."""
    buckets = {v: {"same_class": [], "cross_class": []} for v in SWAP_VARIANTS}
    for r in results:
        for swap_cls, variants in r["swaps"].items():
            relation = "same_class" if swap_cls == r["true_class"] else "cross_class"
            baseline_swap_prob = r["baseline"]["probs"][swap_cls]
            for variant, pred in variants.items():
                buckets[variant][relation].append(pred["probs"][swap_cls] - baseline_swap_prob)
    return {
        variant: {relation: (sum(vals) / len(vals) if vals else float("nan")) for relation, vals in rel.items()}
        for variant, rel in buckets.items()
    }


def print_summary(summary: dict) -> None:
    print("\n=== Summary: mean delta in P(swap_class) vs baseline ===")
    print(f"{'variant':>6s} {'same_class (noise floor)':>26s} {'cross_class (shortcut signal)':>30s} {'cross - same':>14s}")
    for variant, rel in summary.items():
        same, cross = rel["same_class"], rel["cross_class"]
        print(f"{variant:>6s} {same:>26.4f} {cross:>30.4f} {cross - same:>14.4f}")


def swap_target_spread(results: list[dict]) -> dict:
    """cross - same in summarize() only shows that swapping moved the
    prediction; it can't tell apart "the model reacts to *which* class's
    template was used" (a targeted shortcut) from "any averaged/smoothed
    template moves things the same way regardless of source class" (a
    generic artifact of averaging, not class-specific). This checks that by
    comparing, per image, the 3 swap-target predictions against *each
    other* instead of against baseline -- a small spread here means swap
    identity barely matters, which points at the generic-averaging
    explanation."""
    spreads = {v: [] for v in SWAP_VARIANTS}
    for r in results:
        for variant in SWAP_VARIANTS:
            prob_vectors = [r["swaps"][swap_cls][variant]["probs"] for swap_cls in CLASSES]
            for cls in CLASSES:
                vals = [pv[cls] for pv in prob_vectors]
                spreads[variant].append(max(vals) - min(vals))
    return {variant: {"mean": sum(vals) / len(vals), "max": max(vals)} for variant, vals in spreads.items()}


def print_swap_target_spread(spread: dict) -> None:
    print("\n=== Swap-target spread: does *which* class's template was used matter? ===")
    print(f"{'variant':>6s} {'mean spread':>12s} {'max spread':>12s}")
    for variant, stats in spread.items():
        print(f"{variant:>6s} {stats['mean']:>12.4f} {stats['max']:>12.4f}")


def save_probe_results(results: list[dict], summary: dict, spread: dict, out_dir: Path = EVAL_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"counterfactual_probe_{stamp}.json"
    out_path.write_text(json.dumps({"results": results, "summary": summary, "swap_target_spread": spread}, indent=2))
    print(f"\nSaved probe results -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--template-samples", type=int, default=20, help="samples per class used to build each artifact template")
    parser.add_argument("--probe-samples", type=int, default=10, help="held-out samples per class to run the probe on")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataset = ForgeryDataset(args.split)
    indices_by_class = _class_indices(dataset)

    templates = extract_templates(dataset, indices_by_class, args.template_samples)
    results = run_probe(model, dataset, indices_by_class, templates, args.probe_samples, args.template_samples)
    summary = summarize(results)
    print_summary(summary)
    spread = swap_target_spread(results)
    print_swap_target_spread(spread)
    save_probe_results(results, summary, spread)


if __name__ == "__main__":
    main()
