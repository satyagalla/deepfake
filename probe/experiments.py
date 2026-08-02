"""Pre-registered experiments E1, E2, E5, E6, E7 (`0004` §9, build plan S5-S6).

E1 is the headline and the only one that needs anything beyond the S3
feature cache (it draws N-shot training subsets and refits Head A per
draw). E2/E5/E6/E7 are free -- everything they need is already on disk.

E3 (degradation ladder) and E4 (off-domain) live in separate modules
(`probe/degradation.py`, folded into this file's `run_e4`) since E3 needs
its own re-encoded image variants and E4 only needs a domain filter this
module already has via `rows_for`.

Usage:
    python -m probe.experiments --all
    python -m probe.experiments --e1-only --target gemini
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURES_DIR,
    FEATURES_STANDARD_DIR,
    HELDOUT_GENERATOR,
    NSHOT_TRIALS,
    NSHOT_VALUES,
    PROBE_OUTPUTS_DIR,
    SEED,
    SELFGEN_GENERATORS,
)
from probe.features import POOL_METHODS, FeatureRow, build_matrix, read_index, rows_for
from probe.heads import fit_head_a, score_head_a
from probe.thresholds import threshold_report


def draw_nshot(pool_rows: list[FeatureRow], n: int, seed: int) -> list[FeatureRow]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    if n >= len(pool_rows):
        return list(pool_rows)
    return rng.sample(pool_rows, n)


def evaluate_head_a(
    head, eval_rows: list[FeatureRow], pool_method: str, standard_arm: bool = False, seed: int = SEED
) -> dict:
    """The `0005` §4 metric set. Metrics run on **raw logits** (cut point
    0.0, not 0.5) -- there is no calibration stage any more (I8).

    AUC/TPR/FPR use the full eval set: each is prevalence-invariant, so
    none of them is damaged by the class imbalance that makes raw accuracy
    uninterpretable here (I9, `0005` §5). Only `alpha` is computed on a
    balanced subsample, inside `threshold_report`."""
    X_eval, y_eval, _ = build_matrix(eval_rows, pool_method=pool_method, standard_arm=standard_arm)
    if len(y_eval) == 0:
        return threshold_report(np.zeros(0), np.zeros(0), seed=seed)
    return threshold_report(score_head_a(head, X_eval), y_eval, seed=seed)


def _fit_base_head_a(all_rows: list[FeatureRow], pool_method: str, extra_train_rows: list[FeatureRow] | None = None):
    """Returns (head, val_rows). `val_rows` is a genuinely held-out split
    now that nothing is fitted on it (`0005` §3) -- which is what makes E5
    an honest measurement rather than a self-evaluation."""
    train_rows = rows_for(all_rows, split="train") + (extra_train_rows or [])
    val_rows = rows_for(all_rows, split="val")
    X_train, y_train, _ = build_matrix(train_rows, pool_method=pool_method)
    return fit_head_a(X_train, y_train), val_rows


def run_e1(
    all_rows: list[FeatureRow],
    target_generator: str,
    pool_method: str = "mean",
    Ns: list[int] = NSHOT_VALUES,
    n_trials: int = NSHOT_TRIALS,
    seed: int = SEED,
) -> list[dict]:
    """N-shot adaptation curve for `target_generator` (`0004` §9 E1)."""
    pool_rows = rows_for(all_rows, split="selfgen_pool", generator=target_generator, domain="indomain")
    eval_ai_rows = rows_for(all_rows, split="selfgen_eval", generator=target_generator, domain="indomain")
    eval_real_rows = rows_for(all_rows, split="val", generator="real")
    eval_rows = eval_ai_rows + eval_real_rows

    if not pool_rows or not eval_ai_rows:
        print(f"E1[{target_generator}]: no selfgen pool/eval rows -- run data/selfgen_organize.py, "
              "probe.split, and probe.extract first. Skipping.")
        return []

    if len(eval_ai_rows) * 4 < len(eval_real_rows):
        print(f"E1[{target_generator}]: eval set is {len(eval_ai_rows)} AI vs {len(eval_real_rows)} real. "
              "AUC/TPR/FPR are prevalence-invariant and unaffected, but raw accuracy is not -- read "
              "`auc` and `balanced_accuracy`, never `accuracy` (0005 §5).")

    base_train_rows = rows_for(all_rows, split="train")
    # Read the COCO_AI train cache from disk once -- every trial below reuses this
    # matrix and only builds the small N-shot increment, instead of re-reading all
    # ~15k base .npz files per trial (62 trials otherwise, worse on Drive-mounted
    # storage in Colab).
    X_base, y_base, _ = build_matrix(base_train_rows, pool_method=pool_method)

    results = []
    for n in Ns:
        trials = n_trials if n > 0 else 1  # 0-shot is deterministic
        max_n = min(n, len(pool_rows))
        if max_n < n:
            print(f"E1[{target_generator}] N={n}: pool only has {len(pool_rows)} images, using {max_n}.")
        for trial in range(trials):
            trial_seed = seed * 1000 + n * 10 + trial
            drawn = draw_nshot(pool_rows, n, seed=trial_seed)
            X_drawn, y_drawn, _ = build_matrix(drawn, pool_method=pool_method)
            X_train = np.concatenate([X_base, X_drawn], axis=0) if len(X_drawn) else X_base
            y_train = np.concatenate([y_base, y_drawn], axis=0) if len(y_drawn) else y_base
            head = fit_head_a(X_train, y_train)
            # trial_seed also varies alpha's balanced draw, so the spread
            # across trials covers subsample noise as well as draw noise
            metrics = evaluate_head_a(head, eval_rows, pool_method, seed=trial_seed)
            results.append({"generator": target_generator, "n": n, "trial": trial, **metrics})
    return results


def plot_e1(results: list[dict], target_generator: str, out_path: Path) -> None:
    """Two lines (`0005` §4): **AUC** -- the claim's quantity, threshold-free
    and prevalence-invariant, and the metric the knee is now read on -- and
    balanced accuracy, the user-visible number on a footing where 0.5 really
    is chance. TPR/FPR at the endpoints are annotated because their *shape*
    reads out the failure mode: TPR~0.05/FPR~0.01 is a misplaced threshold,
    TPR~0.5/FPR~0.5 is the Chameleon regime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(r["n"] for r in results))

    def _stat(metric, n, fn):
        vals = [r[metric] for r in results if r["n"] == n and not np.isnan(r[metric])]
        return float(fn(vals)) if vals else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for metric, colour, label in (
        ("auc", "C0", "AUC (separability)"),
        ("balanced_accuracy", "C1", "balanced accuracy @ 0"),
    ):
        for r in results:
            ax.scatter(r["n"], r[metric], alpha=0.25, color=colour, s=13, zorder=1)
        ax.errorbar(ns, [_stat(metric, n, np.mean) for n in ns], yerr=[_stat(metric, n, np.std) for n in ns],
                    marker="o", capsize=4, color=colour, zorder=2, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", lw=1, zorder=0, label="chance")
    for n in (ns[0], ns[-1]):
        tpr, fpr = _stat("tpr", n, np.mean), _stat("fpr", n, np.mean)
        ax.annotate(f"TPR {tpr:.2f}\nFPR {fpr:.2f}", xy=(n, _stat("auc", n, np.mean)),
                    xytext=(0, -34), textcoords="offset points", ha="center", fontsize=7, color="dimgray")

    ax.set_xlabel("N images from target generator")
    ax.set_ylabel("AUC / balanced accuracy")
    ax.set_title(f"E1: N-shot adaptation curve -- {target_generator}")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_e2(all_rows: list[FeatureRow], pool_method: str = "mean") -> dict:
    """Held-out Midjourney, 0-shot (`0004` §9 E2) -- I7: never trained on."""
    heldout_rows = rows_for(all_rows, split="heldout", generator=HELDOUT_GENERATOR)
    real_eval_rows = rows_for(all_rows, split="val", generator="real")
    head, _ = _fit_base_head_a(all_rows, pool_method)
    metrics = evaluate_head_a(head, heldout_rows + real_eval_rows, pool_method)
    metrics["generator"] = HELDOUT_GENERATOR
    return metrics


def run_e5(all_rows: list[FeatureRow], pool_method: str = "mean") -> dict:
    """AUC on the val split (`0004` §9 E5) -- first measurement on this
    project, and now an honest one: pre-`0005` this evaluated on the split
    the Platt calibrator had been fitted on. Read against
    `plan-c-source-verification.md` §5 -- AUC >~0.75 with accuracy ~50% is
    a threshold problem; AUC ~0.5 is the Chameleon regime."""
    head, val_rows = _fit_base_head_a(all_rows, pool_method)
    return evaluate_head_a(head, val_rows, pool_method)


def run_e6(all_rows: list[FeatureRow]) -> dict:
    """Aggregator ablation (§6.5) -- same cached patch array, different pooling."""
    heldout_rows = rows_for(all_rows, split="heldout", generator=HELDOUT_GENERATOR)
    real_eval_rows = rows_for(all_rows, split="val", generator="real")
    results = {}
    for method in POOL_METHODS:
        head, val_rows = _fit_base_head_a(all_rows, method)
        results[method] = {
            "val": evaluate_head_a(head, val_rows, method),
            "midjourney_0shot": evaluate_head_a(head, heldout_rows + real_eval_rows, method),
        }
    return results


def run_e7(pool_method: str = "mean") -> dict:
    """Native patches vs standard resize+centre-crop (§9 E7) -- does §6 pay?"""
    native_rows = read_index(FEATURES_DIR)
    if not FEATURES_STANDARD_DIR.exists() or not (FEATURES_STANDARD_DIR / "index.csv").exists():
        print("E7: standard arm not extracted -- run `python -m probe.extract --arm standard` first.")
        return {}
    standard_rows = read_index(FEATURES_STANDARD_DIR)

    def _eval_arm(rows, standard_arm):
        train_rows = rows_for(rows, split="train")
        val_rows = rows_for(rows, split="val")
        heldout_rows = rows_for(rows, split="heldout", generator=HELDOUT_GENERATOR)
        real_eval_rows = rows_for(rows, split="val", generator="real")
        X_train, y_train, _ = build_matrix(train_rows, pool_method=pool_method, standard_arm=standard_arm)
        head = fit_head_a(X_train, y_train)
        return {
            "val": evaluate_head_a(head, val_rows, pool_method, standard_arm),
            "midjourney_0shot": evaluate_head_a(head, heldout_rows + real_eval_rows, pool_method, standard_arm),
        }

    return {"native_patches": _eval_arm(native_rows, False), "standard_resize": _eval_arm(standard_rows, True)}


def run_e4(all_rows: list[FeatureRow], pool_method: str = "mean") -> dict:
    """Off-domain vs in-domain accuracy at equal N (`0004` §9 E4) -- is the
    model reading synthesis or COCO's content distribution?"""
    results = {}
    for generator in SELFGEN_GENERATORS:
        indomain_eval = rows_for(all_rows, split="selfgen_eval", generator=generator, domain="indomain")
        offdomain_eval = rows_for(all_rows, split="selfgen_eval", generator=generator, domain="offdomain")
        real_eval_rows = rows_for(all_rows, split="val", generator="real")
        if not indomain_eval and not offdomain_eval:
            continue
        head, _ = _fit_base_head_a(all_rows, pool_method)
        results[generator] = {
            "indomain": evaluate_head_a(head, indomain_eval + real_eval_rows, pool_method),
            "offdomain": evaluate_head_a(head, offdomain_eval + real_eval_rows, pool_method),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-method", default="mean", choices=POOL_METHODS)
    parser.add_argument("--target", default=None, help="restrict E1/E4 to one selfgen generator")
    parser.add_argument("--e1-only", action="store_true")
    args = parser.parse_args()

    all_rows = read_index(FEATURES_DIR)
    if not all_rows:
        raise SystemExit("No extracted features found -- run `python -m probe.extract` first.")

    PROBE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    targets = [args.target] if args.target else SELFGEN_GENERATORS
    for target in targets:
        e1 = run_e1(all_rows, target, pool_method=args.pool_method)
        if e1:
            results[f"e1_{target}"] = e1
            plot_e1(e1, target, PROBE_OUTPUTS_DIR / f"e1_{target}.png")

            def _mean(metric, n):
                return float(np.mean([r[metric] for r in e1 if r["n"] == n]))

            # Knee is read on AUC, not accuracy (0005 §5): accuracy on this
            # eval set is prevalence-dependent, and "always real" would score
            # >=90% -- which would falsify the claim on set composition alone.
            knee_n = next((n for n in NSHOT_VALUES if _mean("auc", n) >= 0.90), None)
            print(f"E1[{target}] AUC:  {[(n, round(_mean('auc', n), 3)) for n in NSHOT_VALUES]}")
            print(f"E1[{target}] bacc: {[(n, round(_mean('balanced_accuracy', n), 3)) for n in NSHOT_VALUES]}")
            print(f"E1[{target}] 0-shot TPR={_mean('tpr', 0):.3f} FPR={_mean('fpr', 0):.3f} "
                  f"(n_ai={int(_mean('n_ai', 0))}, n_real={int(_mean('n_real', 0))})"
                  f"  -> first N reaching AUC 0.90: {knee_n}")

    if args.e1_only:
        (PROBE_OUTPUTS_DIR / "experiments.json").write_text(json.dumps(results, indent=2))
        return

    results["e2_midjourney_0shot"] = run_e2(all_rows, args.pool_method)
    print("E2 (Midjourney 0-shot):", results["e2_midjourney_0shot"])

    results["e5_auc"] = run_e5(all_rows, args.pool_method)
    print("E5 (val AUC):", results["e5_auc"])

    results["e6_aggregator_ablation"] = run_e6(all_rows)
    print("E6 (aggregator ablation):", json.dumps(results["e6_aggregator_ablation"], indent=2))

    e7 = run_e7(args.pool_method)
    if e7:
        results["e7_preprocessing_arms"] = e7
        print("E7 (native vs standard):", json.dumps(e7, indent=2))

    results["e4_offdomain"] = run_e4(all_rows, args.pool_method)
    print("E4 (off-domain vs in-domain):", json.dumps(results["e4_offdomain"], indent=2))

    (PROBE_OUTPUTS_DIR / "experiments.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll results written to {PROBE_OUTPUTS_DIR / 'experiments.json'}")


if __name__ == "__main__":
    main()
