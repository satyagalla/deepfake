"""Fits and saves the production Head A, Head B, and Mahalanobis gate that
`probe/demo.py` loads -- the "seconds to fit" full-corpus versions, as
opposed to the many refits `probe/experiments.py` does per N-shot draw or
aggregator choice. Also fits and saves the Spectral card's head.

Usage:
    python -m probe.fit_production
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_POOL_METHOD,
    FEATURES_DIR,
    GATE_NAME,
    HEAD_A_NAME,
    HEAD_B_NAME,
    SPECTRAL_HEAD_NAME,
)
from probe.features import build_matrix, read_index, rows_for
from probe.heads import fit_head_a, fit_mahalanobis_gate, fit_head_b, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-method", default=DEFAULT_POOL_METHOD)
    parser.add_argument("--skip-spectral", action="store_true", help="skip the CPU-only spectral head (slower: re-reads every raw image)")
    args = parser.parse_args()

    all_rows = read_index(FEATURES_DIR)
    if not all_rows:
        raise SystemExit("No extracted features found -- run `python -m probe.extract` first.")

    train_rows = rows_for(all_rows, split="train")
    val_rows = rows_for(all_rows, split="val")
    if not train_rows or not val_rows:
        raise SystemExit("Missing train/val rows -- run `python -m probe.split` first.")

    X_train, y_train, kept_train = build_matrix(train_rows, pool_method=args.pool_method)
    X_val, y_val, _ = build_matrix(val_rows, pool_method=args.pool_method)
    generator_train = np.array([r.generator for r in kept_train])

    head_a = fit_head_a(X_train, y_train)
    save(head_a, HEAD_A_NAME)
    print(f"Head A fitted on {len(y_train)} train rows (no calibration stage -- 0005 §3), saved as '{HEAD_A_NAME}'.")

    # Head B (generator ID) and the gate also get the self-generated adaptation pool
    # (never selfgen_eval -- that stays held out for E1) so they can recognize
    # gemini/gptimage at all: config.GENERATOR_CLASSES names 8 classes, but
    # split="train" is COCO_AI-only, so without this Head B could never predict
    # gemini/gptimage and the demo's live-upload -> E1-curve match would never fire.
    # Head A stays COCO_AI-only -- it's the head E1/E2/E5 measure, and training it on
    # the pool would make every live upload look pre-adapted, contradicting the curve
    # it's plotted against.
    selfgen_pool_rows = rows_for(all_rows, split="selfgen_pool")
    X_train_b, _, kept_train_b = build_matrix(train_rows + selfgen_pool_rows, pool_method=args.pool_method)
    generator_train_b = np.array([r.generator for r in kept_train_b])

    head_b = fit_head_b(X_train_b, generator_train_b)
    save(head_b, HEAD_B_NAME)
    print(f"Head B fitted on {len(generator_train_b)} rows ({len(selfgen_pool_rows)} selfgen pool) -- "
          f"classes: {head_b.classes}, saved as '{HEAD_B_NAME}'.")

    gate = fit_mahalanobis_gate(X_train_b, generator_train_b)
    save(gate, GATE_NAME)
    print(f"Mahalanobis gate fitted -- threshold={gate.threshold:.3f}, saved as '{GATE_NAME}'.")

    from config import VERDICT_FPR_BUDGETS, VERDICT_THRESHOLDS_NAME
    from probe.heads import prob_head_a
    from probe.thresholds import fit_cut_points

    real_val_rows = rows_for(all_rows, split="val", generator="real")
    X_real, _, _ = build_matrix(real_val_rows, pool_method=args.pool_method)
    if len(X_real):
        cuts = fit_cut_points(prob_head_a(head_a, X_real), VERDICT_FPR_BUDGETS)
        save(cuts, VERDICT_THRESHOLDS_NAME)
        print(f"Verdict cut points fitted on {len(X_real)} real val images: {cuts} "
              f"(budgets {VERDICT_FPR_BUDGETS}), saved as '{VERDICT_THRESHOLDS_NAME}'.")

    if not args.skip_spectral:
        from probe.spectral import fit_spectral_head

        spectral_fitted = fit_spectral_head()
        save(spectral_fitted, SPECTRAL_HEAD_NAME)
        print(f"Spectral head fitted, saved as '{SPECTRAL_HEAD_NAME}'.")


if __name__ == "__main__":
    main()
