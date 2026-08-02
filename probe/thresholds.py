"""Threshold placement and the metric set (`0005` §4, §5).

Everything here operates on **raw logits** (`probe.heads.score_head_a`),
where the natural cut point is `0.0`, not `0.5`. There is no fitted stage
between a score and a verdict anywhere in this build (I8) -- what used to
be a Platt calibrator is replaced by measuring *where the cut point is*
and reporting how much of any failure is placement rather than blindness.

The governing distinction is **prevalence-dependence**, because the eval
sets' class ratio is a construct and deployment prevalence is ~0.5%:

- `auc`, `tpr`, `fpr` -- prevalence-invariant. Reported directly, on the
  full eval set. AUC is also the discriminator `plan-c-source-verification.md`
  §5 states the project's decision rule in: AUC ~>=0.75 with accuracy ~50%
  is a misplaced threshold; AUC ~0.5 is the Chameleon regime.
- balanced accuracy -- `(tpr + 1 - fpr)/2`, so it is free and needs no
  subsampling.
- **precision -- derived at a named prevalence, never stored** (§4.3).
  Storing it would freeze the eval set's arbitrary class ratio into the
  number.
- `alpha = E[z]` -- Yang et al.'s label-free correction. Spends no labels,
  which is the only reason it is admissible *inside* E1: a supervised
  correction would consume target-generator labels outside the stated N.
  It **does** require a roughly balanced batch (moment-balancing is the
  whole argument), which is what `balanced_indices` is for.
- the oracle threshold -- a labelled diagnostic **ceiling**.
  `oracle - default` is the decomposition. Never quoted as a result.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sweep(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every achievable (threshold, tpr, fpr) in one pass, via cumulative
    counts over the sorted scores. Prediction is `score > threshold`."""
    order = np.argsort(scores, kind="mergesort")
    s, yy = scores[order], y[order]
    n_pos, n_neg = int((yy == 1).sum()), int((yy == 0).sum())

    cum_pos = np.concatenate([[0], np.cumsum(yy == 1)])
    cum_neg = np.concatenate([[0], np.cumsum(yy == 0)])
    tpr = (n_pos - cum_pos) / max(n_pos, 1)
    fpr = (n_neg - cum_neg) / max(n_neg, 1)

    # candidate cut i predicts the top (n - i) scores positive
    cuts = np.empty(len(s) + 1)
    cuts[0] = s[0] - 1.0
    cuts[-1] = s[-1] + 1.0
    if len(s) > 1:
        cuts[1:-1] = (s[:-1] + s[1:]) / 2.0
    return cuts, tpr, fpr


def rates_at(scores: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> tuple[float, float]:
    """(tpr, fpr) at a cut point. Both are computed within a single class,
    so both are prevalence-invariant -- they survive the eval-set imbalance
    that makes raw accuracy uninterpretable here (`0005` §5)."""
    pred = scores > threshold
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr = float(pred[y == 1].sum() / n_pos) if n_pos else float("nan")
    fpr = float(pred[y == 0].sum() / n_neg) if n_neg else float("nan")
    return tpr, fpr


def balanced_accuracy_at(scores: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    """Mean of per-class recall. Estimates accuracy on a hypothetically
    balanced set while using every available image -- which is why E1 does
    not subsample for this (`0005` §5)."""
    tpr, fpr = rates_at(scores, y, threshold)
    return float((tpr + (1.0 - fpr)) / 2.0)


def accuracy_at(scores: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    """Raw accuracy. Prevalence-dependent, kept only for continuity with
    the pre-`0005` numbers -- read `balanced_accuracy_at` instead."""
    if len(y) == 0:
        return float("nan")
    return float(((scores > threshold).astype(int) == y).mean())


def label_free_threshold(scores: np.ndarray) -> float:
    """Yang et al.'s `alpha* = E[z]`: the optimal cut is the expected logit
    under the estimated distribution `[confirmed]`. **No labels**, so it
    costs nothing against E1's N-shot budget.

    Pass a *balanced* batch (`balanced_indices`) -- on a >=90%-real batch
    the mean is dragged toward the real class and alpha is actively wrong.
    Using labels to build the batch is an experimental-design choice, not
    a fit, so the label-free property survives; the resulting number is
    optimistic vs. a deployment batch that cannot be balanced.
    """
    return float(np.mean(scores)) if len(scores) else float("nan")


def oracle_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """The cut point maximising balanced accuracy. Uses labels: this is a
    **diagnostic ceiling**, not a deployable threshold. The gap from the
    default cut is the answer to "was this a placement problem or a
    blindness problem"."""
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    cuts, tpr, fpr = _sweep(scores, y)
    return float(cuts[int(np.argmax((tpr + (1.0 - fpr)) / 2.0))])


def tpr_at_fpr(scores: np.ndarray, y: np.ndarray, target_fpr: float = 0.01) -> float:
    """Recall at a fixed false-positive budget -- the operating-point view,
    and what the detection literature reports. Needs no class balance."""
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    _, tpr, fpr = _sweep(scores, y)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """`precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR)`.

    Because TPR and FPR are prevalence-invariant, precision at *any* prior
    follows from them -- so it is derived on demand rather than stored
    (`0005` §4.3). This is `2026-08-01-calibration-and-thresholds.md` §1.2
    computed rather than asserted: at TPR 0.95 / FPR 0.01 and pi=0.005 it
    reproduces the ~32% already on record in `2026-07-31-production-deployment.md`.
    """
    num = prevalence * tpr
    den = num + (1.0 - prevalence) * fpr
    return float(num / den) if den > 0 else float("nan")


def balanced_indices(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices of a class-balanced subsample, larger class downsampled.
    Only `alpha` needs this -- AUC, TPR and FPR keep the full set (I9)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    k = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])


def fit_cut_points(real_scores: np.ndarray, fpr_budgets: dict) -> dict:
    """Verdict cut points from a **false-positive budget** rather than a
    guessed constant (`0005` §7). `{name: fpr}` -> `{name: threshold}`,
    each the `(1-fpr)` quantile of scores on known-real images, so the
    threshold carries a stated meaning: "one real photo in a hundred gets
    called synthetic."

    Works on any score scale -- pass fused card scores in [0,1] for the
    verdict cuts, or logits for a Head A cut.
    """
    if len(real_scores) == 0:
        return {name: float("nan") for name in fpr_budgets}
    return {name: float(np.quantile(real_scores, 1.0 - fpr)) for name, fpr in fpr_budgets.items()}


def threshold_report(scores: np.ndarray, y: np.ndarray, seed: int = 0, default_threshold: float = 0.0) -> dict:
    """The `0005` §4 metric set for one eval set.

    `alpha` is fitted on a balanced subsample; every rate is then measured
    on the **full** set, since TPR and FPR need no balancing and the extra
    reals are free statistical power.
    """
    from sklearn.metrics import roc_auc_score

    n_ai, n_real = int((y == 1).sum()), int((y == 0).sum())
    if len(y) == 0 or n_ai == 0 or n_real == 0:
        return {"auc": float("nan"), "tpr": float("nan"), "fpr": float("nan"),
                "balanced_accuracy": float("nan"), "accuracy": float("nan"),
                "alpha": float("nan"), "tpr_alpha": float("nan"), "fpr_alpha": float("nan"),
                "balanced_accuracy_alpha": float("nan"), "balanced_accuracy_oracle": float("nan"),
                "tpr_at_fpr01": float("nan"), "n_ai": n_ai, "n_real": n_real}

    tpr, fpr = rates_at(scores, y, default_threshold)
    alpha = label_free_threshold(scores[balanced_indices(y, seed=seed)])
    tpr_a, fpr_a = rates_at(scores, y, alpha)

    return {
        "auc": float(roc_auc_score(y, scores)),
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": float((tpr + (1.0 - fpr)) / 2.0),
        "accuracy": accuracy_at(scores, y, default_threshold),
        "alpha": alpha,
        "tpr_alpha": tpr_a,
        "fpr_alpha": fpr_a,
        "balanced_accuracy_alpha": float((tpr_a + (1.0 - fpr_a)) / 2.0),
        "balanced_accuracy_oracle": balanced_accuracy_at(scores, y, oracle_threshold(scores, y)),
        "tpr_at_fpr01": tpr_at_fpr(scores, y, 0.01),
        "n_ai": n_ai,
        "n_real": n_real,
    }
