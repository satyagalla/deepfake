"""Head A (fused binary score), Head B (generator ID), and the Mahalanobis
OOD gate (`0004` §7). All three fit on the same L2-normalized
concat(pooled-patches, whole) feature matrix from `probe/features.py` --
"seconds to fit," per §7's framing of what the frozen backbone buys.

- **Head A**: logistic regression, real vs AI. **No post-hoc calibration
  stage** (I8, `0005` §3): the Platt calibrator this module used to fit
  was fitted on COCO_AI val and applied to 1024px Gemini and to
  Midjourney, was monotone so could never move AUC, and contaminated E5
  by evaluating on the split it was fitted on. Metrics run on
  `score_head_a` (raw logits); the cards' `[0,1]` number comes from
  `prob_head_a`, the classifier's own sigmoid, not a second fitted stage.
  Threshold placement lives in `probe/thresholds.py`.
- **Head B**: multiclass generator ID, feeds the AI Model card.
- **Mahalanobis gate**: per-class means, one pooled covariance with
  Ledoit-Wolf shrinkage (never per-class -- singular at 768+-d against
  ~150 self-generated samples).
"""
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROBE_CHECKPOINT_DIR


@dataclass
class HeadA:
    clf: LogisticRegression


def fit_head_a(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> HeadA:
    return HeadA(clf=LogisticRegression(max_iter=2000, class_weight="balanced", C=C).fit(X_train, y_train))


def score_head_a(head: HeadA, X: np.ndarray) -> np.ndarray:
    """Raw logits, shape (n,). The natural cut point is **0.0, not 0.5** --
    this is what every metric in `probe/thresholds.py` operates on."""
    return head.clf.decision_function(X)


def prob_head_a(head: HeadA, X: np.ndarray) -> np.ndarray:
    """P(AI-generated) from the classifier's own sigmoid, shape (n,) -- for
    the cards, which need a number in [0,1] to fuse. Fitted with
    `class_weight="balanced"` against a ~1:5 real:fake corpus, so these are
    stated **as-if-balanced**; that is the right frame for the balanced
    eval sets (`0005` §5) and it is stated rather than assumed."""
    return head.clf.predict_proba(X)[:, 1]


@dataclass
class HeadB:
    clf: LogisticRegression
    classes: list[str]


def fit_head_b(X_train: np.ndarray, generator_train: np.ndarray, C: float = 1.0) -> HeadB:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C).fit(X_train, generator_train)
    return HeadB(clf=clf, classes=list(clf.classes_))


def predict_head_b(head: HeadB, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_label, prob) -- prob is the predicted class's own probability."""
    probs = head.clf.predict_proba(X)
    idx = probs.argmax(axis=1)
    labels = np.array(head.classes)[idx]
    return labels, probs[np.arange(len(X)), idx]


@dataclass
class MahalanobisGate:
    class_means: dict  # generator -> (D,)
    precision: np.ndarray  # (D, D), Ledoit-Wolf pooled inverse covariance
    threshold: float  # abstention cutoff (distance units)


def _min_mahalanobis(x: np.ndarray, class_means: dict, precision: np.ndarray) -> float:
    best = float("inf")
    for mu in class_means.values():
        d = x - mu
        dist = float(np.sqrt(max(d @ precision @ d, 0.0)))
        best = min(best, dist)
    return best


def fit_mahalanobis_gate(X_train: np.ndarray, generator_train: np.ndarray, percentile: float = 95.0) -> MahalanobisGate:
    lw = LedoitWolf().fit(X_train)  # pooled over ALL classes -- ~thousands of rows for one (D,D) matrix
    class_means = {g: X_train[generator_train == g].mean(axis=0) for g in sorted(set(generator_train))}
    dists = np.array([_min_mahalanobis(x, class_means, lw.precision_) for x in X_train])
    threshold = float(np.percentile(dists, percentile))
    return MahalanobisGate(class_means=class_means, precision=lw.precision_, threshold=threshold)


def gate_distance(gate: MahalanobisGate, X: np.ndarray) -> np.ndarray:
    return np.array([_min_mahalanobis(x, gate.class_means, gate.precision) for x in X])


def gate_abstain(gate: MahalanobisGate, X: np.ndarray) -> np.ndarray:
    """True where the sample is farther from every known class mean than
    the fitted threshold -- 'this generator is outside everything I was
    fitted on' (0004 §7.2)."""
    return gate_distance(gate, X) > gate.threshold


def save(obj, name: str, out_dir: Path = PROBE_CHECKPOINT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load(name: str, out_dir: Path = PROBE_CHECKPOINT_DIR):
    with open(out_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
