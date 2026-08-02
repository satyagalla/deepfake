"""Loads cached CLIP features (`probe/extract.py`'s output) into fit-ready
matrices, with the patch-aggregation choice (`0004` §6.5 / E6) applied at
load time -- everything downstream (heads, experiments, cards) shares this
so the aggregator ablation only ever means "call this with a different
`pool_method`," never a re-extraction.

**I6, applied here, not just at extraction**: the pooled + concatenated
feature vector is re-L2-normalized before being handed to any fit, since
mean/max/top-k/trimmed pooling of unit-norm patch vectors does not itself
produce a unit-norm result, and Mahalanobis conditioning (`0004` §7.2)
needs it.
"""
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_DIR, FEATURES_STANDARD_DIR, N_PATCHES

POOL_METHODS = ["mean", "max", "topk", "trimmed", "flat_filtered"]


def pool_patches(
    patch_feats: np.ndarray,
    flatness: np.ndarray | None = None,
    method: str = "mean",
    k: int = 4,
    trim_frac: float = 0.2,
    flat_drop_frac: float = 0.25,
) -> np.ndarray:
    """patch_feats: (N_PATCHES, D). Returns (D,). All methods are
    element-wise across patches -- no cross-dimension mixing."""
    if method == "mean":
        return patch_feats.mean(axis=0)
    if method == "max":
        return patch_feats.max(axis=0)
    if method == "topk":
        # element-wise top-k mean per dimension -- avoids needing a scalar
        # per-patch score to rank whole feature vectors by.
        sorted_desc = np.sort(patch_feats, axis=0)[::-1]
        return sorted_desc[:k].mean(axis=0)
    if method == "trimmed":
        n = patch_feats.shape[0]
        cut = int(n * trim_frac)
        sorted_arr = np.sort(patch_feats, axis=0)
        if cut > 0:
            sorted_arr = sorted_arr[cut : n - cut]
        return sorted_arr.mean(axis=0)
    if method == "flat_filtered":
        if flatness is None:
            raise ValueError("flat_filtered requires patch_flatness")
        thresh = np.quantile(flatness, flat_drop_frac)
        keep = flatness >= thresh
        if not keep.any():
            keep = np.ones_like(flatness, dtype=bool)
        return patch_feats[keep].mean(axis=0)
    raise ValueError(f"unknown pool method: {method}")


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


@dataclass
class FeatureRow:
    feature_path: str
    split: str
    generator: str
    key: str
    label: int
    domain: str
    container: str


def read_index(features_dir: Path = FEATURES_DIR) -> list[FeatureRow]:
    index_path = features_dir / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"{index_path} missing -- run `python -m probe.extract` first.")
    rows = []
    with open(index_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                FeatureRow(
                    feature_path=r["feature_path"],
                    split=r["split"],
                    generator=r["generator"],
                    key=r["key"],
                    label=int(r["label"]),
                    domain=r["domain"],
                    container=r["container"],
                )
            )
    return rows


def build_matrix(
    rows: list[FeatureRow], pool_method: str = "mean", standard_arm: bool = False
) -> tuple[np.ndarray, np.ndarray, list[FeatureRow]]:
    """Returns (X, y, kept_rows). X is L2-normalized concat(pooled_patches,
    whole) for the native arm (2D-dim), or just L2-normalized `whole` for
    the standard arm (D-dim, E7). kept_rows drops any row whose .npz is
    missing/corrupt so X/y/kept_rows always stay aligned."""
    xs, ys, kept = [], [], []
    for row in rows:
        try:
            data = np.load(row.feature_path)
        except Exception as e:
            print(f"  skip unreadable feature file {row.feature_path}: {e}")
            continue
        whole = data["whole"]
        if standard_arm:
            vec = whole
        else:
            patches = data["patches"]
            assert patches.shape[0] == N_PATCHES, f"I1 violation reading {row.feature_path}"
            flatness = data["patch_flatness"] if "patch_flatness" in data else None
            pooled = pool_patches(patches, flatness=flatness, method=pool_method)
            vec = np.concatenate([pooled, whole])
        xs.append(vec)
        ys.append(row.label)
        kept.append(row)
    if not xs:
        return np.zeros((0, 0)), np.zeros((0,)), []
    X = l2_normalize_rows(np.stack(xs).astype(np.float32))
    y = np.array(ys, dtype=np.int64)
    return X, y, kept


def rows_for(rows: list[FeatureRow], split: str | None = None, generator: str | None = None,
             domain: str | None = None, container: str | None = None) -> list[FeatureRow]:
    out = rows
    if split is not None:
        out = [r for r in out if r.split == split]
    if generator is not None:
        out = [r for r in out if r.generator == generator]
    if domain is not None:
        out = [r for r in out if r.domain == domain]
    if container is not None:
        out = [r for r in out if r.container == container]
    return out
