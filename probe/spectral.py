"""Radial FFT spectral profile -- the Spectral evidence card's engine
(`0004` §8). Deliberately independent of the CLIP feature cache: it is a
different *dimension* of evidence (frequency-domain statistics of the
pixel grid, not a semantic embedding), which is the whole point of
evidence-level fusion (§8's "cards must be independently computed").

The profile is azimuthally averaged and binned by *normalized* radius
(fraction of Nyquist), not absolute frequency, so it is a fixed-length
vector regardless of source resolution -- comparable across the
270-1024px range this corpus spans.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_RADIAL_BINS = 32


def radial_fft_profile(image: Image.Image, n_bins: int = N_RADIAL_BINS) -> np.ndarray:
    """Returns an (n_bins,) log-power radial profile, resolution-invariant
    and independent of aspect ratio (image is centre-cropped to square
    first so the frequency grid is isotropic)."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    h, w = gray.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    gray = gray[top : top + side, left : left + side]

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    power = np.log1p(np.abs(spectrum) ** 2)

    cy, cx = side / 2, side / 2
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()  # 0 (DC) .. 1 (corner)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1])
        profile[i] = power[mask].mean() if mask.any() else 0.0

    # normalize to remove overall brightness/contrast scale, keep shape
    profile = profile - profile.mean()
    std = profile.std()
    if std > 1e-8:
        profile = profile / std
    return profile.astype(np.float32)


def _profiles_for(items) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for it in items:
        try:
            image = Image.open(it.image_path).convert("RGB")
        except Exception as e:
            print(f"  skip unreadable {it.image_path}: {e}")
            continue
        xs.append(radial_fft_profile(image))
        ys.append(it.label)
    if not xs:
        return np.zeros((0, N_RADIAL_BINS)), np.zeros((0,))
    return np.stack(xs), np.array(ys, dtype=np.int64)


def fit_spectral_head() -> dict:
    """Independent of the CLIP feature cache -- reads raw images directly
    via the same train/val split `probe/extract.py` uses (`load_index`),
    so no GPU or CLIP extraction is required to fit this card.

    No calibration split: `0005` §3 removed the Platt stage, so val stays
    genuinely held out for this card too."""
    from sklearn.preprocessing import StandardScaler

    from probe.extract import load_index
    from probe.heads import fit_head_a

    items = load_index()
    train_items = [it for it in items if it.split == "train"]
    if not train_items:
        raise SystemExit("No train items found -- run `python -m probe.split` first.")

    X_train, y_train = _profiles_for(train_items)

    scaler = StandardScaler().fit(X_train)
    head = fit_head_a(scaler.transform(X_train), y_train)
    return {"scaler": scaler, "head": head}


def spectral_score(fitted: dict, image: Image.Image) -> float:
    """P(AI-generated) from the spectral profile alone, from the
    classifier's own sigmoid (`0005` §3 -- no second fitted stage)."""
    from probe.heads import prob_head_a

    profile = radial_fft_profile(image).reshape(1, -1)
    X = fitted["scaler"].transform(profile)
    return float(prob_head_a(fitted["head"], X)[0])


def main() -> None:
    from probe.heads import save

    fitted = fit_spectral_head()
    path = save(fitted, "spectral_head")
    print(f"Spectral head fitted and saved to {path}")


if __name__ == "__main__":
    main()
