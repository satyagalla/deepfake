"""Fixed-N native-resolution patch sampling (`0004` §6).

No resampling on the patch path -- a native crop is the same operation
regardless of source resolution, so it introduces no resolution-dependent
signature and preserves the high-frequency band the artifact lives in
(§6.2). **N is fixed at 16 for every image** (I1): patch count correlates
with source resolution which correlates with label, so a variable count
would leak through mean, max and top-k alike (§6.3, the bag-size leak).

Small images (native size below PATCH_SIZE on some axis) sample with
overlap; large images sample N at random from the non-overlapping grid.
Sampling is deterministic per image (seeded from path + a run seed) so
extraction is reproducible.
"""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_PATCHES, PATCH_SIZE, SEED


def _image_seed(key: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:16], 16)


def sample_patch_boxes(
    w: int, h: int, n: int = N_PATCHES, patch_size: int = PATCH_SIZE, seed: int = SEED, key: str = ""
) -> list[tuple[int, int]]:
    """Returns n (left, top) boxes, patch_size x patch_size, within a w x h
    image. Assumes w >= patch_size and h >= patch_size (the caller upscales
    undersized images first -- see extract_patches)."""
    rng = random.Random(_image_seed(key, seed))

    xs = list(range(0, w - patch_size + 1, patch_size)) or [0]
    ys = list(range(0, h - patch_size + 1, patch_size)) or [0]
    grid_boxes = [(x, y) for x in xs for y in ys]
    rng.shuffle(grid_boxes)

    if len(grid_boxes) >= n:
        return grid_boxes[:n]

    # Not enough non-overlapping room -- take every grid box once, then fill
    # the rest with random overlapping boxes (uniform over valid top-left
    # positions). This is the "small images sample with overlap" case;
    # near patch_size x patch_size it degenerates to repeating ~the same
    # box, which is correct: patching is close to a no-op there (§6.4).
    boxes = list(grid_boxes)
    x_max, y_max = w - patch_size, h - patch_size
    while len(boxes) < n:
        boxes.append((rng.randint(0, x_max), rng.randint(0, y_max)))
    return boxes[:n]


def load_upscaled_if_needed(image: Image.Image, patch_size: int = PATCH_SIZE) -> Image.Image:
    """The only resampling permitted on the patch path: if native size is
    below patch_size on either axis, upscale (never downscale) just enough
    that a native-equivalent crop is possible at all. Rare on this corpus
    (native short side is 270-1024px per bottlenecks.md §9.1) -- documented
    fallback, not the common path."""
    w, h = image.size
    if w >= patch_size and h >= patch_size:
        return image
    scale = max(patch_size / w, patch_size / h)
    new_w, new_h = max(patch_size, round(w * scale)), max(patch_size, round(h * scale))
    return image.resize((new_w, new_h), Image.BICUBIC)


def extract_patches(image: Image.Image, n: int = N_PATCHES, patch_size: int = PATCH_SIZE,
                     seed: int = SEED, key: str = "") -> list[Image.Image]:
    """Returns n native-resolution patch_size x patch_size PIL crops."""
    image = load_upscaled_if_needed(image, patch_size)
    w, h = image.size
    boxes = sample_patch_boxes(w, h, n=n, patch_size=patch_size, seed=seed, key=key)
    assert len(boxes) == n, f"I1 violation: sampled {len(boxes)} boxes, expected {n} (key={key})"
    return [image.crop((x, y, x + patch_size, y + patch_size)) for x, y in boxes]


def patch_flatness(patch: Image.Image) -> float:
    """Mean gradient magnitude of a patch -- low for a flat sky/wall patch,
    high for a textured one. Cached alongside features at extraction time
    so the flat-patch filter ablation (§6.5) is free at aggregation time
    instead of requiring the source images to be re-read."""
    gray = np.asarray(patch.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    return float(np.sqrt(gx**2 + gy**2).mean())
