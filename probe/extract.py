"""Frozen CLIP ViT-L/14 feature extraction (`0004` §6-§7, build plan S3).

Two arms, cached separately:

- **native patches** (default): N_PATCHES=16 native-resolution 224 crops
  per image (no resampling, I1/I5) + 1 whole-image resized view. This is
  the arm the rest of the pipeline (heads, E1) is built on.
- **standard** (`--arm standard`, E7's control): the ordinary
  resize-short-side-then-centre-crop pipeline, one view per image. This
  is UniversalFakeDetect's exact preprocessing and exists only to answer
  "does native-patch preprocessing actually pay for itself."

Every feature is L2-normalized before being cached (I6 -- required for
well-conditioned Mahalanobis distance downstream, §7.2). Patch counts are
asserted per image (I1) before anything is written to disk.

Output layout: FEATURES_DIR/<split>/<generator>/<key>.npz, each holding
`patches` (N_PATCHES, D) and `whole` (D,) arrays (native arm), or
FEATURES_STANDARD_DIR/<split>/<generator>/<key>.npz holding `whole` (D,)
only (standard arm). An index CSV at the arm's root lists every item with
its label/domain/container metadata, so downstream code never has to
re-derive it from folder structure.

Usage:
    python -m probe.extract                    # native-patch arm, everything in the splits
    python -m probe.extract --arm standard      # E7 control arm
    python -m probe.extract --limit 200         # smoke test on a small slice
"""
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLIP_MEAN,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_STD,
    COCO_AI_RAW_DIR,
    DEVICE,
    EXTRACT_BATCH_SIZE,
    FEATURES_DIR,
    FEATURES_STANDARD_DIR,
    HELDOUT_GENERATOR,
    N_PATCHES,
    PATCH_SIZE,
    SEED,
    SELFGEN_RAW_DIR,
    SPLITS_DIR,
)
from probe.patches import extract_patches, patch_flatness
from probe.split import RowSplit, SelfgenSplit, expand_row_split_to_images


@dataclass
class ExtractItem:
    image_path: Path
    split: str  # 'train' | 'val' | 'heldout' | 'selfgen'
    generator: str
    key: str  # unique within (split, generator)
    label: int  # 0 = real, 1 = AI-generated
    domain: str = "indomain"  # 'indomain' | 'offdomain' -- selfgen only, COCO_AI is always indomain
    container: str = ""  # 'api' | 'web' -- selfgen only


def load_index() -> list[ExtractItem]:
    items: list[ExtractItem] = []

    row_split_path = SPLITS_DIR / "coco_ai_row_split.json"
    if row_split_path.exists():
        row_split = RowSplit(**json.loads(row_split_path.read_text()))
        images = expand_row_split_to_images(row_split)
        for split_name, rows in images.items():
            for rel_path, row_id, generator in rows:
                items.append(
                    ExtractItem(
                        image_path=COCO_AI_RAW_DIR / rel_path,
                        split=split_name,
                        generator=generator,
                        key=row_id,
                        label=0 if generator == "real" else 1,
                    )
                )
    else:
        print(f"NOTE: {row_split_path} missing -- run `python -m probe.split` first. Skipping COCO_AI.")

    selfgen_split_path = SPLITS_DIR / "selfgen_split.json"
    if selfgen_split_path.exists():
        selfgen_split = SelfgenSplit(**json.loads(selfgen_split_path.read_text()))
        for generator, domains in selfgen_split.generators.items():
            for domain, parts in domains.items():
                for subset, rel_paths in parts.items():  # subset: 'pool' | 'eval'
                    for rel_path in rel_paths:
                        p = Path(rel_path)
                        container = p.parts[2] if len(p.parts) > 2 else ""
                        items.append(
                            ExtractItem(
                                image_path=SELFGEN_RAW_DIR / rel_path,
                                split=f"selfgen_{subset}",
                                generator=generator,
                                key=p.stem,
                                label=1,
                                domain=domain,
                                container=container,
                            )
                        )
    else:
        print(f"NOTE: {selfgen_split_path} missing -- run `python -m probe.split` first. Skipping selfgen.")

    return items


def build_clip(device: str = DEVICE):
    import open_clip

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess_val


def _patch_tensor(patch: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    arr = torch.from_numpy(np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0)
    t = arr.permute(2, 0, 1)  # (3, H, W)
    return (t - mean) / std


@torch.no_grad()
def encode_batch(model, tensors: torch.Tensor, device: str, dtype: torch.dtype) -> np.ndarray:
    """tensors: (B, 3, H, W) in [0,1] CLIP-normalized. Returns L2-normalized
    (B, D) float32 features (I6)."""
    feats = model.encode_image(tensors.to(device=device, dtype=dtype))
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return feats.float().cpu().numpy()


def featurize_single_image(
    image: Image.Image, model, preprocess_val, pool_method: str = "mean", seed: int = SEED, key: str = "live"
) -> np.ndarray:
    """Same pipeline as extract_native, for one arbitrary image at request
    time (the Gradio demo's live-upload path, S8) -- N native patches +
    whole view, pooled + concatenated, L2-normalized (I6). Shares the exact
    preprocessing the offline cache used, so a live prediction is built
    the same way a training/eval row was."""
    from probe.features import l2_normalize_rows, pool_patches

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)

    patches = extract_patches(image, n=N_PATCHES, patch_size=PATCH_SIZE, seed=seed, key=key)
    assert len(patches) == N_PATCHES, "I1 violation on live image"
    patch_batch = torch.stack([_patch_tensor(p, mean, std) for p in patches], dim=0)
    whole_tensor = preprocess_val(image).unsqueeze(0)

    patch_feats = encode_batch(model, patch_batch, str(device), dtype)
    whole_feat = encode_batch(model, whole_tensor, str(device), dtype)[0]

    pooled = pool_patches(patch_feats, method=pool_method)
    vec = np.concatenate([pooled, whole_feat]).astype(np.float32)
    return l2_normalize_rows(vec.reshape(1, -1))[0]


def extract_native(items: list[ExtractItem], out_dir: Path, seed: int = SEED, batch_size: int = EXTRACT_BATCH_SIZE) -> None:
    model, preprocess_val = build_clip()
    device = DEVICE
    dtype = torch.float16 if device == "cuda" else torch.float32
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)

    index_rows = []
    buf_tensors, buf_meta = [], []  # meta: (item_idx, view) where view in {'patch0'..'patch15','whole'}
    per_item_feats: dict[int, dict[str, np.ndarray]] = {}
    per_item_flatness: dict[int, np.ndarray] = {}

    def flush():
        if not buf_tensors:
            return
        batch = torch.stack(buf_tensors, dim=0)
        out = encode_batch(model, batch, device, dtype)
        for (item_idx, view), vec in zip(buf_meta, out):
            per_item_feats.setdefault(item_idx, {})[view] = vec
        buf_tensors.clear()
        buf_meta.clear()

    for idx, item in enumerate(tqdm(items, desc="extract[native]", unit="img")):
        dest = out_dir / item.split / item.generator / f"{item.key}.npz"
        if dest.exists():
            index_rows.append(_index_row(item, dest))
            continue
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception as e:
            print(f"  skip unreadable {item.image_path}: {e}")
            continue

        patches = extract_patches(image, n=N_PATCHES, patch_size=PATCH_SIZE, seed=seed, key=f"{item.split}:{item.generator}:{item.key}")
        assert len(patches) == N_PATCHES, f"I1 violation for {item.key}"
        per_item_flatness[idx] = np.array([patch_flatness(p) for p in patches], dtype=np.float32)
        for pi, patch in enumerate(patches):
            buf_tensors.append(_patch_tensor(patch, mean, std))
            buf_meta.append((idx, f"patch{pi}"))
        whole_tensor = preprocess_val(image)
        buf_tensors.append(whole_tensor)
        buf_meta.append((idx, "whole"))

        if len(buf_tensors) >= batch_size:
            flush()

    flush()  # trailing partial batch

    for idx, item in enumerate(items):
        dest = out_dir / item.split / item.generator / f"{item.key}.npz"
        if dest.exists():
            continue
        feats = per_item_feats.get(idx)
        if feats is None:
            continue  # unreadable, already warned above
        patch_keys = [f"patch{i}" for i in range(N_PATCHES)]
        if not all(k in feats for k in patch_keys) or "whole" not in feats:
            print(f"  WARNING: incomplete features for {item.key}, skipping write")
            continue
        patch_arr = np.stack([feats[k] for k in patch_keys], axis=0)  # (N_PATCHES, D)
        assert patch_arr.shape[0] == N_PATCHES, "I1 violation at write time"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest,
            patches=patch_arr.astype(np.float32),
            whole=feats["whole"].astype(np.float32),
            patch_flatness=per_item_flatness[idx],
        )
        index_rows.append(_index_row(item, dest))

    _write_index(out_dir, index_rows)


def extract_standard(items: list[ExtractItem], out_dir: Path, batch_size: int = EXTRACT_BATCH_SIZE) -> None:
    """E7 control arm: standard resize+centre-crop, one view per image, no patches."""
    model, preprocess_val = build_clip()
    device = DEVICE
    dtype = torch.float16 if device == "cuda" else torch.float32

    index_rows = []
    to_extract = [it for it in items if not (out_dir / it.split / it.generator / f"{it.key}.npz").exists()]
    for it in items:
        dest = out_dir / it.split / it.generator / f"{it.key}.npz"
        if dest.exists():
            index_rows.append(_index_row(it, dest))

    buf_tensors, buf_items = [], []

    def flush():
        if not buf_tensors:
            return []
        batch = torch.stack(buf_tensors, dim=0)
        out = encode_batch(model, batch, device, dtype)
        rows = []
        for it, vec in zip(buf_items, out):
            dest = out_dir / it.split / it.generator / f"{it.key}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, whole=vec.astype(np.float32))
            rows.append(_index_row(it, dest))
        buf_tensors.clear()
        buf_items.clear()
        return rows

    for it in tqdm(to_extract, desc="extract[standard]", unit="img"):
        try:
            image = Image.open(it.image_path).convert("RGB")
        except Exception as e:
            print(f"  skip unreadable {it.image_path}: {e}")
            continue
        buf_tensors.append(preprocess_val(image))
        buf_items.append(it)
        if len(buf_tensors) >= batch_size:
            index_rows.extend(flush())
    index_rows.extend(flush())

    _write_index(out_dir, index_rows)


def _index_row(item: ExtractItem, dest: Path) -> dict:
    return {
        "feature_path": dest.as_posix(),
        "split": item.split,
        "generator": item.generator,
        "key": item.key,
        "label": item.label,
        "domain": item.domain,
        "container": item.container,
    }


def _write_index(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_path", "split", "generator", "key", "label", "domain", "container"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["patches", "standard"], default="patches")
    parser.add_argument("--limit", type=int, default=None, help="cap number of images (smoke test)")
    args = parser.parse_args()

    items = load_index()
    if not items:
        raise SystemExit("No items to extract -- run `python -m probe.split` (and download.py / "
                          "selfgen_organize.py) first.")
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} image(s) queued for extraction (arm={args.arm}, device={DEVICE})")
    if args.arm == "patches":
        extract_native(items, FEATURES_DIR)
    else:
        extract_standard(items, FEATURES_STANDARD_DIR)


if __name__ == "__main__":
    main()
