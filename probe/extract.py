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

Features for a batch are written as soon as that batch is encoded, so an
interrupted run keeps what it finished and re-running resumes from there.

Usage:
    python -m probe.extract                    # native-patch arm, everything in the splits
    python -m probe.extract --arm standard      # E7 control arm
    python -m probe.extract --limit 200         # smoke test on a small slice
    python -m probe.extract --workers 0         # decode in-process (Windows fallback)
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
from torch.utils.data import DataLoader, Dataset
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
    EXTRACT_NUM_WORKERS,
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


def build_clip(device: str = DEVICE, dtype: torch.dtype | None = None):
    """Frozen CLIP, cast whole-model to the compute dtype (fp16 on cuda).

    The cast is what makes `next(model.parameters()).dtype` the single source
    of truth for callers -- weights and activations cannot drift apart, and
    the live demo path (`featurize_single_image`) encodes in the same
    precision the offline cache was built in."""
    import open_clip

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess_val


def _patch_tensor(patch: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    arr = torch.from_numpy(np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0)
    t = arr.permute(2, 0, 1)  # (3, H, W)
    return (t - mean) / std


class _ViewDataset(Dataset):
    """Yields every CLIP view of one image, already normalized: the N native
    patches followed by the whole-image view (native arm), or the whole view
    alone (standard arm).

    Everything in `__getitem__` is CPU work -- JPEG decode, N crops, N
    flatness gradients, N normalizations -- and it dominates extraction
    wall-clock, so it belongs on DataLoader workers rather than in the
    encode loop. Unreadable images yield None and are dropped by
    `_collate_views`; the item is simply absent from the cache and the index.
    """

    def __init__(self, items: list[ExtractItem], preprocess_val, seed: int = SEED, patches: bool = True):
        self.items = items
        self.preprocess_val = preprocess_val
        self.seed = seed
        self.patches = patches
        self.mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        try:
            image = Image.open(item.image_path).convert("RGB")
        except Exception as e:
            print(f"  skip unreadable {item.image_path}: {e}")
            return None
        whole = self.preprocess_val(image)
        if not self.patches:
            return {"idx": i, "views": whole.unsqueeze(0), "flatness": torch.zeros(0)}

        crops = extract_patches(
            image, n=N_PATCHES, patch_size=PATCH_SIZE, seed=self.seed,
            key=f"{item.split}:{item.generator}:{item.key}",
        )
        assert len(crops) == N_PATCHES, f"I1 violation for {item.key}"
        return {
            "idx": i,
            "views": torch.stack([_patch_tensor(c, self.mean, self.std) for c in crops] + [whole], dim=0),
            "flatness": torch.tensor([patch_flatness(c) for c in crops], dtype=torch.float32),
        }


def _collate_views(batch: list) -> dict:
    """Flattens per-image view stacks into one (sum_views, 3, H, W) tensor.
    `n_raw` is the pre-filter count so the progress bar still advances over
    unreadable images."""
    kept = [b for b in batch if b is not None]
    return {
        "n_raw": len(batch),
        "idx": [b["idx"] for b in kept],
        "flatness": [b["flatness"] for b in kept],
        "views": torch.cat([b["views"] for b in kept], dim=0) if kept else torch.empty(0),
    }


def _make_loader(items: list[ExtractItem], preprocess_val, batch_size: int, num_workers: int,
                 seed: int = SEED, patches: bool = True) -> DataLoader:
    """batch_size is in *views* (EXTRACT_BATCH_SIZE sizes the GPU batch), so
    convert to images per batch here -- the native arm emits N_PATCHES+1
    views per image."""
    views_per_image = N_PATCHES + 1 if patches else 1
    return DataLoader(
        _ViewDataset(items, preprocess_val, seed=seed, patches=patches),
        batch_size=max(1, batch_size // views_per_image),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_views,
        pin_memory=(DEVICE == "cuda"),
        # A native-arm batch is ~225 MB of float32 views, so the default
        # prefetch of 2 would hold several GB in flight across workers --
        # enough to OOM a 12.7 GB Colab runtime. Depth 1 costs nothing here
        # because the workers are the bottleneck: the GPU waits on them, not
        # the other way round.
        prefetch_factor=1 if num_workers else None,
    )


@torch.no_grad()
def encode_batch(model, tensors: torch.Tensor, device: str, dtype: torch.dtype) -> np.ndarray:
    """tensors: (B, 3, H, W) in [0,1] CLIP-normalized. Returns L2-normalized
    (B, D) float32 features (I6)."""
    feats = model.encode_image(tensors.to(device=device, dtype=dtype)).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # normalize in fp32 (I6)
    return feats.cpu().numpy()


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


def extract_native(items: list[ExtractItem], out_dir: Path, seed: int = SEED,
                   batch_size: int = EXTRACT_BATCH_SIZE, num_workers: int = EXTRACT_NUM_WORKERS) -> None:
    """Each batch is written to disk as soon as it is encoded, so a run that
    dies partway keeps everything it finished and the `dest.exists()` skip
    below turns into a real resume point."""
    model, preprocess_val = build_clip()
    device = DEVICE
    dtype = next(model.parameters()).dtype

    index_rows, pending = _partition_done(items, out_dir)
    loader = _make_loader(pending, preprocess_val, batch_size, num_workers, seed=seed, patches=True)

    with tqdm(total=len(pending), desc="extract[native]", unit="img") as bar:
        for batch in loader:
            bar.update(batch["n_raw"])
            if not batch["idx"]:
                continue
            out = encode_batch(model, batch["views"], device, dtype)
            out = out.reshape(len(batch["idx"]), N_PATCHES + 1, -1)  # (B, views, D)
            for item_idx, feats, flatness in zip(batch["idx"], out, batch["flatness"]):
                item = pending[item_idx]
                patch_arr = feats[:N_PATCHES]
                assert patch_arr.shape[0] == N_PATCHES, "I1 violation at write time"
                dest = out_dir / item.split / item.generator / f"{item.key}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    dest,
                    patches=patch_arr.astype(np.float32),
                    whole=feats[N_PATCHES].astype(np.float32),
                    patch_flatness=flatness.numpy(),
                )
                index_rows.append(_index_row(item, dest))

    _write_index(out_dir, index_rows)


def extract_standard(items: list[ExtractItem], out_dir: Path, batch_size: int = EXTRACT_BATCH_SIZE,
                     num_workers: int = EXTRACT_NUM_WORKERS) -> None:
    """E7 control arm: standard resize+centre-crop, one view per image, no patches."""
    model, preprocess_val = build_clip()
    device = DEVICE
    dtype = next(model.parameters()).dtype

    index_rows, pending = _partition_done(items, out_dir)
    loader = _make_loader(pending, preprocess_val, batch_size, num_workers, patches=False)

    with tqdm(total=len(pending), desc="extract[standard]", unit="img") as bar:
        for batch in loader:
            bar.update(batch["n_raw"])
            if not batch["idx"]:
                continue
            out = encode_batch(model, batch["views"], device, dtype)
            for item_idx, vec in zip(batch["idx"], out):
                it = pending[item_idx]
                dest = out_dir / it.split / it.generator / f"{it.key}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez(dest, whole=vec.astype(np.float32))
                index_rows.append(_index_row(it, dest))

    _write_index(out_dir, index_rows)


def _partition_done(items: list[ExtractItem], out_dir: Path) -> tuple[list[dict], list[ExtractItem]]:
    """Splits items into (index rows for what is already cached, items still
    to encode). The already-cached side is what makes a re-run resume rather
    than redo -- and it still rebuilds a complete index.csv."""
    index_rows, pending = [], []
    for it in items:
        dest = out_dir / it.split / it.generator / f"{it.key}.npz"
        if dest.exists():
            index_rows.append(_index_row(it, dest))
        else:
            pending.append(it)
    return index_rows, pending


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
    parser.add_argument("--workers", type=int, default=EXTRACT_NUM_WORKERS,
                        help="DataLoader workers for decode/crop/flatness; 0 runs in-process")
    args = parser.parse_args()

    items = load_index()
    if not items:
        raise SystemExit("No items to extract -- run `python -m probe.split` (and download.py / "
                          "selfgen_organize.py) first.")
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} image(s) queued for extraction "
          f"(arm={args.arm}, device={DEVICE}, workers={args.workers})")
    if args.arm == "patches":
        extract_native(items, FEATURES_DIR, num_workers=args.workers)
    else:
        extract_standard(items, FEATURES_STANDARD_DIR, num_workers=args.workers)


if __name__ == "__main__":
    main()
