"""Uniform MTCNN face detect+crop across all three raw sources, plus a
source-image-level stratified train/val split and manifest.csv.

Run after data/download.py has populated data_raw/{real_src,deepfake_src,edited_src}.
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CLASSES,
    DATASET_DIR,
    DEEPFAKE_SRC,
    DEVICE,
    EDITED_SRC,
    FACE_MARGIN,
    IMAGE_SIZE,
    MANIFEST_PATH,
    REAL_SRC,
    SEED,
    VAL_FRACTION,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BATCH_SIZE = 32
DEEPFAKE_SURVIVAL_FLOOR = 300  # per data_download.md: stop and flag if fewer survive
SOURCE_NAME = {"real": "COCO_AI", "deepfake": "COCO_AI", "edited": "CASIA_v2.0"}


def find_casia_tampered(root: Path) -> list[Path]:
    """CASIA v2.0 marks tampered images with a 'Tp_' filename prefix (vs 'Au_'
    for authentic) regardless of the exact folder layout a mirror unzips to."""
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    tampered = [p for p in all_imgs if p.name.lower().startswith("tp_")]
    if not tampered:
        tampered = [p for p in all_imgs if "tp" in p.parent.name.lower() and "au" not in p.parent.name.lower()]
    print(f"CASIA: {len(all_imgs)} images found under {root}, {len(tampered)} identified as tampered (edited).")
    return tampered


def list_source_images(class_name: str) -> list[Path]:
    if class_name == "real":
        return sorted(p for p in REAL_SRC.iterdir() if p.suffix.lower() in IMG_EXTS)
    if class_name == "deepfake":
        return sorted(p for p in DEEPFAKE_SRC.iterdir() if p.suffix.lower() in IMG_EXTS)
    if class_name == "edited":
        return sorted(find_casia_tampered(EDITED_SRC))
    raise ValueError(class_name)


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def detect_and_crop(mtcnn: MTCNN, paths: list[Path], class_name: str) -> list[tuple[Path, Image.Image]]:
    results = []
    seen = detected = 0
    pbar = tqdm(total=len(paths), desc=f"[{class_name}] detect+crop", unit="img")
    for batch_paths in batched(paths, BATCH_SIZE):
        imgs, valid_paths = [], []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                print(f"  [{class_name}] skip unreadable file {p}: {e}")
        if not imgs:
            pbar.update(len(batch_paths))
            continue
        seen += len(imgs)
        # mtcnn(list) requires every image in the call to share identical
        # pixel dimensions -- group by actual size so same-size images still
        # go through as one batched GPU call (no resize/distortion of source
        # pixels, which would blur the SRM/FFT branches' signal).
        faces: list = [None] * len(imgs)
        by_size: dict[tuple[int, int], list[int]] = {}
        for idx, im in enumerate(imgs):
            by_size.setdefault(im.size, []).append(idx)
        for idxs in by_size.values():
            group = [imgs[i] for i in idxs]
            group_faces = mtcnn(group) if len(group) > 1 else [mtcnn(group[0])]
            for i, f in zip(idxs, group_faces):
                faces[i] = f
        for p, face in zip(valid_paths, faces):
            if face is None:
                continue
            arr = face.clamp(0, 255).byte().permute(1, 2, 0).numpy()
            results.append((p, Image.fromarray(arr)))
            detected += 1
        pbar.update(len(batch_paths))
        pbar.set_postfix(detected=detected, rejected=seen - detected)
    pbar.close()
    rejected = seen - detected
    pct = 100 * rejected / max(seen, 1)
    print(f"[{class_name}] seen: {seen}, faces detected: {detected}, rejected: {rejected} ({pct:.1f}%)")
    return results


def pair_key(class_name: str, path: Path) -> str:
    """real/deepfake share filenames from download.py's paired COCO_AI export --
    group by that shared stem so a pair always lands in the same split."""
    return f"pair:{path.stem}" if class_name in ("real", "deepfake") else f"single:{path.stem}"


def stratified_split(items: list[tuple[str, tuple]], seed: int, val_fraction: float):
    groups: dict[str, list] = {}
    for key, payload in items:
        groups.setdefault(key, []).append(payload)
    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val_groups = max(1, int(len(keys) * val_fraction)) if keys else 0
    val_keys = set(keys[:n_val_groups])
    train, val = [], []
    for key in keys:
        (val if key in val_keys else train).extend(groups[key])
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=FACE_MARGIN,
        select_largest=True,  # one crop per source image: keep only the largest face
        keep_all=False,
        post_process=False,  # keep raw 0-255 pixels for saving to disk
        device=DEVICE,
    )

    for split in ("train", "val"):
        for cls in CLASSES:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    balance_counts: Counter = Counter()

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "split", "source_dataset"])

        for cls in CLASSES:
            src_paths = list_source_images(cls)
            if not src_paths:
                print(f"WARNING: no source images found for class '{cls}' -- did data/download.py run?")
                continue

            cropped = detect_and_crop(mtcnn, src_paths, cls)

            if cls == "deepfake" and len(cropped) < DEEPFAKE_SURVIVAL_FLOOR:
                raise SystemExit(
                    f"STOP: only {len(cropped)} deepfake faces survived filtering "
                    f"(floor: {DEEPFAKE_SURVIVAL_FLOOR}). Too few to train on -- revisit "
                    f"the deepfake source choice (see architecture_decisions.md's "
                    f"accepted-tradeoffs section) before building the rest of the pipeline."
                )

            items = [(pair_key(cls, p), (p, img)) for p, img in cropped]
            train_items, val_items = stratified_split(items, seed=SEED, val_fraction=args.val_fraction)

            for split_name, split_items in (("train", train_items), ("val", val_items)):
                for i, (src_path, img) in enumerate(
                    tqdm(split_items, desc=f"[{cls}] saving {split_name}", unit="img", leave=False)
                ):
                    out_name = f"{cls}_{src_path.stem}_{i:05d}.jpg"
                    out_path = DATASET_DIR / split_name / cls / out_name
                    img.save(out_path, quality=95)
                    rel_path = out_path.relative_to(DATASET_DIR).as_posix()
                    writer.writerow([rel_path, cls, split_name, SOURCE_NAME[cls]])
                    balance_counts[(cls, split_name)] += 1
            f.flush()
            print(f"[{cls}] train: {len(train_items)}, val: {len(val_items)}")

    print("\n=== Class balance check ===")
    for cls in CLASSES:
        for split_name in ("train", "val"):
            print(f"  {cls:>9s} / {split_name:>5s}: {balance_counts.get((cls, split_name), 0)}")
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
