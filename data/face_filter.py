"""Uniform MTCNN face detect+crop across all raw sources, plus a
source-image-level stratified train/val split and manifest.csv.

Run after data/download.py has populated data_raw/{real_src,deepfake_src,
edited_src,edited_src_psbattles}.
"""
import argparse
import csv
import random
import sys
import time
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
    EDITED_SRC_PSBATTLES,
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


def find_casia_tampered(root: Path) -> list[Path]:
    """CASIA v2.0 marks tampered images with a 'Tp_' filename prefix (vs 'Au_'
    for authentic) regardless of the exact folder layout a mirror unzips to."""
    if not root.exists():
        return []
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    tampered = [p for p in all_imgs if p.name.lower().startswith("tp_")]
    if not tampered:
        tampered = [p for p in all_imgs if "tp" in p.parent.name.lower() and "au" not in p.parent.name.lower()]
    print(f"CASIA: {len(all_imgs)} images found under {root}, {len(tampered)} identified as tampered (edited).")
    return tampered


def classify_ps_battles_path(p: Path) -> str | None:
    """Folder-name heuristic for the unverified timocasti/psbattles Kaggle
    layout: "derivative" (Photoshopped, what we want for `edited`),
    "original" (ignored, same reason CASIA's authentic set is ignored), or
    None if neither signal is present. Shared between find_ps_battles_derivatives
    (checked against real downloaded files) and download.py's test_sources()
    (checked against Kaggle's file listing before downloading anything) so
    both use the exact same rule."""
    parts = [s.lower() for s in p.parts]
    if any("photoshop" in s or "derivativ" in s or "manipulat" in s for s in parts):
        return "derivative"
    if any("original" in s for s in parts):
        return "original"
    return None


def find_ps_battles_derivatives(root: Path) -> list[Path]:
    """PS-Battles ships an originals/derivatives split (originals.tsv vs.
    photoshops.tsv on the source repo). The on-disk layout of the
    timocasti/psbattles Kaggle mirror this project downloads from hasn't
    been independently inspected, so this looks for a folder-name signal
    and REFUSES to guess if it can't find one -- silently treating all
    ~103k images as "edited" would mislabel ~11k pristine originals.
    If this raises, inspect the actual unzipped structure and adjust
    classify_ps_battles_path() above. Run `python data/download.py --test`
    first to check this against Kaggle's file listing before downloading."""
    if not root.exists():
        return []
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not all_imgs:
        return []

    labeled = [(p, classify_ps_battles_path(p)) for p in all_imgs]
    derivatives = [p for p, lab in labeled if lab == "derivative"]
    originals = [p for p, lab in labeled if lab == "original"]
    if derivatives:
        print(
            f"PS-Battles: {len(all_imgs)} images found under {root}, {len(derivatives)} identified "
            f"as derivatives (edited), {len(originals)} as originals (ignored)."
        )
        return derivatives

    raise RuntimeError(
        f"PS-Battles: could not distinguish original vs. derivative images under {root} by folder "
        f"naming (inspect it, e.g. list a few subfolders/files under {root}, and adjust the label() "
        "heuristic in find_ps_battles_derivatives()). Refusing to guess -- silently including "
        "originals would mislabel pristine images as edited."
    )


def list_source_images(class_name: str) -> list[tuple[Path, str]]:
    """Returns (path, source_dataset_name) pairs -- edited is multi-source
    (CASIA + PS-Battles) for manipulation-technique diversity; real/deepfake
    stay single-source (COCO_AI) since they must be paired, not diverse."""
    if class_name == "real":
        return [(p, "COCO_AI") for p in sorted(REAL_SRC.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if class_name == "deepfake":
        return [(p, "COCO_AI") for p in sorted(DEEPFAKE_SRC.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if class_name == "edited":
        casia = [(p, "CASIA_v2.0") for p in find_casia_tampered(EDITED_SRC)]
        psb = [(p, "PS-Battles") for p in find_ps_battles_derivatives(EDITED_SRC_PSBATTLES)]
        return sorted(casia + psb, key=lambda t: t[0])
    raise ValueError(class_name)


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def sort_by_dimensions(paths: list[Path]) -> list[Path]:
    """mtcnn(list) only batches images that share identical pixel
    dimensions (see detect_and_crop) -- within an unsorted file order,
    same-size images rarely land in the same BATCH_SIZE window (diagnostic
    run measured 74% of size-groups falling back to one-at-a-time for the
    varied-resolution `real` class, vs. 0% for uniform-resolution
    `deepfake`, and `real` ran at ~0.6x deepfake's img/s as a direct
    result). Sorting by (width, height) first -- a cheap header-only read,
    no pixel decode -- clusters same-size images together so batching
    actually triggers, without changing which images get processed."""
    sized = []
    for p in paths:
        try:
            with Image.open(p) as im:
                sized.append((im.size, p))
        except Exception:
            sized.append(((0, 0), p))  # unreadable -- let detect_and_crop's open() report it
    sized.sort(key=lambda t: t[0])
    return [p for _, p in sized]


def detect_and_crop(mtcnn: MTCNN, paths: list[Path], class_name: str) -> list[tuple[Path, Image.Image]]:
    results = []
    seen = detected = 0
    t_decode = t_infer = t_pack = 0.0
    group_size_counts: Counter = Counter()
    paths = sort_by_dimensions(paths)
    pbar = tqdm(total=len(paths), desc=f"[{class_name}] detect+crop", unit="img")
    for batch_paths in batched(paths, BATCH_SIZE):
        imgs, valid_paths = [], []
        t0 = time.perf_counter()
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                print(f"  [{class_name}] skip unreadable file {p}: {e}")
        t_decode += time.perf_counter() - t0
        if not imgs:
            pbar.update(len(batch_paths))
            continue
        seen += len(imgs)
        # mtcnn(list) requires every image in the call to share identical
        # pixel dimensions -- group by actual size so same-size images still
        # go through as one batched GPU call. facenet-pytorch's batched path
        # (select_boxes) builds a numpy array over per-image results and
        # crashes with "inhomogeneous shape" on NumPy >= 1.24 whenever a
        # batch mixes images with and without a detected face -- fall back
        # to one-at-a-time for just that group when it happens.
        faces: list = [None] * len(imgs)
        by_size: dict[tuple[int, int], list[int]] = {}
        for idx, im in enumerate(imgs):
            by_size.setdefault(im.size, []).append(idx)
        for idxs in by_size.values():
            group_size_counts[len(idxs)] += 1
            group = [imgs[i] for i in idxs]
            t1 = time.perf_counter()
            if len(group) == 1:
                group_faces = [mtcnn(group[0])]
            else:
                try:
                    group_faces = mtcnn(group)
                except ValueError:
                    group_faces = [mtcnn(im) for im in group]
            t_infer += time.perf_counter() - t1
            for i, f in zip(idxs, group_faces):
                faces[i] = f
        t2 = time.perf_counter()
        for p, face in zip(valid_paths, faces):
            if face is None:
                continue
            arr = face.clamp(0, 255).byte().permute(1, 2, 0).numpy()
            results.append((p, Image.fromarray(arr)))
            detected += 1
        t_pack += time.perf_counter() - t2
        pbar.update(len(batch_paths))
        pbar.set_postfix(detected=detected, rejected=seen - detected)
    pbar.close()
    rejected = seen - detected
    pct = 100 * rejected / max(seen, 1)
    print(f"[{class_name}] seen: {seen}, faces detected: {detected}, rejected: {rejected} ({pct:.1f}%)")

    total_groups = sum(group_size_counts.values())
    singleton_groups = group_size_counts.get(1, 0)
    singleton_pct = 100 * singleton_groups / max(total_groups, 1)
    print(
        f"[{class_name}] timing -- decode: {t_decode:.1f}s, mtcnn: {t_infer:.1f}s, "
        f"pack: {t_pack:.1f}s (save time reported separately in main)"
    )
    print(
        f"[{class_name}] GPU-batch fragmentation -- {singleton_groups}/{total_groups} "
        f"size-groups ({singleton_pct:.1f}%) ran as singletons (no batching); "
        f"group-size histogram: {dict(sorted(group_size_counts.items()))}"
    )
    return results


def pair_key(class_name: str, path: Path, source: str) -> str:
    """real/deepfake share filenames from download.py's paired COCO_AI export --
    group by that shared stem so a pair always lands in the same split.
    edited is multi-source (CASIA + PS-Battles) now, so key by source too --
    otherwise two different sources reusing the same stem (e.g. both writing
    "00001.jpg") would incorrectly get treated as one group."""
    if class_name in ("real", "deepfake"):
        return f"pair:{path.stem}"
    return f"single:{source}:{path.stem}"


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Diagnostic mode: cap source images per class to this many, skip the "
            "deepfake survival-floor check, and write output to a sibling "
            "'dataset_diag' dir instead of the real dataset dir -- for quickly "
            "timing the decode/mtcnn/save phases on a small slice."
        ),
    )
    args = parser.parse_args()
    diag = args.limit is not None
    out_dir = (DATASET_DIR.parent / "dataset_diag") if diag else DATASET_DIR
    manifest_path = (out_dir / "manifest.csv") if diag else MANIFEST_PATH

    print(f"Using device: {DEVICE}")
    if diag:
        print(f"[diag mode] limit={args.limit} images/class, writing to {out_dir}")
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
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    balance_counts: Counter = Counter()
    t_save_total = 0.0

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "split", "source_dataset"])

        # deepfake first: it's the class most likely to trip the survival
        # floor below, so a doomed run fails in minutes instead of after
        # also paying for CASIA's much larger edited-class pass.
        for cls in sorted(CLASSES, key=lambda c: 0 if c == "deepfake" else 1):
            src_items = list_source_images(cls)
            if not src_items:
                print(f"WARNING: no source images found for class '{cls}' -- did data/download.py run?")
                continue
            source_of = {p: s for p, s in src_items}
            src_paths = [p for p, _ in src_items]
            if diag and len(src_paths) > args.limit:
                # random.sample, not a prefix slice -- CASIA's paths come out of
                # rglob+sorted (grouped by folder/tamper-type, not shuffled), so a
                # prefix can land entirely inside one non-face category and read as
                # a false "0% survival" that's really a sampling artifact.
                src_paths = random.Random(SEED).sample(src_paths, args.limit)

            cropped = detect_and_crop(mtcnn, src_paths, cls)

            if cls == "deepfake" and not diag and len(cropped) < DEEPFAKE_SURVIVAL_FLOOR:
                raise SystemExit(
                    f"STOP: only {len(cropped)} deepfake faces survived filtering "
                    f"(floor: {DEEPFAKE_SURVIVAL_FLOOR}). Too few to train on -- revisit "
                    f"the deepfake source choice (see architecture_decisions.md's "
                    f"accepted-tradeoffs section) before building the rest of the pipeline."
                )

            items = [(pair_key(cls, p, source_of[p]), (p, img)) for p, img in cropped]
            train_items, val_items = stratified_split(items, seed=SEED, val_fraction=args.val_fraction)

            t_save_cls = 0.0
            for split_name, split_items in (("train", train_items), ("val", val_items)):
                for i, (src_path, img) in enumerate(
                    tqdm(split_items, desc=f"[{cls}] saving {split_name}", unit="img", leave=False)
                ):
                    out_name = f"{cls}_{src_path.stem}_{i:05d}.jpg"
                    out_path = out_dir / split_name / cls / out_name
                    t0 = time.perf_counter()
                    img.save(out_path, quality=95)
                    t_save_cls += time.perf_counter() - t0
                    rel_path = out_path.relative_to(out_dir).as_posix()
                    writer.writerow([rel_path, cls, split_name, source_of[src_path]])
                    balance_counts[(cls, split_name)] += 1
            f.flush()
            t_save_total += t_save_cls
            print(f"[{cls}] train: {len(train_items)}, val: {len(val_items)}, save time: {t_save_cls:.1f}s")

    print("\n=== Class balance check ===")
    for cls in CLASSES:
        for split_name in ("train", "val"):
            print(f"  {cls:>9s} / {split_name:>5s}: {balance_counts.get((cls, split_name), 0)}")
    print(f"\nTotal save (JPEG encode + write) time: {t_save_total:.1f}s")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
