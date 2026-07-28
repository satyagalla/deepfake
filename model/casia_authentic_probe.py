"""CASIA authentic-image fingerprint probe: face-crop CASIA v2.0's own
untouched 'Au_' images -- data/face_filter.py's find_casia_tampered()
explicitly excludes them from the `edited` class, so they've never been
downloaded into the manifest or seen by training/eval. These are genuine,
unmanipulated photos drawn from the exact same corpus (same camera/sensor
population, same JPEG compression history) as `edited`'s CASIA half. Since
they contain no manipulation, any 'edited' prediction the model makes on them
cannot be a manipulation cue -- it can only be a corpus/compression/resolution
fingerprint that survived the face crop.

No retraining, no new downloads: reuses the exact MTCNN config from
data/face_filter.py (detect_and_crop) and the exact per-item channel
construction from model/dataset.py's ForgeryDataset, including the
quality=95 JPEG re-encode step face_filter.py applies before ever saving a
crop to disk -- so a probe prediction is built the same way a training/eval
sample is, not a rougher approximation of it.
"""
import argparse
import io
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, CLASSES, DEVICE, EDITED_SRC, EVAL_DIR, FACE_MARGIN, IMAGE_SIZE, SEED
from data.face_filter import IMG_EXTS, detect_and_crop
from model.branches import SRMFilter
from model.dataset import IMAGENET_MEAN, IMAGENET_STD, ForgeryDataset
from model.eval import load_model


def find_casia_authentic(root: Path) -> list[Path]:
    """Mirror of data/face_filter.py's find_casia_tampered, for the 'Au_'
    (authentic) half instead of 'Tp_' -- never referenced anywhere else in
    the pipeline."""
    if not root.exists():
        return []
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    authentic = [p for p in all_imgs if p.name.lower().startswith("au_")]
    if not authentic:
        authentic = [p for p in all_imgs if "au" in p.parent.name.lower() and "tp" not in p.parent.name.lower()]
    print(f"CASIA: {len(all_imgs)} images found under {root}, {len(authentic)} identified as authentic.")
    return authentic


def _jpeg_roundtrip(img: Image.Image, quality: int = 95) -> Image.Image:
    """Matches face_filter.py's img.save(out_path, quality=95) -- every
    manifest image was re-encoded once before ever reaching the model, so
    skipping this here would understate the real pipeline's JPEG history."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_channels(img: Image.Image, srm: SRMFilter, to_tensor, normalize) -> dict:
    raw01 = to_tensor(img)
    return {
        "rgb": normalize(raw01.clone()),
        "fft_mag": ForgeryDataset._fft_magnitude(raw01),
        "srm_residual": srm(raw01 * 255.0),
    }


@torch.no_grad()
def predict(model, channels: dict, device: str) -> dict:
    rgb = channels["rgb"].unsqueeze(0).to(device)
    fft_mag = channels["fft_mag"].unsqueeze(0).to(device)
    srm_residual = channels["srm_residual"].unsqueeze(0).to(device)
    logits, _gate = model(rgb, fft_mag, srm_residual)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    return {"pred": CLASSES[probs.argmax().item()], "probs": dict(zip(CLASSES, probs.tolist()))}


def run_probe(model, crops: list[tuple[Path, Image.Image]], device: str) -> list[dict]:
    srm = SRMFilter()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    results = []
    for path, face in crops:
        channels = build_channels(_jpeg_roundtrip(face), srm, to_tensor, normalize)
        pred = predict(model, channels, device)
        results.append({"path": str(path), "pred": pred["pred"], "probs": pred["probs"]})
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    counts = Counter(r["pred"] for r in results)
    mean_probs = {cls: sum(r["probs"][cls] for r in results) / n for cls in CLASSES}
    return {
        "n": n,
        "pred_counts": dict(counts),
        "pred_fraction": {cls: counts.get(cls, 0) / n for cls in CLASSES},
        "mean_probs": mean_probs,
    }


def print_summary(summary: dict) -> None:
    print(f"\n=== CASIA authentic (Au_) probe: n={summary['n']} ===")
    print(f"{'class':>10s} {'pred_count':>12s} {'pred_fraction':>14s} {'mean_prob':>12s}")
    for cls in CLASSES:
        print(
            f"{cls:>10s} {summary['pred_counts'].get(cls, 0):>12d} "
            f"{summary['pred_fraction'][cls]:>14.3f} {summary['mean_probs'][cls]:>12.3f}"
        )
    edited_rate = summary["pred_fraction"]["edited"]
    print(
        f"\n{edited_rate:.1%} of untouched, never-seen CASIA authentic images were classified 'edited'. "
        "These images contain no manipulation, so this fraction is attributable only to a corpus/"
        "compression/resolution fingerprint, not a manipulation cue -- it's the ceiling on how much of "
        "edited's real-world val performance the fingerprint alone could be explaining."
    )


def save_probe_results(results: list[dict], summary: dict, out_dir: Path = EVAL_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"casia_authentic_probe_{stamp}.json"
    out_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2))
    print(f"\nSaved probe results -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--n", type=int, default=300, help="number of CASIA authentic source images to sample")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    authentic_paths = find_casia_authentic(EDITED_SRC)
    if not authentic_paths:
        raise SystemExit(f"No CASIA authentic (Au_) images found under {EDITED_SRC}")
    sample_paths = random.Random(SEED).sample(authentic_paths, min(args.n, len(authentic_paths)))

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=FACE_MARGIN,
        select_largest=True,  # one crop per source image, same as face_filter.py
        keep_all=False,
        post_process=False,
        device=DEVICE,
    )
    crops = detect_and_crop(mtcnn, sample_paths, "casia_authentic")
    if not crops:
        raise SystemExit("No faces detected in the sampled CASIA authentic images.")
    if len(crops) < len(sample_paths):
        print(
            f"Note: {len(sample_paths) - len(crops)}/{len(sample_paths)} sampled images had no face "
            "detected and were dropped, same as the main pipeline's face-detect+crop step."
        )

    results = run_probe(model, crops, DEVICE)
    summary = summarize(results)
    print_summary(summary)
    save_probe_results(results, summary)


if __name__ == "__main__":
    main()
