"""Evaluation: macro-F1, per-class precision/recall/OvR ROC-AUC, 3x3 confusion
matrix, Grad-CAM on the spatial branch + gate contribution weights (paired,
per model_code.md section 6 -- never reported separately)."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, CLASSES, DEVICE, EMBED_DIM, EVAL_DIR
from model.dataset import ForgeryDataset, get_dataloader
from model.fusion import ForgeryClassifier

BRANCH_NAMES = ["spatial", "spectral", "noise_residual"]


def load_model(checkpoint_path: str, device: str = DEVICE) -> ForgeryClassifier:
    model = ForgeryClassifier(embed_dim=EMBED_DIM, num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def compute_metrics(model: ForgeryClassifier, loader, device: str = DEVICE) -> dict:
    all_logits, all_labels, all_gates = [], [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        fft_mag = batch["fft_mag"].to(device)
        srm = batch["srm_residual"].to(device)
        logits, gate_weights = model(rgb, fft_mag, srm)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
        all_gates.append(gate_weights.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    gates = torch.cat(all_gates).numpy()  # (N, 3) -- spatial/spectral/noise_residual contribution
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    mean_gate = dict(zip(BRANCH_NAMES, gates.mean(axis=0).tolist()))
    mean_gate_per_class = {
        cls: dict(zip(BRANCH_NAMES, gates[labels == i].mean(axis=0).tolist())) if (labels == i).any() else None
        for i, cls in enumerate(CLASSES)
    }

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASSES))), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))

    roc_auc = {}
    for i, cls in enumerate(CLASSES):
        try:
            roc_auc[cls] = roc_auc_score((labels == i).astype(int), probs[:, i])
        except ValueError:
            roc_auc[cls] = float("nan")  # only one class present in this val split

    return {
        "macro_f1": macro_f1,
        "precision": dict(zip(CLASSES, precision)),
        "recall": dict(zip(CLASSES, recall)),
        "roc_auc_ovr": roc_auc,
        "confusion_matrix": cm,
        "mean_gate_weights": mean_gate,
        "mean_gate_weights_per_class": mean_gate_per_class,
    }


def print_report(metrics: dict) -> None:
    print(f"Macro-F1: {metrics['macro_f1']:.4f}\n")
    print(f"{'class':>10s} {'precision':>10s} {'recall':>10s} {'roc_auc_ovr':>12s}")
    for cls in CLASSES:
        print(f"{cls:>10s} {metrics['precision'][cls]:>10.3f} {metrics['recall'][cls]:>10.3f} {metrics['roc_auc_ovr'][cls]:>12.3f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>10s}" for c in CLASSES))
    cm = metrics["confusion_matrix"]
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>8s}" + "".join(f"{v:>10d}" for v in cm[i]))

    ei, di = CLASSES.index("edited"), CLASSES.index("deepfake")
    print(
        f"\nedited->deepfake confusions: {cm[ei, di]}  deepfake->edited confusions: {cm[di, ei]}  "
        "(the novel failure mode this 3-class setup introduces -- watch this cell, not just accuracy)"
    )

    print("\nMean gate contribution weights (per-branch share of the fused prediction):")
    print("  overall  : " + ", ".join(f"{b}={w:.3f}" for b, w in metrics["mean_gate_weights"].items()))
    for cls in CLASSES:
        per_class = metrics["mean_gate_weights_per_class"][cls]
        if per_class is None:
            print(f"  {cls:>8s} : (no val samples)")
        else:
            print(f"  {cls:>8s} : " + ", ".join(f"{b}={w:.3f}" for b, w in per_class.items()))


def save_metrics(metrics: dict, out_dir: Path = EVAL_DIR, tag: str = "val") -> Path:
    """Write metrics to <out_dir>/metrics_<tag>_<timestamp>.json (confusion
    matrix as nested list). Returns the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classes": CLASSES,
        "mean_gate_weights": metrics["mean_gate_weights"],
        "mean_gate_weights_per_class": metrics["mean_gate_weights_per_class"],
    }
    out_path = out_dir / f"metrics_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics -> {out_path}")
    return out_path


def grad_cam(model: ForgeryClassifier, rgb, fft_mag, srm_residual, target_class: int, device: str = DEVICE):
    """Grad-CAM on the spatial branch's last conv feature map for one sample.
    Returns (heatmap ndarray [H,W] in [0,1], gate_weights dict)."""
    model.eval()
    model.enable_gradcam(True)
    rgb = rgb.unsqueeze(0).to(device)
    fft_mag = fft_mag.unsqueeze(0).to(device)
    srm_residual = srm_residual.unsqueeze(0).to(device)

    logits, gate_weights = model(rgb, fft_mag, srm_residual)
    model.zero_grad(set_to_none=True)
    logits[0, target_class].backward()

    feat_map = model.spatial.last_feature_map  # (1, C, H', W'), grad retained
    weights = feat_map.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat_map).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    model.enable_gradcam(False)
    gate = dict(zip(BRANCH_NAMES, gate_weights[0].detach().cpu().tolist()))
    return cam.numpy(), gate


def explainability_dump(
    model: ForgeryClassifier,
    dataset: ForgeryDataset,
    samples_per_class: int = 3,
    device: str = DEVICE,
    save_dir: Path | None = EVAL_DIR,
):
    """Grad-CAM heatmap + gate contribution weights for a handful of val
    samples per class -- the paired explainability deliverable. If save_dir is
    set (default EVAL_DIR), writes each overlay as a PNG plus one
    explainability.json index so the dump survives a runtime recycle and any
    sample can be re-viewed later without re-running inference."""
    by_class = {cls: [] for cls in CLASSES}
    for i in range(len(dataset)):
        cls = dataset.df.iloc[i]["class"]
        if len(by_class[cls]) < samples_per_class:
            by_class[cls].append(i)

    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / "gradcam").mkdir(parents=True, exist_ok=True)

    results = []
    for cls, indices in by_class.items():
        for idx in indices:
            sample = dataset[idx]
            cam, gate = grad_cam(
                model, sample["rgb"], sample["fft_mag"], sample["srm_residual"], CLASSES.index(cls), device
            )
            record = {"path": sample["path"], "true_class": cls, "gradcam": cam, "gate_weights": gate}

            if save_dir is not None:
                overlay_path = save_dir / "gradcam" / f"{cls}_{idx}.png"
                _save_gradcam_overlay(sample["path"], cam, gate, cls, overlay_path)
                record["overlay_path"] = str(overlay_path)

            results.append(record)
            gate_str = {k: round(v, 3) for k, v in gate.items()}
            print(f"{sample['path']} [{cls}]: gate={gate_str}")

    if save_dir is not None:
        index_path = save_dir / "explainability.json"
        index = [
            {
                "path": r["path"],
                "true_class": r["true_class"],
                "gate_weights": r["gate_weights"],
                "overlay_path": r.get("overlay_path"),
            }
            for r in results
        ]
        index_path.write_text(json.dumps(index, indent=2))
        print(f"Saved explainability index -> {index_path}")

    return results


def _save_gradcam_overlay(image_path: str, cam: np.ndarray, gate: dict, true_class: str, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(cam, cmap="jet", alpha=0.4)
    gate_str = ", ".join(f"{k}:{v:.2f}" for k, v in gate.items())
    ax.set_title(f"{true_class}\n{gate_str}", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_explainability_index(save_dir: Path = EVAL_DIR) -> list[dict]:
    """Read back a previously saved explainability.json index (path, true_class,
    gate_weights, overlay_path) -- use to browse/re-visualize past runs without
    re-running inference."""
    index_path = Path(save_dir) / "explainability.json"
    return json.loads(index_path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    val_loader = get_dataloader("val", batch_size=args.batch_size, shuffle=False, num_workers=4)
    metrics = compute_metrics(model, val_loader)
    print_report(metrics)
    save_metrics(metrics)

    print("\n=== Explainability dump (Grad-CAM + gate weights) ===")
    val_dataset = ForgeryDataset("val")
    explainability_dump(model, val_dataset, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
