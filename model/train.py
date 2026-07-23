"""Training loop: class-weighted CE, AdamW + warmup-cosine schedule, AMP,
early stopping on macro-F1, per-class precision/recall logged every epoch."""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, CLASSES, DEVICE, EMBED_DIM
from model.dataset import compute_class_weights, get_dataloader
from model.fusion import ForgeryClassifier


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.05) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        fft_mag = batch["fft_mag"].to(device, non_blocking=True)
        srm = batch["srm_residual"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
            logits, _ = model(rgb, fft_mag, srm)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(batch["label"])
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASSES))), zero_division=0
    )
    return macro_f1, precision, recall


def train(
    epochs: int = 18,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    patience: int = 5,
    checkpoint_name: str = "best_model.pt",
):
    device = DEVICE
    print(f"Training on device: {device}")

    train_loader = get_dataloader("train", batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_dataloader("val", batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_weights = compute_class_weights(split="train").to(device)
    print(f"Class weights ({CLASSES}): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = ForgeryClassifier(embed_dim=EMBED_DIM, num_classes=len(CLASSES)).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = make_warmup_cosine_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device == "cuda"))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    best_path = CHECKPOINT_DIR / checkpoint_name

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            fft_mag = batch["fft_mag"].to(device, non_blocking=True)
            srm = batch["srm_residual"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                logits, _ = model(rgb, fft_mag, srm)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * rgb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        macro_f1, precision, recall = evaluate(model, val_loader, device)

        print(
            f"\nEpoch {epoch}/{epochs} ({time.time() - epoch_start:.1f}s) "
            f"train_loss={train_loss:.4f} val_macro_f1={macro_f1:.4f}"
        )
        for i, cls in enumerate(CLASSES):
            print(f"  {cls:>9s}: precision={precision[i]:.3f} recall={recall[i]:.3f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "macro_f1": macro_f1},
                best_path,
            )
            print(f"  -> new best macro-F1 {macro_f1:.4f}, saved to {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no macro-F1 improvement in {patience} epochs)")
                break

    print(f"\nTraining done. Best val macro-F1: {best_macro_f1:.4f} ({best_path})")
    return best_path, best_macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
