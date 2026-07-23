"""Dataset/DataLoader reading manifest.csv, producing the three per-sample
model inputs (RGB, FFT-magnitude, SRM residual). All three computed per-item,
not precomputed/cached -- see model_code.md section 1."""
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, CLASS_TO_IDX, DATASET_DIR, IMAGE_SIZE, MANIFEST_PATH
from model.branches import SRMFilter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ForgeryDataset(Dataset):
    def __init__(self, split: str, manifest_path: Path = MANIFEST_PATH, dataset_dir: Path = DATASET_DIR):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")
        self.dataset_dir = dataset_dir
        self.to_tensor = transforms.ToTensor()  # PIL -> (3,H,W) float in [0,1]
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.srm = SRMFilter()

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _fft_magnitude(raw01: torch.Tensor) -> torch.Tensor:
        gray = raw01.mean(dim=0, keepdim=True)
        mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(gray)))
        log_mag = torch.log1p(mag)
        lo, hi = log_mag.min(), log_mag.max()
        return (log_mag - lo) / (hi - lo + 1e-8)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = self.dataset_dir / row["path"]
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        raw01 = self.to_tensor(img)

        return {
            "rgb": self.normalize(raw01.clone()),
            "fft_mag": self._fft_magnitude(raw01),
            "srm_residual": self.srm(raw01 * 255.0),
            "label": CLASS_TO_IDX[row["class"]],
            "path": str(img_path),
        }


def get_dataloader(split: str, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    dataset = ForgeryDataset(split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def compute_class_weights(manifest_path: Path = MANIFEST_PATH, split: str = "train") -> torch.Tensor:
    """Inverse-frequency class weights from actual manifest counts -- never
    hardcoded, per model_code.md section 4."""
    df = pd.read_csv(manifest_path)
    counts = df[df["split"] == split]["class"].value_counts()
    total = counts.sum()
    weights = [total / (len(CLASSES) * counts.get(c, 1)) for c in CLASSES]
    return torch.tensor(weights, dtype=torch.float32)
