"""Gated fusion: concatenate the 3 branch embeddings, feed to two independent
heads -- a gate (per-branch contribution, softmax over 3) and a classifier
(3-way real/edited/deepfake logits). Both read the same concatenation."""
import torch
import torch.nn as nn

from model.branches import NoiseResidualBranch, SpatialBranch, SpectralBranch

BRANCH_NAMES = ["spatial", "spectral", "noise_residual"]


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 3, num_branches: int = 3, gate_hidden: int = 128):
        super().__init__()
        concat_dim = embed_dim * num_branches
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, num_branches),
        )
        self.classifier = nn.Linear(concat_dim, num_classes)

    def forward(self, spatial_embed: torch.Tensor, spectral_embed: torch.Tensor, noise_embed: torch.Tensor):
        concat = torch.cat([spatial_embed, spectral_embed, noise_embed], dim=1)
        gate_weights = torch.softmax(self.gate(concat), dim=1)  # (B, 3) -- spatial/spectral/noise contribution
        logits = self.classifier(concat)
        return logits, gate_weights


class ForgeryClassifier(nn.Module):
    """3-branch fusion model: EfficientNet-B4 spatial + FFT spectral CNN +
    SRM noise-residual CNN -> gated fusion -> 3-way softmax + gate weights."""

    def __init__(self, embed_dim: int = 256, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.spatial = SpatialBranch(embed_dim, pretrained=pretrained)
        self.spectral = SpectralBranch(embed_dim)
        self.noise = NoiseResidualBranch(embed_dim)
        self.fusion = GatedFusion(embed_dim, num_classes=num_classes)

    def forward(self, rgb: torch.Tensor, fft_mag: torch.Tensor, srm_residual: torch.Tensor):
        spatial_embed = self.spatial(rgb)
        spectral_embed = self.spectral(fft_mag)
        noise_embed = self.noise(srm_residual)
        logits, gate_weights = self.fusion(spatial_embed, spectral_embed, noise_embed)
        return logits, gate_weights

    def enable_gradcam(self, enabled: bool = True) -> None:
        self.spatial.enable_gradcam(enabled)
