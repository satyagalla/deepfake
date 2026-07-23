"""Spatial (EfficientNet-B4), spectral (small CNN), and noise-residual
(SRM + small CNN) branches. Each projects to the same EMBED_DIM so fusion.py
can concatenate them.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed SRM (Steganalysis Rich Model) high-pass kernels -- the compact 3-filter
# bank from Zhou et al., "Learning Rich Features for Image Manipulation
# Detection" (CVPR 2018). Never trained; SRMFilter.parameters() stays empty.
_SRM_KERNELS = torch.tensor(
    [
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ],
    dtype=torch.float32,
) / torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1)


class SRMFilter(nn.Module):
    """Applies the 3-kernel SRM bank to each RGB channel independently
    (depthwise), producing a 9-channel residual map at the input's spatial
    size. Fixed weights -- this module has no trainable parameters."""

    def __init__(self):
        super().__init__()
        weight = _SRM_KERNELS.unsqueeze(1).repeat(3, 1, 1, 1)  # (9, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) or (3, H, W)
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        out = F.conv2d(x, self.weight, padding=2, groups=3)
        return out.squeeze(0) if squeeze else out


class SmallCNN(nn.Module):
    """4x conv-bn-relu, stride-2 downsampling, GAP, linear projection to embed_dim.
    Shared shape for the spectral and noise-residual branches."""

    def __init__(self, in_channels: int, embed_dim: int, base_channels: int = 32):
        super().__init__()
        channels = [in_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        pooled = self.pool(feat).flatten(1)
        return self.proj(pooled)


class SpectralBranch(SmallCNN):
    """Small CNN trained from scratch on log-scaled FFT-magnitude (1 channel)."""

    def __init__(self, embed_dim: int):
        super().__init__(in_channels=1, embed_dim=embed_dim)


class NoiseResidualBranch(SmallCNN):
    """Small CNN trained from scratch on the (already computed) 9-channel SRM
    residual -- the filter bank itself lives in SRMFilter / dataset.py, not here."""

    def __init__(self, embed_dim: int):
        super().__init__(in_channels=9, embed_dim=embed_dim)


class SpatialBranch(nn.Module):
    """EfficientNet-B4, full fine-tune, pooled + projected to embed_dim.
    Keeps the last conv feature map (pre-pool) available for Grad-CAM."""

    def __init__(self, embed_dim: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)
        self._feature_map = None
        self._gradcam_enabled = False

    def enable_gradcam(self, enabled: bool = True) -> None:
        """Call before a Grad-CAM backward pass so the feature map retains its
        gradient; leave off during normal training to avoid the extra memory."""
        self._gradcam_enabled = enabled

    @property
    def last_feature_map(self) -> torch.Tensor:
        return self._feature_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(x)  # (B, C, H', W')
        if self._gradcam_enabled:
            feat_map.retain_grad()
        self._feature_map = feat_map
        pooled = self.pool(feat_map).flatten(1)
        return self.proj(pooled)
