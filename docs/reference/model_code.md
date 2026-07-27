# Model Implementation — Instructions

Implements the 3-branch fusion architecture from `architecture_decisions.md`. Reads from the `train/val` folders and `manifest.csv` produced per `data_download.md`.

## File layout

- `model/branches.py` — the spatial, spectral, and noise-residual branch modules.
- `model/fusion.py` — the gating/fusion module and 3-way classifier head.
- `model/dataset.py` — `Dataset`/`DataLoader` reading the manifest, producing the three per-sample inputs (RGB, FFT-magnitude, SRM residual).
- `model/train.py` — training loop, loss, optimizer, checkpointing.
- `model/eval.py` — metrics, confusion matrix, Grad-CAM + gate-weight dump.
- Colab notebook sections "Model," "Train," "Eval" — import from the above and drive execution; keep the notebook thin, logic lives in the `.py` files so it's testable outside Colab too.

## 0. GPU-tier check first

Confirm the assigned GPU immediately. Everything below assumes the A100 40GB path from `architecture_decisions.md`'s GPU-tier table. If a T4 or L4 is assigned instead, switch to that table's fallback config (smaller backbone, partial freeze, stacked-channel input) rather than attempting the full 3-branch version under time pressure.

## 1. Per-sample inputs

Each sample is one 380x380 face crop, from which three parallel inputs are derived:
- **RGB** — the crop itself, ImageNet-normalized (matches EfficientNet-B4's pretraining stats).
- **FFT-magnitude** — 2D FFT of the grayscale crop, magnitude, log-scaled, normalized to a fixed range. Single channel.
- **SRM residual** — a small bank of fixed, non-learned high-pass filters (standard steganalysis-rich-model kernels from the forensics literature) applied as a convolution with frozen weights. Produces a multi-channel residual map at the same spatial size as the input.

Compute all three inside the dataset's per-item logic. Only precompute and cache SRM/FFT ahead of time if the dataloader turns out to be a bottleneck during the dry run — don't add that complexity preemptively.

## 2. Branch modules

**Spatial branch:** EfficientNet-B4, ImageNet-pretrained, full fine-tune (no frozen layers at the A100 tier), pooled feature output projected to a shared embedding dimension. Keep a hook into the backbone's last convolutional feature map from the start — it's needed for Grad-CAM in the eval stage, and retrofitting it later means re-running training.

**Spectral branch and noise-residual branch:** each a small CNN trained from scratch (roughly 4 conv-batchnorm-relu blocks with stride-2 downsampling, global average pooling, then a linear projection to the *same* embedding dimension as the spatial branch, so all three can be concatenated). The noise-residual branch's input already comes from the fixed SRM filter bank (section 1) — this CNN just learns on top of that fixed residual, the filter bank itself is never trained.

## 3. Fusion gate

Concatenate the three branch embeddings. Feed the concatenation through two small heads:
- A **gate head** (small MLP ending in a softmax over 3 values) producing per-branch contribution weights — this is a first-class output, not a diagnostic bolted on after the fact. Surface it alongside every prediction.
- A **classifier head** (linear layer) producing the 3-way real/edited/deepfake logits.

Both heads read the same concatenated embedding; they are two separate output projections, not a pipeline where one feeds the other.

## 4. Loss & optimizer

- **Loss:** cross-entropy, class-weighted using weights derived from the actual per-class counts in `manifest.csv` (inverse frequency, normalized) — never hardcode assumed proportions, `edited` and `deepfake` are expected to be much smaller than `real`.
- **Optimizer:** AdamW with a small weight decay.
- **Schedule:** cosine annealing over the planned epoch count, with a brief linear warmup at the start (roughly the first 5% of training steps).
- **Precision:** mixed precision (AMP) is mandatory at this time budget, not optional.
- **Batch size:** start at 64-96 on the A100; drop to 32 on OOM before considering a resolution reduction.

## 5. Training loop

- 15-20 epochs, early stopping tracked on **macro-F1**, not accuracy — accuracy hides poor performance on the smaller `edited` class.
- Evaluate on the val split every epoch; keep the checkpoint with the best macro-F1 seen so far, not simply the final epoch's weights.
- Log per-class precision/recall every epoch — cheap to compute, and catches a collapsing/ignored class early instead of only discovering it at the final confusion matrix.

## 6. Evaluation deliverables

- Macro-F1, per-class precision/recall, per-class one-vs-rest ROC-AUC.
- Full 3x3 confusion matrix, with explicit attention to the edited↔deepfake cell — that's the one novel failure mode this task setup introduces, and it's the cell most likely to be misleadingly hidden by an overall-accuracy number.
- Grad-CAM on the spatial branch's last conv feature map for a handful of validation samples per class.
- The gate's per-branch contribution weights for those same samples, reported alongside the Grad-CAM heatmap — the paired output (heatmap + branch-contribution percentages) is the explainability deliverable described in `architecture_decisions.md`'s overview diagram; don't report one without the other.

## Time budget checkpoints (A100 path, from architecture_decisions.md)

| Stage | Target |
|---|---|
| Model instantiation + a single dry-run batch through the full pipeline (catch shape/dimension errors here) | ~5 min |
| Training (15-20 epochs) | ~40-55 min |
| Validation + confusion matrix + macro-F1 | ~5 min |
| Grad-CAM + gate-weight dump | ~10 min |
| Buffer | ~10-20 min |

Run the dry-run batch before committing to the full training loop — a shape mismatch between the three branch embeddings, or a dataloader bug, costs 5 minutes if caught at step 1 and costs the entire session if only discovered after 20 epochs.
