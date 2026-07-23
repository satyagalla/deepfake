# Forgery Classifier: Real / Edited / Deepfake

3-class image classifier distinguishing **real** photos, **human-edited** images (Photoshop/splicing/retouching), and **AI-generated deepfakes** (diffusion output), with per-class explainability signals rather than a single opaque logit.

## Main architecture decisions

- **EfficientNet-B4 over Xception.** Both are spatial-texture backbones — same evidence type, so running both would be two opinions on one signal rather than diverse evidence. EfficientNet's compound scaling gives a better accuracy/parameter tradeoff at this compute budget, and it's flagged as the cheapest strong option in `deepfake_detection_research.md`. Full comparison (incl. why not ResNet-50, CLIP-ViT, DINOv2) in `architecture_decisions.md` → Branch 1.
- **Why a 3-stream pipeline, not 1, 2, or 4+.** 1 branch (RGB-only) is a single opaque logit — fails the explainability requirement and risks learning dataset fingerprints instead of manipulation cues. 2 branches (spatial + spectral) leaves `edited` with no branch built for its actual failure mode: a splice doesn't disturb global frequency statistics much, so it needs the noise-residual branch to catch it. 4+ branches has diminishing returns and makes the gate's output harder to read. 3 branches map cleanly onto 3 manipulation types — structural, generative-frequency, boundary-splice. See `architecture_decisions.md` → "Why 3 signal branches."

See `deepfake_detection_research.md` for the SOTA survey and `architecture_decisions.md` for the finalized architecture with full reasoning.

## Architecture

3-branch fusion model feeding a gated classifier head:

```
RGB image (~380x380)
   |-- EfficientNet-B4 (full fine-tune)        --> spatial embedding
   |-- FFT-magnitude (log-scaled)  -> small CNN --> spectral embedding
   |-- SRM high-pass residual      -> small CNN --> noise-residual embedding
                                          |
                          concat -> gating/attention MLP
                                          |
                  +------------------------+------------------------+
             3-way softmax                          per-branch contribution weights
        (real / edited / deepfake)                 (spatial / spectral / noise-residual
                                                      -- the explainability signal)
```

Grad-CAM on the spatial branch adds a visual "where" heatmap alongside the gate's "which evidence type" weighting. Full reasoning for each branch choice, the fusion mechanism, and discarded alternatives is in `architecture_decisions.md`.

## Dataset

One dataset per class (see `architecture_decisions.md` → Dataset for why):

| Class | Source |
|---|---|
| Real | Pristine images paired 1:1 with the DALL-E 3 slice below |
| Edited | CASIA v2.0 tampered set |
| Deepfake | DALL-E 3 slice of COCO_AI/SynthBuster |

All three classes are face-detected and cropped uniformly (MTCNN via `facenet_pytorch`) — see `data_download.md` for the full pipeline.

## Environment

```
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

CUDA is used automatically when available (`DEVICE` in `config.py`).

## Repo layout

| Path | Contents |
|---|---|
| `config.py` | Shared paths/constants for data and model code |
| `data/download.py`, `data/face_filter.py` | Dataset download + face-detect/crop pipeline |
| `model/branches.py`, `model/fusion.py`, `model/dataset.py`, `model/train.py`, `model/eval.py` | Model, training, and eval code |
| `forgery_classifier.ipynb` | Colab notebook driving Data/Train/Eval end to end |
| `architecture_decisions.md`, `data_download.md`, `model_code.md` | Design docs behind the code |
| `deepfake_detection_research.md` | Background research |

## GPU

1x NVIDIA A100 (40GB VRAM), Google Colab Pro.

## Time spent

| Window | Activity |
|---|---|
| 2:00 – 2:30 PM | Setup, understanding the problem statement |
| 2:30 – 3:00 PM | Domain research, SOTA approaches |
| 3:00 – 4:00 PM | Architecture decisions |
| 4:00 – 4:30 PM | Scaffolding code files |
| 4:30 – 5:00 PM | Squashing dependency and code bugs |
| 5:00 – 6:00 PM | Downloading + processing data — halted on the deepfake class: only 197 faces survived MTCNN filtering against a 300 floor (`data/face_filter.py`'s `DEEPFAKE_SURVIVAL_FLOOR` check). Training hasn't started as a result. |

## Future work (unconstrained)

Beyond the current single 4-hour session:

**Data**
- Fix the deepfake-class shortfall properly, not just retroactively — source a larger and/or face-specific diffusion dataset rather than a general-purpose slice that loses most of its hardest (most malformed, most informative) examples to MTCNN's landmark-detection filter.
- Train on more than one diffusion generator. Currently DALL-E 3 only; hold out SynthBuster's other generator slices (Midjourney, Stable Diffusion) as eval-only to actually measure the cross-generator generalization gap before deciding whether it needs closing.
- Add face-swap/reenactment coverage (FaceForensics++, Celeb-DF) if the real-world threat model includes GAN-based face swaps and not just full-image diffusion generation — deliberately dropped for v1 since the stated eval target is diffusion output.

**Model**
- Add DINOv2 or CLIP-ViT as a 4th "generalization" branch — contingent on eval actually showing a cross-generator gap the current 3 branches don't close, not worth the added compute/complexity speculatively.
- Revisit EfficientNet-B4 vs. a ViT-based spatial backbone once there's more than a single hackathon session to fine-tune one properly.

**Pipeline breadth**
- Metadata forensics as a cheap first-pass filter — EXIF, dimensions, compression, timestamps — ahead of the pixel branch. Spoofable and increasingly unavailable on stripped upload pipelines, but free, so worth keeping as a first signal rather than the primary one.
- Post-training design: what breaks once this ships against real KYC traffic — adversarial recompression/cropping to evade the noise-residual branch, new generators the spectral branch hasn't seen, drift in what "edited" looks like — and how a post-training loop would close each gap.

## To-dos

- [x] decide the execution environment -> Colab (300 compute units)
- [x] understand the dataset -> real / edited / deepfake, one source dataset per class
- [x] check SOTA for deepfake detection on real, deepfake, and edited
- [x] finalize a model (see `architecture_decisions.md` for reasoning)
- [x] decide metrics and why (macro-F1, per-class precision/recall, ROC-AUC — see `model_code.md`)
- [x] decide loss function (class-weighted cross-entropy)
- [x] find data
- [x] set up model, data, metrics pipeline
- [ ] start training — blocked: deepfake class under the 300-face floor after filtering (197 survived, see Time spent / Future work)
- [ ] fold prioritized Future work items back into this README as they're picked up
- [ ] document Gaussian noise/blur augmentation status — none implemented yet; note here if/when added (e.g. noise augmentation for spectral-branch robustness, or blur to soften splice boundaries for the edited class)
- [x] include GPU info in this README (see "GPU" above)
- [x] log time spent in this README (see "Time spent" above)

## Notes

- `edited` means human-edited (Photoshop, splicing, filtering, etc.); AI-based editing is classified as `deepfake`.
