# Forgery Classifier: Real / Edited / Deepfake

3-class image classifier distinguishing **real** photos, **human-edited** images (Photoshop/splicing/retouching), and **AI-generated deepfakes** (diffusion output), with per-class explainability signals rather than a single opaque logit.

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

One dataset per class (see `architecture_decisions.md`'s Dataset section for why):

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

- `config.py` — shared paths/constants for data and model code
- `data/download.py`, `data/face_filter.py` — dataset download + face-detect/crop pipeline
- `model/branches.py`, `model/fusion.py`, `model/dataset.py`, `model/train.py`, `model/eval.py` — model, training, and eval code
- `forgery_classifier.ipynb` — Colab notebook driving Data/Train/Eval end to end
- `architecture_decisions.md`, `data_download.md`, `model_code.md` — design docs behind the code
- `deepfake_detection_research.md` — background research

## GPU

<!-- TODO: placeholder — fill in with the GPU tier actually assigned for the training run and any tier-specific config used (see the GPU-tier table in architecture_decisions.md: T4/L4/A100 fallback configs). Reference build targeted an A100 40GB. -->

## Time spent

<!-- TODO: placeholder — log actual wall-clock time spent per phase (prep/data, model+training, eval) and compare against the budgets in data_download.md and model_code.md. -->

## Gaussian usage

<!-- TODO: placeholder — no Gaussian noise/blur augmentation currently implemented in the codebase. Document here if/when added (e.g. Gaussian noise augmentation for spectral-branch robustness, or Gaussian blur as a splice-boundary softening augmentation for the edited class). -->

## Future work (unconstrained)

Ideas beyond the current 1.5hr/single-Colab-session scope (see "Open items carried forward" in `architecture_decisions.md`):

- A 4th generalization branch (DINOv2 or CLIP-ViT), contingent on eval showing a cross-generator generalization gap the current 3 branches don't close
- Training signal from more than one diffusion generator (currently DALL-E 3 only) — SynthBuster's other generator slices (Midjourney, SD) are held out as eval-only
- Face-swap/reenactment deepfake representation (FF++/Celeb-DF), deliberately dropped for v1 since the eval target is full-image diffusion generation, not GAN face-swap

<!-- TODO: placeholder — expand with any other decisions that were made under the time/compute constraint and would be revisited given unlimited time/compute. -->

## To-dos

- [x] decide the execution environment -> Colab (300 compute units)
- [x] understand the dataset -> real / edited / deepfake, one source dataset per class
- [x] check SOTA for deepfake detection on real, deepfake, and edited
- [x] finalize a model (see `architecture_decisions.md` for reasoning)
- [x] decide metrics and why (macro-F1, per-class precision/recall, ROC-AUC — see `model_code.md`)
- [x] decide loss function (class-weighted cross-entropy)
- [x] find data
- [ ] set up model, data, metrics and start training
- [ ] incorporate future decisions made without constraints into this README (see "Future work" above)
- [ ] write up Gaussian usage in this README (see "Gaussian usage" above)
- [ ] include GPU info in this README (see "GPU" above)
- [ ] log time spent in this README (see "Time spent" above)

## Notes

- `edited` means human-edited (Photoshop, splicing, filtering, etc.); AI-based editing is classified as `deepfake`.
