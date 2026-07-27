## Project info


## Environment

Python venv at `.venv/`. Activate with `.venv\Scripts\activate` (Windows). CUDA is available and used automatically (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`).

## Subagents

```
Do NOT spawn subagents unless I explicitly ask for parallel work. Do all tasks inline.
```

Use agents only for:
- Parallel searches across unrelated areas
- Risky experiments you want isolated (worktree)
- Genuinely independent tasks that benefit from concurrent execution

Never for: sequential steps, code review of your own recent edits, single-file tasks.

## File/folder index

- `notes.md` - guidelines and to-dos for the user
- `docs/README.md` - index of all docs below, with the lifecycle convention (decisions / reference / research / investigations) explained
- `docs/decisions/001_architecture_decisions.md` - doc describing requirements and architecture decisions with reasoning, incl. finalized dataset choice
- `docs/reference/data_download.md` - instructions (no code) for downloading + face-filtering the 3 class datasets; code in `data/download.py` and `data/face_filter.py`
- `docs/reference/model_code.md` - instructions (no code) for the 3-branch fusion model, training, and eval; code in `model/*.py` + `forgery_classifier.ipynb`
- `docs/research/deepfake_detection_research.md` - deep research doc on deepfake detection models for classifying real, edited (AI generated and human) and deepfake classes
- `docs/investigations/2026-07-26-upscale-artifact.md` - investigation into severe blur seen in `docs/screenshots/demo_*.png` across all 3 classes: traces it to MTCNN's fixed bilinear crop-resize (`data/face_filter.py`/`model/demo.py`) upscaling small detected face boxes ~10-12x median across real/edited/deepfake alike, and the resulting (revised, still open) shortcut-learning hypothesis with sources
- `docs/screenshots/*.png` - live-demo screenshots (one per class) referenced by the README and the upscale investigation doc
- `config.py` - shared config for data and model paths
- `data/download.py`, `data/face_filter.py` - COCO_AI/CASIA download and face-filter pipeline
- `model/branches.py`, `model/fusion.py`, `model/dataset.py`, `model/train.py`, `model/eval.py` - 3-branch fusion model, dataset, training, and eval code
- `model/demo.py` - standalone Gradio demo (face-crop + 3-branch prediction + Grad-CAM overlay) for arbitrary uploaded images
- `forgery_classifier.ipynb` / `forgery_classifier_final.ipynb` - Colab notebook wiring together Data/Train/Eval sections; `_final` is the executed run with outputs committed
- `requirements.txt` - frozen environment dependencies

## Current State

**Done:**

- Architecture finalized: 3-branch fusion (EfficientNet-B4 spatial + FFT spectral CNN + SRM noise-residual CNN -> gated fusion), see `docs/decisions/001_architecture_decisions.md`
- Dataset finalized: one dataset per class (deepfake = DALL-E 3 slice of COCO_AI/SynthBuster, real = its paired originals, edited = CASIA v2.0), uniform face-detect+crop (MTCNN via `facenet_pytorch`) across train and test since the test set guarantees a face is present
- Data-download and model-implementation instructions written (`docs/reference/data_download.md`, `docs/reference/model_code.md`)
- `data/*.py`, `model/*.py`, and `forgery_classifier.ipynb` implemented per those instructions; full run executed (`forgery_classifier_final.ipynb`), results/limitations written up in `README.md`
- Environment dependencies frozen to `requirements.txt`
- Investigated the severe blur visible in all three demo screenshots (`docs/investigations/2026-07-26-upscale-artifact.md`): confirmed it's MTCNN's unconditional bilinear crop-resize amplifying small detected face boxes, measured at a similar ~10-12x median upscale factor across `real`, `deepfake`, and `edited` alike (not a clean per-class blur-magnitude confound, as initially assumed before `edited` and `real` were actually measured) — open hypothesis is a subtler interaction between that shared resampling artifact and each class's underlying pixel statistics, not yet confirmed against the trained model

**Next:**

- Run the tiered ablation in `docs/investigations/2026-07-26-upscale-artifact.md` to test the shortcut-learning hypothesis against the actual trained checkpoint (diagnostic classifier on a resampling-artifact proxy -> counterfactual resolution-swap probe -> resolution-matched retrain, in that order)
- Address the already-known cross-generator generalization failure (README: confidently wrong on Gemini/gpt-image-1 images) once the above clarifies how much of it is shortcut-driven vs. simply narrow training data

