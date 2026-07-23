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
- `deepfake_detection_research.md` - deep research doc on deepfake detection models for classifying real, edited (AI generated and human) and deepfake classes
- `architecture_decisions.md` - doc describing requirements and architecture decisions with reasoning, incl. finalized dataset choice
- `data_download.md` - instructions (no code) for downloading + face-filtering the 3 class datasets; code goes in `data/download.py` and `data/face_filter.py`
- `model_code.md` - instructions (no code) for the 3-branch fusion model, training, and eval; code goes in `model/*.py` + a single Colab notebook with Data/Model/Train/Eval sections

## Current State

**Done:**
- Architecture finalized: 3-branch fusion (EfficientNet-B4 spatial + FFT spectral CNN + SRM noise-residual CNN -> gated fusion), see `architecture_decisions.md`
- Dataset finalized: one dataset per class (deepfake = DALL-E 3 slice of COCO_AI/SynthBuster, real = its paired originals, edited = CASIA v2.0), uniform face-detect+crop (MTCNN via `facenet_pytorch`) across train and test since the test set guarantees a face is present
- Data-download and model-implementation instructions written (`data_download.md`, `model_code.md`)

**Next:**
- Confirm exact download source (Kaggle slug/URL) for COCO_AI/SynthBuster and CASIA v2.0
- Write the actual `data/*.py` and `model/*.py` code + the Colab notebook, following the two instruction docs
- Run in Colab, check DALL-E 3 slice's post-face-filter survival rate before committing to a full training run

