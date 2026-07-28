# Documentation Index

Docs are split by lifecycle, not just topic, so it's obvious whether a file is safe to edit or is a historical record:

- `decisions/` - architecture/design decisions with reasoning and discarded alternatives. Living doc for now; if it grows unwieldy, split new decisions into numbered ADRs (`NNNN-slug.md`, with a `Status: Accepted/Superseded` field) instead of editing old reasoning in place.
- `reference/` - instructions describing the pipeline/model as currently implemented. Edited in place as the system changes.
- `research/` - external literature and background research. Evergreen.
- `investigations/` - dated, point-in-time investigation/debug reports (`YYYY-MM-DD-slug.md`). Immutable once written - if a conclusion later changes, add a new dated report and link back rather than rewriting the old one.

## Contents

**Decisions**
- [001_architecture_decisions.md](decisions/001_architecture_decisions.md) - 3-branch fusion architecture, backbone choices, dataset selection, with reasoning and discarded alternatives.

**Reference**
- [data_download.md](reference/data_download.md) - download + face-filter pipeline for the 3 class datasets (code: `data/download.py`, `data/face_filter.py`).
- [model_code.md](reference/model_code.md) - 3-branch fusion model, training, and eval implementation (code: `model/*.py`, `forgery_classifier.ipynb`).
- [resolution_swap_probe.md](reference/resolution_swap_probe.md) - spec (not yet implemented) for a probe that manipulates `rgb` pixels directly to test the upscale-artifact shortcut hypothesis, addressing the scope gap found in the 2026-07-27 investigation below.

**Research**
- [deepfake_detection_research.md](research/deepfake_detection_research.md) - survey of deepfake detection models/approaches informing the architecture decisions.

**Investigations**
- [2026-07-26-upscale-artifact.md](investigations/2026-07-26-upscale-artifact.md) - MTCNN's bilinear crop-resize blurs all 3 classes at a similar ~10-12x median upscale factor; open shortcut-learning hypothesis and a tiered ablation plan recommended as the next step.
- [2026-07-27-fft-srm-template-swap-probe.md](investigations/2026-07-27-fft-srm-template-swap-probe.md) - the built counterfactual probe (`model/counterfactual_probe.py`) rules out a targeted per-class fft/srm-template shortcut, but never varies `rgb` and so doesn't reach the original hypothesis; see `resolution_swap_probe.md` for the follow-up spec.

**Screenshots**
- `screenshots/*.png` - live-demo screenshots (one per class), referenced by the README and the upscale investigation doc.
