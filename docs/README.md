# Documentation Index

## Start here

New to the project, in reading order:

1. [`decisions/0001-architecture-decisions.md`](decisions/0001-architecture-decisions.md) — what was built and why (3-branch fusion model).
2. [`../README.md`](../README.md) — results, evidence, and known limitations of that build.
3. [`decisions/0002-frozen-backbone-generalization.md`](decisions/0002-frozen-backbone-generalization.md) — **current decision.** What the eval revealed, and where the project goes next.

## Conventions

Docs are split by lifecycle, not just topic, so it's obvious whether a file is safe to edit or is a historical record:

| Folder | Contains | Editing rule |
|---|---|---|
| `decisions/` | Numbered ADRs (`NNNN-slug.md`) with reasoning and discarded alternatives | New decisions get a new number. Superseded reasoning is **marked inline, never rewritten** — the doc is the record of what was decided and why. |
| `reference/` | How the pipeline/model works **as currently implemented** | Edited in place as the system changes. |
| `research/` | External literature and background | Kept as a record of what a research pass found. If its conclusions are overtaken, add a status header pointing forward rather than editing the findings. |
| `investigations/` | Dated point-in-time reports (`YYYY-MM-DD-slug.md`) | **Immutable.** If a conclusion later changes, write a new dated report and link back. |

**Status legend:** `Current` · `Accepted (conditional)` — pending a stated experiment · `Partially superseded` — still largely valid, specific sections overtaken · `Historical` — preserved as a record, not as guidance.

Evidence tags used across decision and research docs: `[confirmed]` (fact-checked or multi-source) · `[measured]` (measured in this project) · `[unverified]` (single source, treat as a lead).

---

## Decisions

| Doc | Status | Summary |
|---|---|---|
| [0001-architecture-decisions.md](decisions/0001-architecture-decisions.md) | Partially superseded by 0002 | 3-branch fusion architecture (EfficientNet-B4 + FFT + SRM → gated fusion), backbone choices, fusion mechanism, compute budget, dataset selection. Architecture and fusion decisions still stand; the CLIP-ViT/DINOv2 rejection and the Dataset section are superseded. |
| [0002-frozen-backbone-generalization.md](decisions/0002-frozen-backbone-generalization.md) | Accepted (conditional) | Adopts **frozen** foundation-model features + a linear probe as an additional generalization path, alongside (not replacing) the 3-branch model. Traces the full arc: what the survey found → what was decided → what the eval showed → what changed. Conditional on the gating experiment in its §9. |

## Reference

| Doc | Status | Summary |
|---|---|---|
| [data_download.md](reference/data_download.md) | Current | Download + face-filter pipeline for the class datasets. Code: `data/download.py`, `data/face_filter.py`. Note: describes the 3-class pipeline; `edited` is deferred per 0002 §8.3. |
| [model_code.md](reference/model_code.md) | Current | 3-branch fusion model, training, and eval implementation. Code: `model/*.py`, `forgery_classifier.ipynb`. |
| [resolution_swap_probe.md](reference/resolution_swap_probe.md) | Historical (probe complete) | Spec for the `rgb`-direct probe built to test the upscale-artifact hypothesis. Implemented as `model/resolution_swap_probe.py`; findings in the 2026-07-27 investigation below. Kept in `reference/` rather than moved, because the investigation doc links to it and investigations are immutable. |

## Research

| Doc | Status | Summary |
|---|---|---|
| [deepfake_detection_research.md](research/deepfake_detection_research.md) | Historical — recommendations superseded | The 2026-07-23 SOTA survey. Survey content (architecture families, dataset tables, metrics, risks) still useful; §1's recommendations superseded by 0001 (dataset) and 0002 (architecture). Carries an errata header covering the family (c) framing error that 0002 §6.2 analyses. |

## Investigations

Dated, immutable. All three concern the same thread: whether the model's in-distribution success rested on a resampling/blur shortcut. Read in order.

| Doc | Outcome |
|---|---|
| [2026-07-26-upscale-artifact.md](investigations/2026-07-26-upscale-artifact.md) | MTCNN's bilinear crop-resize blurs all 3 classes at a similar ~10-12x median upscale factor (remeasured n=200: real 8.18x, deepfake 8.59x, edited 12.43x). **Rules out** a clean per-class blur-magnitude confound; leaves an interaction hypothesis open. |
| [2026-07-27-fft-srm-template-swap-probe.md](investigations/2026-07-27-fft-srm-template-swap-probe.md) | **Rules out** a targeted per-class shortcut in the *averaged* fft/srm channels. Records its own scope gap: never varies `rgb`, so it cannot reach the spatial branch or the core hypothesis. |
| [2026-07-27-resolution-swap-probe.md](investigations/2026-07-27-resolution-swap-probe.md) | Varies `rgb` directly. **Rules out** a general resolution-magnitude shortcut across 2 of 3 achievable pairs, replicated across a >30% swing in injected perturbation size. One small residual (`real→edited`) left open and unpursued. |

**Net result of the thread:** the cross-generator failure is *not* explained by a resolution/blur shortcut. That conclusion is what 0002 §5.3 builds on.

## Screenshots

`screenshots/*.png` — live-demo screenshots (one per class), referenced by the root README and the upscale-artifact investigation.
