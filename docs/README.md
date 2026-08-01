# Documentation Index

## Start here

New to the project, in reading order:

1. [`decisions/0001-architecture-decisions.md`](decisions/0001-architecture-decisions.md) — what was built and why (3-branch fusion model).
2. [`../README.md`](../README.md) — results, evidence, and known limitations of that build.
3. [`decisions/0002-frozen-backbone-generalization.md`](decisions/0002-frozen-backbone-generalization.md) — what the eval revealed, and the frozen-features direction it set.
4. [`decisions/0003-frozen-probe-demo-build.md`](decisions/0003-frozen-probe-demo-build.md) — **current decision.** Closes the research chapter, retires v1 from the demo slot, and builds the frozen CLIP probe under a 2026-08-03 deadline.

## Conventions

Docs are split by lifecycle, not just topic, so it's obvious whether a file is safe to edit or is a historical record:

| Folder | Contains | Editing rule |
|---|---|---|
| `decisions/` | Numbered ADRs (`NNNN-slug.md`) with reasoning and discarded alternatives | New decisions get a new number. Superseded reasoning is **marked inline, never rewritten** — the doc is the record of what was decided and why. |
| `reference/` | How the pipeline/model works **as currently implemented** | Edited in place as the system changes. |
| `research/` | External literature and background | Kept as a record of what a research pass found. If its conclusions are overtaken, add a status header pointing forward rather than editing the findings. |
| `investigations/` | Dated point-in-time reports (`YYYY-MM-DD-slug.md`) | **Immutable.** If a conclusion later changes, write a new dated report and link back. |
| `notes/` | Standalone background/explainer material (methods, math, tooling) — not a project finding or decision | Edited in place; low-stakes, no lifecycle tracking. |

**Status legend:** `Current` · `Accepted (conditional)` — pending a stated experiment · `Partially superseded` — still largely valid, specific sections overtaken · `Historical` — preserved as a record, not as guidance.

Evidence tags used across decision and research docs: `[confirmed]` (fact-checked or multi-source) · `[measured]` (measured in this project) · `[unverified]` (single source, treat as a lead).

---

## Decisions

| Doc | Status | Summary |
|---|---|---|
| [0001-architecture-decisions.md](decisions/0001-architecture-decisions.md) | Partially superseded by 0002 | 3-branch fusion architecture (EfficientNet-B4 + FFT + SRM → gated fusion), backbone choices, fusion mechanism, compute budget, dataset selection. Architecture and fusion decisions still stand; the CLIP-ViT/DINOv2 rejection and the Dataset section are superseded. |
| [0002-frozen-backbone-generalization.md](decisions/0002-frozen-backbone-generalization.md) | Partially superseded by 0003 — §9, and the conditional status | Adopts **frozen** foundation-model features + a linear probe as an additional generalization path, alongside (not replacing) the 3-branch model. Traces the full arc: what the survey found → what was decided → what the eval showed → what changed. §8's direction is what 0003 executes; §9's gating experiment is **not being run** and the "Accepted (conditional)" status is resolved by 0003 §3 on deadline grounds rather than by evidence. |
| [0003-frozen-probe-demo-build.md](decisions/0003-frozen-probe-demo-build.md) | Accepted | **Current decision (2026-08-01).** Closes the research chapter on the v1 checkpoint and retires v1 from the Monday demo slot. Builds a frozen CLIP ViT-L/14 + logistic-regression probe, binary real vs. AI-generated, on **whole images** (face cropping dropped — MTCNN rejected 83.9% of general content, and dropping it retires the upscale-artifact confound outright). Records that the §9 falsification gate is being spent rather than passed, and that the binding constraint is **data, not modelling**: COCO_AI is DALL·E-3-only, and single-generator training is the already-identified cause of v1's failure, so ≥2 training generators is a requirement. Lists what is deliberately not built. |

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
| [2026-07-31-production-deployment.md](research/2026-07-31-production-deployment.md) | Current | What changes when the objective stops being "separate classes on a held-out split." Base-rate math (95% TPR / 1% FPR at 0.5% prevalence → ~32% precision), the four layers (provenance / classifier / abstention / review), generator acquisition as the real asset, abstention as the only label-free monitoring signal, post-processing robustness as a spec, label feedback without data rights, and which research principles survive / invert / die. **Decision input, not a decision** — the material a future `0003` would draw on. |
| [2026-07-31-literature-triage.md](research/2026-07-31-literature-triage.md) | Current | Method for deciding what to keep from a contradictory literature. Three causes of apparent disagreement, claim half-life tiers 0–4, a 90-second triage filter, and the claims-ledger / decisions-log / kill-list trio. Tracks **claims, not papers** — a paper carries claims with four different expiry dates. Ends with the two open, load-bearing claims for this project and their discriminating tests. |
| [2026-08-01-calibration-and-thresholds.md](research/2026-08-01-calibration-and-thresholds.md) | Current | **Reverses the 07-31 conclusion that calibration cannot address the cross-generator failure.** Yang et al. (AAAI 2026) attributes that failure substantially to *misaligned decision thresholds rather than lost feature separability*, correctable post-hoc with a scalar logit adjustment on a frozen backbone, with a label-free variant. Also records the prerequisite chain: a score is not a probability until measured (Guo; Minderer's qualification), base-rate shift changes the conclusion without changing the evidence (0.9 → ~0.043 at 0.5% prevalence), and prior correction *requires* a calibrated model (Alexandari) — plus where the theory stops, since an unseen generator changes `p(x\|fake)` and violates the label-shift assumption every prior-shift method rests on. Carries a `[second-hand]` tag for numbers not re-checked against primary sources, including the calibration-set size a current build decision depends on. Amends 0002 §10, §11. |
| [2026-07-31-claim-verification.md](research/2026-07-31-claim-verification.md) | Partially superseded — §1.2, §5 | Narrow audit of the mechanistic claims the project was reasoning from, several previously unsourced. **Confirms** that detectors key on the global VAE decode (AlignedForensics / AEROBLADE / INP-X), that failure on unseen generators is confident and biased toward `real`, and that generator count is the dominant training lever. **Corrects** the "shared f8 VAE lineage → family-level transfer" story that 0002 §6.6 rested on — VAE configs diverge, and Flux Dev sits at 21% despite being latent diffusion. Amends 0002 §6.6, §8.5, §10, §11. |

## Investigations

Dated, immutable. All three concern the same thread: whether the model's in-distribution success rested on a resampling/blur shortcut. Read in order.

| Doc | Outcome |
|---|---|
| [2026-07-26-upscale-artifact.md](investigations/2026-07-26-upscale-artifact.md) | MTCNN's bilinear crop-resize blurs all 3 classes at a similar ~10-12x median upscale factor (remeasured n=200: real 8.18x, deepfake 8.59x, edited 12.43x). **Rules out** a clean per-class blur-magnitude confound; leaves an interaction hypothesis open. |
| [2026-07-27-fft-srm-template-swap-probe.md](investigations/2026-07-27-fft-srm-template-swap-probe.md) | **Rules out** a targeted per-class shortcut in the *averaged* fft/srm channels. Records its own scope gap: never varies `rgb`, so it cannot reach the spatial branch or the core hypothesis. |
| [2026-07-27-resolution-swap-probe.md](investigations/2026-07-27-resolution-swap-probe.md) | Varies `rgb` directly. **Rules out** a general resolution-magnitude shortcut across 2 of 3 achievable pairs, replicated across a >30% swing in injected perturbation size. One small residual (`real→edited`) left open and unpursued. |

**Net result of the thread:** the cross-generator failure is *not* explained by a resolution/blur shortcut. That conclusion is what 0002 §5.3 builds on.

## Notes

| Doc | Summary |
|---|---|
| [linear_probe_loss_functions.md](notes/linear_probe_loss_functions.md) | Logistic regression vs. Ridge classifier as linear probes — the loss-function difference, why arXiv:2606.26384 likely uses both (not stated in the paper), and a runnable demo (`loss_boundary_demo.py`) showing Ridge's boundary rotate toward far-but-easy points while logistic regression's doesn't. Relevant background for `0002` §9. |

## Screenshots

`screenshots/*.png` — live-demo screenshots (one per class), referenced by the root README and the upscale-artifact investigation.
