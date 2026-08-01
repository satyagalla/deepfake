# 0003 — Frozen CLIP Probe as the Monday Demo; v1 Retired from the Demo Slot

**Status:** Accepted
**Date:** 2026-08-01
**Deadline:** Monday 2026-08-03 (Plurall AI second interview)
**Supersedes:** [`0002-frozen-backbone-generalization.md`](0002-frozen-backbone-generalization.md) §9 (the gating experiment) and its "Accepted (conditional)" status — see §3.
**Relates to:** `0002` §8 (which this executes), [`../research/2026-07-31-production-deployment.md`](../research/2026-07-31-production-deployment.md), `../../notes.md` §Interview.

---

## 1. Context: what changed on 2026-08-01

Two decisions were taken by the user on this date, in order:

1. **The research chapter is closed.** No further analysis of the v1 3-branch checkpoint — no calibration measurement, no reliability diagram, no threshold correction, no OOD eval set built against v1, no remaining `edited`/PS-Battles fingerprint work. These were live open items in `0002` §11 and in the 2026-08-01 calibration research doc §5. They are now **out of scope**, not disproven.
2. **v1 is not the Monday demo.** Something new is built instead, and the choice made was the frozen CLIP/DINOv2 + linear-probe path.

This doc records what that means and what is being accepted to get there. It is written under a hard two-day deadline and should be read as a deadline decision, not a research decision.

## 2. The decision

Build a **frozen foundation-model backbone + linear probe**, binary **real vs. AI-generated**, operating on **whole images**, as the artifact demonstrated on Monday. v1 stays in the repo as the documented prior build and as the explainability story; it is not the thing being run live.

This is `0002` §8.1 executed as a build rather than as an experiment.

## 3. What this changes about `0002`

`0002` was **Accepted (conditional)**, and the condition was §9: extract frozen embeddings, fit a probe, and revert the decision if the probe fails to beat v1 on the images v1 demonstrably fails. **That gate is not being run.** The comparison against v1 requires exactly the v1 analysis that §1 just placed out of scope, and there is no time for both.

Stated plainly: **the frozen-probe path is being built without the falsification experiment that was written to justify it.** `0002` §9 exists precisely so this decision could be cheaply reverted on evidence; that safety mechanism is being spent. The justification is deadline, not evidence, and it should not be presented on Monday as though the gate had passed.

What survives from `0002` unchanged: §8.1 (candidates), §8.2 (v1 kept, not deleted), §8.3 (`edited` deferred), §8.5 (multi-generator training, held-out eval generators), §10's cost list — in particular the frozen-public-backbone adversarial liability, which this build inherits whole.

## 4. Design decisions

### 4.1 Whole images, not face crops `[decided]`

v1's pipeline MTCNN-crops to a face before inference. This build **drops face cropping**.

Three reasons, in order of weight:

- **Live-demo survival.** No face detected means no prediction. The CASIA authentic probe measured MTCNN rejecting **83.9%** of a general-content sample (`../investigations/2026-07-29-casia-authentic-probe.md`). If an uploaded image is a landscape, a product shot, or a screenshot, a face-cropping demo simply fails to answer.
- **It matches the recipe being copied.** UniversalFakeDetect's CLIP + linear-probe setup operates on whole images at the backbone's native resolution. Deviating from a recipe that is known to work, under a two-day deadline, is unforced risk.
- **It retires the entire upscale-artifact confound.** The three 2026-07-26/27 investigations all concern distortions introduced by MTCNN's bilinear crop-resize. Removing the crop removes the confound rather than continuing to argue about it.

Cost accepted: face-region forensic detail is lost, and the person-caption filtering in `data/download.py` becomes unnecessary for this path.

### 4.2 Binary: real vs. AI-generated `[decided]`

`edited` is dropped from this build, consistent with `0002` §8.3 and for the reason recorded there — no source provides all three classes from a common distribution, and the CASIA corpus fingerprint is measured at a non-trivial 17.4% (n=161). Reintroducing it under deadline would reintroduce a known confound into the one artifact being demonstrated.

### 4.3 Backbone: CLIP ViT-L/14 first `[decided]`

`0002` §8.1 lists three candidates. Under deadline the order is: **CLIP ViT-L/14** (UniversalFakeDetect's exact setup — the most documented and most reproducible), with DINOv2 as a swap **only if** the extraction code is factored so the backbone is a one-line change and time remains. LNCLIP-DF is dropped for this build: it is face-domain, and §4.1 just removed the face crop.

### 4.4 Probe: logistic regression `[decided]`

Per `../notes/linear_probe_loss_functions.md` — logistic regression rather than Ridge, because Ridge's boundary rotates toward far-but-easy points, which is the wrong behaviour when one class is a heterogeneous mixture of generators.

## 5. The binding constraint is data, not modelling

**This is the part of the plan most likely to fail, and it is not the model.**

`data/download.py:4` pairs COCO_AI's `coco_image` (real) 1:1 with `dalle_image` — **DALL·E 3 only**. Single-generator training is the *already-identified cause* of v1's cross-generator failure (`0001` Dataset amendment, `0002` §6.4). A frozen backbone changes the feature space; it does not change the fact that the probe would see one generator.

**A CLIP probe trained on DALL·E-3-only data can reproduce v1's exact failure mode on Monday.** Freezing mitigates, it does not immunise.

Therefore: **≥2 training generators is a requirement of this build, not an enhancement**, and sourcing them is the first task, ahead of any model code.

Preferred route under deadline: an existing public multi-generator corpus (SynthBuster's non-DALL·E slices, GenImage) — download-and-extract is hours, whereas generating locally is a day of GPU babysitting. Generating with SDXL/FLUX on the A100 is the fallback if licensing or availability blocks the download.

Held out, never trained on: a small hand-made set from **GPT Image / Gemini / Nano Banana** — the generator class v1 already fails on, and the most likely provenance of anything uploaded live. This set exists to *state the boundary*, not to pass it.

## 6. What is explicitly not being built

Recorded so the omissions are deliberate rather than discovered on Monday:

- Grad-CAM / explainability on the probe. `0002` §10 already accepted that spatial explainability does not transfer cleanly to a ViT. v1 remains the explainability story.
- Calibration, temperature scaling, base-rate correction, tunable thresholds. Out of scope per §1, despite mapping neatly onto Plurall's product surface. If asked, this is a conceptual answer backed by the 2026-08-01 research doc, not a demoed feature.
- Abstention / OOD distance gating.
- The `edited` class (§4.2), augmentation robustness (`0002` §8.4), per-branch auxiliary heads.
- Any v1-vs-probe comparison numbers (§3).

## 7. Risks accepted

| Risk | Status |
|---|---|
| The `0002` §9 gate is unrun, so this path is unvalidated on our own data | Accepted, §3. Do not claim otherwise on Monday. |
| Multi-generator data may not land in time, leaving a single-generator probe with v1's failure mode | **Highest risk.** §5 is first in the build order for this reason. |
| The probe may underperform v1 in-distribution | Expected — `0002` §10 predicted it. Not a failure of this decision. |
| Frozen public backbone is a gray-box adversarial liability | Inherited from `0002` §10, unchanged. No adversary in scope. |
| Two days of build time competing with live-coding prep, which `notes.md` identifies as the actual round | Real. Build stops when it stops; a working small thing beats an unfinished large one. |

## 8. Positioning consequence

`notes.md` §Positioning carried **"don't demo v1 alone"** — written when a v2 path was planned, then made stale when the research chapter closed, and now live again for a different reason: v1 is not the demo at all.

The stated differentiator is unchanged and this build must serve it: *state the failure boundary before it is tested.* With the §5 held-out set, the honest Monday sentence is a measured one — this is what it was trained on, this is the generator class it has never seen, here is what it does on that class. That sentence is the deliverable. The model is what makes it credible.
