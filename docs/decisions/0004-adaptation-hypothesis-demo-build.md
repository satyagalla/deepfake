# 0004 — The Adaptation Hypothesis: Patch-Based Frozen CLIP Probe with Evidence Cards

**Status:** Accepted — **partially superseded by [`0005-measurement-and-verdict-semantics.md`](0005-measurement-and-verdict-semantics.md)** (§6.6, §7.1, §8's verdict vocabulary, §9 E1/E5 criteria). The claim, the data constraints, the patch design and the frozen backbone are untouched.
**Date:** 2026-08-02
**Deadline:** Monday 2026-08-03 (Plurall AI second interview)
**Supersedes:** [`0003-frozen-probe-demo-build.md`](0003-frozen-probe-demo-build.md) §4.3 (backbone), §5 (data route), and most of §6 (the not-built list) — see §3.
**Relates to:** [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md) (the inventory this clears), [`../research/2026-08-02-patch-inference-and-encoder-choice.md`](../research/2026-08-02-patch-inference-and-encoder-choice.md) (the evidence), [`../notes/2026-08-02-build-plan.md`](../notes/2026-08-02-build-plan.md) (the execution), `../../notes.md` §Interview.

---

## 1. Context: what changed on 2026-08-02

`0003` was written on 2026-08-01 to build a frozen CLIP probe as the Monday demo. It remains the right shape. Three things have changed since, and together they change the *claim* the build is making:

1. **The "data and time were the constraint" story is spent.** It was delivered at interview 1. Repeating it on Monday is not a new finding — it is the same finding, a week later. The week between interviews has to have bought something, and the demo's job is to say what.
2. **The test scenario is fixed and known** `[stated]`. The founder will generate an image spontaneously with a current commercial generator (Gemini, ChatGPT) and upload it. Any deliverable whose evidence lives entirely in held-out benchmark splits loses the only test that actually happens in the room.
3. **A day of verification against primary sources established that no cheap construction generalizes to those generators.** Reconstruction-pairing collapses outside the SD-VAE family (`2026-08-02-dataset-alignment.md`); the corpus occupies a resolution band and an architecture family the demo distribution does not (`bottlenecks.md` §2.1–2.2); composition beats scale by 400× but our composition is one family (§2.2).

Point 3 is usually read as a wall. It is better read as a **result**: the thing everyone tries to buy — generalization to an unseen generator — cannot be bought at this budget, and that is a measurable, statable fact rather than a personal failure.

## 2. The decision

**Build a system that measures the price of adaptation rather than claiming generalization.**

The claim being demonstrated on Monday:

> You cannot buy generalization to an unseen generator. You *can* buy **adaptation**, and it costs roughly 30–50 labelled images and a few seconds of compute.

The deliverable is a number with a unit: **images-per-new-generator**. That is a procurement fact a company that ships a detector actually needs, and it is the answer to *"what would you do with more time and resources"* expressed as a measurement rather than a wish.

This survives the fixed test scenario in a way `0003`'s framing does not. When the founder uploads a Gemini image and the model is wrong, the demo is not over — that is the **0-shot point on a curve that was stated in advance**, and the next thing shown is what 30 images does to it.

## 3. What this changes about `0003`

| `0003` section | Status under `0004` |
|---|---|
| §2 (frozen backbone + linear probe, binary, whole images) | **Stands.** Whole-image is generalized to patches (§6), not reversed — no face crop returns. |
| §4.1 (no face crop) | **Stands, reinforced.** `bottlenecks.md` §7 corroborates against SSAFE and NTIRE. |
| §4.2 (binary real vs AI) | **Stands** as the fused score. Extended by a second head (§7.2), not replaced. |
| §4.3 (backbone: CLIP first, DINOv2 as a swap) | **Superseded.** DINOv2 is dropped outright, not deferred — §6.4. |
| §4.4 (logistic regression probe) | **Stands.** |
| §5 (data: ≥2 generators, prefer external corpus) | **Superseded.** The generators were already in the corpus we had — §5. |
| §6 (not built: calibration, OOD gating, explainability) | **Mostly superseded.** Calibration, OOD gating and per-dimension explainability are now *in* scope, because the card system is what the product surface is — §8. Grad-CAM stays out. **Amended 2026-08-02 by `0005` §3:** calibration is *out* of scope again — `0003` §6 was right and this row was wrong. What was built was one step of the chain (a Platt fit on the val split), fitted on the wrong distribution for every headline number. OOD gating and explainability stand. |
| §7 (risks) | Carried forward and extended — §11. |

`0003` §3's honesty note carries forward unchanged: **the `0002` §9 falsification gate was spent, not passed.** Nothing in `0004` restores it. What `0004` adds is a set of *pre-registered* experiments (§9) that are gates in their own right.

## 4. Why this hypothesis and not the alternatives

Three framings were drafted and two were discarded. Recorded so the choice is visible.

**Discarded — "state the failure boundary."** This was `0003` §8's positioning and it is still correct, but it is no longer *sufficient*: it was communicated at interview 1 as the data/time constraint. Delivered again it reads as an excuse that has had a week to improve and didn't.

**Discarded — the leave-one-generator-out curve as the headline.** LOGO with volume held constant is a clean measurement of cross-generator transfer and it does prove something true. It fails the §1.2 test: all its evidence is internal to the COCO_AI corpus, so a live Gemini upload sits outside every axis of the chart. **Retained as the 0-shot baseline** (§9, E2) rather than the headline.

**Adopted — the N-shot adaptation curve.** It contains the LOGO result as its own left endpoint, and it extends onto the axis the live test occupies. It is also the only one of the three whose conclusion is *actionable by the company*: a number of images.

## 5. Data

The binding correction to `0003` §5: **the multi-generator data was already in the corpus.** `data/download.py:68` calls `ds.select_columns(["caption", "coco_image", "dalle_image"])`, discarding five of six generator columns `[confirmed]`. COCO_AI's real schema is:

```
caption, coco_image (real),
sd35_image, sd3_image, sd21_image, sdxl_image, dalle_image, midjourney_image
```

`0003` §5 planned to source a second generator externally — hours of download and a licensing risk — to fix a problem caused by one line of column selection. That is the single highest-value finding of the 08-02 pass.

**Composition:**

| Source | Count | Role |
|---|---|---|
| COCO_AI `coco_image` | ~3,000 | Real |
| COCO_AI × 5 columns (sd21, sdxl, sd3, sd35, dalle) | ~15,000 | Train/val fake |
| COCO_AI `midjourney_image` | ~3,000 | **Held out entirely** — the honesty anchor |
| Self-generated Gemini + GPT-Image, COCO captions | ~150–300 | The N-shot pool |
| Self-generated, off-domain prompts | ~20/generator | Domain-shift control (E4) |
| Self-generated via web UI (not API) | ~20/generator | Container control |

**Constraints that are not negotiable:**

- **The 1:1 row pairing is preserved.** `bottlenecks.md` §7 identifies it as the most valuable property the data has, and §2.4 as the only thing preventing the corpus shortcut SSAFE never controlled for. **Splits are at the row level** — one row's real and all six of its fakes land on the same side of the split, or the real/fake pairing leaks the split.
- **No unpaired real source is added to training.** `bottlenecks.md` §2.4, standing constraint. Adding modern high-res reals to a low-res fake corpus makes "high-res photo = real" the cheapest available hypothesis, which would produce a good validation number and a failed live demo — the exact failure mode of round 1.
- **Uniform container.** All classes re-saved at JPEG q95 (already implemented, `data/download.py:80-82`). §2.5 prices the container↔label shortcut at ~11 points; leaving it in would make the headline number meaningless.
- **The person-caption filter is dropped.** `PERSON_KEYWORDS` / `caption_has_person()` existed to serve the face crop, which `0003` §4.1 removed.

**Midjourney is held out and never trained on, at any N.** It is the one generator where a 0-shot number can be quoted without the adaptation story attached, and it is the answer to "how do I know the curve isn't just memorisation."

## 6. Preprocessing: fixed-N native patches

This is the largest change from `0003` and the one most likely to decide whether the demo works.

### 6.1 The problem

CLIP ViT-L/14 takes **224×224** `[confirmed]`. The standard pipeline resizes the short side to 224 and centre-crops. Applied to this corpus:

| Source | Native | Downscale to 224 |
|---|---|---|
| COCO real | ~480px `[second-hand, §9.1]` | ~2.1× |
| Gemini / GPT-Image | 1024px | ~4.6× |

Two failures at once. Downsampling is a low-pass filter, so it **destroys the high-frequency band the artifact lives in**. And the downscale *ratio* differs systematically by class, so **the resampling signature itself becomes label-correlated**. This is `bottlenecks.md` §2.1(b) recurring in a new model — the same bug that killed round 1, one resize further down the pipeline.

### 6.2 The decision

**Extract features from N fixed-count 224px patches taken at native resolution, plus one whole-image resized view.**

- **No resampling on the patch path.** A native crop is the same operation regardless of source resolution, so it introduces no resolution-dependent signature and preserves the high-frequency band intact.
- **N is fixed at 16, independent of source resolution.** Small images sample with overlap; large images sample 16 at random from the grid. See §6.3 — this is a correctness requirement, not a tuning choice.
- **One whole-image resized view is retained alongside.** Patches cannot see composition, object coherence, anatomy or lighting, and CLIP is a semantic encoder — patching removes much of what CLIP is best at. The two views are concatenated into the probe.
- **CLIP's own normalization constants**, `mean=(0.48145466, 0.4578275, 0.40821073)`, `std=(0.26862954, 0.26130258, 0.27577711)` — **not** the `IMAGENET_MEAN/STD` currently in `model/demo.py:36`. Carrying v1's constants into a CLIP path would be a silent, plausible-looking bug.

### 6.3 The bag-size leak — a correctness requirement

A 480px image yields ~4 non-overlapping 224 patches; a 1024px image yields ~20. Source resolution correlates with label. Therefore **patch count would encode the label**.

This leaks through every aggregator, not just the obvious ones. Max and top-k leak directly — the maximum of 20 draws is stochastically larger than the maximum of 4. Mean leaks through *variance* — larger bags produce tighter means, and a threshold can exploit the spread.

**Fixed N=16 for every image, enforced as an invariant with an assertion in the extraction loop.** This is precisely the class of silent data bug that produced round 1's blurred submission, and the lesson recorded there (`notes.md` line 79) is that the root cause was not inspecting the data after processing.

### 6.4 What patching does and does not fix

It converts a **resolution** confound into a **field-of-view** confound. A 224 patch is 47% of a 480px image but 22% of a 1024px image, so the two still see content at different scales. That is real and it is not eliminated.

It is the right trade because **field-of-view is controllable and resolution mismatch is not.** Patch count and patch size are ours to hold fixed; the native resolution of COCO_AI is not.

The benefit is asymmetric, and correctly so: at 270px a 224 patch is 83% of the image, so patching is nearly a no-op at the low end of the corpus. The resize was crushing 1024px generated images 4.6× while barely touching COCO — **patching removes the damage exactly where the damage was being done.**

### 6.5 Aggregation is measured, not assumed

Not every patch carries evidence. A flat sky patch from Gemini may be genuinely indistinguishable from a flat sky from a camera, so training every patch on its image's label injects label noise. This is a multiple-instance learning problem: the bag is positive if *any* instance is.

Mean assumes a spatially uniform artifact (which the decode signature plausibly is). Top-k and max assume a localised one (which flat-region emptiness argues for). **This is settled by measurement, not by argument, and the measurement is free** — patch scores are computed once, and mean / max / top-k / trimmed-mean all come off the same cached array at zero marginal cost. A gradient-variance filter that drops flat patches before scoring is ablated the same way.

### 6.6 Calibration is fitted on aggregated image-level scores

> **Moot as of 2026-08-02 — [`0005`](0005-measurement-and-verdict-semantics.md) §3.** Nothing is fitted post-hoc any more, so nothing can be fitted at the wrong level. The section is not *wrong*; it no longer has a referent. The nesting rule below stands and still governs every split (build-plan I2/I3).

16 patches from one image are not 16 independent samples. Fitting Platt/temperature scaling on patch-level scores would produce a systematically overconfident model. The nesting is **row → image → patch**, and all three levels respect the split.

## 7. Model

Frozen CLIP ViT-L/14. Features extracted once and cached; every experiment thereafter is a refit measured in seconds. **The frozen backbone is load-bearing, not a compromise** — see §10.

### 7.1 Head A — binary, the fused score

Logistic regression, real vs AI-generated, on the concatenated patch + whole-image features. ~~Platt/temperature calibrated on a held-out split.~~ This produces the number the verdict thresholds apply to.

> **Superseded 2026-08-02 — [`0005`](0005-measurement-and-verdict-semantics.md) §3.** The calibrator is removed. It was fitted on `split="val"` (COCO_AI, SD/DALL·E, 270–480px) and applied to Gemini/GPT-Image at 1024px and to Midjourney — the wrong distribution for every headline number. It is monotone, so it could not move AUC; and it contaminated E5, which evaluated on the split the calibrator was fitted on. Card scores now come from the classifier's own `predict_proba`. **This build emits no calibrated probability**, and `0005` §8 records why that is the honest position rather than a gap.

### 7.2 Head B — multiclass generator ID, and the OOD gate

Multiclass logistic regression over `{real, sd21, sdxl, sd3, sd35, dalle, gemini, gptimage}`. Two jobs:

- **Populates the AI Model card** with a named generator rather than a bare score — which is what the product surface actually displays.
- **Serves as the OOD gate** via **Mahalanobis distance on the frozen features**: per-class means with a single shared covariance. Low distance to the nearest class mean means "I have seen this kind of image"; high means "this generator is outside everything I was fitted on," and the system abstains rather than guessing.

Two implementation requirements:

- **L2-normalize features before fitting.** SSAFE does `[confirmed]`; `bottlenecks.md` §3.2 records that whether our path does is unverified. Mahalanobis on unnormalized CLIP features is badly conditioned.
- **Pooled covariance with Ledoit-Wolf shrinkage, never per-class.** With 768+ dimensions and possibly ~150 samples in the Gemini class, a per-class empirical covariance is singular. Pooling over all classes gives ~21k samples for one matrix, which is comfortable.

The Mahalanobis choice is also the honest version of the founder's **Gaussian** interest — but per `notes.md` line 106, the Mahalanobis↔Gaussian-density link was our own construction, **not her stated point**, and must not be presented as though it were.

## 8. Evidence cards

Plurall returns six cards, schema `{dimension, label, score, verdict, detail}`, verdicts `AUTHENTIC | SUSPICIOUS | SYNTHETIC | PLAUSIBLE | STRIPPED`, fused thresholds SYNTHETIC ≥0.85 / SUSPICIOUS ≥0.5 `[stated]`.

> **Superseded 2026-08-02 — [`0005`](0005-measurement-and-verdict-semantics.md) §6.** The sentence above is accurate about *their* product and stays as the record of it; the error was adopting it as *our* vocabulary, since it describes a different model's observables. Our system now emits two fields — `verdict ∈ {DECLARED_SYNTHETIC, LIKELY_SYNTHETIC, WEAK_EVIDENCE, NO_EVIDENCE}` and `reliability ∈ {IN_DISTRIBUTION, UNKNOWN_SOURCE}` — because the score and the model's entitlement to it are orthogonal, and the Mahalanobis gate is the property this build is a claim about. Cards carry no verdicts at all: they have a `score` or a `silent_because`. `AUTHENTIC` is dropped as unreachable (absence of synthesis evidence is not evidence of capture). Thresholds are re-derived from a false-positive budget (`0005` §7). **The P1 drill still implements their spec, unchanged** — `0005` §12.

| Card | Status | Source |
|---|---|---|
| **AI Model** | Built | Head B — named generator + confidence |
| **Spectral** | Built | Radial FFT profile |
| **EXIF** | Built | EXIF + C2PA parse, deterministic, no learning |
| **Diffusion** | `NOT_IMPLEMENTED`, documented | See §10 |
| **Temporal** | `NOT_APPLICABLE` on stills | Scope declared |
| **Web Intelligence** | `NOT_IMPLEMENTED` | Out of scope, no index |

**Fusion is evidence-level, not feature-level.** v1's gate weights are not explanations — they are learned scalars with no per-dimension semantics. Cards must be independently computed and independently reportable, or the "explainable score" claim is decoration.

**Rules:**

- A `STRIPPED` card is **excluded from the fused score**, not scored as 0.5. Absent metadata is absent evidence; scoring it as neutral evidence lets missing data move the number. It widens the confidence interval instead. — *Rule stands; re-expressed as card silence, `0005` §6.3.*
- Per-card **scope**: a card that cannot speak to an input returns `NOT_APPLICABLE` rather than a low-confidence guess. — *Rule stands; now `silent_because`, `0005` §6.3.*
- **Abstention on an unknown generator** (§7.2) overrides the fused verdict. — **Superseded, `0005` §6.2:** abstention no longer overrides the verdict, it *accompanies* it as the `reliability` field. Suppressing the verdict hid the gate's most informative output; Midjourney at 0-shot reading `LIKELY_SYNTHETIC / UNKNOWN_SOURCE` is the better demonstration.
- Thresholds are **parameters**, not constants — the product exposes them under Detection Settings, so the code must too. — *Rule stands; only the defaults' provenance changes, `0005` §7.*

> **Two fusion defects found 2026-08-02 and left open — [`0005`](0005-measurement-and-verdict-semantics.md) §6.4.** The unweighted mean made the top verdict *unreachable* whenever the EXIF card fired on camera metadata (ceiling `(1.0+1.0+0.05)/3 = 0.68`), and it gives the weakest card half the vote once metadata-free API output silences the EXIF card. Documented rather than fixed: weighted fusion is not new behaviour to introduce the day before the deadline.

### 8.1 The watermark and post-processing problem

Gemini embeds **SynthID** in pixel values at generation time, engineered to survive crop, compression and resize `[confirmed]`. `bottlenecks.md` §4.2 records the consequence precisely: **it passes the container control by design.** A detector could be reading a watermark and scoring 99% for a reason that evaporates on any generator that does not watermark.

Handled structurally rather than by hoping:

1. **Provenance is quarantined in its own card.** EXIF/C2PA/watermark evidence never enters the learned score. If provenance says SYNTHETIC and the classifier says AUTHENTIC, that disagreement is *displayed*, not averaged away.
2. **The degradation ladder measures the leakage** (§9, E3). Score each card across a ladder of re-encodes, rescales and re-renders. A learned score that survives degradation identically to the provenance card is reading the watermark.
3. **Gemini vs GPT-Image is a free natural experiment.** They differ in watermarking policy. If per-generator accuracy tracks watermark presence rather than architecture, the confound is real and measured.

## 9. Pre-registered experiments

Registered before running, because `0002` §9's gate was already spent once by deciding after the fact (`0003` §3).

| | Experiment | What it answers | Success criterion, stated in advance |
|---|---|---|---|
| **E1** | **N-shot adaptation curve** — ~~accuracy~~ **AUC** on a held-out generator vs N training images from it, N ∈ {0, 5, 10, 20, 30, 50, 100} | The headline. What does adaptation cost? | The *knee* is the finding, whatever its location. A curve with no knee falsifies the claim. **Knee criterion moved to AUC ≥ 0.90** (`0005` §5); balanced accuracy, TPR and FPR reported alongside. |
| **E2** | **Held-out Midjourney, 0-shot** | Does the model transfer to a generator it has genuinely never seen? | Reported as measured. A low number is a result, not a failure. |
| **E3** | **Degradation ladder, per card** | Which evidence survives post-processing; is the score reading a watermark? | Per-card curves. Learned score tracking the provenance card = confound confirmed. |
| **E4** | **Off-domain prompts** | Is the model reading synthesis or COCO's content distribution? | Accuracy drop on off-domain vs in-domain prompts at equal N. |
| **E5** | **AUC, not just accuracy** | `bottlenecks.md` §4.1: **AUC has never been computed on this project.** | First measurement. Separates "lost separability" from "misaligned threshold" — the open question in `plan-c-source-verification.md`. **Absorbed, `0005` §4.1:** AUC is no longer one experiment but the primary metric of all of them; E5 remains as the val-split measurement, now on a genuinely held-out val (the calibrator that contaminated it is gone). |
| **E6** | **Aggregator ablation** (§6.5) | mean vs max vs top-k vs trimmed | Free; comes off cached patch scores. |
| **E7** | **Preprocessing arms** — standard resize vs native patches | Does §6 actually pay? | One extra extraction pass. Standard resize is the reproducible control. |

E1 and E5 are the two that must land. E6/E7 are free given the cache. E3 is the one with the most product relevance.

## 10. Discarded alternatives

**DINOv2 — dropped outright, not deferred (supersedes `0003` §4.3).** Self-supervised training optimises for *invariance* across augmented views, which is an explicit instruction to discard the local high-frequency detail synthesis artifacts live in. DINOv2-Giant is the worst row in SSAFE's encoder ablation at 72.8% `[confirmed]`. Decisively for this build: **DINOv2's multi-crop self-distillation objective trains crops of an image to embed alike** `[confirmed]` — under §6's patch design we would be selecting the one encoder explicitly trained to make patches indistinguishable.

`bottlenecks.md` §3.1 recommends a two-arm encoder comparison, noting NTIRE 2026's winners used DINOv3-**7B** so the multimodal>self-supervised rule may be measuring scale rather than objective. **That recommendation is overridden deliberately.** The confound is not resolvable here — a 7B model does not run on a Colab A100 in a day — so our two-arm run would not settle the question it was proposed to settle. That makes it a research question, and the research chapter is closed.

**PE-Core-G14-448 — dropped.** Three reasons: §2.1's inversion argument (on a 270–480px corpus its extra resolution reads an empty or interpolation-filled band, so it "could underperform CLIP-224 and could score well for the wrong reason"); availability in this environment is **unverified** (§3.2/§9.2), an unknown-cost dependency the day before a deadline; and its central claim is that the best embeddings are in *intermediate* layers, which opens an unbudgeted hyperparameter search. Under §6, a 448 patch cannot be taken natively from a 270px image at all.

**CLIP ViT-L/14@336 — considered and dropped.** It was the right call *before* the patch decision, as a mitigation for the downsampling asymmetry. Patching addresses that cause directly, and a larger input costs corpus coverage at the low end. @224 is confirmed.

**SigLIP2 — deferred as a stretch arm, not cut.** If a second encoder ever runs it should be this one: it is the arm that could plausibly win, it loads from HF without availability risk, and it is the direction SSAFE's ablation actually points. Strictly after E1–E3.

**Full fine-tuning — rejected, and this is load-bearing.** The headline claim is that adaptation costs ~30 images and seconds. **That is only true for a linear refit on cached features.** If adaptation required backbone fine-tuning it would cost a GPU-hour and hundreds of images, and the entire argument collapses. The frozen backbone is *the thing being demonstrated*, not a shortcut around training. Independently corroborated on our own data: `notes.md` line 81, round 1 — *"Full fine-tuning further degraded results."*

The honest counter is recorded rather than hidden: `bottlenecks.md` §3.1 is right that a frozen encoder is a **ceiling** — if CLIP discarded the artifact, no probe recovers it. That constrains how high E1's curve goes; it does not affect whether the adaptation claim holds. And it is measurable: **E1's plateau is the ceiling.**

**AEROBLADE / the Diffusion card — demoted to a documented stub.** It was pitched as an independent corroborator against the watermark confound. It is not one. VAE round-trip reconstruction error is low when the decoder *matches the generator's*; `2026-08-02-dataset-alignment.md` records AEROBLADE's 0.992 mAP as a **matched-decoder** number, with cross-decoder mAP at 0.543–0.623. Gemini and GPT-Image decode through discrete VQ tokenizers, not an SD-family VAE, so the branch is least competent exactly where the live test happens. **The per-card scope reasoning is worth more here than the code would be** — a card that declares what it cannot speak to is the explainability story.

**Adding modern high-res real images — rejected.** `bottlenecks.md` §2.4, standing constraint. It is the fastest route to a good validation number and a failed demo.

## 11. Risks accepted

| Risk | Status |
|---|---|
| E1's curve may have no knee, or a knee far above 50 images | **This is the experiment.** Pre-registered in §9; a flat curve falsifies the claim and that is reported, not buried. |
| Patch-based features may underperform the standard resize on a corpus whose low end is already below patch size | Real. E7 runs the standard pipeline as a control arm for exactly this reason. |
| Self-generated N-shot pool is small (~150–300) and single-session, so it may carry a session-specific signature | Real and partly unmeasurable at this budget. E4's off-domain arm and the web-UI container control are the mitigations. |
| A live upload may still be scored wrong at 0-shot | **Expected, and it is the left endpoint of the headline chart.** Stated before she uploads, per `notes.md` §Positioning. |
| Watermark leakage may not be fully separable from architecture at n≈150/generator | Real. E3 measures it; the result may be inconclusive, and that is reportable. |
| Frozen public backbone is a gray-box adversarial liability | Inherited from `0002` §10, unchanged. No adversary in scope. |
| The `0002` §9 gate remains unrun | Carried from `0003` §3, unchanged. §9's pre-registration does not retroactively restore it. |
| Build time competes with live-coding prep, which `notes.md` identifies as the actual round | **Real and binding.** The build plan carries a hard cap and an explicit cut order; P0/P1 are not touched. |

## 12. Positioning consequence

`notes.md` §Positioning: *"The differentiator is not 'it works.' … The differentiator is stating where it fails before she tests it, and being right."*

`0004` keeps that and adds the half it was missing. Stating a boundary is a defensive move; on its own it invites "so what would you do about it." **The N-shot curve is the answer to that question, expressed as a price.**

The Monday sequence, unchanged in shape from `0003` §8 but with a different second beat:

1. Round-1 post-mortem — the failure was data, here is how it was found, here is the write-up.
2. Here is what the week bought: **generalization is not purchasable at this budget; adaptation is, and here is what it costs in images.**
3. Here is exactly where it fails — stated before you test it.
4. Then let her upload, and let the result land on a curve that was already on screen.
