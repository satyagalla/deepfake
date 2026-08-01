# 0002 — Frozen Foundation-Model Features as the Generalization Path

**Status:** Partially superseded by [`0003-frozen-probe-demo-build.md`](0003-frozen-probe-demo-build.md) (2026-08-01) — §9 and the conditional status
**Date:** 2026-07-28

> **Amended 2026-08-01.** The "Accepted (conditional)" status above is **resolved by deadline, not by evidence.** `0003` executes §8's direction as a build but does **not** run §9's gating experiment, because the v1-side comparison it requires was placed out of scope the same day. The falsification mechanism §9 was written to provide is therefore spent rather than passed — see `0003` §3. §11's open items are likewise out of scope, not disproven. §8.1's candidate order is narrowed by `0003` §4.3, and §8.2's face-cropped pipeline does not carry into that build (`0003` §4.1).
**Supersedes:** nothing outright. Partially supersedes [`0001-architecture-decisions.md`](0001-architecture-decisions.md) §Branch 1 (the CLIP-ViT / DINOv2 rejection) and its Dataset section.
**Relates to:** [`../research/deepfake_detection_research.md`](../research/deepfake_detection_research.md) §1–2, and the three investigations dated 2026-07-26 / 2026-07-27.

---

## Evidence tags

This doc reuses the research doc's convention and adds one:

- **`[confirmed]`** — passed the original research pass's adversarial fact-check, or corroborated by two or more independent sources here.
- **`[unverified]`** — a single source, not independently checked. Treat as a lead, not a fact.
- **`[measured]`** — measured in this project, against our own data or checkpoint. The strongest tag available here; it does not mean the *interpretation* is settled, only the number.

The central architectural claim in this doc is `[confirmed]` at the family level and `[unverified]` at the specific-model level. That distinction is load-bearing and is kept explicit throughout.

---

## 1. What the literature survey found

The research pass (`deepfake_detection_research.md`, 2026-07-23) grouped the field into three architecture families (§2):

| Family | Examples | What the survey recorded |
|---|---|---|
| (a) single CNN backbone | Xception, EfficientNet-B0/B4 | cheap, strong, well-understood baseline |
| (b) CNN+ViT hybrid | Xception + EffNet-B4 → Transformer | ~2x backbone cost for a marginal gain |
| (c) PEFT of a large frozen ViT/CLIP backbone | DeepFake-Adapter, MoE-FFD, OSDFD, LNCLIP-DF | adapters/LoRA on a frozen backbone |

Two findings from that pass matter directly here, and both were recorded correctly at the time:

- **`[confirmed]`** LNCLIP-DF (LayerNorm-only tuning of a frozen CLIP backbone, ~0.03% of params trained) achieved **SOTA cross-dataset AUROC** on CDFv2/FFIW. The survey's own table annotated it: *"strong cross-dataset generalizer if you later need it."*
- **`[confirmed]`** A CLIP-based detector trained on the older but more manipulation-diverse FF++ generalized better across 13 newer benchmarks than models trained on recent datasets — **diversity of manipulation type mattered more than dataset recency**.

The survey also flagged, as Risk 2 (§6), that a model can learn *which dataset an image came from* rather than which manipulation produced it — citing GM-DF's finding that naively combining multi-source datasets degrades accuracy purely from collection/generation differences. This risk is revisited in §6.4 below.

## 2. What was concluded, and the decision made

The survey recommended family (a) and rejected (c) on this reasoning (§1, verbatim):

> Family (c) … **exists specifically to make fine-tuning affordable when the backbone is too large to fully train** (300M+ parameter ViT/CLIP models). At Colab scale, EfficientNet-B4 (~19M params) is already cheap enough to fully fine-tune directly, so the added engineering of custom adapter/LoRA modules buys **compute savings you don't need yet**.

`0001-architecture-decisions.md` carried that forward into the built system: a 3-branch fusion model (EfficientNet-B4 spatial, **fully fine-tuned** + FFT-magnitude CNN + SRM noise-residual CNN → gated fusion), with CLIP-ViT and DINOv2 explicitly considered and set aside (§Branch 1).

## 3. Why that decision was defensible at the time

Recorded plainly, because the decision was not careless and this doc is not a retraction of the reasoning that produced it:

1. **Hard 1.5-hour training budget** on a GPU tier not known in advance (T4/L4/A100). Full fine-tuning a ~19M-param CNN fits; standing up a large-ViT pipeline may not have.
2. **Explainability was requirement 2**, and Grad-CAM's spatial-feature-map assumption transfers cleanly to a CNN but not to a ViT (needs attention-rollout, less mature, doesn't reduce to a clean "spectral: 62%" figure).
3. **CNN inductive bias** (locality, translation-equivariance) is the better bet than a ViT when fine-tuning once on a small, time-boxed dataset.
4. **No eval numbers existed yet.** There was no measured generalization gap to weigh against these costs — only a literature expectation of one.

All four still hold. Points 1 and 3 are specifically arguments against *fine-tuning* a ViT, which is not what this doc proposes (see §7.2).

## 4. The contingency it pre-registered

`0001` did not close the door. It wrote the reopening condition twice:

> **CLIP-ViT (frozen or LayerNorm-tuned)** — … Held in reserve as a v2 branch **if cross-generator generalization proves to be the bottleneck once real eval numbers exist.**

> **DINOv2** — … **Reserved as the v2 generalization branch (superseding CLIP in that role) if a real generalization gap shows up in evaluation.**

`0001`'s Dataset section pre-registered the measurement too: *"hold out a couple of SynthBuster's other generator slices (Midjourney, SD) as eval-only … to measure the cross-generator generalization gap the FFT/SRM branches are supposed to close."*

**This decision doc exists because that condition fired.** The primary basis for what follows is the project's own pre-registered plan, not a new paper.

## 5. What happened when the decision ran

### 5.1 In-distribution: the architecture works `[measured]`

Validation set, best checkpoint (early-stopped on macro-F1): **macro-F1 0.9079**.

| class | precision | recall | ROC-AUC (OvR) |
|---|---|---|---|
| real | 0.848 | 0.921 | 0.962 |
| edited | 0.942 | 0.895 | 0.990 |
| deepfake | 0.949 | 0.898 | 0.976 |

Gate contribution weights came out close to the architecture's stated intent: spectral led for `deepfake` (0.485), noise-residual was relatively stronger for `edited` (0.363 vs. deepfake's 0.225).

### 5.2 Out-of-distribution: it fails, confidently `[measured, weakly]`

Images from Gemini and gpt-image-1, tested ad hoc after training, were **confidently misclassified as `real` (>75% real score)**.

This is tagged weakly on purpose: it was a spot check with no recorded sample size, no fixed image set, and no reproducible script. It is strong enough to establish *that* a gap exists — the direction and confidence are unambiguous — and too weak to quantify it. Quantifying it is part of §9.

### 5.3 Three investigations ruled out the leading alternative explanation `[measured]`

The failure was initially hypothesized to stem from a resampling/blur shortcut, since MTCNN bilinear-upscales small face boxes ~8-12x median across all classes. That hypothesis was pursued to exhaustion:

| Investigation | Result |
|---|---|
| `2026-07-26-upscale-artifact.md` | All three classes land in the same ~10-12x median upscale range — no clean per-class blur-magnitude confound. Remeasured at n=200: real 8.18x, deepfake 8.59x, edited 12.43x. |
| `2026-07-27-fft-srm-template-swap-probe.md` | Rules out a targeted per-class shortcut in the *averaged* fft/srm channels. Scope gap: never varied `rgb`, so it could not reach the spatial branch. |
| `2026-07-27-resolution-swap-probe.md` | Varies `rgb` directly. Rules out a general resolution-magnitude shortcut across 2 of 3 achievable pairs, replicated across a >30% swing in injected perturbation size. One small residual (`real→edited`, mean ΔP +0.0613, 1 flip in 30) left open. |

**Conclusion carried forward:** the cross-generator failure is *not* explained by a resolution/blur shortcut. That leaves the generalization gap needing a different explanation — which §6 and §7 supply.

## 6. New facts established since the original decision

### 6.1 The contingency condition is met `[measured]`

Real eval numbers exist. Cross-generator generalization is the bottleneck. No shortcut explanation survived. The reopening condition written in `0001` is satisfied on its own terms.

### 6.2 The family (c) rejection rested on a framing error `[confirmed]`

The survey characterized frozen backbones as a technique for *making fine-tuning affordable*. That is an accurate description of why LoRA/adapters were invented, and it is not why freezing matters for this problem.

For cross-generator transfer, freezing is not the price of using a large backbone — it is **the mechanism**. Full fine-tuning lets gradient descent rewrite the backbone's representation toward whatever separates the training set fastest, which in a real-vs-DALL-E-3 corpus is that generator's specific artifact signature. The general-purpose pretrained features are not needed to fit the data, so they degrade. Freezing removes that pathway entirely.

The survey recorded the generalization evidence (LNCLIP-DF, §2) but filed the whole family under "compute savings you don't need yet," so the generalization argument was discounted alongside the compute argument. **The evidence was found and correctly noted; the categorization is what caused it to be set aside.**

### 6.3 A fourth option was never enumerated `[confirmed]`

"Frozen" spans a spectrum the survey collapsed into one row:

| Approach | Backbone params trained | Representation |
|---|---|---|
| Full fine-tune — **what we built** | ~19M (100%) | rewritten |
| LoRA / adapters — family (c) as defined | ~1–5% | adapted |
| LayerNorm-only — LNCLIP-DF | ~0.03% | lightly rescaled |
| **Frozen features + linear probe** | **0%** | **untouched** |

The last row is not family (c). It inserts nothing into the network: extract embeddings once, fit a classifier on top. It carries **less** engineering than the 3-branch model we built, not more — which removes the specific objection §1 of the survey raised.

### 6.4 There is no 3-class dataset drawn from a common distribution `[measured]`

This is the root cause of a problem `0001` half-solved. `0001` paired `real` and `deepfake` 1:1 from the same COCO_AI rows specifically so the model could not shortcut on dataset fingerprint. **That protection never extended to `edited`**, which comes from CASIA — a different corpus with different cameras, JPEG histories, and native resolutions.

Nothing in the pipeline closes that gap. The MTCNN crop normalizes framing and output resolution, and the q95 re-save normalizes the *final* compression step, but compression history, sensor statistics, and resolution provenance all survive. And `model/dataset.py` applies **zero augmentation** — no JPEG jitter, no noise, no blur, not a flip — so nothing forces invariance to any of it.

The original evidence for this was circumstantial: errors concentrate where provenance is *shared* (real↔deepfake, both COCO_AI: 46 errors) and nearly vanish where it differs (edited↔deepfake: 1 error), and `edited` posts the **highest ROC-AUC of any class (0.990)** despite being the smallest after face-filtering. The README previously read "edited→deepfake confusions: 0" as unambiguous architectural success, and it is not — but on its own this pattern is also consistent with a genuinely distinctive manipulation signature, not just a fingerprint.

**This has since been measured directly** (`2026-07-29-casia-authentic-probe.md`): CASIA's own untouched `Au_` (authentic) images — never downloaded into the manifest, same corpus/compression/sensor identity as `edited`'s CASIA half, zero manipulation — get classified `edited` **17.4% of the time** (n=161), vs. 2.7% for true COCO_AI `real` images on the same checkpoint. That ~6.4x gap confirms the fingerprint is real and non-trivial. It is a **partial**, not primary, explanation, though: 78.3% of these authentic images are still correctly called `real`, far more than a mostly-fingerprint account of `edited`'s 89.5% recall would predict.

This is also why `edited` is deferred rather than solved (see §8.3): the class cannot be sourced from the same distribution as the other two, because no such dataset exists, and it now carries a measured (if partial) fingerprint confound.

### 6.5 The generator landscape moved `[confirmed]`

As of July 2026: **GPT Image 2** (OpenAI), **Nano Banana Pro** (Google/Gemini), **FLUX.2** (Black Forest Labs, open weights for `dev`), **Midjourney V8.1**, Seedream, Ideogram 4, Recraft V4.1.

**Imagen 4 is deprecated and shuts down 2026-08-17.** It appears as a target in `0001`'s requirement 1 and in the README's future-work section; those references are now stale and should not be built on.

### 6.6 The artifact families have split `[unverified]`

Most open generators (FLUX, SD, Qwen-Image) are latent diffusion: a VAE decoder upsamples from latent space, leaving periodic high-frequency traces. GPT Image is autoregressive/token-based — a different synthesis path with a different artifact profile. DALL-E 3, our only training generator, sits on the diffusion side.

This offers a sharper diagnosis than "unseen generator": our FFT and SRM branches learned diffusion-VAE upsampling traces, and gpt-image-1 output does not carry them. Tagged `[unverified]` — it is a mechanistic hypothesis consistent with 5.2, not something we have measured.

> **Amended 2026-07-31** — see [`../research/2026-07-31-claim-verification.md`](../research/2026-07-31-claim-verification.md). The section above is **half right and half wrong**, and the wrong half was load-bearing.
>
> **Upheld and upgraded to `[confirmed]`:** detectors do key on the global VAE **decode** rather than on synthesized content. Three independent lines: AlignedForensics (ICLR 2025) trains on real vs. autoencoder reconstructions *with no denoising* and transfers to real generated images; AEROBLADE reaches 0.992 mAP training-free from reconstruction error alone; and INP-X collapses pretrained detectors from ~91-94% to ~55% by restoring pixels outside an inpainted region — keeping generated content, removing the global decode.
>
> **Corrected:** the implication that a shared "diffusion family" implies shared artifacts, and therefore transfer. It does not.
> - VAE configurations **diverge** across models and are not interchangeable (SD/SDXL 4ch — mutually incompatible; SD3/FLUX.1 16ch; FLUX.2 32ch). The trend is divergence, not convergence.
> - **Flux Dev is latent diffusion and sits at 21% detection accuracy** in an independent 16-detector benchmark — among the hardest targets measured. Family membership does not predict transfer.
> - Corvi et al. (ICASSP 2023) found strong spectral peaks for SD/LDM but **weak** artifacts for ADM and DALL·E 2 — the artifact is not uniform even within latent diffusion.
>
> **Revised mechanism, replacing the family story:** detectors read the global decode, but sharing "a VAE" buys nothing, because configurations diverge and artifact strength shrinks each generation. **Recency is an axis independent of family.**
>
> **Also reclassified:** the gpt-image-1 autoregressive claim moves from `[unverified]` to `[reported by vendor, not independently confirmed]`. OpenAI describes next-visual-token prediction inside a natively multimodal transformer, but there is no architecture paper and no independent replication. The *token-grid artifact profile* asserted above remains unestablished. Note also that GPT Image 2 and Nano Banana Pro are **products, not raw models** — what we see is decoder output plus an undisclosed post-pipeline.

## 7. What these facts change

### 7.1 Why not fine-tuning helps

Four mechanisms, in decreasing order of how well-established they are:

1. **Fine-tuning is destructive, not additive** `[confirmed]`. Gradient descent finds the fastest-separating feature. Generator artifacts are near-perfect in-distribution predictors, so capacity reallocates toward them and away from broad pretrained features. This is the shortcut-learning mechanism of Geirhos et al. — already cited in `2026-07-26-upscale-artifact.md` — applied to the backbone's own representation.
2. **Freezing removes the pathway** `[confirmed]`. If weights cannot move, no shortcut can be written into the representation.
3. **Low capacity is the regularizer** `[confirmed]`. A ~8k-parameter probe can only read directions *already present* in the frozen space; it cannot construct a DALL-E-3 artifact detector because it has nowhere to put one. Contrast 19M params fitting ~3k images, which is heavily overparameterized.
4. **The empirical signature** `[unverified]`. Frozen CLIP reportedly outperformed its own fine-tuned counterpart by 6.3 points on So-Fake-OOD — same backbone, same data, weights moving as the only difference. Single source; see §7.3.

### 7.2 Why the original objections don't transfer

`0001`'s objections to a ViT branch were: attention cost at high resolution, poor fine-tuning economics on small data, and Grad-CAM incompatibility.

- **Fine-tuning economics** — moot. Nothing is fine-tuned. The objection argued *against* an operation this proposal doesn't perform.
- **Compute** — largely moot. A frozen backbone permits **precomputing embeddings once** for the whole dataset; every subsequent probe experiment is then minutes on CPU. Iteration cost drops rather than rises.
- **Grad-CAM incompatibility** — still valid, and it is the real cost. It is why this is an *additional* path, not a replacement: the 3-branch model retains the explainability deliverable (§8.2).

### 7.3 What is **not** established

The specific claim that **DINOv3** is the current best instance rests on a single arXiv preprint (2511.22471) that was retrieved via automated fetch-and-summarize and **has not been read in full**. It is not peer-reviewed as far as we have checked, its SOTA numbers are self-reported, and no independent replication was located.

Everything specific from it is **`[unverified]`**: the 87.5%/92.6% benchmark figures, DINOv3-over-CLIP, "1k images suffices," the 6.3-point fine-tuning-hurts figure, Fisher-Guided Token Selection.

**LNCLIP-DF is the better-supported first candidate** — already `[confirmed]` in our own bibliography, and evaluated in the face domain (CDFv2/FFIW) rather than on general images.

## 8. Decision

### 8.1 Adopt frozen foundation-model features as a second, additive detection path

Extract embeddings from a **frozen** large pretrained vision backbone; train a **linear probe** on top. No backbone fine-tuning, no adapters, no LoRA. Candidates to compare head-to-head on our own data: **CLIP ViT-L/14** (baseline; UniversalFakeDetect's setup), **LNCLIP-DF** (`[confirmed]`, face-domain), **DINOv2/v3** (`[unverified]`, strongest reported).

### 8.2 Keep the 3-branch fusion model

It is not replaced. It holds macro-F1 0.9079 in-distribution and it is the **only** component satisfying requirement 2 (explainability via gate weights + Grad-CAM). The frozen path is expected to *lose* to it in-distribution and win out-of-distribution; those are complementary, not competing, roles. This is exactly the "v2 generalization branch" `0001` reserved.

### 8.3 Narrow to real vs. deepfake, provisionally

`edited` is **deferred, not dropped**. The reason is §6.4: no dataset provides all three classes from a common distribution, so `edited` cannot currently be added without reintroducing a corpus fingerprint. Binary real/deepfake is the tractable exploration; folding `edited` back in — with matched provenance, or with the fingerprint measured rather than assumed — remains an open goal, tracked in §11.

### 8.4 Augmentation becomes mandatory

Currently zero (`model/dataset.py`). Adopt the standard protocol: Gaussian blur σ ∈ [0,2] and JPEG quality ∈ {60,70,80,90,100} at training time `[confirmed]`. This is independent of the frozen-backbone decision and applies to both paths. Noise robustness is the documented weak point of frozen-feature detectors (reported degradation to ~55% at Gaussian σ=10, `[unverified]`), so it must be trained for, not assumed.

### 8.5 Multi-generator data, with a held-out eval split

Train on ≥2 generators (FLUX.2-dev and SDXL are open-weight and runnable on the existing A100 at no marginal cost). Hold out **GPT Image 2, Nano Banana Pro, and Midjourney V8.1** as eval-only, never trained on. This finally executes the measurement `0001` pre-registered and never ran. Not Imagen 4 — see §6.5.

> **Amended 2026-07-31** — see [`../research/2026-07-31-claim-verification.md`](../research/2026-07-31-claim-verification.md) §4.1.
>
> Do **not** design the held-out split by grouping generators into architecture families and taking one representative each. Family is not the right unit (§6.6 amendment): Flux Dev is latent diffusion and is among the hardest generators measured, at 21%. A FLUX / SD3 / SDXL holdout is *not* redundant, and FLUX must not be dropped from an eval set on the grounds that it shares a family with the training data.
>
> Two additions to the protocol:
> - **Report a robustness surface, not a point.** TPR at fixed low FPR across a JPEG-quality × downscale grid. Augmentation with common post-processing improves generalization even when test images are not post-processed (Wang et al. 2020, 92.6% AP) — this reinforces §8.4 independently.
> - **Budget generator count with the diminishing-returns curve in mind.** Community Forensics improves monotonically with generator count but flattens beyond ~1,000 models. The curve is steep early, so the first few additional generators carry most of our achievable marginal return — which is the argument for ≥2 being a floor, not a target.

## 9. The gating experiment — how this gets falsified

§8 is **Accepted (conditional)**. The condition is a single experiment, runnable against data already on disk, with no new sourcing:

1. Extract frozen embeddings for the existing val split from each candidate backbone.
2. Fit a linear probe on the existing train split.
3. Compare against the 0.9079 checkpoint on **(a)** the in-distribution val split and **(b)** the Gemini / gpt-image-1 images that already fail.

**Falsification condition:** if frozen features do not beat the current model on the images it demonstrably fails (3b), this decision is wrong and should be reverted, at a cost of roughly one afternoon. In-distribution parity (3a) is *not* required for the decision to hold — see §10.

This also settles CLIP vs. DINOv2/v3 vs. LNCLIP-DF empirically, on our data, rather than by deferring to one preprint's claim.

Fixing 5.2 into a reproducible, versioned OOD image set is a prerequisite and part of this work.

## 10. Accepted costs and risks

- **In-distribution accuracy will likely drop on the frozen path.** Fine-tuned models usually beat frozen probes in-distribution. 0.9079 may not be matched. Accepted, because §8.2 keeps the fine-tuned model for that role.
- **Explainability does not transfer cleanly to a ViT.** Grad-CAM assumes spatial feature maps. Accepted for the same reason.
- **Frozen features only work if pretraining encoded something relevant.** The approach depends on web-scale pretraining having captured natural-image coherence. If the frozen space has no separating axis, a linear probe cannot manufacture one. §9 tests this directly.
- **Freezing does not confer corruption robustness.** It addresses generator transfer only; §8.4 is a separate, independent requirement.
- **Freezing mitigates but does not eliminate corpus fingerprint.** A probe can still find a corpus-separating hyperplane if one exists in the frozen space. Relevant when `edited` returns (§8.3).
- **The strongest supporting result is `[unverified]`.** Mitigated by §9 replacing deference with measurement, and by treating LNCLIP-DF (`[confirmed]`) as the primary candidate.
- **A frozen public backbone makes the feature space public — an adversarial liability.** `[confirmed]`, added 2026-07-31. Knowledge of the frozen ViT backbone architecture *alone* is sufficient to craft gray-box adversarial examples reaching near-white-box attack success rates, even under complete training misalignment ([Backbone is All You Need](https://arxiv.org/abs/2605.13381)). This is a property of the frozen-backbone design, not of contamination, and it does not affect §9 — but it is a real cost of this path and should be stated rather than discovered later. Accepted: the generalization benefit is the reason for the path, and no adversary is in scope for the current evaluation.
- **Confidence is not a usable abstention signal on this failure mode.** `[confirmed]`, added 2026-07-31. Failure on unseen generators is *confident* and biased toward `real` (§5.2 is an instance of a documented general property — see the verification doc §1.2). The model is not uncertain, it is wrong and sure, so calibration cannot recover it. Any abstention mechanism must key on **distance to the training distribution** (kNN / Mahalanobis on frozen features), not on output probability.

  **Amended 2026-08-01** — the two conclusions above are **withdrawn**; the premise is retained. See [`../research/2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md). This is the project's first overturned `[confirmed]` tag.
  - *"Calibration cannot recover it"* — **contradicted.** [Yang et al., AAAI 2026](https://arxiv.org/abs/2602.01973) attributes the cross-generator failure to **misaligned decision thresholds rather than a loss of feature separability**, and corrects it with a learnable scalar logit adjustment fitted on a small target-distribution set, backbone frozen, with a label-free variant. A confidently-wrong model can still rank correctly: a monotone rescaling moves the boundary without changing the ordering. The 07-31 reasoning did not separate *boundary placement* from *separability*.
  - *"Abstention must key on distance, not confidence"* — **unsupported.** [Jaeger et al., ICLR 2023](https://arxiv.org/abs/2211.15259) `[unverified — second-hand]` reports that no evaluated confidence-scoring method beats a Maximum Softmax Response baseline across a realistic range of failure sources, with Mahalanobis winning only on far-OOD. Revised position: **measure both distance and confidence on our own data**; do not assume distance wins.
  - **New prerequisite, not previously recorded.** [Alexandari et al., ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html) establishes that prior/base-rate correction *assumes a calibrated `p(y|x)`*. Calibration is a precondition for the base-rate work in [`../research/2026-07-31-production-deployment.md`](../research/2026-07-31-production-deployment.md), not an alternative to it. **Our checkpoint's calibration has never been measured** — see §11.
  - **Boundary of the fix.** Prior-shift methods assume label shift (`p(y)` changes, `p(x|y)` fixed). An unseen generator changes `p(x|fake)`, so they are outside their proven assumptions here. Any correction applied is empirically motivated, not theoretically licensed, and should be reported as such.

## 11. Open items

- Run §9. Nothing downstream should be built before it reports.
- Build the reproducible OOD eval set (§5.2 is currently unquantified).
- Fold `edited` back in once a provenance-matched source exists. The fingerprint itself is now partially measured (§6.4, `2026-07-29-casia-authentic-probe.md`: 17.4% false-`edited` rate on authentic CASIA images, n=161) — PS-Battles' "original" half is still unmeasured, and a tighter n on the CASIA side would help before this closes out.
- The `real→edited` residual from `2026-07-27-resolution-swap-probe.md` remains open and unpursued — small, isolated, low priority.
- Per-branch auxiliary classifier heads (noted in `notes.md`, README future work) — still wanted, still requires retraining, sequenced after §9.

Added 2026-07-31, from [`../research/2026-07-31-claim-verification.md`](../research/2026-07-31-claim-verification.md) §5:

- **Test the decode-artifact mechanism against our own checkpoint.** §1.1 of the verification doc establishes that detectors key on the global VAE decode *for the field*; it does not establish it for our FFT/SRM branches. Testable on data we already hold, and it would convert §6.6 from hypothesis to measurement.
- ~~**Decompose the 18-30% commercial-generator figures.**~~ **Answered 2026-08-01 — substantially misplaced thresholds.** Those are accuracies at each method's own threshold on balanced sets; the benchmark does not separate lost separability from misplaced thresholds. Whether the signal is gone or merely mis-thresholded materially changes how pessimistic the outlook should be — and it is the difference between "retrain" and "recalibrate." [Yang et al., AAAI 2026](https://arxiv.org/abs/2602.01973) resolves this in favour of **recalibrate**: the failure is attributed to misaligned decision thresholds rather than lost feature separability, recoverable post-hoc with the backbone frozen. The outlook is correspondingly less pessimistic than §6.5 assumed. See [`../research/2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md) §2.

Added 2026-08-01, from [`../research/2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md) §5:

- **Measure this checkpoint's in-distribution calibration.** Never done. A reliability diagram + ECE on the existing val set needs no new data, no API cost, and no retraining — and it is the precondition Alexandari establishes for any base-rate or threshold correction (§10). Cheapest open item on this list.
- **Verify the calibration-set size in Yang et al.** The abstract says only "a small validation set"; the ~100-image figure the current plan assumes is second-hand. The OOD eval set (§11, above) is specced to serve as both the failure measurement *and* the calibration set — if the real requirement is materially larger, that plan needs revisiting.
- **Adapt the scalar logit correction to 3 classes.** Yang et al. is binary. Temperature scaling extends natively; the per-class correction does not, and how it is adapted is a design decision.
- **Test whether ranking survives on our OOD set.** The local replication of the §2 mechanism. If AUC collapses rather than the threshold merely shifting, the correction does not apply here — this is the discriminating test.
- **`edited` may be the wrong shape of problem, not just the wrong dataset.** §8.3 deferred it on sourcing grounds. INP-X now adds a mechanistic reason: detectors trained on global synthesis fall to ~55% on localized manipulation, with ~75% the ceiling even when trained directly for it. If `edited` returns, it likely returns as a **localization** task (per-pixel mask + pooled image score), not as a third head on a global classifier.
- **The one-class / camera-anchor direction needs its own pass before it is relied on.** PRNU is losing per-device uniqueness to computational photography, and neural ISPs hallucinate content. No positive definition of the `real` class that survives this was located.
- **Resolve the FLUX.2 autoencoder compression-ratio conflict** (verification doc §2.1) if FLUX.2 becomes a training or eval target.

## Sources

Carried from the research doc's bibliography:

- [3] [arXiv:2306.00863 — DeepFake-Adapter](https://arxiv.org/pdf/2306.00863)
- [11] [arXiv:2406.20078 — GM-DF, domain gap across deepfake sources](https://arxiv.org/pdf/2406.20078)
- [12] [arXiv:2404.08452 — MoE-FFD](https://arxiv.org/pdf/2404.08452)
- [13] [arXiv:2508.06248 — LNCLIP-DF](https://arxiv.org/html/2508.06248v1) — `[confirmed]`, primary candidate
- [15] [arXiv:2408.12791 — OSDFD](https://arxiv.org/pdf/2408.12791)

New to this decision:

- [Rethinking Cross-Generator Image Forgery Detection through DINOv3](https://arxiv.org/html/2511.22471v1) — `[unverified]`, not read in full, retrieved via automated summary. See §7.3.
- [Methods and Trends in Detecting Generated Images: A Comprehensive Review (2025)](https://arxiv.org/html/2502.15176v1)
- [A Bias-Free Training Paradigm for More General AI-generated Image Detection](https://arxiv.org/pdf/2412.17671)
- [Detection of Synthetic Face Images: Accuracy, Robustness, Generalization](https://arxiv.org/html/2406.17547) — augmentation protocol in §8.4
- Geirhos et al., [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z), Nat. Mach. Intell. 2 (2020) — mechanism in §7.1
- [Best AI Image Generators July 2026](https://www.buildmvpfast.com/articles/best-llms-2026-guide/image-generation-ai) — generator landscape, §6.5

Added by the 2026-07-31 verification pass — full bibliography and per-claim status in [`../research/2026-07-31-claim-verification.md`](../research/2026-07-31-claim-verification.md):

- [Aligned Datasets Improve Detection of Latent Diffusion-Generated Images (ICLR 2025)](https://arxiv.org/abs/2410.11835) — decode-artifact mechanism, §6.6 amendment
- [AEROBLADE (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.pdf) — same
- [Inpainting Exchange / INP-X](https://arxiv.org/html/2602.00192) — same, and the `edited`-as-localization item in §11
- [Community Forensics (CVPR 2025)](https://arxiv.org/abs/2411.04125) — generator-count lever, §8.5 amendment
- [Open-sourced detector benchmark](https://arxiv.org/html/2602.07814v1) — the 18-30% commercial-generator figures, §6.6 and §8.5 amendments
- [Breaking Latent Prior Bias](https://arxiv.org/pdf/2506.00874) / [GenDet](https://arxiv.org/html/2312.08880) — confident-failure-toward-`real`, §10
- [Backbone is All You Need](https://arxiv.org/abs/2605.13381) — adversarial risk of a public frozen backbone, §10
- [notes_on_sd_vae](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9) — VAE divergence across model families, §6.6 amendment
- [CNN-generated images are surprisingly easy to spot… for now (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf) — independent support for §8.4
