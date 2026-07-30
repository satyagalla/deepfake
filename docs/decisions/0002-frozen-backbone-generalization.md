# 0002 — Frozen Foundation-Model Features as the Generalization Path

**Status:** Accepted (conditional — see §9, the gating experiment)
**Date:** 2026-07-28
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

The results are consistent with the fingerprint having been learned: errors concentrate where provenance is *shared* (real↔deepfake, both COCO_AI: 46 errors) and nearly vanish where it differs (edited↔deepfake: 1 error), and `edited` posts the **highest ROC-AUC of any class (0.990)** despite being the smallest after face-filtering. This is not proof — a genuinely distinctive manipulation signature would look similar — but the README previously read "edited→deepfake confusions: 0" as unambiguous architectural success, and it is not.

This is also why `edited` is deferred rather than solved (see §8.3): the class cannot be sourced from the same distribution as the other two, because no such dataset exists.

### 6.5 The generator landscape moved `[confirmed]`

As of July 2026: **GPT Image 2** (OpenAI), **Nano Banana Pro** (Google/Gemini), **FLUX.2** (Black Forest Labs, open weights for `dev`), **Midjourney V8.1**, Seedream, Ideogram 4, Recraft V4.1.

**Imagen 4 is deprecated and shuts down 2026-08-17.** It appears as a target in `0001`'s requirement 1 and in the README's future-work section; those references are now stale and should not be built on.

### 6.6 The artifact families have split `[unverified]`

Most open generators (FLUX, SD, Qwen-Image) are latent diffusion: a VAE decoder upsamples from latent space, leaving periodic high-frequency traces. GPT Image is autoregressive/token-based — a different synthesis path with a different artifact profile. DALL-E 3, our only training generator, sits on the diffusion side.

This offers a sharper diagnosis than "unseen generator": our FFT and SRM branches learned diffusion-VAE upsampling traces, and gpt-image-1 output does not carry them. Tagged `[unverified]` — it is a mechanistic hypothesis consistent with 5.2, not something we have measured.

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

## 11. Open items

- Run §9. Nothing downstream should be built before it reports.
- Build the reproducible OOD eval set (§5.2 is currently unquantified).
- Fold `edited` back in once a provenance-matched source exists, or measure the fingerprint via a held-out second edited corpus (§6.4, §8.3).
- The `real→edited` residual from `2026-07-27-resolution-swap-probe.md` remains open and unpursued — small, isolated, low priority.
- Per-branch auxiliary classifier heads (noted in `notes.md`, README future work) — still wanted, still requires retraining, sequenced after §9.

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
