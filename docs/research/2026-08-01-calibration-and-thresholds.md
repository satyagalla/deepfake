# Calibration, Base Rates, and Threshold Placement — Why the Cross-Generator Failure Is Partly Recoverable

**Status:** Current as literature; **build recommendations descoped** — see [`../decisions/0003-frozen-probe-demo-build.md`](../decisions/0003-frozen-probe-demo-build.md) §1, §6
**Date:** 2026-08-01

> **Note added later on 2026-08-01.** §4's build order (in-distribution calibration measurement → OOD eval set → threshold correction on v1) and §5's open items are **out of scope**, not disproven: the research chapter on the v1 checkpoint closed the same day. The literature findings below stand unchanged and remain the basis for a conceptual answer on calibration, base rates, and threshold placement — they are simply not being implemented before 2026-08-03.
**Relates to:** [`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) §1.2 and §5 — **this pass reverses §1.2's calibration conclusion** · [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §5.2, §10 · [`2026-07-31-production-deployment.md`](2026-07-31-production-deployment.md) (base-rate section)

The 07-31 verification pass concluded that calibration could not address the cross-generator failure, and tagged it `[confirmed]`. That conclusion was wrong. A dedicated pass over the calibration literature finds the failure is **substantially a threshold-placement problem**, not purely a loss of separability — which makes a large part of it recoverable at near-zero cost, without retraining and without touching the checkpoint.

This is the first time a `[confirmed]` tag in this project has been overturned. The tag was applied to a mechanistic argument that was internally sound but had never been checked against the calibration literature specifically. Recorded here as a methodology result as well as a technical one — see §6.

**Evidence tags.** Repo convention: `[confirmed]` · `[measured]` · `[unverified]`. One addition used throughout this doc: `[second-hand]` — reported by a subagent literature survey on 2026-07-31 and **not** re-checked against the primary source. Second-hand numbers are flagged individually because several sit underneath build decisions.

---

## 1. The prerequisite theory

Three facts, in dependency order. Each is needed to make sense of the one after it.

### 1.1 A model's output is not a probability until measured `[confirmed]`

A score in [0,1] is a number. Whether `0.9` means "90% of images like this are fake" is a separate empirical fact, established by bucketing scores and counting outcomes — a *reliability diagram*.

**Temperature scaling** is the standard correction: divide the logits by a single learned scalar before softmax, fitted on a held-out split ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)). Two properties from that paper matter here:

- It is a single parameter, optimized for held-out likelihood.
- **It does not change the most-confident prediction**, so accuracy under 0-1 loss is unchanged. It rescales confidence only.

Guo et al. also identify depth, width, weight decay, and Batch Normalization as factors influencing calibration.

**Qualification — do not assume the direction of miscalibration.** [Minderer et al., *Revisiting the Calibration of Modern Neural Networks*](https://openreview.net/pdf?id=QRBvLayFXI) reports that newer architectures are better calibrated than the generation Guo tested. The "modern nets are overconfident" heuristic is architecture-dependent, not a law. `[confirmed that the paper makes this challenge; not read in full]`

**Consequence for this project:** our checkpoint's calibration is an open `[measured]` item, not something to be assumed. A reliability diagram on the existing val set requires no new data and settles it.

### 1.2 Base-rate shift changes the conclusion without changing the evidence `[derivable]`

Training is balanced (50/50). Deployment prevalence is on the order of 0.5%. The model carries the training prior baked in.

Worked in odds form:

```
model outputs 0.9, assumed calibrated at training prevalence

odds_train       = 0.9   / 0.1   = 9.0
prior_odds_train = 0.5   / 0.5   = 1.0
prior_odds_prod  = 0.005 / 0.995 = 0.005025

odds_prod = 9.0 × (0.005025 / 1.0) = 0.0452
p_prod    = 0.0452 / 1.0452        = 0.043
```

**A "90% confident fake" is ~4.3% likely to be fake at 0.5% prevalence** — roughly 1 in 23. The evidence strength is unchanged; the conclusion changed because the prior did.

This is Bayes in odds form and needs no citation. It is the same phenomenon as the precision figure in [`2026-07-31-production-deployment.md`](2026-07-31-production-deployment.md), derived from the other direction.

For rare events, the sample/population mismatch affects only the intercept, so a closed-form logit offset suffices ([King & Zeng 2001](https://gking.harvard.edu/files/0s.pdf)) `[second-hand]`. The arithmetic above stands independently of that citation.

### 1.3 Calibration is a prerequisite for base-rate correction, not an alternative to it `[confirmed]`

Re-run §1.2 assuming the model is overconfident — its `0.9` empirically corresponds to a true `0.6`:

```
odds_train = 0.6 / 0.4 = 1.5
odds_prod  = 1.5 × 0.005025 = 0.00754
p_prod     = 0.0075
```

**0.043 vs 0.0075 — nearly 6× apart.** Same image, same model, same correction. The only difference is whether the input to the correction was an honest probability.

[Alexandari, Kundaje & Shrikumar, ICML 2020](https://proceedings.mlr.press/v119/alexandari20a.html) states this as the motivation for their method: the Saerens et al. (SLD/EM) maximum-likelihood prior-correction algorithm **"assumes p(y|x) is calibrated, which is not true of modern neural networks."** Their contribution is that maximum likelihood combined with **bias-corrected calibration** outperforms BBSL and RLLS.

The correct term is *bias-corrected calibration*, not "bias-corrected temperature scaling."

**The pipeline is ordered and the order is not optional:**

```
raw logits → calibration → honest probability → prior correction → deployment probability
```

Feeding an uncalibrated score into a prior correction does not raise an error. It returns a plausible-looking wrong number, which is the more dangerous outcome.

### 1.4 Where the theory stops: label shift vs. a changed `p(x|fake)` `[confirmed]`

Every method in §1.3 assumes **label shift**, defined in Alexandari's opening sentence: `p(y)` changes between train and test while `p(x|y)` stays fixed. The class *proportions* move; what each class *looks like* does not.

An unseen generator violates this. A DALL·E 3 image and a FLUX image are both labeled `fake`, but they are written by different decoders leaving different artifacts (see [`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) §1.1, §2.1). The `fake` class has not merely become rarer — **it looks different. `p(x|fake)` changed shape.**

**So SLD/EM, BBSE, and King & Zeng are formally outside their assumptions on this problem.** Not known to fail — *unlicensed*. Any use of them here is empirical, and should be stated as such.

This is a genuine boundary of the theory and it is worth carrying explicitly. It is also why §2 is a notable empirical result rather than a routine application.

---

## 2. The reversal: threshold placement, not lost separability

### 2.1 The finding `[confirmed from abstract]`

[Yang et al., AAAI 2026 — *Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated*](https://arxiv.org/abs/2602.01973). Verified against the abstract:

- Detectors systematically misclassify generated images as real.
- The cause is attributed to distributional shift and overfitting to superficial artifacts producing **misaligned decision thresholds rather than a loss of feature separability.**
- The method is a post-hoc calibration framework grounded in Bayesian decision theory: a **learnable scalar correction to the model's logits**, optimized on a small validation set from the target distribution, **backbone frozen**.
- **A variant requiring no ground-truth labels exists.**

### 2.2 What is not yet verified `[unverified / second-hand]`

These were reported by the 07-31 calibration subagent and have **not** been checked against the paper body. They are listed separately because one of them sits underneath a build decision.

| item | reported value | status |
|---|---|---|
| Calibration set size | ~100 images | **Not stated in the abstract.** The abstract says only "a small validation set." Load-bearing — see §4. |
| Per-detector gains | RINE 81.78→97.94, AIDE 60.22→75.61, Effort 79.35→89.48, Fusing 68.85→78.42, CNNSpot 70.83→78.22 | `[unverified]` |
| Unsupervised form | `α* = E[z]`, centre of mass of the logit distribution; trails supervised by 1–2 points | `[unverified]` |
| Runtime | ~0.5–0.9 ms | `[unverified]` |
| Stated limitation | On Chameleon, where real and fake logit distributions genuinely *overlap* rather than merely shift, scalar correction stops working | `[unverified]` — but conceptually the most important of the five |

The Chameleon caveat, if it holds, defines the method's boundary: **a shifted-but-separable distribution is fixable by moving a threshold; a collapsed one is not.**

### 2.3 What this overturns

[`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) §1.2 asserts, tagged `[confirmed]`:

> *"Calibration cannot fix it. The model is not uncertain; it is wrong and confident. Temperature scaling rescales a confidently-wrong score."*
> *"Abstention must key on distance to the training distribution, not on output confidence — confidence is precisely the broken signal."*

**Both halves are contradicted.**

The first by §2.1 above. The reasoning behind the original claim was not wrong about the *mechanism* — the model genuinely is confident and wrong — but it drew an incorrect inference: a decision boundary can sit in the wrong place while the ranking of scores remains informative. Ordering survives what thresholding destroys. That distinction was not made.

The second by [Jaeger et al., ICLR 2023](https://arxiv.org/abs/2211.15259) `[second-hand]`, reported as the largest unified study of failure detection (15+ confidence scoring functions across covariate shift, sub-class shift, new-class shift, and i.i.d.), concluding that no evaluated method beats the Maximum Softmax Response baseline across a realistic range of failure sources — with Mahalanobis and MaxLogit winning only on non-semantic new-class (far-OOD) shifts. Also reported: softmax-response scores are scale-sensitive, so temperature scaling alters abstention behaviour without altering accuracy, and a scale-invariant **margin** (top-two logit gap) outperforms softmax-based scores under shift.

Honest position: **use both distance and confidence, and measure which wins on our data.** The 07-31 doc asserted distance beats confidence; that is unsupported.

This also closes 07-31 §5's open item — *"whether the 18–30% figures reflect lost separability or misplaced thresholds."* The answer, per §2.1, is substantially **misplaced thresholds**, which makes the outlook meaningfully less pessimistic than that doc concluded.

---

## 3. Calibration under a varying evidence set — an unresolved structural problem

Relevant to the evidence-card / fusion direction rather than to the current checkpoint.

### 3.1 The problem, by construction `[derivable]`

A fused score computed as a mean over whichever evidence dimensions are available is not on a stable scale:

| case | fused score |
|---|---|
| 6 cards, each 0.6 | 0.60 |
| 2 cards at 0.6, 4 absent | 0.60 |

Identical number, very different evidential weight. A fixed threshold cannot separate them. A single calibration map fitted on a pool dominated by full-card images will be wrong on the sparse-card slice — and sparse-card images (platform-sourced, metadata stripped, recompressed) are the common case, not the edge case.

### 3.2 The theory predicts this `[confirmed]`

[Hébert-Johnson, Kim, Reingold & Rothblum, ICML 2018 — *Multicalibration*](https://proceedings.mlr.press/v80/hebert-johnson18a.html) formalizes that a predictor can be calibrated in aggregate while badly miscalibrated on subgroups, and that the guarantee holds only for subgroups within a **specified class of computations**. "Which evidence dimensions fired" is computationally identifiable — it can be enumerated — so it is exactly the kind of subgroup the theory covers.

### 3.3 What could not be found — and how weak that is `[unverified]`

The 07-31 subagent searched calibration + missing features, calibration under variable input sets, ensemble calibration with member dropout, missing-modality calibration, and multibiometric fusion with missing scores, and reported finding no work addressing the question head-on.

**A bounded search returning nothing is the weakest form of evidence in this document.** The defensible statement is: *the theory predicts this failure, and no positive evidence was found that anyone has measured it for a fused forensic system.* It is **not** "genuinely open in the literature," which is how it was phrased on 07-31.

The nearest mature treatment is multibiometric score fusion. [Nandakumar, Chen, Dass & Jain, TPAMI 2008](https://www.ncbi.nlm.nih.gov/pubmed/18084063) is `[confirmed]` as the likelihood-ratio fusion framework, modelling genuine/impostor score distributions as finite Gaussian mixtures and explicitly handling arbitrary score scales, correlated matchers, and sample quality. The specific claim that *most score-level fusion rules assume all matcher scores are available and are not equipped for missing scores* was **not** located in a primary source and is `[unverified]` — do not quote it.

---

## 4. What this changes

### 4.1 Build order

Threshold correction moves ahead of the frozen probe. The argument is cost, not novelty:

| | threshold correction | frozen probe (0002 §9) |
|---|---|---|
| new data required | none beyond the OOD eval set | none, but full extraction pass |
| touches checkpoint | no | no |
| training | none | probe fit |
| targets | the exact measured failure (0002 §5.2) | a different architecture, unvalidated |

A step is added ahead of both: **measure in-distribution calibration on the existing val set.** Zero new data, zero API cost, and it is the prerequisite §1.3 establishes. Doing prior or threshold correction on an unmeasured model is exactly the error Alexandari identifies.

Revised order: **in-distribution calibration measurement → OOD eval set → threshold correction on v1 → per-branch decomposition → frozen probe (first to cut).**

### 4.2 An unresolved dependency

The OOD eval set is currently specced at ~150 images and is assumed to double as the calibration set for §2. **That assumption rests on the unverified "~100 images" figure in §2.2.** If the real requirement is materially larger, the OOD set is undersized for both jobs and the plan needs revisiting. Verify against the paper body before relying on it.

### 4.3 A 3-class adaptation is required

Yang et al. is binary; this project's checkpoint is 3-class. Temperature scaling extends natively (one scalar over the logit vector), but the per-class scalar logit correction requires a deliberate adaptation. This is a design decision, not an implementation detail.

### 4.4 Downstream of §1.4

Any prior-shift correction applied here is empirically motivated, not theoretically licensed, because an unseen generator changes `p(x|fake)`. This should be stated wherever such a correction is reported.

---

## 5. Still open

- **The calibration-set size in Yang et al. (§2.2, §4.2).** Highest priority — a build decision depends on it.
- The remaining `[unverified]` items in §2.2: gains table, unsupervised formula, Chameleon limitation.
- [Jaeger et al.](https://arxiv.org/abs/2211.15259) has not been read against the primary source (§2.3).
- Minderer et al. has not been read in full; the extent to which its finding applies to EfficientNet-B4-based architectures is unknown (§1.1).
- **This project's own checkpoint has never had its calibration measured.** `[measured]` — pending, and the cheapest open item on the list.
- Whether ordering (AUC/ranking) survives on our OOD set the way Yang et al. reports for theirs. This is the local replication of §2.1 and determines whether the correction is applicable here at all.

---

## 6. Methodology note — why a `[confirmed]` tag failed

The 07-31 §1.2 claim was mechanistically reasoned and multi-sourced on its *premise* (detectors fail confidently toward `real`, supported by Breaking Latent Prior Bias and GenDet). The `[confirmed]` tag was applied to the premise, but the doc then carried an **inference** — *therefore calibration cannot help* — at the same confidence, without a source and without a search of the calibration literature.

The failure mode is specific and worth naming: **a confirmed premise laundering an unconfirmed inference.** The premise (the model is confident and wrong) does not entail the conclusion (confidence cannot be corrected), because a monotone rescaling can move a boundary without changing an ordering.

Consistent with [`2026-07-31-literature-triage.md`](2026-07-31-literature-triage.md)'s principle of tracking claims rather than papers, this suggests the unit should be finer still: **premise and inference need separate tags even inside a single bullet.**

## Sources

- [On Calibration of Modern Neural Networks (Guo et al., ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html)
- [Revisiting the Calibration of Modern Neural Networks (Minderer et al.)](https://openreview.net/pdf?id=QRBvLayFXI)
- [Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation (Alexandari, Kundaje & Shrikumar, ICML 2020)](https://proceedings.mlr.press/v119/alexandari20a.html)
- [Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated (AAAI 2026)](https://arxiv.org/abs/2602.01973)
- [Multicalibration: Calibration for the (Computationally-Identifiable) Masses (ICML 2018)](https://proceedings.mlr.press/v80/hebert-johnson18a.html)
- [A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification (Jaeger et al., ICLR 2023)](https://arxiv.org/abs/2211.15259)
- [Likelihood Ratio-Based Biometric Score Fusion (Nandakumar, Chen, Dass & Jain, TPAMI 2008)](https://www.ncbi.nlm.nih.gov/pubmed/18084063)
- [Logistic Regression in Rare Events Data (King & Zeng, 2001)](https://gking.harvard.edu/files/0s.pdf)
- [Label shift experiments (code, Kundaje lab)](https://github.com/kundajelab/labelshiftexperiments)
