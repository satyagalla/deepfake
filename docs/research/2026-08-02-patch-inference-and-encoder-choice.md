# Patch-Based Inference and Encoder Choice

**Date:** 2026-08-02 (fourth 08-02 pass)
**Status:** Current
**Feeds:** [`../decisions/0004-adaptation-hypothesis-demo-build.md`](../decisions/0004-adaptation-hypothesis-demo-build.md) §6, §7.2, §10
**Amends:** [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md) §2.1 (adds a fourth resolution problem the table missed), §3.1 (overrides the two-arm encoder recommendation), §3.2 (closes the L2-normalization item by decision)

Evidence tags per `../README.md`: `[confirmed]` · `[measured]` · `[unverified]` · `[second-hand]`.

---

## 1. The question this pass answers

`bottlenecks.md` §2.1 splits "resolution" into three independent problems and assigns each a fix. The third — *source images are natively 270²–480px* — is marked **"Nothing cheap."**

This pass asks whether that is actually true, and finds that it is true *as stated* but that the table is missing a fourth problem which is both cheaper and more damaging than any of the three listed.

## 2. The fourth resolution problem: asymmetric downsampling

§2.1's three problems are all about *how much resolution reaches the encoder*. The missing one is about **how much resolution is removed, and whether that amount correlates with the label.**

CLIP ViT-L/14 takes 224×224 `[confirmed]`. Under the standard resize-and-centre-crop pipeline:

| Source | Native | Downscale to 224 |
|---|---|---|
| COCO real | ~480px `[second-hand — §9.1 open]` | ~2.1× |
| Gemini / GPT-Image | 1024px | ~4.6× |

Two distinct harms:

1. **Signal destruction.** Downsampling is a low-pass filter. The high-frequency band is where both the generator fingerprint (artifact type A) and the decode signature (type B) live, per the taxonomy in `2026-08-02-session-findings.md`. A 4.6× downscale removes most of it.
2. **Confound creation.** The downscale *ratio* differs systematically by class, and a resampling kernel leaves a ratio-dependent signature. So the resize step manufactures a label-correlated artifact that was not present in the source data.

Harm 2 is the serious one, and it is structurally identical to `bottlenecks.md` §2.1(b) — the `IMAGE_SIZE = 380` square resize in `model/dataset.py:48`. **The same bug survives the move to a new model, one stage further down the pipeline.** §2.1(b)'s prescribed fix ("crop instead of resize") is correct and was never carried into the frozen-probe plan.

This also reframes §2.1(c). "Nothing cheap" fixes the *absolute* resolution of the corpus, which is true. But the *asymmetry* is fixable and is the part that produces a confound rather than merely a weak signal.

## 3. Patch-based inference

### 3.1 Prior art

**Chai, Bau, Lim & Isola, "What Makes Fake Images Detectable? Understanding Properties that Generalize," ECCV 2020** `[confirmed]` — [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710103.pdf) · [code](https://github.com/chail/patch-forensics).

Confirmed at the abstract/summary level: the method truncates a classifier after an intermediate block to obtain predictions from a *local region*, with earlier truncation giving a smaller receptive field; the paper's subject is which properties of fake images generalize across architectures, datasets and training variations; and it reports that even a generator adversarially finetuned against a detector still leaves detectable artifacts **in certain image patches**.

`[unverified]` — that patch aggregation *outperforms* whole-image classification on cross-generator transfer, and the specific aggregation rule used. Not read at the body. This is flagged deliberately: this project has twice been damaged by second-hand claims (`plan-b-verification.md` §1's reliance on the since-withdrawn DailyBench; §4's misattribution of a real-accuracy figure to SSAFE). The patch decision in `0004` §6 does **not** rest on this claim — it rests on §2 above, which is a mechanism argument about our own pipeline.

What the confirmed part does establish: **the artifact is not uniformly distributed across an image**, and *some patches carry it while others do not*. That is the premise §4 builds on.

### 3.2 What patching fixes

A native-resolution crop is the **same operation regardless of source resolution**. It therefore:

- introduces no resolution-dependent resampling signature (fixes harm 2)
- preserves the high-frequency band intact (fixes harm 1)
- requires no upsampling of small images, which §2.1's inversion argument identifies as actively harmful

The benefit is **asymmetric, and correctly so.** At 270px a 224 patch is 83% of the image, so patching is nearly a no-op at the low end of the corpus. At 1024px it is a genuine change. The resize was crushing generated images 4.6× while barely touching COCO — patching removes the damage precisely where the damage was.

### 3.3 What patching does not fix

It converts a **resolution** confound into a **field-of-view** confound. A 224 patch is 47% of a 480px image and 22% of a 1024px image, so the two views see content at different scales. This is real and not eliminated.

The trade is nonetheless correct, and the reason is general enough to state as a principle:

> **Prefer a confound you can hold fixed to one you cannot.** Patch count and patch size are free parameters we control. The native resolution of COCO_AI is not.

### 3.4 What patching costs

Patches cannot see composition, object-level coherence, anatomy, or globally inconsistent lighting. CLIP is a *semantic* encoder trained on image-caption pairs; patching deliberately discards much of what it is best at.

Hence `0004` §6.2's two-view design — N native patches (forensic evidence) **plus** one whole-image resized view (semantic evidence), concatenated. The two are different kinds of evidence, and the card system is a natural place to expose that.

## 4. Aggregation is a multiple-instance learning problem

Given §3.1's confirmed finding that only *certain* patches carry the artifact, assigning the image-level label to every patch injects label noise. A flat sky patch from a diffusion model may be genuinely indistinguishable from a flat sky from a camera.

This is the standard MIL setup: a bag is positive if at least one instance is positive. Consequences:

- **The aggregator encodes an assumption.** Mean assumes a spatially uniform artifact — defensible for the decode signature (type B), which is a global property of the decoder. Max/top-k assume a localised one — defensible given that flat regions have no high-frequency content to carry a fingerprint. Both are plausible; the taxonomy says the two artifact types should behave *differently* under aggregation, which is itself a testable prediction.
- **The measurement is free.** Patch scores are computed once and cached. Mean, max, top-k and trimmed-mean all come off the same array at zero marginal GPU cost. There is no reason to pick by argument.
- **A variance/gradient filter is a principled option**, dropping flat patches before scoring to reduce label noise — and it ablates for free the same way.

## 5. The bag-size leak

**This is the most important finding of this pass, and it is a correctness issue rather than a performance one.**

A 480px image yields ~4 non-overlapping 224 patches; a 1024px image yields ~20. Source resolution correlates with label. **Therefore patch count encodes the label.**

It leaks through *every* aggregator:

- **Max and top-k leak directly.** The maximum of 20 draws is stochastically larger than the maximum of 4, whatever the underlying distribution.
- **Mean leaks through variance.** Expectation is bag-size-invariant, but the variance of the mean falls as 1/N. Larger bags produce tighter score distributions, and a threshold can exploit the difference in spread even when the centres coincide.

**Mitigation: fix N per image, independent of source resolution.** Small images sample with overlap, large images sample N from the grid. Enforce with an assertion in the extraction loop rather than by convention.

This belongs in the same family as round 1's blurred-data failure: a preprocessing step that quietly encodes the label, invisible in the metrics, discovered too late. `notes.md` line 79 records the root cause as *not inspecting the data after processing* — the generalisation of that lesson is that preprocessing invariants should be **asserted in code**, not verified by eye once.

## 6. Statistical nesting

Patching introduces a third level: **row → image → patch.**

- COCO_AI rows already require row-level splitting (one row = 1 real + 6 fakes; splitting within a row leaks the pairing).
- Patches from one image must not straddle train/val.
- **Calibration must be fitted on aggregated image-level scores, not patch scores.** 16 patches from one image are not 16 independent samples; treating them as such produces systematic overconfidence — which matters here specifically, because the product surface ships "the calibrated probability that the label is correct" `[stated]`.

## 7. Encoder objective vs. artifact preservation

`bottlenecks.md` §3.1 characterises the three encoder families by training objective. This pass adds a mechanism that is decisive under a patch-based design.

**DINOv2's multi-crop self-distillation objective explicitly trains crops of an image to embed alike** `[confirmed]`. The training signal *is* crop-invariance.

Under `0004` §6, the model is fed native patches and asked whether one patch's low-level statistics differ from another's. **DINOv2 was trained to answer no.** This is not a marginal disadvantage — it makes the self-supervised family structurally the worst choice for this specific design, on top of DINOv2-Giant already being the worst row in SSAFE's encoder ablation at 72.8% `[confirmed]`.

CLIP's position is unchanged and worth restating precisely, because it is also the honest limit: **no caption describes upsampler ringing**, so image-text contrastive training neither preserves nor destroys forensic artifacts. They survive *incidentally*. That is exactly why UniversalFakeDetect works at all and exactly why it is fragile — and it is why E1's plateau, not any argument, is the ceiling.

### 7.1 Overriding §3.1's two-arm recommendation

`bottlenecks.md` §3.1 recommends running our own encoder comparison rather than accepting SSAFE's ranking, because NTIRE 2026's winners used DINOv3-**7B** alongside SigLIP2-Giant — so the ablation may be measuring *scale* rather than *objective* `[confirmed]`.

The reasoning is sound and the recommendation is still overridden, for a reason §3.1 could not have known when written: **our two-arm run cannot resolve the confound it was proposed to resolve.** The confound is specifically about behaviour at 7B scale. A CLIP-L vs DINOv2-L comparison holds size fixed and would cleanly measure objective — which is a good experiment, and a *research* experiment. The research chapter is closed.

If a second arm is ever run it should be **SigLIP2**: it is the arm that could plausibly win, it loads from HF without the availability risk §3.2 flags for PE-Core, and it is the direction SSAFE's ablation points.

## 8. Mahalanobis on frozen features

Adopted in `0004` §7.2 as the OOD gate. Two implementation facts that decide whether it works:

- **L2-normalize before fitting.** SSAFE L2-normalizes embeddings before its linear layer `[confirmed]`; `bottlenecks.md` §3.2 lists whether our path does as unverified. `0004` closes this by decision rather than by measurement — normalize. Mahalanobis on unnormalized CLIP features is badly conditioned.
- **Pooled covariance with shrinkage, never per-class.** CLIP ViT-L/14 gives 768-d (post-projection) or 1024-d (pre-projection) features against possibly ~150 samples in the smallest class. A per-class empirical covariance is singular by construction. Pooling across classes gives ~21k samples for a single matrix; Ledoit-Wolf shrinkage handles the remainder.

**Which feature layer** — post-projection 768-d vs pooled pre-projection 1024-d — remains the open free variable `bottlenecks.md` §3.2 identifies, and the Perception Encoder result (best embeddings are not at the output) suggests it may matter as much as the backbone choice `[confirmed]`. Not resolved here. Default to what UniversalFakeDetect uses, and record it as an assumption rather than a finding.

**Presentation caveat, carried from `notes.md` line 106:** the founder engaged with a *Gaussian* concept, but the Mahalanobis↔Gaussian-density link was this project's own construction, **not her stated point.** It must not be presented as though it were hers.

## 9. Corrections to this session's own reasoning

Recorded per the convention established in `2026-08-02-dataset-alignment.md` §10.

1. **CLIP ViT-L/14@336 was recommended, then withdrawn.** It was proposed as the mitigation for §2's asymmetric downsampling — 2.25× the pixels for compute that is free on an A100. It is the right answer only if resizing is assumed. Patching addresses the cause directly, and a larger input *costs* coverage at the corpus's low end. Superseded within the same session.

2. **The AEROBLADE/Diffusion branch was oversold.** Pitched as an independent corroborator against watermark contamination and ranked the top stretch item. Two errors: reconstruction error is low only when the decoder matches the generator's, so it cannot corroborate on Gemini or GPT-Image (discrete VQ tokenizers, not SD-family VAEs); and ranking it top-stretch put effort where its competence does not overlap the live test. Demoted to a documented `NOT_IMPLEMENTED` stub — the per-card scope reasoning is the deliverable, not the code.

3. **"Native multi-crop with averaged embeddings" was proposed before the bag-size leak was noticed.** The initial proposal took as many crops as the image allowed. That is §5's leak, and it would have shipped as a silent label channel. The fixed-N requirement is a correction to this session's own design, not a refinement of it.

4. **`bottlenecks.md` §2.2's "single architecture family" is imprecise.** All six COCO_AI columns are latent diffusion, which is correct, but SD3/SD3.5 use a 16-channel VAE with MMDiT while SD2.1/SDXL use a 4-channel VAE with a U-Net. That is precisely the magnitude of jump that broke AlignedForensics on FLUX (`dataset-alignment.md`). The corpus is *less* homogeneous than §2.2 states — which slightly improves the outlook for E1, and does not change §2.2's conclusion that it would still collapse under SSAFE's Farthest-Point Sampling.

## 10. Open items

Carried forward; none block `0004`.

1. **`bottlenecks.md` §9.1 — native resolution distribution per COCO_AI column.** Still unmeasured, still `[second-hand]`. §2's downscale table above is built on those figures and inherits their status. One histogram closes it, and it should be run *before* the extraction pass, since it determines whether N=16 is the right patch count.
2. **Feature layer selection** (§8). An assumption, not a finding.
3. **Chai et al. at the body** (§3.1) — specifically whether patch aggregation beats whole-image cross-generator, and what aggregator they use. Would corroborate `0004` §6 but is not load-bearing for it.
4. **DDA (arXiv:2505.14359)** — carried from `dataset-alignment.md`, still unread at the body. It is the paper most directly about building an aligned real:fake distribution and its JPEG-matching result (FLUX 3.6 → 50.2) is the strongest single number in that doc. Not on the critical path for Monday.

## Sources

- [Chai, Bau, Lim & Isola — What Makes Fake Images Detectable? (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710103.pdf)
- [patch-forensics — code](https://github.com/chail/patch-forensics)
- [Springer Nature Link — chapter record](https://link.springer.com/chapter/10.1007/978-3-030-58574-7_7)
