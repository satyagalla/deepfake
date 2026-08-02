# Plan B verification — does augmentation-forced invariance transfer diffusion → transformer?

**Written:** 2026-08-02. Deadline Mon 2026-08-03.
**Verifies:** [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) §3 (Plan B), with corrections to its §5 ordering and §6.4.
**Relates to:** [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §8.4, §9 · [`../decisions/0003-frozen-probe-demo-build.md`](0003) §4.3 · [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §2

**Status:** Decision input. Nothing below was measured on our data — every number is from published literature, tagged per the repo convention (`[confirmed]` · `[unverified]` · `[measured]`). No code was written and no experiment was run this session.

**Question asked:** is Plan B legitimate, and would it prove the hypothesis?

**Answer in one line:** the mechanism is real and has a published existence proof — but the configuration Plan B specifies (CLIP ViT-L/14, augmentation as the mechanism) has a published **0.00%** result on our exact test distribution. Plan B is verified as a hypothesis and falsified as configured.

---

## 1. The direct negative — our architecture has already been run on our test distribution

[DailyBench (arXiv:2607.24016)](https://arxiv.org/html/2607.24016v1) `[confirmed]` evaluates 17 detectors on seven 2026 generators. **UniFD is frozen CLIP ViT-L/14 + linear probe** — the architecture `0003` §4.3 specifies. Table 2:

| Generator | real acc | **fake acc** |
|---|---|---|
| Nano Banana 2 | 99.59% | **0.00%** |
| GPT-Image 2 | 98.54% | **0.00%** |
| FLUX.2 | 98.83% | 0.50% |
| Z-Image | 98.80% | 1.03% |
| Qwen-Image | 98.63% | 0.51% |
| FLUX.1 | 98.72% | 0.19% |
| SD3.5 | 98.90% | 2.70% |

**Overall average 49.78%.** Peers on the same benchmark: NPR 67.52 · Effort 60.21 · SAFE 54.49 · B-Free 52.70.

Three things this establishes:

1. This is `0002` §5.2's failure mode — confident, biased toward `real` — reproduced at scale with a fixed sample size and a published protocol. §5.2 is `[measured, weakly]`; this is the external quantification it was missing.
2. It is **not** degradation. 0.00% is total collapse. There is no partial-credit reading of this number.
3. It fails on **SD3.5 (2.70%)**, which is a COCO_AI training column. So the collapse is not specific to the transformer paradigm — UniFD (ProGAN-trained) does not reach modern diffusion either. Recency, not family, per `2026-07-31-claim-verification.md` §2.2.

Field-level context from the same paper: detectors reporting **91–96% balanced accuracy on GenImage** drop to **60–76% on FakeBench** and 54–66% on ManipulationBench. That is the honest prior for a live demo.

## 2. The direct positive — SSAFE is Plan B, and it works

[SSAFE (arXiv:2606.08634)](https://arxiv.org/html/2606.08634v1) `[unverified — 2026 preprint, self-reported, no replication located]` was recorded in `three-candidate-plans.md` §6.4 as an architectural coincidence. It is materially more than that, because **its training set contains no transformer and no commercial generator.**

Training set (Table 8) — 10K images curated from a 50K pool, 28 domain–generator combinations reduced to 8:

- **Generators:** SD1.4, BigGAN, GLIDE, Midjourney (ImageNet domain); SDXL-LoRA, SD2.1, SD1.5 (COCO domain); SD3.5
- **Reals:** LSUN, ImageNet, COCO 2017, OpenImages V7, Unsplash, Pexels, Pixabay

Evaluation on generators of an unseen paradigm (AIGI-Holmes benchmark, Table 2):

| Generator | Acc | AP |
|---|---|---|
| GPT-Image-1 | 98.7% | 99.8% |
| Nano-Banana | 98.4% | 99.8% |
| Janus-7B (autoregressive) | 99.9% | 100.0% |
| LlamaGen (autoregressive) | 99.9% | 100.0% |

**This is Plan B's hypothesis demonstrated:** diffusion + GAN training data, frozen encoder, single linear layer, transferring to autoregressive and commercial generators. It is the strongest existing evidence that mechanism (B) — decode-stage statistics, `three-candidate-plans.md` §1 — carries across decode paradigms. It also corroborates §6.1's correction: a VQ tokenizer is a lossy latent→pixel decode and leaves a readable trace.

Consistent with this: OpenFake 99.0% average TPR / 99.4% ROC AUC from 10K images, surpassing a SwinV2 detector trained on the full 4M set — a 133× data reduction.

## 3. What SSAFE credits, and why it is not what Plan B specifies

Three levers, none of which is augmentation.

**3.1 The encoder, and it is the largest single effect measured `[unverified]`**

Encoder ablation (Table 12), same probe, same data:

| Encoder | Real acc | Fake acc | Overall | AP |
|---|---|---|---|---|
| PE-Core-G14-448 | 95.1% | 95.7% | **95.4%** | 99.4% |
| SigLIP2-G16-384 | 85.5% | 89.3% | 87.4% | 95.5% |
| DINOv3 ViT-L/16 | 79.9% | 87.7% | 83.8% | 92.7% |
| DINOv2-Giant | 76.0% | 69.5% | 72.8% | 78.1% |

A **22.6-point spread from encoder choice alone**, holding everything else fixed. **CLIP ViT-L/14 is not in the ablation** — and its nearest published instance is §1's 0.00% row.

This is a stronger statement than `three-candidate-plans.md` §6.4's "worth one head-to-head." The encoder is not a tiebreaker; on this evidence it is the dominant variable, and `0003` §4.3 currently specifies the one option with a direct collapse result against it.

**3.2 A modern real class, treated as central rather than as hygiene `[confirmed from the paper's own text]`**

SSAFE augments training with high-resolution images from Open Images V7, Pexels and Pixabay to close the "domain gap" with "outdated" sources, stating explicitly that ImageNet and LSUN "do not reflect modern smartphone and camera sensors."

This is exactly the prerequisite `three-candidate-plans.md` §5 names and marks as removable by none of the three plans. The paper that best supports Plan B treats it as a design requirement, not a caveat.

**3.3 Augmentation is absent `[confirmed — by absence]`**

The paper does not mention blur or JPEG augmentation in its training procedure, and does not report JPEG / resize / post-processing robustness at all.

**Plan B's stated mechanism does not appear in the strongest paper supporting Plan B's claim.**

## 4. The caveat inside the supporting paper — and it resolves an open discrepancy

SSAFE reaches 99.9% accuracy / 100.0% AP across AIGI-Holmes test generators. On real-world subsets — **SocialRF, CommunityAI, Chameleon** — performance "significantly degrades," dropping to **41.8–66.9% real accuracy.** The paper reports this itself as a key limitation.

The collapse is on the **real** class. That is a false-positive explosion on in-the-wild photographs, which is §1(C)'s container problem appearing inside the paper you would cite for Plan B.

**This resolves the discrepancy `three-candidate-plans.md` §6.4 recorded as unresolved** ("does not obviously reconcile with SSAFE's figures"). The two numbers do reconcile: SSAFE's 98–99% is benchmark-conditioned, DailyBench's 60–76% is in-the-wild, and SSAFE's own real-world subset numbers sit in and below the DailyBench band. The flattering number and the pessimistic number are measurements of different distributions, not a contradiction to be adjudicated.

## 5. The independent check — what actually wins in the wild

[NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild (arXiv:2604.11487)](https://arxiv.org/html/2604.11487) `[confirmed]` — 108,750 real and 185,750 generated images, 42 generators, 36 transformation types, scored on robust ROC AUC.

| Team | Robust ROC AUC | Clean ROC AUC |
|---|---|---|
| MICV (1st) | 0.9723 | 0.9974 |
| Ant International (2nd) | 0.9721 | 0.9972 |
| 3rd | ~0.925 | |

What the winners used:

- **Backbones:** DINOv3-7B, DINOv3-Huge, SigLIP2-Giant. Organizers: *"larger models and higher input resolutions consistently improved performance."*
- **Data:** ~1M images; MICV integrated academic benchmarks + cutting-edge open generators + **closed-source commercial API output**.
- **Augmentation:** 4-level severity-graded offline augmentation; the winner's runner-up went to 5 distortion types × 5 severity levels.
- **Ensembles:** every top finisher fused 2–6 models.

**The reading that matters for Plan B:** aggressive augmentation is present in *every* winning solution — and in *none* of them alone, and in none of them with single-family training data. Augmentation is a robustness technique that protects a signal; it is not a transfer technique that manufactures one. Plan B isolates the single ingredient nobody wins with by itself.

This also independently reinforces `0002` §8.4 (augmentation is mandatory) while contradicting `three-candidate-plans.md` §3's framing of it as the mechanism.

## 6. Plan C's precondition — status unchanged, one new hazard

[Yang et al., AAAI 2026 (arXiv:2602.01973)](https://arxiv.org/abs/2602.01973). The abstract was read verbatim this session. It **confirms** the mechanism — distribution shift in fake samples produces a misaligned decision threshold; a learnable scalar logit correction fitted on a small target-distribution validation set, backbone frozen, with a label-free variant — and it names **no detector, no generator, no accuracy figure, no AUC, and no calibration-set size.**

So `0002` §11's open item stands unresolved: **the ~100-image figure remains `[second-hand]`** and could not be verified from the abstract. The full text was not accessible in either fetched form.

**New hazard, not previously recorded.** [When AUC Misleads (arXiv:2606.19184)](https://arxiv.org/pdf/2606.19184) `[unverified — surfaced via search, abstract-level]` reports AUC ≈ 1 achievable per-dataset while dropping sharply once datasets are **pooled**, because scores stop being commonly ordered across sources.

Consequence for `0002` §11's discriminating test: the AUC precondition must be evaluated on the **pooled** real-vs-fake set, not per-generator. A per-generator AUC that survives can still mean no single threshold exists — which is precisely the thing Plan C needs to be true.

## 7. Loopholes in Plan B

Recorded so a passing number can be read correctly.

**Producing a good number that means nothing:**

| Loophole | Mechanism | Status |
|---|---|---|
| Container shortcut | `coco_image` natively JPEG, every generator column PNG | Known, priced at ±11 pts ([Fake or JPEG?](https://arxiv.org/html/2403.17608v1)) |
| Row-level leakage | One COCO_AI row = one real image + six fakes | Known, mandatory fix (`context-transfer.md` §8) |
| Resolution as label | `dalle` 270², `midjourney` 436², reals 480×640 | Addressed by the surgical cut, if it actually ships |
| A real class from 2014 | Demo-time false positives on modern phone photos | **Now measured externally** — §4, 41.8–66.9% |

The fourth is the one that does not show up in validation. It arrives on the founder's photo, not on the fake.

**Failing silently:**

- **Augmentation can be a null intervention or worse.** It destroys (C) and blunts (A). If (B) is not present in the frozen space, signal was deleted and nothing gained. Wang et al.'s SAN regression is the documented precedent.
- **The target is a product, not a model.** GPT Image 2 and Nano Banana Pro are decoder output plus an undisclosed post-pipeline that can change without notice (`claim-verification.md` §4.1).
- **Resolution ceiling.** Nano Banana Pro defaults to 4K; training data tops out at 1024².
- **Screenshot path.** A screenshotted demo image destroys every fragile signal.

**In the evaluation itself:**

- **No pre-registered success criterion.** A single live image is n = 1. Against a 98.5%-real / 0.00%-fake prior, "real" on a fake proves nothing and "fake" on a fake may be luck. Fix the OOD set (≥50/class) and the threshold **before** looking — otherwise `0002` §9's falsification gate is spent a second time, as it already was once (`0003` §3).

## 8. Alternatives not in the three-plan doc

**8.1 Provenance layer — deterministic for the stated acceptance test `[confirmed]`**

OpenAI has embedded SynthID + C2PA manifests in every ChatGPT and API image since 2026-05-19; Gemini and Nano Banana carry SynthID, with verification in the Gemini app, Chrome and Search. A file handed over live from either product is identifiable by **lookup**, not classification, while intact.

Two hard limits: stripping tools are freely available and a screenshot defeats it entirely; and **it is the container shortcut, made explicit**. It proves nothing about detection. Correct use is as layer 1 of the four-layer stack in [`2026-07-31-production-deployment.md`](2026-07-31-production-deployment.md), labelled as provenance — never as the classifier, never as the research claim.

**8.2 Off-the-shelf checkpoint — no training at all `[confirmed]`**

NPR scores 67.52% on DailyBench FakeBench; Community-Forensics is the best out-of-box performer at **75.0% mean / 82.1% median** across 12 datasets, 291 generators ([arXiv:2602.07814](https://arxiv.org/html/2602.07814v1)). Both plausibly exceed what Plan B produces, for an afternoon of environment work. Absent from the three-plan doc entirely. Cost: no novelty, and an inherited failure mode that cannot be diagnosed.

**8.3 Few-shot adaptation — a fourth point between A and C `[unverified]`**

[Fleet (arXiv:2606.31082)](https://arxiv.org/pdf/2606.31082) adapts a frozen backbone to a target generator from a few exemplar images, reporting results on GPT-Image, FLUX and Nano Banana. **Numeric tables could not be extracted from the PDF — treat the performance claim as a lead, not a fact.** If it holds it is Plan C's economics with Plan A's mechanism (~10 images rather than ~300, adaptation rather than threshold-shifting) and it does not depend on ranking surviving. Worth reading before committing to a 300-image purchase.

**8.4 The NTIRE recipe.** §5. Out of budget as stated (~1M images, 7B backbones, 6-model ensembles), but its *ordering* of levers is the transferable part: encoder scale > data breadth > augmentation > ensemble.

## 9. Recommendation

Plan B stays as the first experiment — its selection basis is sound and unchanged: **it is the cheapest experiment that discriminates between the remaining options**, its failure is informative, and Plan C is blocked behind it because C needs a trained probe and a scored OOD set to run its precondition at all. That basis was never "most likely to pass the demo," and §1 says it probably will not.

Run it as a **two-arm ablation rather than a single plan**:

1. **PE-Core-G14-448 vs CLIP ViT-L/14**, frozen, identical probe and data. Half a day, $0, and it discriminates *encoder* as well as *mechanism* — which §1-vs-§2 says is the larger variable. Amends `0003` §4.3.
2. **Modernize the real class.** Not deferrable (§3.2, §4). Add Open Images V7 / Pexels / Pixabay-tier reals to sit beside COCO 2014.
3. **Keep augmentation, demote it.** Wang et al.'s p = 0.1–0.5 protocol, per `0002` §8.4 — as robustness hygiene, not as the transfer mechanism (§5).
4. **Fix the success criterion and the OOD set before looking** (§7).
5. **Run the container control** from `context-transfer.md`: re-save at JPEG q85, rescale 95%, re-score. If the score collapses, the probe reads the delivery pipeline.

In parallel and independent of the outcome: stand up **8.1 as the demo's provenance layer** and pull **8.2 as a floor**. That produces a working Monday demo regardless of how the experiment lands, and a real result either way.

## 10. Corrections to record

**10.1 — `three-candidate-plans.md` §6.4 understates SSAFE.** It records the architecture match and the encoder ablation but not the training-set composition. SSAFE trains on diffusion + GAN only and evaluates on autoregressive and commercial generators — which makes it the **single strongest existing evidence for Plan B**, not a general architectural note. §2 above.

**10.2 — The SSAFE / DailyBench discrepancy is resolved, not open.** §6.4 records it as not obviously reconcilable. §4 above reconciles it: benchmark-conditioned vs in-the-wild, with SSAFE's own real-world subsets (41.8–66.9%) sitting in and below the DailyBench band.

**10.3 — Augmentation is not the mechanism.** §3's framing of Plan B rests on augmentation forcing the probe onto (B). The strongest supporting paper uses no augmentation (§3.3), and every NTIRE winner pairs augmentation with encoder scale and data breadth (§5). Augmentation protects signal; it does not manufacture it. `0002` §8.4 is unaffected and independently reinforced.

**10.4 — The encoder is the dominant variable, not a tiebreaker.** §6.4 suggests "one head-to-head before defaulting to CLIP ViT-L/14." The measured spread is 22.6 points, CLIP ViT-L/14 is not in the ablation, and its nearest published instance scores 0.00% on our two test generators. This should gate the build, not follow it.

**10.5 — Plan C's calibration-set size remains unverified.** The Yang et al. abstract names no numbers. `0002` §11's open item is unchanged, and §6 adds the pooled-AUC hazard.

---

## Sources

New to this doc:

- [*DailyBench*, arXiv:2607.24016](https://arxiv.org/html/2607.24016v1) — §1, §4, §8.2
- [*NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*, arXiv:2604.11487](https://arxiv.org/html/2604.11487) — §5
- [*Fleet: Few Shots Lead Effective AI-generated Image Detection*, arXiv:2606.31082](https://arxiv.org/pdf/2606.31082) — §8.3, `[unverified]`
- [*When AUC Misleads: Polarization-Aware…*, arXiv:2606.19184](https://arxiv.org/pdf/2606.19184) — §6, `[unverified]`
- [ChatGPT C2PA + SynthID embedding](https://slopornot.ai/blog/how-to-detect-chatgpt-generated-images) · [Nano Banana SynthID & C2PA](https://www.nenobanana.com/blogs/nano-banana-synthid-and-c2pa) — §8.1

Re-read this session at primary source, previously in the bibliography:

- [*SSAFE*, arXiv:2606.08634](https://arxiv.org/html/2606.08634v1) — §2, §3, §4. Tables 2, 4, 5, 8, 12
- [Yang et al., AAAI 2026, arXiv:2602.01973](https://arxiv.org/abs/2602.01973) — §6, abstract only
- [*How well are open sourced AI-generated image detection models out-of-the-box*, arXiv:2602.07814](https://arxiv.org/html/2602.07814v1) — §8.2
- [*Fake or JPEG?*, arXiv:2403.17608](https://arxiv.org/html/2403.17608v1) — §7
- [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf) — §7, §9
- [*AIGI-Holmes*, ICCV 2025](https://arxiv.org/html/2507.02664v1) — the benchmark SSAFE's §2 numbers are measured on
