# Real, Deepfake, Edited: a model & dataset strategy for 3-class forgery classification

Extending a 2-class (real/fake) image classifier into a 3-class detector that separates AI-generated deepfakes from conventional (non-AI) photo edits — scoped to a Colab-scale training budget.

- **Scope:** face images, static frames
- **Compute:** ~300 Colab compute units, PyTorch/CUDA
- **Prepared:** 2026-07-23

## How this was built

An automated multi-agent research pass (5 search angles → 23 sources fetched → 79 candidate claims → adversarial 3-vote fact-check) ran out of budget mid-verification when the session hit its rate limit, so the final synthesis step never ran automatically. What follows is a manual synthesis over everything the pass gathered:

- **[confirmed]** — passed adversarial fact-check 2–1 or 3–0
- **[unverified — refuted]** — a majority of checkers could not support the claim in the source; kept only where still directionally useful, flagged clearly
- **[unverified]** — raw findings that never reached a vote before the pass was cut off

Treat anything not marked `[confirmed]` as a lead to check against the primary source before citing it further, not as an established fact.

---

## 1. Recommendation

Fine-tune a single **EfficientNet-B4** backbone (ImageNet-pretrained, full fine-tune, 3-way softmax head) as the primary model. Assemble training data from face-cropped frames of **FaceForensics++ + Celeb-DF** (deepfake), face-filtered crops from **CASIA v2 + IMD2020 + PS-Battles** (edited), and the paired authentic images from both groups (real).

### Why this over the alternatives: a single compact CNN, not a hybrid or a frozen-backbone adapter stack

Three architecture families showed up repeatedly in the research pass: (a) single CNN backbones (Xception, EfficientNet), (b) CNN+ViT hybrids that concatenate two backbones into a transformer, and (c) parameter-efficient tuning (PEFT) of a large frozen ViT/CLIP backbone via adapters or LoRA.

Family (c) — DeepFake-Adapter [3], MoE-FFD [12], OSDFD [15] — exists specifically to make fine-tuning affordable when the backbone is too large to fully train (300M+ parameter ViT/CLIP models). At Colab scale, EfficientNet-B4 (~19M params) is already cheap enough to fully fine-tune directly, so the added engineering of custom adapter/LoRA modules buys compute savings you don't need yet.

Family (b) — the Xception+EfficientNet-B4+Transformer hybrid [7] — reports strong numbers (98.24% DFDC accuracy) but runs two CNN backbones simultaneously, roughly doubling training cost for a gain that a single well-tuned CNN can likely approach.

**Start with (a); reach for (b) only if compute remains after a working 3-class baseline exists, and for (c) only if you later swap in a much larger backbone.**

### Where the 3rd class needs a bespoke design: add a frequency/noise-residual input channel

No dataset or published model in this research pass was built for exactly REAL vs. DEEPFAKE vs. EDITED — see §3. AI-generated and conventionally-edited images differ most reliably not in RGB content but in *generation artifacts*: GAN/diffusion output carries characteristic upsampling and spectral signatures, while splicing/retouching leaves noise-level and compression discontinuities at edit boundaries.

A plain RGB-only CNN risks learning which *dataset* an image came from (resolution, JPEG quality, framing) rather than which *manipulation family* produced it — see Risk 2 in §6. Concatenate a simple SRM noise-residual or FFT-magnitude channel to the RGB input; this is a well-established, cheap addition (no extra backbone) that several forensic-localization papers in this pass lean on for exactly this kind of discrimination [10].

---

## 2. SOTA binary deepfake architectures, and what adapts to 3-class

Every architecture the search surfaced was built and evaluated for *binary* real/fake detection. All can be adapted to 3-class by swapping a sigmoid+BCE head for a 3-way softmax+cross-entropy head — the harder part is data, not architecture (§3–4).

| Family | Example | Reported result | Colab fit |
|---|---|---|---|
| Single CNN | Xception [14] | 89.2% acc, DFDC | Cheap, fast, well-understood baseline |
| Single CNN | EfficientNet-B0/B4 [6] | AUC 0.951, DFDC | Cheapest strong option — **recommended base** |
| CNN + ViT hybrid | Xception + EffNet-B4 → Transformer [7] | 98.24% acc, DFDC | ~2× backbone cost of a single CNN |
| CNN + ViT hybrid | EfficientNet-B0 + ViT [6] | 88.0% F1, DFDC | Moderate; avoids ensembling/distillation overhead |
| Frozen ViT + PEFT | DeepFake-Adapter [3] | n/r `[unverified]` | Only pays off on backbones too large to fully tune |
| Frozen ViT + PEFT | MoE-FFD (LoRA+Adapter) [12] | n/r `[unverified]` | Same — plus Mixture-of-Experts routing complexity |
| Frozen ViT + PEFT | OSDFD [15] | 1.34M vs 85.8M trainable params | Genuinely tiny trainable footprint, but built for open-set generalization, not 3-class |
| Frozen CLIP + PEFT | LNCLIP-DF (LayerNorm-only) [13] | SOTA cross-dataset AUROC, CDFv2/FFIW | Tunes ~0.03% of params; strong cross-dataset generalizer if you later need it |
| Video transformer | TimeSformer [2] | 78.4% acc / 0.801 AUC | Needs frame sequences, not single images — out of scope here |

One notable, generally-applicable finding: a CLIP-based detector trained on the older but more diverse **FaceForensics++ (2019)** generalized better across 13 newer benchmarks than models trained only on recent datasets [13] `[confirmed]` — diversity of manipulation type mattered more than dataset recency. This favors FF++ as your deepfake-class backbone even though newer datasets exist.

---

## 3. The 3-class gap: closest existing analogs

A direct REAL / DEEPFAKE (AI) / EDITED (conventional) 3-way classifier does not appear in the literature surfaced by this pass. Two adjacent efforts are worth knowing about, with different reasons for caution on each:

**`[confirmed]`** A three-branch (identity / spatial / frequency) architecture fused via an MLP performs 3-class classification of **real vs. forged vs. anti-forensically-perturbed forged** — 98.2% accuracy on FaceForensics++, 93.5% cross-dataset AUC (trained FF++ → tested Celeb-DF). This is the closest published 3-class forensic classifier found, but its third class is a deepfake that's been adversarially post-processed to evade detection, *not* a conventionally-edited (Photoshop) image. The architecture pattern (identity + spatial + frequency branches → fusion) is still a reasonable template to borrow.
Source: [PMC12727607](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12727607/), Dec 2025.

**`[unverified]`** A paper describing a "ForensicFormer" reportedly trains on a dataset explicitly mixing AI-generated forgeries (Stable Diffusion 2.1, DALL-E 3, StyleGAN3) with traditional manipulations, and performs joint real/fake + localization + 7-class manipulation-type prediction. This is the one source that directly instantiates an AI-vs-traditional-forgery training mix — exactly what this project needs. **However, every specific numeric claim from this source failed adversarial fact-check (0–3 votes)** during the research pass, meaning independent checkers could not confirm the extracted figures against the source text. Read the primary source yourself before relying on any number from it.
Source: [arxiv.org/html/2601.08873](https://arxiv.org/html/2601.08873) — unconfirmed.

**`[unverified]`** A curated dataset index lists **CocoGlide**, built around GLIDE diffusion-model tampering, suggesting some forensic benchmarking infrastructure already treats AI-generative edits as a manipulation category distinct from splicing/copy-move — a potential bridge dataset between the "edited" and "deepfake" worlds worth investigating further, though it targets general-scene image edits rather than face manipulation specifically.
Source: [IMDLBenCo dataset hub](https://scu-zjz.github.io/IMDLBenCo-doc/imdl_data_model_hub/data/IMDLdatasets.html).

**Practical implication:** you are defining new task framing, not reproducing a benchmark. Budget time for your own held-out validation split that specifically measures EDITED↔DEEPFAKE confusion (§5), since that's the boundary no existing published model has been checked against.

---

## 4. Assembling the 3 classes from public datasets

### Deepfake class

| Dataset | What it gives you | Scale |
|---|---|---|
| FaceForensics++ | Real videos + 4 AI manipulation subtypes (Deepfakes, Face2Face, FaceSwap, NeuralTextures) [15] | ~1,000 videos/class |
| Celeb-DF v2 | Higher-quality face-swap deepfakes, common cross-dataset test target [13] | ~6,000 videos |
| DFDC | Large, diverse deepfake benchmark used by most papers in this pass [6][7][14] | 100,000+ videos |

### Edited class (conventional, non-AI manipulation)

| Dataset | Manipulation types | Scale |
|---|---|---|
| CASIA v2.0 | Splicing, copy-move, removal `[confirmed]` | 7,491 tampered / 5,123 authentic |
| IMD2020 | Splicing, copy-move, removal — largest of this group `[confirmed]` | 35,000 / 35,000 synthetic + 2,010 real-world |
| PS-Battles | Crowd-sourced Photoshop edits (r/PhotoshopBattles) — closest to real "beautification/retouching" use case `[confirmed]` | 11,142 sets / 103,028 images |
| COVERAGE | Copy-move only `[confirmed]` | 100 pairs — too small alone |
| GreatSplicing | Splicing across 335 semantic categories, positioned as more generalizable than CASIA/Columbia/COVERAGE `[unverified]` | 5,000 images |

Note: PS-Battles ships as two manifest files (`originals.tsv` / `photoshops.tsv`) plus fetch scripts, not raw images `[confirmed]` — budget time for link rot when scraping it.

### Real class

Use the paired authentic/original images already bundled with each source above (FF++ real videos, CASIA authentic images, PS-Battles originals). Don't source "real" from a separate unrelated dataset — matching capture conditions to each forged counterpart reduces the spurious-correlation risk described below.

### A mandatory preprocessing step: face-filter the edited class

CASIA v2, IMD2020, and PS-Battles are general-scene datasets — most images contain no face at all. Run a face detector (RetinaFace or MTCNN) over each and keep only images with a detected face, cropped the same way as your deepfake/real face crops. Skipping this step means the model can trivially separate "edited" from the other two classes by learning "contains a face" vs. "doesn't," which is not the manipulation-type signal you want.

---

## 5. Loss functions & metrics

Expect significant class imbalance: deepfake video datasets yield far more frames than the traditional-forgery image datasets have images, especially after face-filtering shrinks the edited class further.

- **Loss:** start with class-weighted cross-entropy (weights inverse to class frequency). Move to focal loss (γ≈2) only if the edited class is still being ignored after weighting — it adds a hyperparameter to tune for a problem weighting usually already fixes. One paper in this pass used a combined cross-entropy + uniformity + alignment loss (pushing same-class embeddings together, classes apart on a hypersphere) for stronger cross-dataset generalization [13] `[confirmed]` — a reasonable stretch goal once the basic 3-way classifier works, not a v1 requirement.
- **Metrics:** macro-F1 as the headline number (equal weight per class, won't hide poor performance on the smaller edited class); per-class precision/recall; per-class one-vs-rest ROC-AUC. Track the full 3×3 confusion matrix explicitly — the edited↔deepfake cell is the one novel failure mode this task introduces, and overall accuracy will hide it if the real class dominates the sample count.

---

## 6. Risks & open questions

1. **No established benchmark for this exact task.** As shown in §3, published 3-class forensic work uses different class definitions. You're defining new evaluation protocol, not reproducing one — plan your own held-out test split rather than expecting a drop-in benchmark.
2. **Domain gap between traditional-forgery and deepfake datasets.** One paper found that naively training a single model on combined multi-source deepfake datasets causes rapid accuracy degradation purely from differences in data collection and generation method across datasets [11] `[confirmed]`. Mixing full-scene press/amateur photos (CASIA, IMD2020) with broadcast-quality face-cropped video frames (FF++, DFDC) is a larger version of the same problem — resolution, compression history, and framing all differ systematically by class, which a CNN can latch onto instead of genuine forgery cues.
3. **Realistic edits are harder to detect than classic splicing benchmarks suggest.** A newer deep-learning-composited splicing dataset (built to look like real-life edits rather than handcrafted splices) dropped a SOTA detector's accuracy from the 84–96% typical of CASIA/Columbia/DSO-1 down to ~72% `[unverified — flagged in verification, treat as directional]`. If your edited-class sources skew toward crude/obvious splices, real-world performance on subtle retouching will likely be worse than validation numbers imply.
4. **Face-filtering will shrink an already-small edited class further.** Most images in CASIA v2/IMD2020/PS-Battles don't contain faces (§4); after filtering, expect the edited class to be the smallest of the three by a wide margin, reinforcing the need for class-weighted loss and macro-F1 tracking (§5).

---

## Sources

1. [PLOS ONE — MobileNetV2+BiLSTM+Transformer binary deepfake detector](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0334980)
2. [Springer MVA — cross-dataset video architecture comparison (TimeSformer et al.)](https://link.springer.com/article/10.1007/s00138-026-01809-w)
3. [arXiv:2306.00863 — DeepFake-Adapter](https://arxiv.org/pdf/2306.00863)
4. [GitHub — Image-Forgery-Datasets-List](https://github.com/greatzh/Image-Forgery-Datasets-List)
5. [arXiv:2310.10070 — GreatSplicing](https://arxiv.org/pdf/2310.10070)
6. [ResearchGate — Combining EfficientNet and Vision Transformers](https://www.researchgate.net/publication/360607256_Combining_EfficientNet_and_Vision_Transformers_for_Video_Deepfake_Detection)
7. [arXiv:2208.05820 — Xception + EfficientNet-B4 + Transformer hybrid](https://arxiv.org/pdf/2208.05820)
8. [IMDLBenCo dataset hub](https://scu-zjz.github.io/IMDLBenCo-doc/imdl_data_model_hub/data/IMDLdatasets.html)
9. [GitHub — PS-Battles dataset](https://github.com/dbisUnibas/PS-Battles)
10. [arXiv:2404.02897 — automated deep-learning-composited splicing dataset](https://arxiv.org/pdf/2404.02897)
11. [arXiv:2406.20078 — GM-DF, domain gap across deepfake sources](https://arxiv.org/pdf/2406.20078)
12. [arXiv:2404.08452 — MoE-FFD](https://arxiv.org/pdf/2404.08452)
13. [arXiv:2508.06248 — LNCLIP-DF](https://arxiv.org/html/2508.06248v1)
14. [Applied Sciences — Xception/ResNet/VGG16 benchmark](https://www.mdpi.com/2076-3417/15/3/1225)
15. [arXiv:2408.12791 — OSDFD, open-set forgery-style mixture](https://arxiv.org/pdf/2408.12791)
16. [PMC — three-branch identity/spatial/frequency 3-class classifier](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12727607/)
17. [arXiv:2601.08873 — "ForensicFormer" (claims unverified)](https://arxiv.org/html/2601.08873)

---

*Compiled from a 5-angle automated search + fact-check pass (23 sources, 79 candidate claims, 25 adversarially verified before the session rate-limited) plus manual synthesis. Confidence tags reflect the automated fact-check where one completed; unverified items are worth a primary-source read before you build on them.*
