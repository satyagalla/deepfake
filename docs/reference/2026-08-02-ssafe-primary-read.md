# SSAFE and the encoder question, read at primary source

**Written:** 2026-08-02 (second pass of the day). Deadline Mon 2026-08-03.
**Status:** Current.
**Scope:** SSAFE read at the paper body rather than the abstract or a summary, plus the Perception Encoder paper, plus corrections to the four 2026-08-02 reference docs written earlier the same day.
**Amends:** [`2026-08-02-plan-b-verification.md`](2026-08-02-plan-b-verification.md) §1, §4 · [`2026-08-02-plan-c-source-verification.md`](2026-08-02-plan-c-source-verification.md) §1, §4 · [`2026-08-02-plan-a-verification.md`](2026-08-02-plan-a-verification.md) §3.2 (reinstated)
**Consequences for the build:** [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md). This doc is what the sources say; that one is what it means for us.

Per the `research/` and `reference/` convention the amended docs are **not edited**; this file is the forward-pointing record.

Evidence tags: `[confirmed]` checked against the paper body this session · `[repo]` · `[second-hand]` · `[unverified]`. **Nothing here is `[measured]`** — no number in this doc came from our data.

---

## 1. Why this pass happened

The four 08-02 docs were written the same day and disagree with each other about their two most-cited sources.

- `plan-b-verification.md` cites SSAFE Tables 2, 4, 5, 8, 12 with specific figures — body read.
- `plan-c-source-verification.md` §4 states the body was **not** read, "abstract page only," and downgrades the same claims to `[unverified]`.

Neither doc knows about the other. Since the encoder recommendation and the whole Plan B assessment rest on which is right, the body was fetched and read directly.

**Result: plan-B's read is accurate. plan-C's downgrade is the stale side.** Details below, correction in §5.2.

---

## 2. SSAFE, from the body `[confirmed]`

[*SSAFE: Simple and Strong AI-Generated Image Detection via Frozen Vision Encoders*, arXiv:2606.08634](https://arxiv.org/abs/2606.08634), submitted 2026-06-07. Preprint, not peer-reviewed, no replication located.

### 2.1 Training data — 10K images, 5K real / 5K synthetic

| | Sources |
|---|---|
| **Real** | LSUN, ImageNet, COCO 2017, OpenImages V7, Unsplash (people/portraits), Pexels, Pixabay |
| **Synthetic — ImageNet domain** | SD1.4, **BigGAN**, GLIDE, Midjourney |
| **Synthetic — COCO domain** | SDXL-LoRA, SD2.1, SD1.5 |
| **Synthetic — additional** | SD3.5 |

Note the fake class spans **four architecture families**, including a GAN. Note also the deliberate **domain pairing**: ImageNet-domain fakes against ImageNet reals, COCO-domain fakes against COCO reals.

### 2.2 The curation method — this is the paper's actual contribution

From a **50K universal pool** (AIGI-Holmes 45K + 5K modern high-resolution reals):

1. Embed the pool with PE-Core.
2. Compute **Maximum Mean Discrepancy** between generators.
3. **Hierarchical clustering.**
4. **Farthest-Point Sampling** to select 8 representative generators from 28.

So the claim is not "10K images is enough." It is **"10K chosen to span generator space is enough."**

### 2.3 Table 5 — composition beats scale by 400×

The single most decision-relevant number found across both passes. RealWorldBench results by training set:

| Training set | Real TNR | Avg TPR | ROC AUC |
|---|---|---|---|
| AIGIBench (288K) | 65.6% | 78.4% | 81.5 |
| OpenFake (4M) | **29.7%** | 94.9% | 87.5 |
| Universal (50K) | 96.8% | 95.3% | 99.5 |
| **Curated 10K** | **98.3%** | 94.4% | 99.0 |

Four million images yield 29.7% real accuracy; ten thousand curated yield 98.3%. This is the evidence that generator *diversity*, not data *volume*, is the lever — and the reason `0003` §5's "≥2 training generators" framing counts the wrong quantity.

### 2.4 Table 12 — the encoder ablation

Same probe, same data, 40K AIGIBench subset excluding face-swap:

| Encoder | Real acc | Fake acc | Overall | AP |
|---|---|---|---|---|
| PE-Core-G14-448 | 95.1% | 95.7% | **95.4%** | 99.4 |
| SigLIP2-G16-384 | 85.5% | 89.3% | 87.4% | 95.5 |
| DINOv3 ViT-L/16 | 79.9% | 87.7% | 83.8% | 92.7 |
| DINOv2-Giant | 76.0% | 69.5% | 72.8% | 78.1 |

Paper's stated conclusion: *"Multimodal encoders (PE-Core and SigLIP2) substantially outperform self-supervised encoders."*

**CLIP ViT-L/14 — the backbone `0003` §4.3 specifies — is not in the ablation.**

### 2.5 Cross-paradigm transfer, which is our hypothesis

Trained on diffusion + GAN only. No transformer, no autoregressive, no commercial generator in training.

| Generator | Result | Paradigm |
|---|---|---|
| Janus-7B | 99.9% | autoregressive (VQ tokenizer) |
| LlamaGen | 100.0% | autoregressive (VQ tokenizer) |
| GPT Image 1 | 98.7% | commercial |
| Nano-Banana | 98.7% | commercial |

Overlap is partial but the relevant rows are clean: SD variants, Midjourney, BigGAN and GLIDE recur in test benchmarks, but **FLUX, Imagen 3/4, Janus, LlamaGen, GPT-Image-1 and Nano-Banana appear nowhere in training.**

This is the closest thing in the literature to our experiment, run by someone else.

### 2.6 Method details

| Item | Value |
|---|---|
| Head | Single linear layer + sigmoid on L2-normalized embeddings; `h(x) = σ(wᵀf + b)`, fake if ≥ 0.5 |
| Loss / optimizer | BCE, AdamW, lr 1×10⁻³, batch 40 |
| Fine-tuning | None. "No text encoder, prompts, or backbone fine-tuning are required" |
| Resolution | Native resolutions preserved; no crop-vs-resize strategy described |
| **Augmentation** | **None.** No blur, JPEG, or resize augmentation anywhere in the training procedure |

`0003` §4.4's logistic regression **is** this head — linear + sigmoid + BCE. No change needed there.

### 2.7 What the paper does not contain

- **No mechanism analysis.** Nothing on decoder/VAE/tokenizer artifacts, no frequency analysis, no low-level-vs-semantic decomposition. The stated intuition is only that multimodal encoders capture "both semantic information and subtle fakeness cues." **Our (A)/(B)/(C) taxonomy is our theory, not theirs** — SSAFE supports the outcome, not the explanation.
- **No compression, format, or resolution bias discussion, and no controls for it.** Not an oversight so much as a structural immunity: 7 real sources × 8 generators × 4 families leaves nothing to shortcut to. Their curation dissolved the problem. Ours manufactures it (`bottlenecks.md` §2.5).
- **No data release.** The only availability statement is *"Code, curated generator lists, and evaluation scripts will be released upon publication"* — the recipe, not the images, and conditional on a publication that has not happened. No GitHub, HuggingFace, or project page located.

What we do have is the generator list and real-source list, which is the expensive output of the curation.

### 2.8 The real class is uncontrolled — audit result

Prompted by the question *"how are iPhone/Samsung photos in-distribution with any of the generator families?"*

First, a premise correction: the ~11K iPhone 12–17 / Samsung photos are **RealWorldBench, the test set.** They were never paired against training fakes. Training reals are the seven sources in §2.1.

The underlying concern survives, and the paper answers it badly:

- The **5K modern reals added to the training pool are unpaired** — general real-image diversity, matched to no generator's output domain. Justification given: AIGI-Holmes contains "only LSUN- and ImageNet-style real images, which do not reflect the characteristics of contemporary smartphone or web-crawled photos."
- **No discussion** of a classifier exploiting source characteristics (camera photo vs render, professional stock style) rather than synthesis artifacts. Framed only as a "domain gap."
- **No ablation isolates the modern reals' contribution.** Table 6 compares random vs curated *generator* selection; nothing separates real-source diversity from improved fake detection.
- **No resolution statistics for RealWorldBench's synthetic images**, while its reals reach 5712×4284 (iPhone 15).

**Why the last gap is load-bearing.** If the benchmark's fakes sit at ~1024² against 4–5K reals, "large image = real" separates it almost perfectly and produces **both** the 98.3% TNR and the 94.4% TPR. A high TPR does not rule the shortcut out, because the shortcut also gets the fakes right. Nothing published distinguishes these.

This does not make SSAFE's numbers wrong. It makes them **unverifiable from what is published**, which matters because the "modernize the real class" recommendation was derived from them. Consequences in [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md) §2.4.

---

## 3. The Perception Encoder `[confirmed]`

[*Perception Encoder*, arXiv:2504.13181](https://arxiv.org/abs/2504.13181).

- **Contrastive vision-language training**, and the paper's claim is that this alone produces strong general embeddings across downstream tasks.
- **Central finding: "the best visual embeddings are not at the output of the network"** — they are hidden in intermediate layers, addressed via language alignment (multimodal) and spatial alignment (dense prediction).

For a frozen-backbone probe this makes **layer selection a free variable we have no plan for**, and potentially as consequential as the backbone choice itself. Recorded as an open item in `bottlenecks.md` §3.2.

Naming convention gives the rest: **PE-Core-G14-448** is ViT-G/14 at 448px input, against CLIP ViT-L/14's 224 — 4× the pixels, in the band where synthesis artifacts live.

---

## 4. Why backbone choice is a ceiling, not a hyperparameter

Synthesised from §2.4, §3 and NTIRE. The mechanism was not stated in any single source.

A frozen encoder is a fixed measurement instrument. The probe reads only what survived into the embedding. **If the encoder discarded the artifact, no probe, no data volume and no augmentation recovers it.** Fine-tuning could partially re-learn it and is ruled out by `0003` §2.

How the candidates differ by training objective:

| Encoder | Objective | Consequence for this task |
|---|---|---|
| **DINOv2 / v3** | Self-supervised, self-distillation across augmented views. Objective is *invariance* | A learned instruction to discard low-level, high-frequency, local detail — exactly where synthesis artifacts live. DINOv2-Giant is the worst ablation row at 72.8% |
| **CLIP ViT-L/14** | Image-text contrastive | No caption describes upsampler ringing, so nothing preserves artifacts — and nothing destroys them. They survive incidentally. Why UniversalFakeDetect works at all, and why it is fragile |
| **PE-Core / SigLIP2** | Contrastive vision-language at larger scale and resolution | SSAFE's grouping: multimodal substantially outperforms self-supervised |

**Confound to record.** NTIRE 2026's winners used **DINOv3-7B** alongside SigLIP2-Giant `[confirmed, plan-b-verification.md §5]`. DINOv3 is self-supervised, so at 7B scale the multimodal-beats-self-supervised rule breaks. PE-Core-G14 and SigLIP2-G16 are both G-class while DINOv3 ViT-L/16 is not — **SSAFE's ablation may be partly measuring model size rather than training objective.** This is the reason to run our own comparison rather than adopt the ranking.

`0003` §4.3 justifies CLIP ViT-L/14 as "UniversalFakeDetect's exact setup — the most documented and most reproducible." That is a sound argument for a **control**; `0003` used it as a performance argument too.

---

## 5. Corrections to the 2026-08-02 docs

### 5.1 ✗ `plan-b-verification.md` §4 misattributes SSAFE's real-class degradation `[confirmed]`

That doc records SSAFE degrading to **41.8–66.9% real accuracy** on real-world subsets and reports it as "the paper's own key limitation," then uses it as §1(C)'s container problem appearing inside the supporting paper.

The body attributes that degradation to models trained on **AIGI-Holmes alone** — it is the paper's *motivation* for adding modern reals, not a result for SSAFE. **SSAFE's own curated model reaches 98.3% TNR** (§2.3).

Downstream consequence: §4's reconciliation of SSAFE against DailyBench ("benchmark-conditioned vs in-the-wild") is **void on both sides** — misread number, and a withdrawn counterparty (§5.3). The discrepancy §6.4 of the plans doc recorded as unresolved returns to unresolved.

### 5.2 ~ `plan-c-source-verification.md` §4's SSAFE downgrade is stale `[confirmed]`

It leaves the generator list, the PE-Core recommendation and all figures `[unverified]` on the grounds that "the full body was not read." The body has now been read and plan-B's figures check out. The `[unverified]` tag on SSAFE as a *whole* should stand — unreplicated preprint — but not on the specific claims plan-C singled out.

### 5.3 ✗ `plan-b-verification.md` §1's central verdict rests on a withdrawn paper `[confirmed]`

§1 builds "Plan B is falsified as configured" on DailyBench's UniFD 0.00% rows for Nano Banana 2 and GPT-Image 2. `plan-c-source-verification.md` §4 independently established that **DailyBench (arXiv:2607.24016) was withdrawn 2026-07-28** — "some errors must be corrected" — but only withdrew the figure's use in the plans doc §6.4, never reaching into plan-B §1.

Also unsourced by the same withdrawal: the 91–96% → 60–76% pessimistic prior (used in three docs) and the NPR 67.52% floor in plan-B §8.2.

**The direction survives only via NTIRE**, which is a real competition, independent, and not withdrawn. The specific 0.00% figure should not be quoted.

### 5.4 ✓ `plan-a-verification.md` §3.2 stands unrefuted — reinstated `[confirmed]`

That doc objects that stock-site reals (Unsplash / Pexels) carry their own processing fingerprint — uniform sRGB export, Lightroom pipeline, professional composition — separable from AI output on style alone, and cites `0002` §6.4's CASIA corpus fingerprint at 17.4%.

Earlier in this session that objection was treated as settled in SSAFE's favour, on the grounds that SSAFE deliberately uses those exact sources. **That reading was wrong.** SSAFE never ran the argument: no source-characteristic discussion, no isolating ablation, no control (§2.8). Its usage is not evidence against the objection.

### 5.5 `plan-c-source-verification.md` §1's prerequisite table is stale on one row `[confirmed]`

`.venv/` **exists** in this working copy. `data_raw/`, `dataset/` and `checkpoints/` are correctly listed as absent.

### 5.6 The anchor document is missing `[confirmed]`

All four 08-02 reference docs are structured as amendments to `docs/notes/2026-08-02-three-candidate-plans.md`, citing it by section number throughout. **That file does not exist in the repo.** `docs/notes/` contains only `2026-08-01-context-transfer.md`, `linear_probe_loss_functions.md`, and the new bottlenecks note. Either it was never committed or it lives outside the checkout.

---

## 6. Repo claims verified against the code this session `[confirmed]`

| Claim | Location | Status |
|---|---|---|
| One line discards five of six generator columns | `data/download.py:68` — `select_columns(["caption", "coco_image", "dalle_image"])` | Holds |
| Both classes re-saved JPEG q95 | `data/download.py:80-82` | Holds. Format handled; compression *history* asymmetry not |
| No augmentation | `model/dataset.py` | Holds |
| Preprocessing destroys signal | `config.py:32` `IMAGE_SIZE = 380`; `model/dataset.py:47-48` resizes to a **square** | Holds — high frequencies *and* aspect ratio lost |
| `model/dataset.py` is v1's loader | returns `rgb` + `fft_mag` + `srm_residual` | New — the frozen path needs its own loader, and need not inherit the 380 square resize |

---

## 7. Open, and cheap

Carried to [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md) §9:

1. Native resolution distribution per COCO_AI column — the 270²/436²/480×640 figures are `[second-hand]` from `plan-b-verification.md` §7 and have never been measured here.
2. Does PE-Core-G14-448 load in this environment?
3. Does our extraction path L2-normalize (§2.6)?
4. GenImage per-generator download granularity.
5. Resolution histogram of any candidate real source against our fakes — decides trainable vs eval-only (§2.8).
6. **Does GenImage ship paired reals per generator, or a shared real pool?** `bottlenecks.md` §8 step 8 asserts paired; unverified. If it is a shared ImageNet-derived pool, adding those slices imports a second unpaired real source and the §2.8 hazard recurs.

---

## Sources

Read at the paper body this session:

- [*SSAFE*, arXiv:2606.08634](https://arxiv.org/abs/2606.08634) · [HTML full text](https://arxiv.org/html/2606.08634v1) — §2. Tables 2, 4, 5, 6, 8, 10, 12
- [*Perception Encoder*, arXiv:2504.13181](https://arxiv.org/abs/2504.13181) — §3

Re-used from earlier passes, not re-checked here:

- [*NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*, arXiv:2604.11487](https://arxiv.org/html/2604.11487) — §4
- [*DailyBench*, arXiv:2607.24016](https://arxiv.org/abs/2607.24016) — **withdrawn 2026-07-28**, §5.3
- [*Fake or JPEG?*, arXiv:2403.17608](https://arxiv.org/html/2403.17608v1) — §2.7
- [*AIGI-Holmes*, ICCV 2025](https://arxiv.org/html/2507.02664v1) — the benchmark SSAFE curates from and tests on
- [Awesome-AIGC-Image-Video-Detection](https://github.com/ant-research/Awesome-AIGC-Image-Video-Detection) — searched for an SSAFE code release; none found
