# Bottlenecks — full list as of 2026-08-02

**Written:** 2026-08-02. Deadline Mon 2026-08-03.
**Status:** Working note. Inventory, not a plan.
**Relates to:** [`../reference/2026-08-02-session-findings.md`](../reference/2026-08-02-session-findings.md) · [`../reference/2026-08-02-plan-a-verification.md`](../reference/2026-08-02-plan-a-verification.md) · [`../reference/2026-08-02-plan-b-verification.md`](../reference/2026-08-02-plan-b-verification.md) · [`../reference/2026-08-02-plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) · [`../decisions/0003-frozen-probe-demo-build.md`](../decisions/0003-frozen-probe-demo-build.md) · [`2026-08-01-context-transfer.md`](2026-08-01-context-transfer.md)

Evidence tags per repo convention: `[confirmed]` checked against a primary source · `[measured]` measured on our data · `[repo]` stated in an existing repo doc · `[second-hand]` a number carried from a doc but never measured here · `[unverified]`.

**Nothing in this file is `[measured]`.** That is itself bottleneck #9.

---

## 0. The one-line statement

`0003` §5 says the binding constraint is *data, not modelling*, and counts only generators. That was half the diagnosis. The full version:

> **Our corpus occupies a resolution band and a single architecture family that neither the recipe we are copying nor the demo distribution occupies.**

Both are sourcing problems. Neither is fixed by more images, better augmentation, or a different probe.

---

## 1. Ranked summary

Ranked by *cost to clear ÷ consequence if not cleared*. "Blocks" = D (Monday demo), R (the research/generalization claim), or both.

| # | Bottleneck | Blocks | Cost to clear | Tag |
|---|---|---|---|---|
| 1 | Resolution band: corpus 270–480px vs demo 1024–4096px | D + R | High — resourcing, not a fix | `[second-hand]` |
| 2 | Single architecture family (all 6 COCO_AI columns are latent diffusion) | R | Medium — GenImage download | `[repo]` |
| 3 | No VQ/tokenizer decode in training — the paradigm Monday tests | D + R | Medium — GenImage VQDM slice | `[confirmed]` |
| 4 | Real class is one source, 2014-era COCO web JPEG | D + R | Low as **eval**; high as **training** — needs pairing (§2.4) | `[repo]` |
| 5 | Container↔label correlation is perfect (JPEG reals / PNG fakes) | D + R | Low — uniform re-encode | `[confirmed]` |
| 6 | Five of six generators discarded at `data/download.py:68` | R | **Trivial — one line** | `[confirmed]` |
| 7 | Row-level leakage (1 row = 1 real + 6 fakes) | R | Low — manifest work | `[repo]` |
| 8 | Nothing materialized: no `data_raw/`, `dataset/`, `checkpoints/` | D + R | Low — hours of download | `[confirmed]` |
| 9 | No OOD eval set, and no AUC has ever been computed | D + R | **Low, and highest information yield** | `[confirmed]` |
| 10 | Backbone unresolved; `0003` §4.3's choice is the worst-evidenced | D + R | Low — but see #1 | `[unverified]` |
| 11 | Preprocessing destroys the signal (380 square resize) | D + R | Low — crop instead | `[confirmed]` |
| 12 | Evidence base has withdrawn / unreplicated / conflicting sources | R | Cannot be cleared — only stated | `[confirmed]` |
| 13 | One day, competing with live-coding prep | D | Cannot be cleared | — |

---

## 2. Data

### 2.1 Resolution band — the bottleneck `0003` did not name

Three separate problems get merged under "resolution." They are independent.

| | Problem | Fixed by |
|---|---|---|
| a | Encoder input: CLIP 224 vs PE-Core 448 | Backbone swap |
| b | Our preprocessing: `model/dataset.py:48` resizes to `IMAGE_SIZE = 380`, **square** — destroys high frequencies *and* aspect ratio `[confirmed]` | Crop instead of resize |
| c | **Source images are natively 270²–480px** `[second-hand]` | Nothing cheap |

**(c) can invert (a).** Upsampling 270 → 380 → 448 means the high-frequency band PE-Core's extra resolution exists to read is empty, or filled with interpolation artifacts. Since each generator column has a different native size, that interpolation signature differs per column — which is *resolution-as-label* amplified, not removed.

Consequence: **PE-Core-448 on this corpus could underperform CLIP-224 and could score well for the wrong reason.** The 22.6-point encoder spread was measured on data that satisfies a precondition ours violates.

The wider version: SSAFE's reals are 3,000–4,500px `[confirmed]`; Nano Banana Pro defaults to 4K `[repo]`; our corpus is 270–480px. **Train and test occupy disjoint resolution bands.**

The 270²/436²/480×640 figures are `[second-hand]` from `plan-b-verification.md` §7 — never measured here. Measuring them is open item #9.1.

### 2.2 Single architecture family

All six COCO_AI generator columns are latent diffusion `[repo]`. SSAFE's curation selects 8 generators from 28 by Farthest-Point Sampling over PE-Core embeddings, and the selection deliberately spans four families: **SD1.4, BigGAN (GAN), GLIDE, Midjourney, SDXL-LoRA, SD2.1, SD1.5, SD3.5** `[confirmed]`.

Run through their own procedure, our six columns would collapse to roughly one representative. We would be supplying exactly the input FPS is built to reject.

The measured consequence of low diversity is SSAFE Table 5 `[confirmed]`:

| Training set | Real TNR | Avg TPR | ROC AUC |
|---|---|---|---|
| AIGIBench (288K) | 65.6% | 78.4% | 81.5 |
| OpenFake (4M) | **29.7%** | 94.9% | 87.5 |
| Universal (50K) | 96.8% | 95.3% | 99.5 |
| **Curated 10K** | **98.3%** | 94.4% | 99.0 |

**Composition beats scale by 400×.** This is the single most decision-relevant number found this session.

### 2.3 No VQ / tokenizer decode in training

`session-findings.md` §5.1 establishes that autoregressive and commercial generators decode through a VQ-VAE / visual tokenizer `[confirmed]` — which is what licenses decode-stage statistics as a shared target at all.

Our training data contains no VQ decode. **GenImage's VQDM slice is a vector-quantized decode and is the cheapest available proxy for the paradigm the demo tests.** It converts "trust that it transfers to VQ decodes" into "measure it."

Caveat: VQDM is a 2022-era tokenizer; its artifacts may not resemble Janus/GPT-Image's. A proxy, not a substitute. SSAFE does not have one either.

### 2.4 Real class — and why the obvious fix is a trap

One source, 2014-era COCO web JPEG `[repo]`. SSAFE uses seven — LSUN, ImageNet, COCO 2017, OpenImages V7, Unsplash, Pexels, Pixabay — and added 5K modern high-res reals because AIGI-Holmes contains "only LSUN- and ImageNet-style real images, which do not reflect the characteristics of contemporary smartphone or web-crawled photos" `[confirmed]`.

The obvious move is to copy that: download OpenImages V7 / Pexels / Unsplash, shoot an hour of phone photos, done. **That move is unsafe here, and the paper does not license it.**

**What SSAFE actually does `[confirmed]`:**

- Its AIGI-Holmes portion **is** domain-paired — SD1.4/BigGAN/GLIDE/Midjourney generate ImageNet-domain content against ImageNet reals; SDXL-LoRA/SD2.1/SD1.5 are COCO-domain against COCO reals.
- The 5K modern reals are **unpaired** — added as general real-image diversity, matched to no generator.
- The iPhone 12–17 / Samsung photos are **RealWorldBench, the test set**, not training. They were never paired against training fakes.
- The paper contains **no discussion** of a classifier exploiting source characteristics (camera photo vs render, stock-photography style) rather than synthesis artifacts. It frames the issue only as a "domain gap."
- **No ablation isolates the modern reals' contribution.** Table 6 compares random vs curated *generator* selection; nothing separates real-source diversity from improved fake detection.
- **No resolution statistics are given for RealWorldBench's synthetic images**, while its reals reach 5712×4284.

**Why the last gap matters.** If RealWorldBench's fakes sit at ~1024² against 4–5K reals, then "large image = real" separates the benchmark almost perfectly — producing *both* the 98.3% TNR and the 94.4% TPR. A high TPR does not rule the shortcut out, because the shortcut also gets the fakes right. Nothing published distinguishes these, and no control was run that could. This is §1's (C) sitting unexamined inside the paper we are copying, consistent with §2.5's finding that SSAFE never discusses compression or format bias either.

**The trap for us, and it is specific.** Adding unpaired modern high-res reals to COCO_AI's 270–480px latent-diffusion fakes makes *"high-resolution modern photo = real"* the cheapest available hypothesis. On Monday the founder generates a ≥1024² Gemini image and the model calls it **real** — the exact failure being fixed, manufactured by the fix. Our resolution bottleneck (§2.1) makes this *worse* than SSAFE's case, because our fakes sit in a lower band than the reals we would add, so the correlation would be cleaner than theirs.

**Therefore:**

1. **Modern reals go into the eval set first.** There they measure false-positive risk on the demo distribution, which is what we need to know.
2. **Training-set additions require pairing** — matched resolution and content domain against fakes. We have no matched modern fakes, so this is not the cheap item it appeared to be.
3. **COCO_AI's 1:1 pairing is our most valuable data property.** Do not break it. See §7.

**Manufacturing the missing paired fakes was proposed and falsified** — see [`../reference/2026-08-02-dataset-alignment.md`](../reference/2026-08-02-dataset-alignment.md). VAE round-tripping a modern real to create its paired fake is AlignedForensics (ICLR 2025), which collapses outside the SD-VAE family (25.87% on FLUX, 3.6% on DDA-COCO/FLUX), and reconstructions measurably cluster with **real** rather than with generated images. Do not re-propose it as the sole fake class. The live lead from that pass is **DDA's dual alignment** — format-matching both sides — which is the only published method addressing this section's problem directly.

Two corrections to record:

- `plan-b-verification.md` §4 attributes 41.8–66.9% real accuracy to SSAFE as a limitation. That belongs to models trained on **AIGI-Holmes alone** — SSAFE's own curated model reaches **98.3% TNR** on RealWorldBench `[confirmed]`. The doc's §4 reconciliation of SSAFE against DailyBench is void on both sides: misread number, withdrawn counterparty.
- `plan-a-verification.md` §3.2's objection — that stock-site reals carry their own processing fingerprint (uniform sRGB export, Lightroom pipeline, professional composition), separable from AI output on style alone — **stands unrefuted.** SSAFE's use of Pexels/Unsplash/Pixabay is not evidence against it; the paper never ran that argument. An earlier reading in this session treated it as settled in SSAFE's favour. It is not.

### 2.5 Container↔label correlation

`coco_image` is natively JPEG; every generator column is PNG `[repo]`. One real source × six same-family generators = the correlation is **perfect**, not merely present. Priced at ~11 points cross-generator by *Fake or JPEG?* `[confirmed]`.

SSAFE never discusses compression, format, or resolution bias and runs no control for it `[confirmed]`. Not an oversight — 7 real sources × 8 generators × 4 families leaves nothing to shortcut to. **Their curation dissolved the problem; ours manufactures it.** So we need a control they did not.

`data/download.py:80-82` already re-saves both classes at JPEG q95 `[confirmed]`, which handles format but not compression *history* — COCO reals double-compress, generator PNGs single-compress.

### 2.6 Mechanical items

- `data/download.py:68` still reads `select_columns(["caption", "coco_image", "dalle_image"])` — one line discarding five of six generators `[confirmed]`. This was `0003` §7's **highest-rated risk** ("multi-generator data may not land in time"), planned as an external corpus download with GPU generation as fallback. It is a one-line fix.
- **Row-level splitting is mandatory** — one COCO_AI row is one real image plus six fakes `[repo]`.
- **Nothing is on disk.** `data_raw/`, `dataset/`, `checkpoints/` all absent `[confirmed]`. (`.venv/` exists — `plan-c-source-verification.md` §1's table is stale on that row.)

---

## 3. Model and architecture

### 3.1 The backbone is a ceiling, not a hyperparameter

A frozen encoder is a fixed measurement instrument. The probe can only read what survived into the embedding; if the encoder discarded the artifact, **no probe, no data, and no augmentation recovers it.** Fine-tuning could partially re-learn it and is ruled out.

`0003` §4.3 specifies CLIP ViT-L/14, justified as "UniversalFakeDetect's exact setup — the most documented and most reproducible." That is a sound argument for a **control**, and `0003` used it as a performance argument too.

How the candidates differ:

- **DINOv2/v3** — self-supervised, trained for *invariance* across augmented views. That is a learned instruction to discard low-level, high-frequency, local detail, which is where synthesis artifacts live. DINOv2-Giant is the worst row in SSAFE's ablation at 72.8%.
- **CLIP ViT-L/14** — image-text contrastive. No caption describes upsampler ringing, so nothing preserves artifacts; nothing destroys them either. They survive incidentally, which is why UniversalFakeDetect works at all and why it is fragile.
- **PE-Core / SigLIP2** — contrastive vision-language at much larger scale and resolution. SSAFE: *"multimodal encoders substantially outperform self-supervised encoders."*

**Confound to record:** NTIRE 2026's winners used **DINOv3-7B** alongside SigLIP2-Giant `[confirmed]`. DINOv3 is self-supervised, so at 7B scale the multimodal>self-supervised rule breaks. PE-Core-G14 and SigLIP2-G16 are both G-class while DINOv3 ViT-L/16 is not — SSAFE's ablation may be partly measuring size rather than objective. This is a reason to run our own two-arm comparison rather than accept the ranking.

### 3.2 Unresolved model-side items

- **PE-Core-G14-448 weight availability in our environment — unverified.** If loading is awkward, half the ablation evaporates.
- **Layer selection.** The Perception Encoder paper's central claim is that *the best visual embeddings are not at the output of the network* but in intermediate layers `[confirmed]`. For a frozen-backbone probe this is a free variable we have no plan for, and it could matter as much as the backbone choice.
- **L2 normalization.** SSAFE L2-normalizes embeddings before the linear layer `[confirmed]`. Whether our extraction path does is unverified.
- **No augmentation implemented** `[confirmed]`. `0002` §8.4 called it mandatory; `0003` §6 dropped it; nothing reinstated it.
- **`model/dataset.py` is v1's three-branch loader** (returns `rgb` + `fft_mag` + `srm_residual`) `[confirmed]`. The frozen path needs its own loader; the 380 square resize is v1's constraint and need not be inherited.
- **Probe form is settled** — `0003` §4.4's logistic regression *is* SSAFE's linear layer + sigmoid + BCE. No change needed.

---

## 4. Evaluation

### 4.1 AUC has never been computed — the highest-yield gap

`0002` §11's open item, never run. One measurement serves three purposes at once:

| Result on a transformer eval set | Reading | Action |
|---|---|---|
| AUC ≳ 0.75, accuracy ≈ 50% | Shifted but separable | Threshold correction applies — a scalar and an afternoon |
| AUC ≈ 0.5 | Overlapped, the Chameleon regime | Threshold correction is dead; buy data |

Two constraints: it must be computed on the **pooled** real-vs-fake set, not per-generator (`plan-b-verification.md` §6), and per `plan-c-source-verification.md` §2.2 this is **not a replication** — Yang et al. reports accuracy only, no AUC, no ranking analysis anywhere in the paper. It is the first recorded measurement of that mechanism.

### 4.2 The rest

- **No OOD eval set exists.** `0002` §5.2 was an ad-hoc spot check with no recorded sample size and no reproducible script `[confirmed]`.
- **No pre-registered success criterion.** Against a ~98% real / ~0% fake prior, one live image proves nothing in either direction. Fix the set (≥50/class) and the threshold *before* looking — `0002` §9's falsification gate has already been spent once (`0003` §3).
- **The container control has never been run** — re-save at JPEG q85, rescale 95%, re-score.
- **That control is blind to watermarking.** SynthID is embedded in pixel values at generation time and engineered to survive cropping, compression and resizing `[confirmed]` — it passes the control by design. Discriminating replacements: codec A/B (API PNG vs web-UI WebP) and re-render/screenshot.
- **No v1 comparison.** `0002` §9's gate required v1 analysis that `0003` §1 placed out of scope. An absolute AUC criterion substitutes for it but does not restore it.

---

## 5. The evidence base is itself a bottleneck

Not clearable — only statable.

- **DailyBench (arXiv:2607.24016) was withdrawn 2026-07-28**, "some errors must be corrected" `[confirmed]`. It carried the UniFD 0.00% collapse on Nano Banana 2 / GPT-Image 2 — the number `plan-b-verification.md` §1 builds its central verdict on — plus the 91–96% → 60–76% pessimistic prior and the NPR floor. All three are now unsourced. The direction survives only via NTIRE, which is independent and not withdrawn.
- **SSAFE is an unreplicated 2026 preprint**, and its **data is not released** — the only statement is *"Code, curated generator lists, and evaluation scripts will be released upon publication,"* which promises the recipe, not the images, and is conditional on a publication that has not happened `[confirmed]`. What we do have is the generator list and the real-source list, which is the expensive part of the curation.
- **SSAFE offers no mechanism.** No frequency analysis, nothing on decoder/VAE/tokenizer artifacts `[confirmed]`. Our (A)/(B)/(C) taxonomy is *our* theory. SSAFE supports the outcome, not the explanation.
- **SSAFE's real class is uncontrolled** (§2.4) `[confirmed]`. Unpaired modern reals, no source-characteristic discussion, no isolating ablation, no synthetic-resolution statistics for the benchmark whose reals reach 5712×4284. Its headline real-accuracy numbers cannot be separated from a corpus shortcut using anything published. This does not make them wrong — it makes them unverifiable, which matters because §2.4's recommendation was originally derived from them.
- **Internal conflicts.** `plan-b-verification.md` cites SSAFE Tables 2/8/12; `plan-c-source-verification.md` §4 says the body was not read and downgrades the same claims. plan-B's read is the accurate one — verified against the body this session — so plan-C's downgrade is the stale side.
- **`docs/notes/2026-08-02-three-candidate-plans.md` does not exist in the repo** `[confirmed]`, and all four 08-02 reference docs are structured as amendments to it by section number.

---

## 6. Time

One day, against `0003` §7's last risk row: *"build time competing with live-coding prep, which `notes.md` identifies as the actual round."* That row argues against scope expansion and it is not wrong.

Separating core from optional, which an earlier framing failed to do:

- **Core** (six-column fix, one encoder, modern reals, row-level split) is comparable to `0003`'s original two-day scope, possibly cheaper — it *removes* `0003` §5's external corpus download and GPU-generation fallback.
- **Optional** (the second encoder arm, the fallback stack) is what pushes it over. These are the first two things to cut.

---

## 7. Not bottlenecks — recorded so they are not re-litigated

- **Probe form.** `0003` §4.4 matches SSAFE exactly.
- **Whole images / no face crop.** `0003` §4.1 is corroborated — SSAFE and NTIRE both operate on whole images.
- **Binary real vs AI.** `0003` §4.2 unchanged.
- **Training-set size.** 10K suffices; scale is not our problem (§2.2).
- **Domain pairing.** COCO_AI's 1:1 real↔fake pairing matches SSAFE's discipline of pairing reals to fake domains. We already do this right — and per §2.4 it is now the **most valuable property our data has**, since it is the one thing preventing the corpus shortcut that SSAFE never controlled for. Protecting it is a constraint on every proposed data addition.
- **Generator count as such.** The lever is family spread, not count. Six latent-diffusion columns are not six generators for this purpose.

---

## 8. Cheapest clearing order

By information yield per hour, not by importance.

1. **Measure the corpus** (§9.1). Everything in §2.1 is `[second-hand]`; if the numbers are wrong in either direction the ordering below changes.
2. **`data/download.py:68`** — one line, clears `0003`'s highest-rated risk.
3. **Modern reals into the *eval* set only** — public downloads plus own phone photos. Cheap, and it measures the false-positive risk on the demo distribution. **Not into training** (§2.4).
4. **Uniform re-encode across all classes and sources**, extending `download.py:80-82` to everything added.
5. **Row-level split** in the manifest.
6. **Frozen extraction + probe, reporting AUC and accuracy separately**, on a pre-registered OOD set.
7. **Container control before fitting anything downstream.**
8. *Optional:* GenImage slices (VQDM, BigGAN, GLIDE) for family spread. Highest research value, largest download. Note these arrive **paired to their own reals**, which is why they are safe to add to training where §2.4's stock reals are not.
9. *Optional:* the second encoder arm.

**Standing constraint on every step above:** no data addition may put a real source into training without matched fakes at comparable resolution and content domain (§2.4, §7).

---

## 9. Open measurements

Cheap, and each one currently blocks a decision on `[second-hand]` evidence.

1. **Native resolution distribution per COCO_AI column** — the whole of §2.1 rests on figures never measured here. `model/resolution_swap_probe.py` already exists from the July investigations.
2. **Does PE-Core-G14-448 load in this environment?**
3. **Does our extraction path L2-normalize?**
4. **GenImage per-generator download granularity** — whether a few thousand images per generator can be pulled without the full corpus.
5. **Resolution distribution of any candidate real source against our fakes** (§2.4). If modern reals overlap our fakes' band they may be trainable; if they sit an order of magnitude above, they are eval-only. This is the measurement that decides it, and it is one histogram.
