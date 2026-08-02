# Session findings — artifact taxonomy, augmentation, and four corrections

**Date:** 2026-08-02. Deadline Mon 2026-08-03.
**Status:** Current as literature. Findings only — **no number here was measured on our data.**
**Relates to:** [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) (the plans these findings feed) · [`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) §1.1, §1.6, §1.7 · [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §6.6, §8.4 · [`../notes/2026-08-01-context-transfer.md`](../notes/2026-08-01-context-transfer.md) §4, §8

A verification pass triggered by one question: *if the hypothesis is that generator decoders leave artifacts, how can heavy augmentation — which destroys those artifacts — be the fix? Is the signature a shortcut or a genuine feature?*

The question exposed a real contradiction in the advice this project had been given. Resolving it produced §1, which reframes the whole problem, and §5, which corrects four claims the project or this session had been reasoning from.

Evidence tags per repo convention: `[confirmed]` · `[measured]` · `[unverified]`.

---

## 1. "The artifact" is three things, not one

This is the session's main result. The three behave differently, and conflating them is what made augmentation look self-contradictory.

| | What it is | Genuine or shortcut? | Transfers? |
|---|---|---|---|
| **(A)** Generator-specific spectral fingerprint | Periodic peaks from one model's upsampler | **Genuine causal evidence** | No |
| **(B)** Decode-stage statistics | Any lossy latent→pixel decode (VAE, VQ-VAE) | **Genuine causal evidence** | Partially — the transfer hope |
| **(C)** Container / corpus artifacts | JPEG-vs-PNG, resolution, delivery pipeline | **Pure shortcut** | No, and it inflates in-distribution numbers |

### 1.1 (A) is genuine, not spurious `[confirmed]`

The decisive evidence is a **negative result inside the augmentation paper itself**. [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf) report that their augmentation **hurts** on SAN (super-resolution), because high-frequency components genuinely differentiate real from fake there and blurring removes those cues. DeepFake degrades too: **98.2% → 89.0%** at Blur+JPEG(0.5), partially recovered at (0.1).

You cannot call (A) a spurious correlation and also explain why destroying it costs accuracy. It is real evidence of synthesis — it is just **generator-specific**, which is a different failure from being spurious.

Consistent with `claim-verification.md` §1.7: Corvi et al. found strong spectral peaks for SD/LDM and notably weaker artifacts for ADM and DALL·E 2. Genuine, and non-uniform even inside one family.

### 1.2 (B) is genuine and causally established `[confirmed]`

Already in the project bibliography, restated here because it is what (A)-vs-(B) turns on. [INP-X](https://arxiv.org/html/2602.00192v1) restores pixels *outside* an edited region — keeping the generated content, removing the global decode — and detectors collapse:

| detector | standard | INP-X |
|---|---|---|
| Corvi2023 | 94.2% | 55.4% |
| Sightengine | 92.6% | 55.0% |

AEROBLADE reaches 0.992 mAP **training-free** from autoencoder reconstruction error alone. Remove the decode, lose the detector — that is a causal test, not a correlation.

### 1.3 (C) is the shortcut, and COCO_AI has it `[confirmed]`

[Fake or JPEG? (arXiv:2403.17608)](https://arxiv.org/html/2403.17608v1) — new to the project — shows GenImage detectors reading the container rather than the image:

- **80.45%** accuracy on uncompressed real images, **100%** on the *same* reals JPEG-compressed at q60. The detector is reading compression, not content.
- Constraining compression and size during training bought **+11.06 points** cross-generator accuracy (ResNet50, 71.68% → 82.74%) and **+11.74** (Swin-T, 74.09% → 85.83%).
- Robustness to compression improved +13.26 points at JPEG95.

**This is our dataset.** `coco_image` is natively JPEG; every generator column is PNG. `context-transfer.md` §4 flagged the asymmetry; this paper prices it at roughly eleven points.

### 1.4 What augmentation actually does

Not *"strip the signature so the model learns robust features"* — that framing was used earlier in this session and §1.1 contradicts it directly.

The correct account: augmentation **destroys (C), blunts (A), and leaves (B) intact.** It buys generalization by removing the *fastest-separating* features so gradient descent has to reach for slower, broader ones — Geirhos's shortcut-learning mechanism, already cited in `0002` §7.1. The cost is real and measured wherever (A) was the honest signal.

---

## 2. The augmentation protocol, as actually published `[confirmed]`

Wang et al.'s setting, which `0002` §8.4 adopts in a narrower variant:

| | |
|---|---|
| Gaussian blur | σ ~ Uniform[0, 3], applied **with probability 0.5** (or 0.1) |
| JPEG | quality ~ Uniform{30…100}, same probability |
| Notation | Blur+JPEG(0.1) = each applied 10% of the time |
| Result | ResNet-50 trained on **ProGAN only**, generalizing to 11 unseen generators: **92.6% AP** at (0.1), **90.8%** at (0.5) |

**The probability is load-bearing.** At p=0.5 half the training images are clean, so the model still sees (A) and still learns it — it just cannot rely on it exclusively. Applying augmentation to every image is a different intervention with no evidence behind it, and §1.1's SAN and DeepFake results are the warning about over-applying it.

The paper's headline generalization claim: augmentation with common post-processing improves generalization **even when the test images are not post-processed.**

`model/dataset.py` currently applies **none** (`0002` §6.4). This is a prerequisite on every path under consideration, not a choice between them.

---

## 3. A published design matches this project's proposed architecture `[unverified]`

[SSAFE (arXiv:2606.08634)](https://arxiv.org/html/2606.08634) — a frozen encoder plus **a single linear layer and sigmoid**. No fine-tuning, no text encoder, no prompts.

- Evaluated across **28 recent diffusion, transformer, and commercial generators**, including **GPT-Image-1, DALL·E 3, Imagen 3/4, Nano-Banana**, and the autoregressive **Janus-7B** and **LlamaGen**. That is our test distribution.
- **10K curated training images** suffice, against 288K / 4M full-dataset baselines. Reported: 89.4% acc / 95.7% AP on AIGIBench; 98.3% avg TPR / 99.9% ROC AUC on OpenFake.
- **Encoder ablation is the actionable part.** Comparing PE-Core, SigLIP2, CLIP variants, DINOv2 and DINOv3, it reports **PE-Core-G14-448** as giving the clearest real/fake separation. Not CLIP, not DINO — worth one head-to-head before defaulting to CLIP ViT-L/14 as `0003` §4.3 specifies.
- Uses a **single** frozen encoder — no ensemble, no fusion. Suggests the frozen branch should be the component that has to work, with spectral / noise / metadata branches serving the explainability requirement (`0001` requirement 2) rather than carrying transfer.

`[unverified]` — 2026 preprint, self-reported numbers, no replication located, paper page read rather than full text. Tagged accordingly, but it is the strongest available evidence that the minimal design is not a compromise.

### Expectation-setting against it `[unverified]`

[DailyBench (arXiv:2607.24016)](https://arxiv.org/html/2607.24016v1) evaluates on **Nano Banana 2, GPT-Image 2, FLUX.2, Qwen-Image, Z-Image**, and reports existing detectors falling from **91–96% balanced accuracy on GenImage to 60–76%** on its FakeBench (and **54–66%** on localized manipulation).

That is the honest prior for "founder generates an image live." It does not obviously reconcile with SSAFE's figures. The discrepancy is **recorded, not resolved in favour of the flattering number.**

---

## 4. Plan C's blocking precondition, restated `[confirmed]`

The threshold-correction route ([Yang et al., AAAI 2026](https://arxiv.org/abs/2602.01973), full treatment in [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md)) works only if the **ranking survives** on transformer images — AUC holding up while accuracy collapses. If AUC collapses too, separability is genuinely gone and no threshold moves it back.

`0002` §11 already lists this as an open item and **it has never been run.** It is the discriminating test between "recalibrate" and "retrain," and it needs both transformer images and a working probe to execute — so it cannot be the first thing built.

Second limit, from `calibration-and-thresholds.md` §1.4: prior-shift methods assume **label shift** (`p(y)` moves, `p(x|y)` fixed). An unseen generator changes what `fake` looks like, so `p(x|fake)` changes shape and these methods are formally **unlicensed** here — not known to fail, but any correction applied is empirical and should be reported that way.

---

## 5. Corrections

Four claims contradicted this session. Recorded here rather than edited into the source docs, per the `research/` convention.

### 5.1 ✗ "Transformer generators have no decode stage" `[confirmed false]`

Asserted earlier in this session as the reason diffusion-trained detectors cannot transfer to GPT Image / Gemini. **False.**

Autoregressive image models decode through a **VQ-VAE / visual tokenizer** — Janus Pro uses the LlamaGen tokenizer at downsample rate 16 ([Janus, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Janus_Decoupling_Visual_Encoding_for_Unified_Multimodal_Understanding_and_Generation_CVPR_2025_paper.pdf), [VTBench](https://arxiv.org/html/2505.13439v1)). Quantization loss is plausibly a *stronger* decode artifact than a continuous VAE's.

`claim-verification.md` §1.6 already had the general form of this: DiT swaps the U-Net for a transformer but keeps SD's VAE — **"transformer" and "diffusion" are orthogonal axes, and the axis that matters is the output stage.** The session restated the error the repo had already corrected.

**Why it matters:** every generator in scope has a lossy latent→pixel decode. §1's (B) is therefore a legitimate shared target, and augmentation-only transfer is worth attempting rather than ruled out a priori.

### 5.2 ✗ TextureCrop's numbers are misquoted in `context-transfer.md` §8 `[confirmed]`

That note records **+4.75% balanced accuracy** over center-crop and **+12.1% BA** for crop-vs-resize. The published abstract reports **+6.1% AUC over center cropping and +15% AUC over resizing**, across Forensynths, Synthbuster and TWIGMA ([arXiv:2407.15500](https://arxiv.org/abs/2407.15500), full title *TextureCrop: Enhancing Synthetic Image Detection through Texture-based Cropping*).

Different metric, different magnitudes. Direction unaffected — crop beats resize, and it is a preprocessing wrapper that plugs into a pretrained detector.

### 5.3 ✗ "Hyper-augmentation," applied to every image `[confirmed false]`

Proposed earlier in this session, with code applying 2–3 augmentations unconditionally. Not the sourced protocol — see §2. The published setting is probabilistic (p = 0.5 or 0.1) and the paper documents cases where augmentation *hurts*.

### 5.4 ~ "Multi-generator training data fixes the transformer gap" `[downgraded]`

Proposed earlier in this session on the strength of Community Forensics' generator-count result. The user's objection was correct: **all six COCO_AI generator columns are latent diffusion.** Adding SD 2.1 / SD3 / SD3.5 / SDXL to a DALL·E 3 training set broadens coverage *within* (A)-space but does not by itself supply a non-diffusion decode.

The generator-count lever (`claim-verification.md` §1.4) still stands — it is just not sufficient on its own for the cross-paradigm case, and it was presented as if it were.

---

## 6. What follows

Detailed in [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md). In short:

1. **Run `0002` §9 first** — frozen backbone + logistic regression on COCO_AI, tested on the Gemini/GPT images that already break v1. No new data, no API spend. It is the unrun gating experiment, and it separates "the mechanism transfers" from "buy transformer data."
2. **Turn augmentation on regardless**, at the §2 protocol. Prerequisite on every path.
3. **Uniform JPEG re-encode across both classes**, and reals that resemble the reals at demo time rather than 2014-era COCO web JPEGs. §1.3 is the failure mode that produces a beautiful validation number and a coin-flip demo.
4. **Run the container control** before trusting any result: take one image the model calls fake, re-save at JPEG q85, rescale 95%, re-score. If the score collapses, the probe is reading the delivery pipeline rather than the image (`context-transfer.md`, final section).

---

## Sources

New to the project:

- [Wang et al., *CNN-generated images are surprisingly easy to spot… for now*, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf) ([ar5iv full text](https://ar5iv.labs.arxiv.org/html/1912.11035)) — §1.1, §2. Previously cited in `claim-verification.md` §1.5 for its headline number only; this pass read the protocol and the negative results.
- [*Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, arXiv:2403.17608](https://arxiv.org/html/2403.17608v1) — §1.3
- [*SSAFE: Simple and Strong AI-Generated Image Detection via Frozen Vision Encoders*, arXiv:2606.08634](https://arxiv.org/html/2606.08634) — §3
- [*DailyBench*, arXiv:2607.24016](https://arxiv.org/html/2607.24016v1) — §3
- [*TextureCrop*, arXiv:2407.15500](https://arxiv.org/abs/2407.15500) — §5.2
- [*Janus*, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Janus_Decoupling_Visual_Encoding_for_Unified_Multimodal_Understanding_and_Generation_CVPR_2025_paper.pdf) · [*VTBench*, arXiv:2505.13439](https://arxiv.org/html/2505.13439v1) — §5.1
- [*SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning*](https://mever-team.github.io/spai/) — not used above; an alternative to patching if the resolution mismatch becomes the blocker

Already in the bibliography, re-used:

- [INP-X, arXiv:2602.00192](https://arxiv.org/html/2602.00192v1) · [AEROBLADE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.pdf) — §1.2
- [Yang et al., AAAI 2026](https://arxiv.org/abs/2602.01973) — §4
- [Community Forensics, CVPR 2025](https://arxiv.org/abs/2411.04125) — §5.4
- Geirhos et al., [*Shortcut Learning in Deep Neural Networks*](https://www.nature.com/articles/s42256-020-00257-z) — §1.4
