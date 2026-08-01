# Claim Verification Pass — Decoder Artifacts, Cross-Generator Transfer, and the Real-Class Anchor

**Status:** Partially superseded — §1.2 and §5 overtaken 2026-08-01
**Date:** 2026-07-31
**Relates to:** [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §6.5, §6.6, §8.5, §10, §11 — several of which this pass amends.

> **Superseded in part by [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md).** Findings below are left unedited per the `research/` convention; read them with these two corrections in hand:
>
> - **§1.2's calibration conclusion is reversed.** The premise (failure on unseen generators is confident and biased toward `real`) stands. The two inferences drawn from it do not: *"calibration cannot recover it"* is contradicted by [Yang et al., AAAI 2026](https://arxiv.org/abs/2602.01973), which attributes the failure substantially to **misaligned decision thresholds rather than lost feature separability** and corrects it with a scalar logit adjustment on a frozen backbone; and *"abstention must key on distance, not confidence"* is contradicted by [Jaeger et al., ICLR 2023](https://arxiv.org/abs/2211.15259), which reports no method beating a Maximum Softmax Response baseline across a realistic range of failure sources. A confidently-wrong model can still rank correctly — a monotone rescaling moves the boundary without changing the ordering, and §1.2 did not make that distinction.
> - **§5's open item on the 18–30% figures is answered.** Substantially misplaced thresholds, not lost separability.
>
> §1.2 is also the project's first overturned `[confirmed]` tag. The failure mode — a confirmed premise carrying an unconfirmed inference at the same confidence — is analysed in the new doc's §6.

A verification pass over the mechanistic claims the project had been reasoning from, several of which had never been sourced. Unlike [`deepfake_detection_research.md`](deepfake_detection_research.md) (a breadth survey of architectures and datasets), this is a narrow audit: each claim was checked against primary sources, and the ones that failed are recorded as failures.

The unit here is the **claim**, not the paper. A single paper contributes claims at different confidence levels with different expiry dates, and filing it as one object destroys that distinction.

---

## 1. Confirmed

### 1.1 Detectors key on the global VAE decoding stage, not on synthesized content `[confirmed]`

Three independent lines of evidence:

- **[AlignedForensics (ICLR 2025)](https://arxiv.org/abs/2410.11835)** trains a detector on real images vs. their LDM-autoencoder reconstructions — *no denoising step at all* — and it transfers to genuinely generated images across latent diffusion models. If the decode pass alone is sufficient training signal, the decode pass is what carries the discriminative artifact.
- **[AEROBLADE (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.pdf)** reaches 0.992 mAP **training-free**, purely from autoencoder reconstruction error, across SD / Kandinsky / Midjourney.
- **[Inpainting Exchange, INP-X (2602.00192)](https://arxiv.org/html/2602.00192)** is the cleanest test. It restores original pixels *outside* the edited region — keeping the generated content, removing the global decode — and detectors collapse:

  | detector | standard inpainting | INP-X | drop |
  |---|---|---|---|
  | Corvi2023 | 94.2% | 55.4% | −38.8 |
  | Sightengine (commercial) | 92.6% | 55.0% | −37.6 |
  | Hive Moderation (commercial) | 91.4% | 54.8% | −36.6 |
  | DNF (best open-source) | 71.0% | 60.4% | −10.6 |

  Best accuracy after *fine-tuning specifically on INP-X*: 75.3%.

**Why this matters here:** it is the mechanism behind 0002 §5.2. Our FFT/SRM branches are, functionally, global-decode-artifact detectors. It also explains why the `edited` class (localized manipulation) is a structurally different problem from `deepfake` (global synthesis) — see §4.2 below.

### 1.2 Failure on unseen generators is confident and biased toward `real` `[confirmed]`

A binary detector learns *"fake = artifact signature X present."* It never learns a positive definition of `real`; `real` is the absence of X. On an unseen decoder, X is absent, the input sits far from the boundary on the no-artifact side, and the model returns a high-confidence `real`.

Corroborated by [Breaking Latent Prior Bias](https://arxiv.org/pdf/2506.00874) (ProGAN-trained detectors misclassify StyleGAN2/Midjourney as real; models "default to classifying as real" when learned artifacts are absent) and [GenDet](https://arxiv.org/html/2312.08880) (real/fake output distributions converge on unseen generators).

Three consequences:

- Errors are **systematic false negatives**, not symmetric noise. Degradation will not look like accuracy drifting toward 50%.
- **Calibration cannot fix it.** The model is not uncertain; it is wrong and confident. Temperature scaling rescales a confidently-wrong score.
- Abstention must key on **distance to the training distribution**, not on output confidence — confidence is precisely the broken signal.

This reframes 0002 §5.2 (>75% `real` on Gemini / gpt-image-1) from a defect in our model to the documented failure mode of the approach.

### 1.3 Current detectors are unusable as shipped on 2026 commercial generators `[confirmed]`

[Open-sourced detector benchmark (2602.07814)](https://arxiv.org/html/2602.07814v1) — 16 methods (23 pretrained variants), 12 datasets, 2.6M images, 291 generators:

| generator | mean detection accuracy |
|---|---|
| Firefly v4 | 18% |
| Flux Dev | 21% |
| Midjourney v7 | 24% |

| detector | mean accuracy across all 12 datasets |
|---|---|
| Community-Forensics (best) | 75.0% |
| AIGCDetectBenchmark_CNNSpot (worst) | 37.5% |

21% accuracy means ~79% of Flux Dev fakes are labeled real — **anti-correlated**, only achievable if the detector is systematically confident in the wrong direction. Same phenomenon as §1.2, measured from the other side.

**Caveat (important, do not drop it):** these are accuracy figures at each method's own threshold on balanced sets. The benchmark does not decompose threshold placement from separability, so a detector at 21% accuracy may retain salvageable AUC. The supported claim is **"unusable as shipped,"** not "no separable signal exists."

Two secondary reads: the **best** detector is the one trained on the most generators (Community-Forensics), an independent replication of the coverage lever; and a **37.5-point spread** across "SOTA" methods means detector *selection* dominates architectural insight.

### 1.4 Generator count is the dominant training lever `[confirmed]`

[Community Forensics (CVPR 2025)](https://arxiv.org/abs/2411.04125): 2.7M images from 4,803 generators (~250× prior datasets). Detection performance improves monotonically with the number of training generators, **even when those generators share an architecture**.

New detail this pass added: **diminishing returns beyond roughly 1,000 models.** Relevant to budgeting — the curve is steep early, so the first handful of additional generators is where our marginal return lives.

### 1.5 Post-processing augmentation is the best-aged result in the field `[confirmed]`

[Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf): ResNet-50 trained on ProGAN only (720k images, 20 LSUN categories), generalizing to 11 unseen CNN generators. Blur+JPEG(0.1) → 92.6% AP; Blur+JPEG(0.5) → 90.8%. Augmentation with common post-processing improves generalization **even when the test images are not post-processed**.

Directly supports 0002 §8.4 (currently zero augmentation in `model/dataset.py`).

### 1.6 A transformer backbone does not create a new artifact family `[confirmed]`

[DiT (Peebles & Xie, ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) replaces the U-Net denoiser with a transformer but uses **Stable Diffusion's pretrained VAE** (256×256×3 → 32×32×4). Same for rectified-flow / flow-matching objectives and consistency distillation: they change the sampling trajectory, not the pixel-writing stage.

"Transformer" and "diffusion" are orthogonal axes. For detection purposes the axis that matters is the **output stage**, not the denoiser or the sampler.

### 1.7 Upsampling-induced spectral artifacts (the historical basis for the FFT branch) `[confirmed]`

[Odena et al., Deconvolution and Checkerboard Artifacts](https://www.semanticscholar.org/paper/Deconvolution-and-Checkerboard-Artifacts-Odena-Dumoulin/d9dab7574d56ae81efe6c90c213c6509b36cf950): zero-insertion during transposed convolution replicates the low-resolution spectrum into the high-frequency band, producing distinctive spectral peaks.

[Corvi et al., ICASSP 2023](https://github.com/grip-unina/DMimageDetection) extended this to diffusion via noise residuals averaged over 1,000 images — **but found strong peaks for Stable Diffusion / LDM and notably weaker artifacts for ADM and DALL·E 2.** The artifact is not uniform across the diffusion family. This is a qualification of, not a support for, the "all LDMs stamp the same thing" reading.

### 1.8 The real class is drifting, from the manufacturer side `[confirmed]`

The one-class framing (*anchor on camera physics; it's constrained and doesn't churn every six months*) is weaker than previously assumed here.

PRNU — sensor-level Photo-Response Non-Uniformity, the classic per-device camera fingerprint — is losing uniqueness on modern smartphones. Computational photography and AI enhancement introduce **Non-Unique Artifacts (NUAs)**: patterns shared by every device of a given model, because they originate in the shared processing pipeline rather than the individual sensor. This causes **false positive attributions** between distinct devices ([Samsung diagonal artifacts](https://arxiv.org/pdf/2510.09509), [Apple synthetic defocus](https://arxiv.org/pdf/2505.07380)).

Further, neural ISP modules are described as sharing the hallucinatory characteristics of generative models used for post-capture editing — i.e. "real" photographs increasingly contain generated content ([Addressing Image Authenticity When Cameras Use Generative AI](https://arxiv.org/pdf/2604.21879)).

**Net:** the physics anchor is eroding independently of anything generator authors do. Any future one-class / camera-anchored direction must budget for this.

---

## 2. Corrected — claims this project was reasoning from that are wrong

### 2.1 ✗ "Most latent diffusion models share an f8 VAE lineage, so a decoder-artifact detector covers the family"

**False.** Per [madebyollin's VAE notes](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9):

| model | latent channels | note |
|---|---|---|
| SD 1.x / 2.x | 4 | |
| SDXL | 4 | latents **incompatible** with SD despite the same architecture |
| SD3 | 16 | VAE trained from scratch; adds a `shift_factor` |
| FLUX.1 | 16 | SD3-style configuration |
| FLUX.2 | 32 | substantially more aggressive compression |

These VAEs are **not interchangeable** — you cannot encode with one and decode with another. And the trend is **divergence over time**, not convergence.

`[unverified]` — one unresolved conflict: [DeepWiki](https://deepwiki.com/black-forest-labs/flux2/3.4-autoencoder-(vae)) reports FLUX.2 at 32:1 spatial compression (H/32 × W/32), 64:1 after patch rearrangement; the gist reports a figure that reads as a *data* compression ratio rather than a spatial one. These measure different quantities and may not actually contradict, but the reconciliation was not derivable with confidence. Recorded rather than guessed. What is certain and load-bearing: **FLUX.2's autoencoder differs materially from FLUX.1's.**

### 2.2 ✗ "Family membership predicts cross-generator transfer"

**Contradicted directly by §1.3.** Flux Dev is a latent diffusion model and sits at **21%** — among the hardest generators in the benchmark. A detector trained on other latent diffusion output does not transfer to it.

**Revised mechanism, replacing the family story:**

> Detectors read the global VAE decode — which is why they collapse when it is removed (§1.1) and why reconstruction-based methods work. But *sharing "a VAE" does not buy transfer*, because VAE configurations diverge across model families (§2.1) and artifact strength shrinks with each generation. **Recency is an axis independent of family.**

Practical consequence for 0002 §8.5: a held-out split cannot be designed by grouping generators into families and taking one representative each. Family is not the right unit. See §4.1.

### 2.3 ✗ "Train-on-SD → test-on-FLUX transfers surprisingly well"

**Unsupported.** No source located. [FlowGuard](https://arxiv.org/html/2604.07879) points the other way: cross-VAE transfer requires lightweight adaptation layers or fine-tuning on target-model latents. Struck from the project's reasoning.

### 2.4 ~ "img2img output inherits the source camera's statistics, making detection harder"

**Downgraded to `[unverified]`.** Mechanistically plausible; no direct source found. The *inpainting* form of the claim is well supported by INP-X (§1.1) and should be used in its place. General noise-inconsistency forensics literature exists but does not establish this specific claim.

---

## 3. Reclassified

### 3.1 gpt-image-1 is autoregressive / token-based — `[unverified]` → `[reported by vendor, not independently confirmed]`

[OpenAI](https://openai.com/index/image-generation-api/) describes it as natively multimodal, generating images by predicting the next visual token within the same transformer that handles text, rather than via a separate diffusion system. This is a stronger footing than 0002 §6.6's original `[unverified]` tag, but it remains a vendor description with no architecture paper or independent replication. Do not treat the token-grid artifact profile as established.

### 3.2 New risk surfaced: frozen backbones are an adversarial liability

[Backbone is All You Need (2605.13381)](https://arxiv.org/abs/2605.13381): knowledge of the detector's **frozen ViT backbone architecture alone** is sufficient to craft gray-box adversarial examples (Surrogate Iterative Adversarial Attack), achieving attack success rates approaching white-box performance — even under limited training data or complete training misalignment.

This is *not* about pretraining contamination, which is what the title suggested. It is a security property: choosing a public frozen backbone makes the detector's feature space public. It does not affect the 0002 §9 gating experiment, but it belongs in 0002 §10 as an accepted risk.

---

## 4. What this changes

### 4.1 Held-out eval split design (amends 0002 §8.5)

The intuition "pick one generator per architecture family" is invalid per §2.2. **Recency and post-processing pipeline matter independently of architecture.** In particular, FLUX must not be excluded from a held-out set on the grounds that it is "the same family as SD" — it is among the hardest targets measured.

Note also that **GPT Image 2 and Nano Banana Pro are products, not raw models**: what reaches us is decoder output plus an undisclosed post-pipeline (sharpening, grain, watermarking, re-encoding). Their architectures are not public. Any mechanistic claim about them stays `[unverified]`.

### 4.2 `edited` is a structurally different problem, not a third class

INP-X (§1.1) puts a number on what 0002 §8.3 deferred on other grounds: detectors trained on global synthesis drop to ~55% on localized manipulation, and 75.3% is the ceiling even when trained for it directly. Combined with the measured corpus fingerprint ([`../investigations/2026-07-29-casia-authentic-probe.md`](../investigations/2026-07-29-casia-authentic-probe.md)), the case for deferring `edited` is now mechanistic rather than purely logistical: it is a **localization** problem, and a global image-level classifier is close to the wrong tool for it.

### 4.3 Coverage speed, not architecture, is the lever

§1.3 + §1.4 together: the best-performing detector in an independent benchmark is the one trained on the most generators, and generator count improves transfer monotonically to ~1,000 models. Architecture selection is not where the remaining headroom is.

---

## 5. Still open

- The FLUX.2 autoencoder compression-ratio conflict (§2.1).
- Whether our specific FFT/SRM branches learned a *decode* artifact or something else — §1.1 establishes the mechanism for the field, not for our checkpoint. Testable against our own data.
- Whether the 18–30% figures reflect lost separability or misplaced thresholds (§1.3 caveat). Decomposing this would materially change how pessimistic the outlook should be.
- No source was located for a positive definition of the `real` class that survives §1.8. If the one-class direction is ever pursued, this needs its own pass.

## Sources

- [Aligned Datasets Improve Detection of Latent Diffusion-Generated Images (ICLR 2025)](https://arxiv.org/abs/2410.11835)
- [AEROBLADE: Training-Free Detection of Latent Diffusion Images (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.pdf)
- [AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange](https://arxiv.org/html/2602.00192)
- [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors (CVPR 2025)](https://arxiv.org/abs/2411.04125)
- [How well are open sourced AI-generated image detection models out-of-the-box: A comprehensive benchmark study](https://arxiv.org/html/2602.07814v1)
- [Breaking Latent Prior Bias in Detectors for Generalizable AIGC Image Detection](https://arxiv.org/pdf/2506.00874)
- [GenDet: Towards Good Generalizations for AI-Generated Image Detection](https://arxiv.org/html/2312.08880)
- [Scalable Diffusion Models with Transformers (DiT, ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
- [CNN-generated images are surprisingly easy to spot… for now (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf)
- [Deconvolution and Checkerboard Artifacts (Odena et al.)](https://www.semanticscholar.org/paper/Deconvolution-and-Checkerboard-Artifacts-Odena-Dumoulin/d9dab7574d56ae81efe6c90c213c6509b36cf950)
- [DMimageDetection — Corvi et al., ICASSP 2023](https://github.com/grip-unina/DMimageDetection)
- [notes_on_sd_vae](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9)
- [FLUX.2 Autoencoder (DeepWiki)](https://deepwiki.com/black-forest-labs/flux2/3.4-autoencoder-(vae))
- [FlowGuard: Lightweight In-Generation Safety Detection via Linear Latent Decoding](https://arxiv.org/html/2604.07879)
- [Diagonal Artifacts in Samsung Images: PRNU Challenges and Solutions](https://arxiv.org/pdf/2510.09509)
- [Apple's Synthetic Defocus Noise Pattern: Characterization and Forensic Applications](https://arxiv.org/pdf/2505.07380)
- [Addressing Image Authenticity When Cameras Use Generative AI](https://arxiv.org/pdf/2604.21879)
- [Backbone is All You Need: Assessing Vulnerabilities of Frozen Foundation Models in Synthetic Image Forensics](https://arxiv.org/abs/2605.13381)
- [Introducing our latest image generation model in the API (OpenAI)](https://openai.com/index/image-generation-api/)
