# Architecture Decisions: 3-Class Forgery Classifier (Real / Edited / Deepfake)

Decisions made for the fine-tuning backbone and overall model architecture, with reasoning and discarded alternatives. Builds on `deepfake_detection_research.md`; this doc records the choices made from that research plus reasoning specific to this project's constraints.

## Requirements driving these decisions

1. **Testing ground:** evaluation will be on AI-generated images from latest-generation models (ChatGPT/gpt-image-1, Gemini/Imagen, and similar — diffusion/autoregressive-decoder based, full-image generators). This is a different artifact family from GAN-based face-swap datasets (FaceForensics++, Celeb-DF, DFDC) that dominate the published research — those leave checkerboard/upsampling artifacts specific to transposed convolutions; diffusion output does not share that specific tell but does deviate from natural camera-sensor statistics in its own way. Backbone/signal choices favor generalizable artifact types over GAN-specific ones.
2. **Explainability:** the model must not just output a class — it needs to surface *why* (e.g., spatial artifacts vs. spectral artifacts vs. noise-residual artifacts), not act as an opaque single-logit classifier.
3. **Hard constraints:** 1.5 hours total training time, Google Colab Pro (300 compute units), GPU assignment not guaranteed — could draw T4, L4, or A100. Final build accessed an **A100 40GB**.

---

## Overview: 3-branch fusion architecture

```
RGB image (~380x380)
   |-- EfficientNet-B4 (full fine-tune)        --> spatial embedding
   |-- FFT-magnitude (log-scaled)  -> small CNN --> spectral embedding
   |-- SRM high-pass residual      -> small CNN --> noise-residual embedding
                                          |
                          concat -> gating/attention MLP
                                          |
                  +------------------------+------------------------+
             3-way softmax                          per-branch contribution weights
        (real / edited / deepfake)                 (spatial / spectral / noise-residual
                                                      -- the explainability signal)
```

Grad-CAM is applied to the spatial branch for a visual "where" heatmap, complementing the gate's "which evidence type" weighting.

---

## Why 3 signal branches, not 1, 2, or more

| Option | Why discarded |
|---|---|
| **1 branch (RGB-only CNN)** | Risks learning which *dataset* an image came from (resolution, JPEG quality, framing) rather than which *manipulation family* produced it — flagged as Risk 2 in the research doc's §6. Also gives one opaque logit with no way to decompose "why," failing the explainability requirement outright. |
| **2 branches (spatial + spectral only)** | Spectral signal is strong for real-vs-deepfake but weak for edited-vs-real — a spliced image still has a normal camera-sensor spectral falloff since the edit doesn't disturb global frequency statistics much. Without a noise-residual branch, the edited class has no branch built for its actual failure mode (local boundary discontinuity). |
| **4+ branches (add CLIP-ViT semantic branch, or an "identity" branch)** | Diminishing returns at this compute/time budget, and — more importantly for explainability — more branches makes the gate's output *harder to read*. A clean 3-way split mapping onto three manipulation types (structural / generative-frequency / boundary-splice) is more useful than a 4-5 way split with marginal per-branch signal. |
| **2 full-size CNN backbones in parallel (e.g. Xception + EfficientNet + Transformer hybrid)** | Both backbones extract the *same kind* of evidence (spatial texture) via different architectures — redundant reasoning, not diverse reasoning. Doubles compute for two opinions on the same question instead of spending that budget on genuinely different evidence types (frequency-domain, noise-domain). |

---

## Branch 1: Spatial — EfficientNet-B4

**Mechanism:** hierarchical texture/edge/shape features via convolutional receptive fields; squeeze-excite attention in MBConv blocks weights the most informative channels.

**How it drives the class decision:**
- Real → consistent natural texture, geometry, lighting throughout
- Edited → local blending seams, shadow/geometry mismatches at splice boundaries, unnatural smoothness from retouching tools
- Deepfake → waxy/over-smooth skin texture, anatomical implausibility, micro-texture inconsistency across face regions from GAN/diffusion blending (training source is now diffusion-specific — see Dataset section — so this framing leans toward the diffusion-artifact end of that spectrum)

Every class leaves *some* trace here, making this the generalist branch — hence it gets the largest capacity (full EfficientNet-B4, not a lightweight CNN like the other two branches).

**Why EfficientNet-B4, and what was discarded:**
- **Xception** — same reasoning category (spatial texture). Picked EfficientNet-B4 first because research showed similar-or-better accuracy with fewer parameters than Xception (EfficientNet-B4: ~19M params, 82.6–83% top-1 on ImageNet vs. Xception: ~22.9M params, 79.0% top-1 — Tan & Le 2019). Further research showed fewer parameters doesn't mean less compute or faster training — a controlled benchmark (DeepfakeBench, NeurIPS 2023) found Xception and EfficientNet-B4 perform about the same on forgery detection, with neither beating the other. EfficientNet-B4 stays the pick to avoid running two backbones on the same evidence type, but on parameter-efficiency grounds only, not a proven accuracy or speed edge for this task.
- **ResNet-50** — less parameter-efficient than EfficientNet at comparable accuracy; no distinct feature type over EfficientNet. Cut earlier in the process as "not worth a branch."
- **CLIP-ViT (frozen or LayerNorm-tuned)** — strong at global semantic plausibility and cross-dataset generalization (research doc's LNCLIP-DF result, §2), but weaker at the localized pixel-level texture forensics Grad-CAM cleanly visualizes, and heavier compute than needed right now. Held in reserve as a v2 branch if cross-generator generalization proves to be the bottleneck once real eval numbers exist.
- **DINOv2 (self-supervised ViT)** — genuinely strong general-purpose features, and its patch tokens retain more local/textural detail than CLIP's globally-pooled semantic embedding (no text-alignment bias pulling it toward global semantics), making it arguably a *better* forensics fit than CLIP in theory. Still discarded for v1 because: (a) attention cost scales O(n^2) with token count, and DINOv2's strong results depend on evaluating near its 336-518px training resolution, which is markedly more expensive per image than a CNN of similar parameter count; (b) full fine-tuning a ViT on a small, time-boxed dataset is a worse bet than fine-tuning a CNN — CNNs carry built-in locality/translation-equivariance inductive bias that ViTs must learn from data, and one shot inside 1.5 hours is a bad setting for that; (c) same explainability mismatch as CLIP — Grad-CAM's spatial-feature-map assumption doesn't transfer cleanly to a ViT, requiring attention-rollout or similar, which is less mature and doesn't reduce as cleanly to the gate's "named artifact, X% contribution" output; (d) it would occupy the same conceptual slot as the existing spatial branch rather than add a new evidence type. **Reserved as the v2 generalization branch (superseding CLIP in that role) if a real generalization gap shows up in evaluation.**

---

## Branch 2: Spectral — FFT-magnitude (log-scaled)

**Mechanism:** 2D FFT magnitude spectrum reveals periodic structure invisible in RGB. GAN upsampling (transposed convs / pixel-shuffle) leaves checkerboard-periodicity spikes; diffusion models don't share that specific tell, but their iterative denoising + VAE decoder still produce a spectral falloff that deviates from the ~1/f natural-image power law real camera captures follow.

**How it drives the class decision:**
- Real → expected natural spectral falloff, no anomalous peaks
- Deepfake/AI-generated → anomalous deviation from natural falloff — primary discriminator for real-vs-deepfake, and the branch most relevant to the actual test target (spectral deviation-from-natural is shared across GAN *and* diffusion outputs even though their specific artifact signatures differ)
- Edited → base image is still a real camera capture (possibly recompressed), so global spectral signature stays close to "real" — this branch stays relatively quiet for edited images, which prevents edited-vs-deepfake confusion

**Why FFT-magnitude, and what was discarded:**
- **DCT** — well-suited to JPEG-block compression-artifact analysis (aligned to 8x8 blocks), which is more of a recompression/edit tell than a generation tell. Assigned conceptually to the noise/edited branch's job instead; using it here would overlap with SRM and miss the global periodic-artifact signal FFT provides.
- **Raw complex FFT (magnitude + phase)** — phase carries positional information that's harder for a small CNN to learn from limited data/time. Log-magnitude alone is the standard, cheaper, numerically stable representation used in the literature the research doc's frequency-branch citation references (source [10]).
- **A large pretrained spectral-forensics model** — none available off-the-shelf for this exact purpose at this budget; a small CNN trained from scratch on FFT-magnitude is the proven-cheap approach.

---

## Branch 3: Noise-residual — SRM (Steganalysis Rich Model)

**Mechanism:** fixed, non-learned high-pass filter kernels from forensics literature that suppress scene content and amplify residual noise — sensor noise, compression quantization noise, and boundary discontinuities.

**How it drives the class decision:**
- Real → consistent noise/compression statistics across the whole frame (one capture, one compression pass)
- Edited (splice/copy-move/retouch) → the actual tell — a spliced region carries noise statistics from a *different* source image or compression history than the rest of the frame, producing a residual discontinuity at the edit boundary. Primary discriminator for edited-vs-everything-else.
- Deepfake → the whole region came from one generative process, so there's no boundary discontinuity the way splicing has — this branch stays comparatively quiet for deepfakes, preventing deepfake-vs-edited confusion

**Why SRM, and what was discarded:**
- **ELA (Error Level Analysis via JPEG recompression)** — cheap and classic, but depends on an original JPEG compression history to exploit. Much AI-generator output is exported as PNG with no meaningful compression artifact. Discarded specifically because of requirement 1 (would degrade exactly on the stated test set).
- **Learned reconstruction-error branch (autoencoder residual)** — requires training/running a separate generative reference model per image; real added compute for a signal SRM's fixed filters already approximate for near-zero cost.
- **Denoiser-residual (run a pretrained denoiser, subtract)** — same problem: an extra full forward pass per image through another network, not justified over SRM's fixed filters at this time budget.

SRM wins on being the cheapest of the three branches (fixed filters, zero training cost for the filter bank itself) while covering the class (edited) expected to have the smallest sample count after face-filtering (per research doc §4).

---

## Fusion mechanism: gated concatenation, not plain MLP or ensembling

**Chosen:** concatenate branch embeddings -> small gating/attention MLP -> 3-way softmax **and** per-branch contribution weights as a first-class output.

| Option | Why discarded |
|---|---|
| Concatenate -> plain MLP, no explicit gate output | Classifies fine, but the "how much did each branch contribute" signal isn't a first-class output — would need post-hoc ablation/occlusion to recover it, defeating the point of this architecture. |
| Late-fusion ensembling (3 independent classifiers, average softmax) | Cleaner per-branch separation, but throws away cross-branch interaction (a spectral anomaly co-occurring with a spatial anomaly can be stronger joint evidence than either alone). Also triples classifier-head complexity and requires calibrating three separate decision boundaries. |
| Cross-attention transformer fusion | More expressive, but attention weights spread across heads/layers don't collapse into a clean, reportable "spectral_artifacts: 62%" figure the way a 3-way softmax gate does. More complexity than a 1.5-hour, 3-branch input needs. |

The gate is the cheapest fusion mechanism that makes "why" a direct model output instead of something reconstructed after the fact.

---

## Compute/infrastructure decisions

**Environment:** Google Colab Pro, 300 compute units. GPU assignment is availability-based (T4, L4, or A100) — plan is GPU-tier-adaptive, not fixed to one assumption.

| GPU tier | What it affords in 1.5 hrs | Config |
|---|---|---|
| T4 (worst case) | Tight | EfficientNet-B0 (not B4), partial fine-tune (freeze ~60-70% of blocks), single stacked 4-channel input (RGB+FFT) instead of separate branch networks, post-hoc attribution (Grad-CAM + integrated gradients on the frequency channel) instead of architectural branch separation |
| L4 | ~2x T4 throughput | EfficientNet-B0 full fine-tune, or step up to partial-fine-tune B2/B3; larger dataset or epoch count |
| **A100 40GB (actual draw)** | ~4-6x T4 throughput | **Full 3-branch architecture as described above**, EfficientNet-B4 full fine-tune at native ~380px resolution |

Compute units are not the binding constraint at any tier (a 1.5-hour A100 session costs roughly 17-22 of the 300 units) — wall-clock time and GPU draw availability are what actually gate scope.

**Time budget on A100 (1.5 hrs total):**

| Stage | Estimate |
|---|---|
| Setup + data load verification | ~10 min |
| Training (15-20 epochs, dataset up to ~40-60k images, batch ~64-96, AMP) | ~40-55 min |
| Validation + confusion matrix + macro-F1 | ~5 min |
| Grad-CAM + gate-weight explainability dump on val sample | ~10 min |
| Buffer | ~10-20 min |

**Loss:** class-weighted cross-entropy (edited class expected smallest after face-filtering, per research doc §4-5).

**Epochs:** 15-20 with early stopping on macro-F1; cosine LR schedule with brief warmup; AdamW.

---

## Dataset assembly (finalized, supersedes the FF++/Celeb-DF plan in the research doc §4)

One dataset per class, chosen for a tight prep window and for matching the actual test target (diffusion output, not GAN face-swap):

| Class | Source | Why |
|---|---|---|
| Deepfake | DALL-E 3 slice of COCO_AI/SynthBuster | Diffusion-generated, matching requirement 1's actual eval target (gpt-image-1/Imagen-class generators) — closes the gap FF++/Celeb-DF (GAN face-swap) left open. Confirm exact source URL/handle before running the download step. |
| Real | The pristine images paired 1:1 with that same DALL-E 3 slice | Reusing the paired original (not an unrelated curated set like FFHQ) avoids the model learning a dataset fingerprint instead of a manipulation cue — same reasoning as §4's "don't source real from a separate dataset." |
| Edited | CASIA v2.0 | Ships as ready image files (no link-rot risk, unlike PS-Battles); single dataset keeps prep inside the time budget. |

**Face detection is applied uniformly to all three classes, train and test alike** — confirmed the test set guarantees a face is present in every image, so this is a scope match, not just a shortcut-avoidance measure the way face-filtering CASIA alone would have been. Use MTCNN (`facenet_pytorch`, already in `.venv`) for detect+crop.

**Known accepted tradeoffs:**
- No face-swap/reenactment deepfake representation at all (FF++/Celeb-DF dropped entirely) — deliberate, since requirement 1 deprioritizes that artifact family for this project.
- Single generator (DALL-E 3 only) — no training signal for other diffusion decoders. Mitigate cheaply by holding out a couple of SynthBuster's other generator slices (Midjourney, SD) as eval-only, never trained on, to measure the cross-generator generalization gap the FFT/SRM branches are supposed to close.
- Face detectors can reject the most artifact-heavy generated faces (malformed features break landmark detection), which risk-filters out exactly the hardest, most informative examples. Check the DALL-E 3 slice's post-filter survival rate before committing to this pipeline — if too few images survive, a bigger or face-specific diffusion source is needed instead.

See `data_download.md` for the concrete download/preprocessing steps and `model_code.md` for the training implementation plan.
