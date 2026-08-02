# Dataset alignment — how to construct a real:fake distribution, and one falsified proposal

**Written:** 2026-08-02 (third pass of the day). Deadline Mon 2026-08-03.
**Status:** Current.
**Prompted by:** the question *"then how to even resolve the real:fake distribution?"* — i.e. how do you build two classes that differ **only** by synthesis, rather than by corpus, container, and resolution.
**Relates to:** [`2026-08-02-ssafe-primary-read.md`](2026-08-02-ssafe-primary-read.md) · [`../notes/2026-08-02-bottlenecks.md`](../notes/2026-08-02-bottlenecks.md) §2.4, §2.5 · [`2026-08-02-session-findings.md`](2026-08-02-session-findings.md) §1 (the (A)/(B)/(C) taxonomy this tests)

Evidence tags: `[confirmed]` = checked against the paper body this session · `[repo]` · `[unverified]`. **Nothing here is `[measured]`** — no number came from our data.

Extracted paper text cached at `…/scratchpad/` as `aeroblade.txt`, `drct.txt`, `aligned.txt`, `dda.txt` for quote-checking.

---

## 0. The question, and the answer

The real:fake distribution problem splits in two, and conflating them is why it looks unresolvable:

| | Question | Fixed by |
|---|---|---|
| **(a)** Training-time confound | Does the probe learn synthesis or corpus? | Alignment |
| **(b)** Test-time mismatch | Does the demo image resemble training data at all? | Resolution/domain sourcing |

They pull against each other — the §2.4 trap in `bottlenecks.md` is exactly (b)'s fix worsening (a).

**The answer is dataset alignment, and the literature has a specific name for doing it properly: aligning *both* sides.** A proposal made this session aligned only one side, and is falsified below. It is recorded in full so it is not re-proposed.

---

## 1. The falsified proposal — recorded so it is not repeated

**What was proposed this session.** Take real photographs, pass them through a Stable Diffusion VAE (encode → decode, **no diffusion sampling**, no editing), label the original `real` and the reconstruction `fake`. The pair is pixel-aligned: same content, resolution, source, compression history. The stated claim was that this *"forces the classifier onto the decode signature, since every other difference has been eliminated by construction"* — and that the signature would transfer to Gemini / GPT-Image, which decode through VQ tokenizers.

**Verdict: falsified.** It is a rediscovery of a published method whose own final experiment falsifies the premise (§4), and the mechanism claim is contradicted by direct measurement (§3).

**Where it came from.** It was reasoned out from AEROBLADE + INP-X, both already in the project bibliography, plus a misreading of DRCT. The reasoning was internally coherent and wrong, which is the interesting part — §7.

---

## 2. AEROBLADE — confirmed, but the headline number is matched-decoder `[confirmed]`

Ricker, Lukovnikov, Fischer, *AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error*, CVPR 2024.

- **0.992 mAP confirmed** — the mean over 7 test sets of the LPIPS₂ + Δ_Min row: SD1.1 0.979, SD1.5 0.978, SD2.1 0.994, KD2.1 0.999, MJ4 0.999, MJ5 0.997, MJ5.1 0.996.
- **Direction confirmed:** *"generated images have a consistently lower reconstruction error than real images."* Real photos reconstruct **worse**.
- **The caveat that matters, and which was omitted when this was cited earlier:** 0.992 is **Δ_Min over three autoencoders** — it assumes you hold the AE that generated the image. Cross-decoder cells collapse:

| Scoring AE | Target | mAP |
|---|---|---|
| Kandinsky 2.1 VQ-VAE | SD images | **0.543 / 0.552 / 0.623** |
| SD1 VAE | Kandinsky images | 0.878 |

The paper concedes it: *"to get the best results, access to the AE of the LDM which actually generated the image is required."*

- **Two mechanism details worth keeping.** Adding denoising steps to the reconstruction **hurts** (t=50, DIRE's setting, gives "a notable decrease in detection performance"). And the error is complexity-driven — for real images reconstruction error rises with patch complexity, for generated images it does not, so **low-complexity real images (logos, flat regions) are false-negative-prone.**

**Correction to record:** `session-findings.md` §1.2 cites "AEROBLADE reaches 0.992 mAP training-free" as evidence that the decode is a causal, shared target. The number is right; the *shared* reading is not. It is a matched-decoder result and does not support cross-generator transfer.

---

## 3. Reconstruction ≠ generation — the mechanism failure `[confirmed]`

The proposal assumed a VAE reconstruction sits where a generated image sits in decode-signature space. **DRCT measured this directly, and it does not.**

- **t-SNE (DRCT Fig. 2a):** *"the Real Rec samples cluster closely to the real samples, whereas the generated samples (via SDv1.4) are noticeably distant from the real samples."*
- **Fourier amplitude spectra (Fig. 6, 5000 samples/class):** *"SDv1.4, SDv1.4 Rec, and SDv2 exhibit a similar and distinctive pattern, significantly different from the amplitude spectrum of real images. Simultaneously, the amplitude spectrum of real reconstructed images shows a greater similarity to that of real images."*

In both feature and frequency space, **a reconstruction clusters with `real`; a generation does not.** A generated image's latent comes from diffusion; a reconstruction's latent comes from encoding a photo, and that difference dominates the decoder's own trace.

The evidence is stronger than it looks, because **DRCT's reconstructions are *more* generation-like than the proposal's** — DRCT does encode → add noise → 50-step DDIM denoise → decode. The proposal deletes the denoising, landing its fake class even closer to `real`.

**So "every other difference has been eliminated by construction" was false.** Two survive:

1. **Compression history.** A float tensor written to PNG does not carry the source JPEG's quantization, even at identical resolution and content.
2. **Latent origin.** Encoder-derived vs diffusion-derived latents are measurably different, per above.

---

## 4. The proposal is AlignedForensics (ICLR 2025), and it collapses outside the SD-VAE family `[confirmed]`

Sundara Rajan et al., *Aligned Datasets Improve Detection of Latent Diffusion-Generated Images*, ICLR 2025.

Their method is the proposal line for line: `ℱ = {φ_dec(φ_enc(x)) | x ∈ ℛ}`, *"without any denoising operation,"* reconstructions as the **sole** fake class, ResNet-50, BCE, 179,257 MS-COCO/LSUN images. Their rationale is the proposal's rationale verbatim: *"we force the fake detector to focus on the fingerprints of the VAE decoder. Since all kinds of generated images always pass through the decoder, they must share the same fingerprints."*

**Inside the SD-VAE family it is excellent.** Clean accuracy: SD 99.31, Midjourney 98.50, Kandinsky 99.92, Playground 94.85, PixArt-α 100, LCM 100, real 99.93. Beats Corvi by +36.98 / +52.09 on Playground under post-processing.

**Outside it, it collapses.**

| Benchmark | Result |
|---|---|
| **FLUX.1-dev** (still a continuous VAE — 16 latent channels vs 4) | **9.59% / 25.87%** |
| ForenSynths, 13 generators | **avg 53.9%** — last of nine methods compared |
| AIGCDetectionBenchmark | VQDM 72.1, DALL·E 2 52.0, ADM 51.6, GLIDE 55.6, ProGAN 50.7, StyleGAN 52.7, CycleGAN 49.5, BigGAN 51.2 — vs SD14 99.7, SD15 99.6, Wukong 99.6, MJ 96.2 |
| Synthbuster | DALL·E 2 50.2, DALL·E 3 48.9, Firefly 51.7, GLIDE 53.5 — vs SD1.3/1.4/2/SDXL all 97–99 |
| DDA-COCO | FLUX.1 **3.6**, SD3.5 55.4 |
| EvalGEN | FLUX 32.0, Infinity 74.0, NOVA 84.8, GoT 72.3, OmniGen 77.0 |

Their own words: *"models with vastly different architectures tend to produce very different kinds of artifacts."*

**The premise is stated in the paper and then falsified by the paper's own last experiment.** Changing the latent channel count from 4 to 16 — staying continuous, staying in the same architectural family — is enough to break it. Gemini and GPT-Image use discrete visual tokenizers, a strictly larger jump.

### 4.1 The one datum in the idea's favour

**Kandinsky 2.1 decodes through MoVQ — a VQ decoder, not the LDM convolutional one** (stated in both AEROBLADE §5.1 and AlignedForensics §5.3), and AlignedForensics scores **99.88–99.92%** on it.

So continuous-VAE → VQ-decoder transfer is not impossible in principle. But MoVQ is an LDM decoder with spatially-conditional normalization added on a near-identical conv stack, with close encoder/latent geometry. A cousin, not a different family — and FLUX shows a much smaller change already breaks it.

### 4.2 Documented failure mode

AlignedForensics reports it directly: *"If real images are originally saved in .webp format, we find that the reconstructions might not inherit those compression artifacts"* — the detector then learns **"webp artifacts ⇒ real,"** with their Fig. 4 showing the score degrading monotonically with webp compression level. This is §1(C) of `session-findings.md` reappearing inside a method designed to eliminate it.

---

## 5. DRCT, correctly described `[confirmed]`

Chen, Zeng, Yang, Yang, *DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images*, **ICML 2024**, PMLR v235:7621–7639.

Two corrections to how it was described earlier this session:

**5.1 — It is not a VAE round-trip.** It is encode → add noise → **50-step DDIM denoise** (via SD inpainting checkpoints: `runwayml/stable-diffusion-inpainting`, `stable-diffusion-2-inpainting`, `sdxl-1.0-inpainting-0.1`) → decode. The denoising pass is the part the falsified proposal removed, and it is the part that imprints the diffusion fingerprint.

**5.2 — Reconstruction-only is their weakest configuration, not their method.** Table 4 ablation (Conv-B, trained on DRCT-2M/SDv1.4, tested on GenImage) runs precisely the proposed experiment:

| Training fakes | GenImage avg ACC |
|---|---|
| real + SDv1.4 generated (baseline) | 68.98 |
| **real + reconstructions only** — the falsified proposal | **73.53** |
| + generated back in | 76.03 |
| + reconstructed-fakes | 76.99 |
| + contrastive loss (full DRCT) | **83.53** |

Reconstruction-only recovers roughly a third of the total gain, with the same family-bound signature: SDv1.4 99.77, SDv1.5 99.61, Wukong 99.61, Midjourney 74.25 — but **ADM 51.73, GLIDE 51.54, VQDM 61.12, BigGAN 50.60.**

Headline numbers for the record: DRCT/UnivFD 87.95% avg on GenImage (+15% over F3Net); +7.1 / +10.04 avg ACC for Conv-B / UnivFD; DRCT/UnivFD (SDv2) 96.90% on DRCT-2M-Wild. Limitations concede GAN improvement is *"less marked."*

DRCT hedges the §4.2 failure mode with heavy augmentation — JPEG at random quality, blur, noise, rotation, grid dropout. **Any reconstruction-based data we build must replicate this or ship a JPEG detector.**

DRCT appears in AIGI-Holmes' training composition (Holmes-Set draws 45K from CNNDetection + GenImage + DRCT), which is how it entered this session.

---

## 6. DDA — dual alignment, and the actual answer to the question `[confirmed]`

*Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable*, arXiv:2505.14359. **New to the project.**

The diagnosis: single-sided alignment (AlignedForensics) leaves the compression-history asymmetry of §3, and what the detector reads off a reconstruction is a **high-frequency asymmetry, not a generator artifact**:

> *"Real images are often with relatively poor high-frequency information, which is due to JPEG compression removing high-frequency details"* — while reconstructions *"restore high-frequency components."*

Their Fig. 5: SAFE reaches ~93% on VAE-reconstructed images, and *"when we mask high-frequency information slightly, the detection rate drops dramatically"* — *"detectors exploit biased features."*

**The fix is cheap: JPEG-compress the reconstruction to match the real image's format during training (p = 0.5).** The effect is large, and largest exactly where we care:

| | before | after |
|---|---|---|
| FLUX on DDA-COCO | 3.6 | **50.2** |
| **Infinity** (bitwise-token autoregressive — closest published proxy for our target paradigm) on EvalGEN | 74.0 | **97.8** |

DDA also posts the best WildRF (80.1) and Chameleon (>70) figures of anything benchmarked in that comparison.

**This is the paper that is directly about the question asked.** It should be read at the body before any dataset construction is committed to.

---

## 7. The easy-task trap `[confirmed]`

Real-vs-reconstruction is trivially separable and produces a validation number that means nothing.

- AlignedForensics: 99.76–100% clean accuracy. DDA-COCO: 99.8 real / 99.2 on MSE-VAE reconstructions.
- **The damning one:** SAFE scores **50.0–54.7% (chance) on every one of the 13 non-reconstruction DRCT-2M subsets**, and **98.2 / 98.5 / 97.3** on the three SDv1-DR / SDv2-DR / SDXL-DR reconstruction subsets.

A detector that has learned nothing generalizable can still ace reconstruction detection. This is `0002` §5.2's failure mode with a flattering number attached, and it is exactly the outcome the project has been trying to avoid.

---

## 8. INP-X, re-read against this `[confirmed]`

INP-X supports "the decode is what detectors read" while making the picture *worse*, not better. Detectors *"overrely on global artifacts,"* because *"VAE-based reconstruction induces a subtle but pervasive spectral shift across the entire image, including unedited regions,"* attributed to high-frequency attenuation. Restoring original pixels outside the edited region drops SOTA and commercial detectors **91% → 55%**.

**Note the apparent contradiction with §6** — INP-X says reconstruction *attenuates* high frequencies; DDA says reconstructions look high-frequency-*rich* relative to JPEG'd reals. Both are consistent once stated properly: the "decode signature" is a **low-level spectral offset relative to whatever the real-image pipeline happened to be.** It is a property of the *dataset's* processing asymmetry as much as of the decoder, and it is destroyed or inverted by re-encoding.

That is the deepest version of `session-findings.md` §1's (B)-vs-(C) distinction: **(B) is not cleanly separable from (C) by construction.** The decode trace and the container trace live in the same frequency band.

---

## 9. What survives

The construction is not worthless — it is mis-scoped. Two honest framings:

1. **As an SD-lineage detector.** Strong, cheap, defensible, state-of-the-art-adjacent: 99%+ on SD / MJ / Kandinsky / PixArt / LCM. The slide sentence is *"detects Stable-Diffusion-lineage images,"* and it would be true.
2. **As an auxiliary hard-negative source** alongside real generated images plus contrastive loss — DRCT's actual configuration. Worth ~5–15 points (§5.2).

**What does not survive:** reconstructions as the sole fake class, claimed to transfer to Gemini / GPT-Image. No source reports that transfer; every adjacent measurement points the other way.

**If the cross-generator claim is kept**, DDA's JPEG-matching (§6) is the cheapest fix in the literature — and even then, do not promise Gemini or GPT-Image without testing on them.

---

## 10. Corrections to this session's own reasoning

Recorded because the reasoning was internally coherent and wrong, which is the reusable lesson.

| # | Claim made this session | Status |
|---|---|---|
| 10.1 | "Reconstruction pairing eliminates every difference except the decode" | ✗ **False.** Compression history and latent origin both survive (§3) |
| 10.2 | "AEROBLADE's 0.992 mAP shows the decode is a shared, causal target" | ✗ **Misused.** Matched-decoder number; cross-decoder is 0.543–0.623 (§2) |
| 10.3 | "DRCT is a VAE round-trip used as reconstruction pairs" | ✗ **Wrong.** encode → noise → 50-step DDIM → decode (§5.1) |
| 10.4 | "DRCT is precedent for reconstruction-based training data" | ✗ **Inverted.** It is their weakest ablation, 73.53 vs 83.53 (§5.2) |
| 10.5 | "This construction resolves the §2.4 pairing trap and the resolution band at once" | ~ **Partially.** It does align the pair, but §4.2's webp finding shows single-sided alignment reintroduces a container shortcut of its own |
| 10.6 | "It gives a control SynthID cannot pass" | ✓ **Stands.** Reconstruction rewrites every pixel. Still a useful control, independent of the training proposal |

**Methodological note.** This is the third instance today of the pattern `calibration-and-thresholds.md` §6 named — *a confirmed premise laundering an unconfirmed inference*. Here the confirmed premises were AEROBLADE's mechanism and INP-X's causal test; the inference — *therefore training on reconstructions transfers across decoder families* — travelled at the same confidence and is false. The check that caught it was reading the papers that had already run the experiment.

---

## Sources

Read at the paper body this session:

- [*AEROBLADE*, CVPR 2024, arXiv:2401.17879](https://arxiv.org/abs/2401.17879) · [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html) — §2
- [*DRCT*, ICML 2024, PMLR v235:7621–7639](https://proceedings.mlr.press/v235/chen24ay.html) · [code](https://github.com/beibuwandeluori/DRCT) — §3, §5
- [*Aligned Datasets Improve Detection of Latent Diffusion-Generated Images*, ICLR 2025, arXiv:2410.11835](https://arxiv.org/abs/2410.11835) — §4
- [*Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable*, arXiv:2505.14359](https://arxiv.org/abs/2505.14359) — §6, §7, and the re-benchmarks in §4
- [*INP-X*, arXiv:2602.00192](https://arxiv.org/abs/2602.00192) · [code](https://github.com/emirhanbilgic/INP-X) — §8

Referenced, not re-read here:

- [*AIGI-Holmes*, ICCV 2025](https://arxiv.org/html/2507.02664v1) — Holmes-Set composition, §5
