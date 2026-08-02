# Plan A verification — does "match the test distribution" prove the hypothesis?

**Written:** 2026-08-02. Deadline Mon 2026-08-03.
**Continues:** [`2026-08-02-three-candidate-plans.md`](2026-08-02-three-candidate-plans.md), which states the three plans but does not adjudicate between them.
**Question asked:** given COCO_AI and the architecture already specced in `0003`, with testing done on a live ChatGPT/Gemini image — is **Plan A** (§2 of the plans doc) legitimate, and would it end up proving the hypothesis?
**Status:** Verification pass. Decision input, not a decision. Nothing below was implemented, and **no number here was measured on our data.**

Provenance follows the repo convention: `[confirmed]` = checked against a primary or near-primary source this session, URL given · `[repo]` = stated in an existing repo doc · `[unverified]` = single source, not independently checked.

---

## 1. Verdict

**Plan A is legitimate as an engineering choice and does not prove the hypothesis. As written it also maximizes the one confound its own control test cannot detect.**

The word "hypothesis" is doing double duty in this project, and Plan A relates to the two readings very differently:

| | The claim | Does Plan A test it? |
|---|---|---|
| **H-demo** | Given a live Gemini / GPT Image upload, the model classifies real vs. AI-generated correctly | **Yes** — by construction, since those generators are in training |
| **H-research** (`0002` §8.1) | Frozen foundation-model features + a linear probe **generalize across generators** | **No.** Plan A deletes the gap rather than closing it |

The plans doc says as much in its §2 "What it proves." The consequence is larger than it prices: `0003` §8 names the deliverable as *"state the failure boundary before it is tested"*, and `context-transfer.md` §5 already recorded that a live-demo goal spends it. `[repo]`

Plan A's "free addition" — hold COCO_AI out as a test set — recovers a generalization number in the **diffusion** direction. That is the direction where a good result is a bonus and a bad result costs nothing, which is why it is free. It is not evidence about the direction being demonstrated on Monday.

## 2. What was verified, and what it does to the plan

Every load-bearing claim in the plans doc's §1(C) and §2 held up. Three of the confirmations make Plan A **worse** rather than better.

### 2.1 The container shortcut is real and correctly priced `[confirmed]`

[Fake or JPEG? (arXiv:2403.17608)](https://arxiv.org/abs/2403.17608) confirms detectors on GenImage learn JPEG-compression and image-size bias rather than generation artifacts, and that removing those biases yields **more than 11 percentage points** of cross-generator improvement for ResNet50 and Swin-T. The plans doc's §1(C) figures are accurate.

### 2.2 SynthID is engineered to survive Plan A's own control test `[confirmed]`

[DeepMind's SynthID page](https://deepmind.google/models/synthid/) confirms the watermark is embedded **at generation time, inside pixel values**, and is designed to survive cropping, compression, resizing and filtering.

**This is the finding that changes the plan.** The plans doc §5 proposes one control: *re-save at JPEG q85, rescale 95%, re-score; if the score collapses, the probe is reading the delivery pipeline.* A probe keying on SynthID **passes that control cleanly**, because surviving exactly that transformation is the watermark's design goal. Plan A's stated safeguard is blind to Plan A's most likely confound. Augmentation (`0002` §8.4) and native patching are equally blind for the same reason.

### 2.3 The ChatGPT delivery pipeline is a second, independent container signal `[confirmed]`

[OpenAI developer-community threads](https://community.openai.com/t/dall-e-images-downloading-as-webp/611090) confirm the ChatGPT web UI serves **lossy WebP** while the API returns PNG, with the change reportedly tied to C2PA provenance marking. This cuts twice:

- **As a shortcut** if web-UI images are what training sees.
- **As a train/test mismatch** if they are not: Plan A's ~$25 budget implies API acquisition (PNG), while a founder generating live on Monday is in the web UI (WebP). The demo image would then differ from every training image in codec *and* compression history.

### 2.4 Generator count is the dominant lever, and Plan A sits at the bottom of the curve `[confirmed]`

[Community Forensics (CVPR 2025)](https://arxiv.org/abs/2411.04125) — 2.7M images from 4,803 generators — reports detection performance improving as generator count rises, *even when the added models share an architecture*, with diversity improving it further. Plan A's n=2 is the thinnest point on that curve. Survivable only because Plan A does not need unseen-generator transfer; fatal to any generalization claim made from it.

### 2.5 The honest prior for Monday `[confirmed]` / `[unverified]`

[DailyBench (arXiv:2607.24016)](https://arxiv.org/html/2607.24016) `[confirmed]` reports detectors at **91–96% balanced accuracy on GenImage falling to 60–76% on FakeBench** and 54–66% on ManipulationBench. That is the expectation to set for a modern-generator demo.

[SSAFE (arXiv:2606.08634)](https://arxiv.org/abs/2606.08634) `[unverified — 2026 preprint, no replication located]` checks out as described in the plans doc §6.4: frozen encoder + linear classifier, PE-Core-G14-448 giving the clearest real/fake separation, 10K training images sufficing against 288K/4M baselines. **But its actual mechanism is a representation-aware curation strategy that selects a compact set of *representative generators*.** It is evidence for generator-diversity curation, not for two commercial generators. Citing it in support of Plan A inverts its lesson.

## 3. Three problems Plan A has that the plans doc undercosts

**3.1 — The container↔label correlation is *perfect* here, not merely present.** Every fake from two commercial web pipelines, every real from a different source, and — unlike Plans B and C — no diffusion data in training to break the correlation. Gradient descent takes the cheapest available hypothesis (Geirhos, already cited in `0002` §7.1). `[repo]` Structurally this is the worst of the three configurations, and §2.2 above establishes that the intended detection method for it does not work.

**3.2 — The "~300 modern reals" is the unsolved half, and it is what actually costs the day.** The plans doc budgets API credits and generation time but never says where the reals come from. Both available answers are bad:

- **Stock sites** (Unsplash / Pexels) carry their own processing fingerprint — uniform sRGB export, Lightroom pipeline, professional composition — separable from AI output on style alone. This is `0002` §6.4's corpus-fingerprint failure, measured at 17.4% on CASIA (n=161). `[repo]`
- **Own phone photos** are the correct answer and cannot be collected at n=300 in half a day.

Plan A's half-day estimate does not survive this. The binding cost is the real class, not the $25.

**3.3 — Statistical power.** 600 images at `0003`'s split gives roughly 90 test images. At 90% accuracy that is about **±6 percentage points** at 95% confidence. Adequate for a live demo; not adequate for a number defended under questioning.

## 4. State of the existing code, relevant to any Plan A build

- `data/download.py:80-82` already re-saves **both** classes as JPEG q95, so the raw format difference is handled. Compression **history** is not: COCO reals were natively JPEG and double-compress, generator PNGs single-compress. Recorded in `context-transfer.md` §4 and `0002` §6.4. `[repo]` Any newly acquired Plan A data needs the same re-encode *plus* a fix for the history asymmetry.
- `data/download.py:68` still selects `["caption", "coco_image", "dalle_image"]` — the one line that discards five of COCO_AI's six generators. Unchanged since `context-transfer.md` Step 4 verified the dataset has six. `[repo]`
- `model/dataset.py` applies **no augmentation**. `0002` §8.4 called it mandatory; `0003` §6 dropped it; nothing has reinstated it. `[repo]`
- No `data_raw/` on disk in this working copy — COCO_AI is available, not yet materialized.

## 5. Recommendation

**Plan A as written: no. Plan A with three changes: yes.** The plans doc's §5 already notes the plans are not mutually exclusive; that is the way out.

1. **Run §5's ordering step 1 first.** Frozen CLIP + logistic regression on COCO_AI already in hand, scored on a handful of Gemini/GPT images. Zero API spend, half a day. It works → done. It fails → the data purchase is made knowing why. Spending the day on acquisition *before* running the free experiment is the one clearly wrong ordering.
2. **Do not abandon COCO_AI — mix it in.** Keep the SD 2.1 / SD3 / SD3.5 / SDXL columns (the surgical cut from `context-transfer.md` Step 6) alongside the transformer fakes. A heterogeneous fake class breaks the container↔label correlation that pure Plan A guarantees, and it costs nothing. **Row-level splitting is mandatory** — the same real image appears in all six generator columns.
3. **Augmentation on, at Wang et al.'s actual probabilistic setting** — σ ~ U[0,3], JPEG quality ~ U{30…100}, each applied at p=0.5 or 0.1, *not* to every image (plans doc §3 and §6.3). Prerequisite on every path.

### 5.1 Replace the control test

§2.2 establishes the plans doc's control is insufficient for Plan A. Two that discriminate:

- **Codec A/B.** Generate the same prompt via the Gemini/OpenAI API (PNG) and via the web UI (WebP/C2PA). If the two scores diverge materially, the probe is reading the container.
- **Re-render.** Score a Gemini image, then score a screenshot / re-render of it. A pixel-embedded watermark degrades under re-rendering differently from synthesis artifacts.

### 5.2 The sentence to keep off Monday's slide

Anything of the form *"generalizes."* Under Plan A the only honest label is **"accuracy on seen generators"** — the discipline `context-transfer.md` Step 3 and Step 5 already committed to. `[repo]`

---

## Sources

New to this doc (all retrieved and checked 2026-08-02):

- [*Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, arXiv:2403.17608](https://arxiv.org/abs/2403.17608) — §2.1
- [SynthID — Google DeepMind](https://deepmind.google/models/synthid/) — §2.2
- [OpenAI developer community: DALL·E images downloading as WebP](https://community.openai.com/t/dall-e-images-downloading-as-webp/611090) — §2.3

Re-checked against the plans doc's citations:

- [*Community Forensics*, CVPR 2025 / arXiv:2411.04125](https://arxiv.org/abs/2411.04125) — §2.4
- [*DailyBench*, arXiv:2607.24016](https://arxiv.org/html/2607.24016) — §2.5
- [*SSAFE*, arXiv:2606.08634](https://arxiv.org/abs/2606.08634) — §2.5
