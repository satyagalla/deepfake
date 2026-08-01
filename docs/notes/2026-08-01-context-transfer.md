# Context transfer — how the demo plan got to where it is

**Written:** 2026-08-01. Deadline Mon 2026-08-03.

A history, not a spec. Nothing below has been implemented — no file changed, no data downloaded, no image generated, no probe trained, no number produced. This records what we believed at each step, what changed it, and why.

Provenance: `[verified]` = checked directly this session, command shown. `[agent]` = reported by a verification subagent with a source, not independently re-checked. `[repo]` = stated in an existing repo doc.

---

## Where we started

From `docs/decisions/0003-frozen-probe-demo-build.md`: `[repo]`

- Frozen CLIP ViT-L/14 + logistic-regression probe, binary real vs AI-generated, whole images, face cropping dropped.
- §5: **"the binding constraint is data, not modelling"** — because `data/download.py` pairs COCO_AI's `coco_image` with `dalle_image`, i.e. DALL·E 3 only. Single-generator training was v1's identified failure cause, so ≥2 training generators was made a *requirement*, and sourcing them the first task ahead of any model code.
- §5 also reserved **GPT Image / Gemini / Nano Banana** as *"held out, never trained on"* — "this set exists to state the boundary, not to pass it."
- §8 made that the deliverable: state the failure boundary before it is tested.

So the opening question was: given DALL·E 3 is all we have, where does a second generator come from?

## Step 1 — Answer: one paid API + one local

Chosen from the options on the table. Immediate consequence noticed: if COCO_AI supplies DALL·E 3, then API + local makes **three** generators, not two, which makes leave-one-generator-out a real diagnostic rather than a single anecdote.

Plan at this point: train on DALL·E 3 + Gemini (API) + FLUX or SDXL (local A100); 3-fold leave-one-out; keep a small GPT Image set as the untrained boundary so §8's sentence survives.

Anti-shortcut measures proposed: prompt the local generator from COCO captions so content matches; re-encode everything identically since generator output is PNG and COCO reals are JPEG.

## Step 2 — "Is it bad to mix old and new generator signatures?"

Position taken: heterogeneity in the fake class is the *point*, not a compromise — a single direction that has to fit several decoders can't encode any one of them, so diversity acts as the regulariser. `0003` §4.4's choice of logistic over Ridge was already made for exactly this geometry.

Caveats flagged: equal per-generator counts (or the largest generator dominates), nuisance variables that correlate with generator, and that mixing doesn't buy family-wide transfer for free — `claim-verification` §2.2 has Flux Dev at 21% despite being latent diffusion. `[repo]`

## Step 3 — "Is it bad that the model trains on the test set's signature?"

Clarified to mean *same generator*, not same image. Two things were being conflated:

- **Leakage** = same images across train and eval. Always invalid.
- **Training on the target generator** = a representative training set. Not cheating.

Position: the data goes in the model, the discipline goes in the reporting. What you can't do is train on Gemini and then quote a Gemini number as evidence of generalization. And the reason to hold *anything* out isn't fairness — it's that a held-out generator is your only estimate of what happens on the generator that ships next.

## Step 4 — First verification agent, and the premise collapsing

Ran an agent against the plan. It reported that COCO_AI is not DALL·E-3-only.

Verified independently, because it overturns an ADR: `[verified]`

```bash
curl -s "https://datasets-server.huggingface.co/info?dataset=NasrinImp%2FCOCO_AI"
```

```
COLUMNS: caption, coco_image, sd35_image, sd3_image,
         sd21_image, sdxl_image, dalle_image, midjourney_image
ROWS:    10,017
```

And `data/download.py:68`:

```python
ds = ds.select_columns(["caption", "coco_image", "dalle_image"])
```

**Six generators, five thrown away by one line.** All from the same COCO caption, all paired 1:1 with the same real image.

`0003` §5's "binding constraint is data" is a fact about our download code, not about the dataset. The paid API run and the local A100 run both existed to fix a shortage that wasn't there.

The agent also surfaced things the plan had missed:

- All six fakes in a row share the same caption **and the same real image** — split naively and identical real pixels land on both sides. That is genuine leakage, unlike the thing in Step 3.
- Identical re-encoding doesn't fix **compression history**: COCO reals are natively JPEG, so a q95 re-save double-compresses them while generator PNGs are single-compressed. `0002` §6.4 had already recorded this fix as inadequate. `[repo]`
- "Same resize path" doesn't fix resolution. Every fake is square, no real is.
- `0002` §8.4 called JPEG+blur augmentation mandatory; `0003` §6 dropped it, and nobody noticed. `[repo]`
- Every anti-shortcut measure targeted the fake side; the **real** class is 2014-era COCO web JPEGs while any live upload is a modern phone photo.

## Step 5 — The goal was restated, and it inverted the holdout

Stated: the founder generates an image from Gemini or ChatGPT live, hands it over, and the model must classify it correctly among real images. That is the acceptance test.

This reverses Step 1's structure. Holding Gemini out optimises `0003` §8's research sentence at the cost of the thing being judged. Under this goal Gemini and GPT Image are primary training targets — legitimate per Step 3, provided the resulting number is labelled "accuracy on seen generators" and never "generalization."

Cost noted: it spends §8's differentiator. Partial recovery available by holding out some *other* generator instead, so the boundary sentence still has a subject.

## Step 6 — Resolution enters, and one wrong turn

Two problems raised: COCO_AI's images are much lower resolution than Gemini/GPT output, and Gemini fakes won't come 1:1 matched to reals the way COCO_AI's do.

I first answered about the wrong dataset — the actual Synthbuster/Zenodo release rather than COCO_AI. Corrected: the dataset in play is COCO_AI, and `coco_image` is the real class.

Per-column dimensions, verified: `[verified]`

```
coco_image  480×640      sd21   768×768
dalle       270×270      sd3    1024×1024
midjourney  436×436      sd35   1024×1024
                         sdxl   1024×1024
```

Observation from this: DALL·E 3 and Midjourney don't natively emit 270² and 436², so those two columns are downsampled — and they're also the entire source of the resolution spread. Proposed a **surgical cut**: drop those two, keep SD 2.1 / SD3 / SD3.5 / SDXL at 768–1024, which is comparable to Gemini/GPT.

Also proposed, to handle the remaining mismatch: native-resolution 224 patches instead of resizing, plus letting the real class span a wider resolution range than the fake class.

Face crops vs whole images was raised here and settled toward whole images — an uploaded image may contain no face at all, artifacts aren't face-specific, and cropping reintroduces the exact MTCNN crop-resize upscale confound from `2026-07-26-upscale-artifact.md`. `[repo]`

## Step 7 — Counter-proposal: drop COCO_AI entirely

Argument: v1's failure was resolution and upscale factor driven by input-size constraints, so data with that property should go, even though it looks valuable.

## Step 8 — Second verification agent

Three things came back that changed the picture.

**Native patching works with frozen CLIP.** I had expected this to fail — CLIP trains on whole resized images, so a native-res crop of a 1024² image is a texture close-up and semantically out-of-distribution. Wrong. TextureCrop (arXiv 2407.15500) tests exactly this on UnivFD, which *is* frozen CLIP ViT-L/14 + linear probe: +4.75% balanced accuracy over center-crop, and crop-vs-resize is +12.1% BA overall. Conditions I'd missed: it uses the top-10 crops by histogram entropy and averages scores, not random crops. `[agent]`

**"Real spans wider than fake" is unsound.** It doesn't decorrelate resolution — it makes it one-way predictive: anything outside the fake range is certainly real, and a probe can learn that half-rule. The literature cites this exact shape as a defect. Since patching removes resolution from the input entirely, the argument is moot either way. `[agent]`

**The container-detector risk, which neither of us was guarding against.** Train on Gemini/GPT web-UI images and test on one, and the cheapest hypothesis available to the probe is *"this file came out of Google's or OpenAI's delivery pipeline"* rather than *"this image was synthesized"* — because every Gemini image carries a SynthID watermark designed to survive compression, resizing and cropping; the ChatGPT web UI serves lossy WebP while the API returns PNG; and the canvas is a fixed 1024 family. Both hypotheses pass the live demo identically. Every safeguard discussed is blind to all three: SynthID survives cropping by design, WebP artifacts survive patching, metadata-only junk probes never see pixels. Same class of error as `0002` §6.4's CASIA corpus fingerprint. `[agent]`

Smaller corrections: cutting `dalle`/`midjourney` is right but *not* because downsampling destroys the signal — it suppresses rather than destroys it. The real reasons are that 270² fake vs 480×640 real makes resolution a perfect label predictor, and that residual high-frequency structure at 270² is the *downsampler's* kernel, a COCO_AI corpus fingerprint rather than a DALL·E one. Also: Nano Banana Pro generates natively at 2K and 4K, with 4K the default marketing setting — so a live generation could sit 4× beyond any training resolution we own. `[agent]`

---

## Where we are now

**Still deciding the data composition and the preprocessing.** Nothing has been built.

What's settled by evidence rather than preference:

- COCO_AI has six generators; `0003` §5's premise is factually false. `[verified]`
- The paid-API and local-generation runs, *as a fix for a data shortage*, are unnecessary. `[verified]`
- Row-level splitting is mandatory, not optional — the same real image appears across all six generator columns. `[agent]`
- Native patching is viable on frozen CLIP. `[agent]`
- "Real spans wider" is not a valid decorrelation strategy. `[agent]`

What's genuinely open:

- **Data.** Surgical cut (drop `dalle`/`midjourney`, keep the four SD columns, add modern reals) versus dropping COCO_AI entirely and building only from Gemini/GPT + own reals. The second agent's verdict was that the surgical cut is better *conditional on* either native patching or a uniform re-encode pipeline actually landing — because those are what make it legitimate to mix 480×640 COCO, 1024² SD, and 12MP phone photos. If neither ships, the surgical cut hides a resolution shortcut behind four generators and dropping COCO_AI is safer.
- **Preprocessing.** Native-resolution entropy-selected patches versus whole-image resize with uniform re-encoding. Patches are better-evidenced; resize is deadline-proof and makes resolution constant rather than decorrelated.
- **How much of `0003` §8's boundary positioning to trade away**, and whether to buy part of it back by holding out a generator that isn't Gemini.

## The immediate next step

The container-detector control, because it's cheap and it tells you whether the whole direction is sound: take one Gemini image the model calls fake, re-save it JPEG q85, rescale 95%, re-score. If the score collapses, the probe is reading delivery-pipeline artifacts rather than synthesis, and augmentation — `0002` §8.4, which `0003` §6 dropped — becomes the first thing to fix rather than the fifth.

It needs a probe to exist first, which needs the data question above answered. That's the decision currently blocking everything downstream.
