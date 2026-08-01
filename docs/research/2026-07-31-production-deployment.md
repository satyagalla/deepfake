# From Research Detector to Production System

**Status:** Current
**Date:** 2026-07-31
**Relates to:** [`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) (sources for every mechanistic claim referenced here), [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §8.4, §8.5, §10.

Everything built in this project so far optimizes a *research* objective: does the model separate classes on a held-out split. Production is a different problem, and carrying the research framing into it builds the wrong system. This doc records what changes, as a record of the 2026-07-30/31 working sessions.

Nothing here has been decided or implemented. It is decision *input* — the material a future `0003` would draw on. Mechanistic claims are sourced in the claim-verification doc and not re-cited.

---

## 1. The reframe

**Zero-day generalization is not winnable.** An independent 16-detector benchmark puts SOTA methods at 18–30% accuracy on current commercial generators (Firefly v4 18%, Flux Dev 21%, Midjourney v7 24%) — worse than chance, and confidently wrong in the `real` direction. That is not a gap to close with a better architecture. It is a property of the problem.

So the production question is not *"how do I build a detector that generalizes to unseen generators?"* It is:

> **How fast can I turn an unseen generator into a seen one, and how do I behave honestly in the window before I do?**

That is a data-pipeline and operations problem wearing a modeling problem's clothes. The standard failure is over-investing in the model.

## 2. The number that should drive the design

Papers evaluate on 50/50 balanced sets. Production traffic is not balanced.

Suppose fakes are 0.5% of screened images and the model ships at 95% TPR / 1% FPR — numbers most papers would call strong. Per 10,000 images:

- 50 fake → 47 caught
- 9,950 real → 99 false flags

**Precision ≈ 32%. Two of every three flags is a real image.**

Run this with the actual base rate and the actual FP:FN cost ratio *before* choosing anything else. It usually shows FPR must sit at 0.1% or below, at which point achievable TPR is far under anything published — and that reality should shape the product (review queues, confidence bands, abstention) rather than being discovered after launch.

Corollary for this project's metrics: **report TPR at low FPR, not accuracy or AUC.** AUC integrates over operating points that will never be used.

## 3. The four layers

The model is one of them.

**Layer 0 — Provenance and metadata.** Non-ML, and the highest-ROI component. C2PA manifests, EXIF/IPTC, encoder fingerprints (JPEG quantization tables identify the encoder; PNG chunk ordering; container structure). Cost: microseconds. FPR when it fires: near zero. The only layer that gets *stronger* over time as C2PA adoption spreads. Usually skipped because it isn't interesting research.

Its limitation defines its use: metadata is trivially stripped, so it is high-precision / low-recall. A hit is strong positive evidence; **absence is never exoneration.**

**Layer 1 — Known-generator classifier.** Trained on maximum generator diversity, refreshed continuously. Where the Community Forensics scaling result cashes out.

**Layer 2 — Abstention / OOD gate.** When an input is far from the training distribution, emit **"unknown," not a score.** See §5.

**Layer 3 — Human review queue + label feedback.** Abstentions and the low-confidence band route here; labels flow back into retraining.

## 4. Generator acquisition is the actual moat

The business asset is not the model. It is **having samples from a new generator within 48 hours of its release.**

Concretely: a monitored feed of model releases (open hubs, commercial API announcements), an automated sampling harness running a fixed prompt battery against every reachable generator on a schedule, and continuous ingestion of in-the-wild samples from user reports.

Two details decide whether this works:

- **The prompt battery must be content-matched to the production domain.** Screening ID documents while sampling landscapes is close to useless.
- **Sampled images must pass through the identical post-pipeline as the real class** — same resize, same crop, same re-encode. Otherwise "more generators" silently becomes "more encoders," which is the corpus-fingerprint failure this project already measured on `edited` ([`../investigations/2026-07-29-casia-authentic-probe.md`](../investigations/2026-07-29-casia-authentic-probe.md)).

Build this *before* training the production model. The model's ceiling is set by this pipeline.

### What "training on N generators" means

Literally: the fake class contains images sampled from N distinct generative models. Nothing more exotic — you only ever touch outputs, never weights.

Granularity is **checkpoint-level, not architecture-level.** Community Forensics reached 4,803 by downloading thousands of open text-to-image checkpoints — mostly fine-tunes and community variants of a few base models, plus a few dozen commercial ones. Two different SDXL fine-tunes count as two. That is precisely why the finding is stated as *performance improves with generator count even when generators are architecturally similar.*

Held-out generators are sampled the same way; the only difference is they never enter training.

For reference: this project currently trains on **one** generator (DALL·E 3, via COCO_AI).

## 5. Abstention

**Mechanism.** Output {real, fake, **unknown**}. "Unknown" means: this input is unlike my training distribution, my score isn't trustworthy, I decline.

**Why it is necessary here specifically.** The failure mode on a new generator is not random error — it is *confident* error biased toward `real`. The new generator's artifacts match nothing the classifier learned, so the input presents as "absence of known fake signatures," which reads as authentic. The model emits 0.03 with high confidence. **Calibration cannot rescue this** — the model is not uncertain, it is wrong and sure. Abstention converts a silent failure into a visible one.

**Implementation**, increasing sophistication:

- kNN distance from the input embedding to a precomputed bank of training embeddings. Cheap, surprisingly strong.
- Mahalanobis distance / density estimate on penultimate features. Also cheap, solid baseline. (Mahalanobis *is* a Gaussian density on the feature distribution — the natural connection point to this project's Gaussian thread.)
- Ensemble disagreement — reliable, costly at inference.
- Outlier exposure: hold out some generators and train the model to abstain on them, so abstention is learned rather than bolted on.

Start with kNN or Mahalanobis on frozen features. Most of the value, days of work — and it composes directly with `0002`'s frozen-embedding path, which produces the required embedding bank as a byproduct.

**Setting the threshold.** Plot the **risk–coverage curve**: coverage (fraction answered) against error rate among answered. Then pick the point from review capacity — if 2,000 images/day can be reviewed, set the threshold so abstentions land near 2,000/day. That is the honest way to choose it.

**Why it is the most important monitoring signal.** Every other production metric needs labels. Accuracy, precision, recall all require ground truth unavailable on live traffic. **Abstention rate needs zero labels** — it is computed purely from input distribution vs. training distribution. Unsupervised, real-time, free.

And its dynamics are right: when a new generator enters traffic, those images are OOD *by construction*, so abstention rate rises **before** accuracy visibly degrades and long before a customer complains. A leading indicator of a coverage gap, obtained without labels.

Operationally:

- Baseline it **per segment** (per customer, per content domain, per source platform). A global rate averages the signal away.
- Alert on deviation from the segment baseline, not an absolute threshold.
- On a spike: sample the abstained images, identify what's new, feed it to generator acquisition, retrain.
- Track abstention rate against *time since last retrain*. The slope gives retraining cadence empirically instead of by guess.

**Caution:** abstention also rises for benign reasons — a new customer with unusual content, a platform changing compression, seasonal shifts. Investigate spikes, don't auto-retrain on them. Segmentation disambiguates: a new generator spikes *many* customers at once; new customer content spikes one.

Abstention is also the honest product behavior. "We can't assess this image confidently" is defensible to a customer and to a court. "This image is authentic," when it wasn't, is neither.

## 6. Post-processing robustness

Post-processing is everything between generation/capture and the moment the detector sees the image. Not adversarial — the ordinary life of an image online:

> saved as JPEG at some quality → uploaded → platform resizes → platform re-encodes → metadata stripped → screenshotted → re-uploaded → repeat

Concretely: JPEG/WebP recompression (often repeated), rescaling with various kernels, aspect-ratio cropping, format conversion, metadata stripping, screenshotting, platform sharpening, blur, noise, brightness/contrast shifts.

**Why it is central to this task specifically:** nearly every low-level forensic signal lives in the **high frequencies** — spectral upsampling peaks, VAE decoder traces, noise residuals, demosaicing correlations. JPEG is *explicitly designed to discard high-frequency content*. Resizing low-passes. So the routine internet post-processing chain is functionally a targeted attack on exactly the signal the detector depends on, **with no attacker involved**. That is the mechanism behind detectors scoring 99% clean and near-chance at JPEG q65 and half resolution — roughly what a social platform does by default.

Four things it means in practice:

- **Train with it.** Randomized augmentation: JPEG QF across a realistic range, random rescale, random crop, format conversion, occasional double compression. Wang et al. 2020's blur+JPEG finding has aged better than almost any other result in this field, and improves generalization *even when test images are not post-processed*. This is independent support for `0002` §8.4.
- **Measure it as a surface, not a number.** The real spec is TPR@fixed-FPR as a 2D grid over JPEG quality × downscale factor. That grid says which deployment channels can actually be served — a commercial fact, not a research one.
- **Replicate the customers' actual chain.** Generic augmentation is the fallback; the real chain is much better.
- **Find and publish the floor.** There is a point where the signal is genuinely destroyed and no model recovers it. Identify it and make it a documented product limit — "images below X resolution or above Y compression are unsupported" — rather than silently returning noise.

## 7. Label feedback without rights to customer data

First, unbundle the permission. "Using our data" is at least five separate asks, refused at very different rates:

> storing the image · storing derived artifacts (embeddings, hashes, metadata) · training on it · a human looking at it · retention beyond the transaction

Most customers who hard-refuse "train on my images" will accept "log a hash, a score, and an embedding" or "human review only on dispute." A blanket no usually means the ask was bundled.

**The structural advantage: ground truth can be manufactured.** Every fake in the training set is self-labeled by construction, because it was sampled from a generator under our control. Reals are licensed or captured. **Customer data is not needed for training labels at all.** What it actually provides is narrower:

- what the traffic distribution looks like (content mix, compression chain, source platforms)
- discovery of failure modes not anticipated

Both are obtainable without a general training license. Industry patterns, roughly by prevalence:

- **Tiered contracts.** Self-serve tier grants improvement rights; enterprise buys zero-retention. The training corpus draws from the tier that granted it. By a wide margin the most common answer.
- **Derived artifacts instead of pixels.** Embeddings, perceptual hashes, scores, metadata — enough for drift monitoring *and* for retraining a head on frozen features. Caveat worth taking seriously: under GDPR, embeddings can count as personal data if re-identifiable, and embedding-inversion attacks are real. A "check with counsel" path, not a loophole.
- **The dispute channel as a consent event.** "Submit this for review" is an explicit, voluntary grant — and disputes are the highest-information samples, concentrated on errors. The appeals path and the best labeling pipeline are the same system; build it that way deliberately.
- **Consented design partners.** A handful opted in for a discount or early access. A *representative* sample suffices, not the whole stream.
- **Federated evaluation.** Ship an eval harness the customer runs internally; they return aggregate metrics only. Federated *training* is usually not worth the complexity at early stage; federated *evaluation* is cheap.

Note the division of labor: **new-generator coverage comes from our own sampling pipeline and the open web, not from customer traffic.** Customer traffic tells us about our *errors*. The piece we can't get consent for is also the piece we need least.

Plan for the review-queue wrinkle: some customers accept automated processing but not human eyes on their images. Options are customer-side review of their own flags, or region- and vendor-restricted reviewers under a DPA.

## 8. Which research principles survive, invert, or die

**Survive — and matter more than in research:**

- *Dataset hygiene / shortcut avoidance.* In research a shortcut inflates a number. In production it causes harm: a detector keyed on JPEG quality flags every high-quality real photograph a customer uploads. And adversaries will find the shortcut and drive through it.
- *Post-processing robustness.* Not a robustness test — the **default condition** of every image seen. Train with the augmentation; treat the robustness curve as a spec.
- *Physical and methodological claims* still anchor architecture. Unchanged.

**Invert:**

- *Paired, controlled evaluation was the research gold standard — in production it is insufficient.* The primary eval must replay production traffic: production base rate, compression pipeline, content mix. Keep the clean paired benchmark, but demote it — it tells you about *mechanism*, not about the business. Two different dashboards.
- *"Held-out generator" was the research bar; in production it is the permanent operating condition, not a test.* Stop treating it as pass/fail; measure it as a **decay curve** — performance vs. months between generator release and last training run. That curve sets retraining cadence directly, and it is the single most useful measurement available.

**Die:**

- *Chasing the universal detector.* It doesn't exist. Betting a product on it is the standard way this fails.
- *SOTA numbers on public benchmarks.* Irrelevant to actual traffic; useful only for picking a starting point.
- *The SSL-contamination worry* — largely. It threatened the validity of a research *inference*. In production you measure on real traffic directly, so it matters only insofar as it makes offline numbers optimistic — which is already handled by not trusting offline numbers.

## 9. Three concerns research doesn't have

**Adversarial pressure.** Papers assume a passive adversary. The moment the detector affects someone's outcome, there is a motivated one. Cheap attacks that beat most detectors: recompress, mild resize, add noise, route through a "humanizer" service. Beyond robustness augmentation, the operational rule: **never expose raw scores.** Return bands or decisions, not floats — a raw score is a gradient-free optimization oracle. Rate-limit, and monitor for probing patterns (many near-identical images with small perturbations).

Risk specific to this project's chosen direction: a **public frozen backbone makes the feature space public.** Knowledge of the backbone architecture alone supports gray-box attacks at near-white-box success rates — recorded in `0002` §10.

**Silent degradation.** You need to know you're failing before customers tell you. Even without labels: score-distribution drift, **abstention rate**, per-domain flag rates. Wire alerts to these.

**Retraining as CI.** Weekly or biweekly refresh, every run gated by a frozen regression suite — a fixed set of generators and reals with per-generator TPR@FPR that must not regress. Without this gate, adding coverage for new generators silently destroys coverage of old ones and nobody notices for months.

And one that is product, not engineering: a false "AI-generated" verdict on a real person's real photo carries legal and reputational exposure that a false negative does not. That asymmetry belongs in the thresholds, the confidence language, and an appeals path.

## 10. Build order

1. **Write the deployment contract first.** Traffic volume, base rate, FP:FN cost ratio, latency budget, what actually happens to a flagged image. Everything downstream is determined by this, and it is the step that gets skipped.
2. **Build generator acquisition + sampling.** Before the model.
3. **Build the production-mirroring eval + freeze the regression suite.**
4. **Train Layer 1** on max generator diversity with heavy post-processing augmentation.
5. **Add abstention + review loop.** Ship at a conservative threshold: low recall, high precision. Expand recall as evidence accumulates.
6. **Monitoring + retraining cadence,** driven by the decay curve from step 3.

## 11. Positioning

Sell a **risk signal with provenance and a review workflow**, not "tells you if an image is AI-generated." The second is a claim the technology cannot honor, and its failure mode is public and unrecoverable — one viral case of the product calling a real journalist's photo fake. The first is defensible, matches what actually works, and survives the next generator release.

Passive detection is a rearguard action; provenance signing (C2PA) and watermarking are the structural answer, and the detection literature is increasingly explicit about being a stopgap. That doesn't make the work pointless — unsigned images will exist for a long time — but it should shape what is claimed.
