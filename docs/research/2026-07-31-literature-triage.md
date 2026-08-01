# Tracking Claims, Not Papers — A Triage Method for This Field

**Status:** Current
**Date:** 2026-07-31
**Relates to:** [`2026-07-31-claim-verification.md`](2026-07-31-claim-verification.md) (the first application of this method), [`deepfake_detection_research.md`](deepfake_detection_research.md) (the survey that motivated it).

Detection literature is unusually contradictory: papers from different years, on different generator families, using different protocols, report incompatible results without labelling why. This doc records a method for deciding what to keep and what to discard, from the 2026-07-30/31 working sessions.

The core move: **the unit of tracking is the claim, not the paper.** A single paper contains claims with four different expiry dates. Filing it as one bibliography entry destroys exactly the information needed to know when it stops being true.

---

## 1. Three causes of apparent disagreement

When two credible papers conflict, it is one of three things. Diagnosing which comes first, because the response differs completely.

**1. Temporal drift — a real disagreement, both papers correct.** "Spectral upsampling peaks separate real from fake" was genuinely true for 2020-era GANs with transposed convolutions. It is false for latent diffusion. Neither paper is wrong; the object of study changed underneath them.
→ **Response:** date-stamp the claim to the *generator family* it was validated on. Never carry it forward without re-testing.

**2. Evaluation artifact — a fake disagreement.** Paper A reports 99%, Paper B reports 61% on nominally the same task, because A's real images came from a web crawl at native compression while A's fakes were saved near-lossless. The disagreement is about protocol, not about the world.
→ **Response:** resolvable **by reading, without running anything** — the answer is in the data-construction section, not the results table.

**3. Genuinely open question.** Why CLIP features work: low-level cues surviving the encoder, vs. semantic bias.
→ **Response:** don't try to resolve it by reading more. Mark it open. If it is load-bearing for a decision, run the experiment.

**Case 2 is the majority of the apparent chaos, and it is the cheapest to settle.** Most of what looks like scientific disagreement is protocol mismatch.

## 2. Sort every claim by half-life

Tier each claim by how long it stays true, because that determines how much weight it can bear.

**Tier 0 — Physical / mathematical constraints. Half-life: decades.**
Bayer CFA demosaicing induces deterministic inter-pixel correlations. JPEG quantization leaves a lattice in DCT space. PRNU is multiplicative and device-specific. These expire only if camera hardware changes, not when generators improve.
*Safe to build architecture on.*

> Caveat added by the 2026-07-31 verification pass: Tier 0 is not immune. PRNU is losing per-device uniqueness to computational photography, and neural ISPs hallucinate content. The physics anchor is eroding from the manufacturer side. Tier 0 means *slowest-moving*, not *static*.

**Tier 1 — Structural facts about a generator family. Half-life: 2–4 years.**
"All latent diffusion outputs pass through a VAE decoder." True while LDMs dominate; dies when the paradigm shifts to pixel-space or token-based models.
*Safe to exploit, unsafe to depend on exclusively.*

**Tier 2 — Empirical scaling and methodology findings. Half-life: 3–5 years. Most underrated tier.**
"More training generators → better cross-generator transfer." "Aggressive JPEG/resize augmentation improves generalization." "Patch-level features transfer better than global." These survive paradigm shifts because they are statements about *learning dynamics*, not about artifacts.
**Weight these heaviest** — they should drive experimental design.

**Tier 3 — Specific artifact/fingerprint claims. Half-life: 12–24 months.**
Checkerboard peaks, azimuthal spectra, a particular reconstruction-error signal. Read for intuition.
*Never build a thesis on one.*

**Tier 4 — SOTA numbers on a named benchmark. Half-life: months; transfer value ≈ 0.**
"97.3% on GenImage." Meaningful only as a *relative* comparison inside one paper's own controlled protocol.

**The rule that falls out:** architecture rests on Tiers 0–2. Tier 3 informs features that are kept ablatable. Tier 4 is noise. Most people's mental model of this field is built almost entirely from Tiers 3 and 4 — which is precisely why it feels like it contradicts itself every six months.

## 3. The 90-second triage filter

Applied in order, cheapest first, to any detection result:

1. **How was the real class sourced vs. the fake class?** Different pipelines → the number measures the pipeline. Check: resolution matched? JPEG quality *and* chroma subsampling matched? Both classes re-encoded identically? Content paired?
2. **What is the held-out axis?** Held-out *images* from a seen generator ≈ worthless. Held-out *generator* is the floor. Held-out generator *released after the backbone's data cutoff* is the real bar.
3. **Is there a post-processing robustness curve?** No accuracy-vs-JPEG-QF and accuracy-vs-downscale plot → assume it dies in the wild. The *absence* is itself evidence.
4. **What operating point?** No TPR at low FPR → the deployment claim is unsupported, whatever the AUC says.
5. **Did the authors ablate their own mechanism story?** Papers that attack their own explanation are worth 5× the ones that don't.

**If 1 and 2 fail, discard without reading the results.** That is most of the triage.

## 4. The three artifacts to maintain

**A claims ledger.** One row per claim:

> **Claim** (stated so it *could* be false) · **Tier** · **Validated on** (which generators, which real source, what year) · **What would falsify it / when to re-check** · **Status**: accepted / rejected / open / to-test

**A decisions log.** For every design decision, record which claim IDs it rests on. When a claim is demoted, query *which decisions just lost their support*. **That inverse link is the entire point.** Without it, dead assumptions aren't removed — they are silently inherited, and you end up with a 2026 model carrying a 2020 premise nobody re-checked.

This project already has the decisions half, in `docs/decisions/`. `0002` §6.2 is a worked example of the inverse query: a claim was re-examined, found to rest on a framing error, and the decision resting on it was reopened.

**A kill list.** Claims explicitly rejected, with the reason. Highest value-per-line document in the set. Without it you *will* meet the same seductive idea in a new paper's framing in three months and re-litigate it from scratch.

The verification doc's §2 (Corrected) and §3 (Reclassified) are the beginnings of both the kill list and the ledger for this project.

## 5. Reading strategy

- **Start from the most recent benchmark/survey and read backwards**, not from the seminal paper forwards. Recent benchmark papers do the negative results for you — they say which older methods fail and by how much. One 2026 paper evaluating 16 detectors beats reading those 16 papers.
- **Weight critique/benchmark papers far above proposal papers.** A paper whose contribution is "we tested existing methods and here's where they break" has no incentive to inflate. A paper proposing a method does. This asymmetry is large and consistent.
- **Methodology papers reshape more than method papers.** A paper whose entire contribution is "this whole line of work has a compression bias" changes more of your model than any individual detector ever will.
- **Anchor on groups, not papers.** A handful of labs have consistent protocol standards. Knowing whose evaluation to trust is a cheap prior that saves re-deriving credibility every time.
- **Time-box with a stopping rule:** stop surveying when new papers stop changing the Tier 0–2 set. Tiers 3–4 churn forever and reading more of them never converges. That is the real answer to "how do I keep up" — you don't, and you don't need to, because the tiers that matter move slowly enough to track.

## 6. The decisive move

When two credible claims conflict and the answer is load-bearing: **stop reading and run the smallest discriminating experiment.** A day of compute beats a week of literature triangulation, and it answers on *our* distribution — the only one being graded.

**Heuristic:** if a disagreement has already survived two papers arguing about it, a third won't settle it.

This is the pattern the project's investigation line already follows — `2026-07-26`, `2026-07-27` ×2, and `2026-07-29` are each a small discriminating experiment run instead of an argument, and each is written up with what it **rules out** rather than what it proves. `0002` §9 is the same move applied prospectively: a falsification condition written down *before* the experiment, at a stated cost of one afternoon.

## 7. Two open, load-bearing claims for this project

Both are Tier 2-ish, both unresolved in the literature, both cheap to settle in-house — which is exactly the signature of a claim to stop reading about:

| Claim | Status | Discriminating test |
|---|---|---|
| CLIP separability is driven by low-level forensic cues, not semantics | open | Patch-shuffle or high-pass the input and re-measure separability. If separation survives, it's forensic; if it evaporates, it was content. |
| SSL backbone pretraining contamination inflates our results | open | Repeat the probe on a backbone with provably pre-generative-AI pretraining (ImageNet-1k MAE or supervised ResNet). If CLIP's advantage survives, contamination isn't driving it; if the IN1k backbone matches, CLIP was never needed. |

Both convert an unfalsifiable anxiety into a number, and both are roughly a day's work.
