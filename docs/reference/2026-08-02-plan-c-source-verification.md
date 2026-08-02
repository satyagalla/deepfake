# Plan C source verification — the calibration plan checked against its primary sources

**Written:** 2026-08-02. Deadline Mon 2026-08-03.
**Status:** Current.
**Scope:** A verification pass over the sources underneath **Plan C** of [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) §4, prompted by the question *"is Plan C legible, and would it prove the hypothesis?"*
**Amends:** [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §2.2, §4.2, §5 · [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) §4, §6.4 · [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §11

Per the `research/` convention, the docs above are **not edited**; this file is the forward-pointing record.

Evidence tags follow the repo convention: `[confirmed]` · `[measured]` · `[unverified]`. Everything below marked `[confirmed]` was checked against the paper body or listing page **this session**, not against a summary of it. Nothing here is `[measured]` — no number in this doc came from our data.

---

## 0. The question, and the answer

**Question.** The acceptance test is `context-transfer.md` §5: the founder generates an image live from Gemini or GPT Image, and the model must call it fake. Is Plan C — train the probe on COCO_AI diffusion data, then fit a scalar logit correction on a handful of transformer images — the plan that gets there?

**Answer. Plan C is legitimate as a bolt-on and is not selectable as the plan.** Two independent reasons:

1. **Structural.** Plan C is downstream of work that does not exist yet. It cannot be first.
2. **Evidential.** The premise Plan C rests on — *"the failure is misplaced thresholds, not lost separability"* — is the source paper's **interpretation**, not a measurement it reports. Verified this session: the paper reports accuracy only. No AUC, no AP, no ranking analysis anywhere in it.

Reason 2 is the material finding of this session and is developed in §2.

---

## 1. Structural: Plan C cannot be the first thing built

`three-candidate-plans.md` §4 already records this — *"Plan C is not selectable until [the AUC precondition test] is run, and it needs the transformer images and a working probe to run at all."* This session confirms the state of the repo makes that stricter than the doc implies.

Checked directly on disk, 2026-08-02: `[confirmed]`

| Prerequisite | State |
|---|---|
| `data_raw/` (COCO_AI images) | **Absent** in this checkout |
| `dataset/` (manifest) | **Absent** |
| `checkpoints/` (any probe or v1) | **Absent** |
| `.venv/` | **Absent** |
| Multi-generator download | **Not wired.** `data/download.py:68` still reads `ds.select_columns(["caption", "coco_image", "dalle_image"])` — the one line that discards five of COCO_AI's six generator columns (`context-transfer.md` §4) |
| Transformer (Gemini / GPT Image) eval set | **Does not exist.** `0002` §5.2 is an ad-hoc spot check with no recorded sample size and no reproducible script |

Choosing Plan C today therefore means executing steps 1 and 2 of `three-candidate-plans.md` §5's ordering anyway, and only then arriving at the decision point. It is not a choice between plans; it is a choice to defer the choice.

---

## 2. Evidential: what the Yang et al. paper actually reports

[Yang et al., AAAI 2026 — *Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated*](https://arxiv.org/abs/2602.01973). Checked against the arXiv full text and the [code release](https://github.com/muliyangm/AIGI-Det-Calib) this session. Authors: Muli Yang, Gabriel James Goenawan, Henan Wang, Huaiyuan Qin, Chenghao Xu, Yanhua Yang, Fen Fang, Ying Sun, Joo-Hwee Lim, Hongyuan Zhu. Presented Singapore, 2026-01-24. `[confirmed]`

Three of the five items that [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §2.2 listed as `[unverified]` are now resolved.

### 2.1 Calibration-set size — resolved, in Plan C's favour `[confirmed]`

`calibration-and-thresholds.md` §5 lists this as **highest priority**, because §4.2 has a build decision resting on it.

The paper uses **100 images (~1% of the test set)** by default, and its Figure 3 ablation reports both the supervised and unsupervised variants performing *"stably with varying amounts of validation data, demonstrating strong performance with as few as 10 samples, which is less than 0.1% of the data in each test set."*

**This closes the open item.** The ~100-image figure carried second-hand since 07-31 is correct, and the true floor is an order of magnitude lower. `calibration-and-thresholds.md` §4.2's worry — that the OOD set might be undersized to serve as both failure measurement and calibration set — does not materialise.

### 2.2 The separability claim is an interpretation, not a measurement `[confirmed]`

**This is the finding that changes the assessment.**

The paper **reports accuracy only. It reports no AUC and no AP, and contains no analysis of whether ranking is preserved while the threshold moves.**

The abstract's attribution of the failure to *"misaligned decision thresholds rather than a loss of feature separability"* is the authors' explanation for why a scalar shift recovers accuracy. It is a plausible explanation and it is consistent with their results. It is not a demonstrated ranking-preservation result, because the metric that would demonstrate it was never computed.

**Consequences for this project:**

- [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) §4's framing — *"the features may already separate transformer fakes from reals, with the threshold in the wrong place"* — inherits an inference and presents it with the weight of a finding.
- [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §2.3's *"Ordering survives what thresholding destroys"* is sound as reasoning and is **not** something the cited paper measured.
- `0002` §11's open item — *"Test whether ranking survives on our OOD set"* — was framed there as a **local replication** of Yang et al.'s mechanism. It is not a replication. It is the **first measurement of that mechanism in a form either paper or project has recorded**, which makes it more valuable and also means there is no prior in its favour.

**Methodology note.** This is the same failure mode [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §6 named — *a confirmed premise laundering an unconfirmed inference* — recurring one level down. The premise (a scalar logit shift recovers substantial accuracy across nine detectors) is `[confirmed]`. The inference (*therefore separability was intact and only the boundary moved*) travelled at the same confidence, this time authored by the paper rather than by us. §6's proposed remedy — separate tags for premise and inference inside a single bullet — would have caught it, and did not get applied to the paper's own claims.

### 2.3 The Chameleon caveat — confirmed, and it is the case we most resemble `[confirmed]`

`calibration-and-thresholds.md` §2.2 flagged this as *"conceptually the most important of the five"* while tagging it `[unverified]`. It holds, close to verbatim:

> on Chameleon, *"the real and fake logit distributions produced by all nine detectors exhibit significant overlap. In these scenarios, simple scalar threshold adjustments may no longer suffice."*

The unsupervised variant degrades further there, gaining little or going **negative** on some detectors.

**Why this is the load-bearing risk for us.** `0002` §5.2 records v1 classifying Gemini and gpt-image-1 images as `real` at >75% confidence — uniformly, across the whole spot-check. That observation is equally consistent with:

- a **shifted** logit distribution — the fake logits sit below the threshold but still above the real ones. Plan C works.
- an **overlapped** logit distribution — the fake logits are interleaved with the real ones. Plan C is the Chameleon case and cannot work.

Nothing in the repo distinguishes these. The distinguishing statistic is AUC on a transformer eval set, and it has never been computed.

### 2.4 Method form and gains — confirmed `[confirmed]`

| Item | Status |
|---|---|
| Correction form | `f̃(x) := f(x) − α`, a single learnable scalar on the logits, backbone frozen |
| Supervised variant | α optimized via kernel density estimation to minimize classification error on labelled target samples |
| Label-free variant | Moment-balancing symmetry — α chosen so the logit distribution is symmetric about the threshold; *"the optimal threshold is simply the expected logit under the estimated distribution."* Matches the `α* = E[z]` formula carried second-hand in `calibration-and-thresholds.md` §2.2 |
| Gains, AIGCDetectBenchmark | +1.62% (AIDE) to +10.13% (Effort), supervised |
| Gains, GenImage | +2.94% (AIDE, unsupervised) to +16.16% (RINE, supervised) |
| CNNSpot | 70.83% → 78.22%, supervised — matches the second-hand table in `calibration-and-thresholds.md` §2.2 |
| Detectors / benchmarks | Nine detectors, two primary benchmarks |

Note the shape of the gains: the corrected detectors start at **60–82%** accuracy, not at the near-total collapse `0002` §5.2 describes. The paper does not contain a case resembling *"confidently `real` on every image of the target class."*

---

## 3. Three further problems specific to our setup

Not defects in the paper — consequences of applying it here.

### 3.1 Plan C's cost advantage over Plan A is smaller than §5's table implies `[derivable]`

The correction is fitted per target distribution, on target-distribution images. At demo time there is **one** image and nothing to fit on, so α must be pre-fitted on a held-out Gemini/GPT set — which must therefore be bought. The label-free variant does not remove this: fitting α as the expected logit presumes a target *pool* that is roughly class-balanced, which a single uploaded image is not.

So Plan C buys ~10–100 transformer images instead of Plan A's ~300. That is a real discount and it is a **discount on Plan A**, not a structurally different data requirement. `three-candidate-plans.md` §5's "Blocked by: an unrun precondition test" is correct; "API spend ~$4" understates how much of Plan A's sourcing work Plan C still inherits.

### 3.2 Calibration is blind to the container shortcut, and makes it look better `[derivable]`

If the probe is reading §1(C) of `three-candidate-plans.md` — JPEG-vs-PNG, the 1024 canvas, Google's or OpenAI's delivery pipeline — then fitting α on Gemini-web-delivered images tunes the threshold of a **container detector** and raises its accuracy on exactly the images the demo will use.

Plan C therefore converts the project's worst failure mode into a passing demo number, silently. The re-encode control (`context-transfer.md` §"immediate next step": re-save JPEG q85, rescale 95%, re-score) is a **prerequisite for Plan C specifically**, not merely a general hygiene step, and must run *before* any α is fitted rather than after.

### 3.3 The claim Plan C produces is not the claim being judged `[derivable]`

Plan C's deliverable, per `three-candidate-plans.md` §4, is *"the failure was recoverable post-hoc without retraining."* That is a research claim, and an interesting one. The hypothesis under test on Monday is *"this image, generated live from Gemini, is classified correctly."* Plan C is a 20-line addition on top of a working probe; it is not the thing that produces a working probe.

---

## 4. A correction to `three-candidate-plans.md` §6.4

**DailyBench (arXiv:2607.24016) has been withdrawn.** `[confirmed]`

Submitted 2026-07-27, **withdrawn by the author 2026-07-28** with the note *"Some errors must be corrected."* The 91–96% GenImage → 60–76% FakeBench figure is confirmed as what the abstract said; the paper stating it no longer stands.

`three-candidate-plans.md` §6.4 uses that figure as *"the honest prior for a live demo,"* and §6.4 also uses the SSAFE-vs-DailyBench discrepancy as a reason not to trust SSAFE's flattering numbers. **Both uses should be withdrawn.** The pessimistic prior may still be right — it is now unsourced.

### Related, unresolved: SSAFE `[unverified]`

[SSAFE (arXiv:2606.08634)](https://arxiv.org/abs/2606.08634), submitted 2026-06-07, preprint, not peer-reviewed. Checked this session against the abstract page only — **the full body was not read**, so this is a partial check and the §6.4 tag stands.

What the abstract page does support: frozen multimodal encoders separate real from generated in embedding space, a linear classifier on top performs strongly without fine-tuning, **10K training images** against AIGIBench's 288K and OpenFake's 4M, and a `RealWorldBench` spanning modern camera photographs, contemporary stock images, and recent commercial generators.

What it does **not** support at this level of checking: the specific generator list (GPT-Image-1, Nano-Banana, Imagen 3/4, Janus-7B, LlamaGen), the **PE-Core-G14-448 over CLIP/DINOv2/DINOv3** encoder recommendation, and any accuracy or AUC figure. §6.4 uses the encoder recommendation as an argument for a head-to-head before defaulting to CLIP ViT-L/14 — that argument currently rests on an unread body of an unreplicated preprint.

The `RealWorldBench` composition is independently interesting: it is the same insight as `context-transfer.md` §4's last bullet — *the real class is 2014-era COCO web JPEGs while any live upload is a modern phone photo* — solved by someone else. Worth reading properly if time allows.

---

## 5. What this changes about the plan

`three-candidate-plans.md` §5's ordering is unchanged and is strengthened by §2.2 above. One addition to step 1.

**1. Run the free experiment — and report AUC and accuracy separately.**

Frozen CLIP ViT-L/14 + logistic regression on COCO_AI, tested on a small fixed set of Gemini / GPT Image files. Requirements carried from the existing docs:

- All six generator columns (`data/download.py:68` is a one-line fix) — `context-transfer.md` §4
- **Row-level split**, since the same `coco_image` appears in every generator column — `context-transfer.md` §"where we are now"
- Augmentation on: blur σ~U[0,3] and JPEG q~U{30…100}, each at **p=0.5**, not applied to every image — `three-candidate-plans.md` §3, `0002` §8.4
- Uniform re-encoding of both classes — `three-candidate-plans.md` §5

The addition: **report AUC alongside accuracy.** That single run is simultaneously the deliverable *and* the Plan C gate:

| Result on the transformer eval set | Reading | Action |
|---|---|---|
| AUC ≳ 0.75, accuracy ≈ 50% | Shifted-but-separable. Yang et al.'s regime | Plan C applies. It is a scalar and an afternoon |
| AUC ≈ 0.5 | Overlapped. The Chameleon regime, §2.3 | Plan C is dead. Buy data — Plan A |

This is the `0002` §11 open item, and per §2.2 it is a first measurement rather than a replication.

**2. Run the re-encode control before fitting anything** — §3.2. On Plan C's path this is a precondition, not a follow-up.

**3. Decide A vs C on that evidence.** Not in advance.

---

## 6. Status of the items this pass touched

| Item | Was | Now |
|---|---|---|
| Yang et al. calibration-set size (`calibration-and-thresholds.md` §5, top priority) | `[unverified]`, second-hand "~100" | **`[confirmed]`** — 100 default, stable to 10. Closed |
| Unsupervised form `α* = E[z]` (§2.2) | `[unverified]` | **`[confirmed]`** — moment-balancing; expected logit under the estimated distribution |
| Chameleon limitation (§2.2) | `[unverified]` | **`[confirmed]`** — quoted in §2.3 |
| Per-detector gains table (§2.2) | `[unverified]` | **`[confirmed]`** — CNNSpot 70.83→78.22 and the ranges match |
| Runtime ~0.5–0.9 ms (§2.2) | `[unverified]` | Still `[unverified]` — not checked, not load-bearing |
| "Misaligned thresholds rather than lost separability" | `[confirmed from abstract]` | **Downgraded** — confirmed as the paper's *attribution*; the paper reports no AUC/AP, so ranking preservation is unmeasured by it. See §2.2 |
| DailyBench 91–96% → 60–76% prior | `[confirmed]` | **Withdrawn paper.** Figure unsourced as of 2026-07-28. See §4 |
| SSAFE generator list, PE-Core recommendation | `[unverified]` | Unchanged `[unverified]` — abstract checked, body not read. See §4 |
| Jaeger et al. (`calibration-and-thresholds.md` §5) | `[second-hand]` | Unchanged — not checked this pass |
| Minderer et al. read in full (§5) | Pending | Unchanged — not checked this pass |

---

## Sources

Checked against the primary source this session:

- [Yang et al., *Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated*, AAAI 2026 — arXiv:2602.01973](https://arxiv.org/abs/2602.01973) · [full text](https://arxiv.org/html/2602.01973v1) · [code](https://github.com/muliyangm/AIGI-Det-Calib) · [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/38146) — §2
- [*DailyBench*, arXiv:2607.24016](https://arxiv.org/abs/2607.24016) — **withdrawn 2026-07-28**, §4
- [*SSAFE*, arXiv:2606.08634](https://arxiv.org/abs/2606.08634) — abstract page only, §4

Repo docs this pass reads against:

- [`../notes/2026-08-02-three-candidate-plans.md`](../notes/2026-08-02-three-candidate-plans.md) §4, §5, §6.4
- [`2026-08-01-calibration-and-thresholds.md`](2026-08-01-calibration-and-thresholds.md) §2.2, §2.3, §4.2, §5, §6
- [`../notes/2026-08-01-context-transfer.md`](../notes/2026-08-01-context-transfer.md) §4, §5, §8, "immediate next step"
- [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §5.2, §8.4, §11
- [`../decisions/0003-frozen-probe-demo-build.md`](../decisions/0003-frozen-probe-demo-build.md) §5, §6
