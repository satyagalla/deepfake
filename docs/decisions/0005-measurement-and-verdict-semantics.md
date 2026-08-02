# 0005 — Measurement and Verdict Semantics: What We Report and What We Are Entitled to Claim

**Status:** Accepted
**Date:** 2026-08-02
**Deadline:** Monday 2026-08-03 (Plurall AI second interview)
**Supersedes:** [`0004-adaptation-hypothesis-demo-build.md`](0004-adaptation-hypothesis-demo-build.md) §6.6, §7.1, §8's verdict vocabulary and fusion rules, and the success criteria of §9 E1/E5 — see §9.
**Relates to:** [`../research/2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md) (the research this executes and, in part, declines to execute), [`../reference/2026-08-02-plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) §2, §5 (the decision rule), [`../research/2026-07-31-production-deployment.md`](../research/2026-07-31-production-deployment.md) (base-rate math), [`../notes/2026-08-02-build-plan.md`](../notes/2026-08-02-build-plan.md) (execution), `../../notes.md` §Interview.

---

## 1. Context: what reading the calibration research against the code found

`0004` §3 recorded calibration as restored to scope after `0003` §6 had excluded it. What was actually built (`probe/heads.py:34-44`) is **one** step of the chain the research doc specifies: a Platt calibrator fitted on the val split. Three things surfaced when the code was read against [`2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md):

1. **The calibrator is fitted on the wrong distribution for every headline number.** It is fitted on `split="val"` — COCO_AI, SD/DALL·E, 270–480px. E1 measures Gemini/GPT-Image at 1024px; E2 measures Midjourney. A monotone squashing function fitted on one distribution and applied to another does not make those numbers more honest.
2. **It cannot help the claim.** Platt is monotone, so it cannot change AUC, and accuracy was the only other number reported. It is a fitted stage between the score and the verdict that earns nothing the demo needs.
3. **It contaminates E5.** `_fit_base_head_a` fits the calibrator on val (`probe/experiments.py:64-65`) and `run_e5` then evaluates on val (`:142-143`). Val is partly fitted-on.

The narrowing from "calibration is in scope" to "the fitting step is in scope" was never recorded. This doc records it, and resolves it in the opposite direction: **remove the fitted stage, and spend the budget on measurement instead.**

## 2. The decision

Three parts, one theme — **report what the system can support and name it accurately.**

1. **Remove post-hoc calibration entirely.** No Platt, no temperature scaling, no fitted map between score and verdict.
2. **Replace accuracy-as-headline with a prevalence-invariant metric set**: AUC, TPR, FPR — plus precision *derived* at a named prevalence rather than stored.
3. **Derive the verdict vocabulary from what this model can observe**, rather than inheriting the product spec's.

## 3. Removing calibration

**Removed:** `HeadA.calibrator`, the Platt fit, `predict_proba_head_a`, and all `X_cal` / `y_cal` plumbing through `probe/experiments.py` and `probe/spectral.py`.

**What replaces it where a `[0,1]` number is genuinely needed** (the card scores, which fusion averages): the classifier's own `predict_proba`. That is the model's sigmoid, not a second fitted stage. Because Head A is fitted with `class_weight="balanced"` against a ~1:5 real:fake corpus, those probabilities are stated as-if-balanced — which is the correct frame for the balanced eval sets §5 introduces, and is stated rather than assumed.

**Three consequences, all favourable:**

- **AUC is unchanged.** Platt is monotone; it cannot reorder. Every AUC this build would have produced is the AUC it now produces. The removal is free on the metric that matters most.
- **E5 becomes honest.** With nothing fitted on val, val is a clean held-out split.
- **The demo narrative loses a stage that could not be defended.** "Calibrated on a held-out split" invites "which split, and does the live upload come from it?" — a question with no good answer when the answer is COCO_AI.

**What is given up, stated plainly:** the system no longer emits anything that can be called a calibrated probability. `0004` §7.1 implied it did. It could not have, on any distribution the live test occupies. Removing the claim is the correction; removing the code is how the claim stops being made accidentally.

## 4. The metric set

The governing distinction is **prevalence-dependence**, because the eval sets' class ratio is a construct and deployment prevalence is ~0.5% ([`2026-07-31-production-deployment.md`](../research/2026-07-31-production-deployment.md)).

| Metric | Prevalence-dependent? | Status |
|---|---|---|
| **AUC** | No — a ranking statistic | **Primary.** The claim's real quantity |
| **TPR** (recall) | No — computed on AI images only | **Primary** |
| **FPR** | No — computed on real images only | **Primary** |
| Balanced accuracy | No — `(TPR + 1−FPR)/2` | Derived, free |
| Accuracy | **Yes** | Demoted. Was the E1 headline |
| Precision | **Yes** | **Derived at a named π, never stored** |
| F1 | **Yes** (inherits precision's) | **Not built** — §8 |

### 4.1 Why AUC is primary

It is threshold-free, so it measures separability and nothing else, and it is invariant to class ratio. It is also the metric [`plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) §5 states the project's live decision rule in: **AUC ≳0.75 with accuracy ≈50% is Yang's regime** (signal present, threshold misplaced, a scalar correction recovers it); **AUC ≈0.5 is the Chameleon regime** (genuinely blind, and the correction idea is dead). `bottlenecks.md` §4.1 records that AUC has never been computed on this project. It is still a first measurement.

### 4.2 Why TPR and FPR rather than balanced accuracy

Balanced accuracy *is* their average, so reporting the pair unaggregated costs nothing and carries strictly more information. Specifically, it **reads out the failure mode directly**, which is the question `0004` §9 E5 exists to answer:

| 0-shot signature | Reading |
|---|---|
| TPR ≈ 0.05, FPR ≈ 0.01 | Calls everything real — **misplaced threshold** |
| TPR ≈ 0.5, FPR ≈ 0.5 | Coin flip — **lost separability**, Chameleon regime |

Both produce the same accuracy. The confusion matrix separates them before AUC or a threshold correction is involved.

It is also the better sentence in the room: *"recall went 0.05 → 0.90 while FPR stayed at 0.02"* says more than *"accuracy went 0.94 → 0.95."*

### 4.3 Precision is derived, not stored

Because TPR and FPR are prevalence-invariant, precision at any prevalence π follows:

```
precision(π) = π·TPR / (π·TPR + (1−π)·FPR)
```

Storing precision would freeze the eval set's arbitrary class ratio into the number. Storing TPR and FPR lets any π be quoted on demand. At a plausible good result — TPR 0.90, FPR 0.02:

| Prevalence | Precision |
|---|---|
| 50% (the eval set) | 0.98 |
| 5% | 0.70 |
| **0.5% (deployment)** | **0.18** |

Four in five `LIKELY_SYNTHETIC` verdicts wrong, at 90% recall and a 2% false-positive rate. This is [`2026-08-01-calibration-and-thresholds.md`](../research/2026-08-01-calibration-and-thresholds.md) §1.2 computed rather than asserted, and it reproduces the figure already on record in [`2026-07-31-production-deployment.md`](../research/2026-07-31-production-deployment.md) (95% TPR / 1% FPR at 0.5% → ~32%) from the same fifteen lines.

### 4.4 The α correction, and why it stays label-free

Yang et al.'s label-free variant — `α* = E[z]`, the mean logit of an unlabelled batch `[confirmed]`, [`plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) §2 — is reported as `accuracy_at_alpha` alongside the default cut. It is admissible **inside E1** precisely because it spends no labels and therefore no N-shot budget; a supervised correction would consume target-generator labels outside the stated N and corrupt the headline.

An **oracle threshold** (best achievable, uses labels) is computed as a diagnostic ceiling. `oracle − default` is the decomposition: how much of the failure was placement rather than blindness. **It is never quoted as a result.**

Note the precondition on record and not yet met: [`plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) §3.2 states the re-encode control (E3) is a **precondition** for fitting α, because a threshold correction is blind to the container shortcut. E3 sits at S9 in the cut order. If E3 is cut, `accuracy_at_alpha` is reported with that caveat attached, not silently.

## 5. The eval-set balance defect

Found while tracing eval sets, and it is a correctness bug in the headline experiment rather than a metric preference.

`run_e1` builds `eval_rows = eval_ai_rows + eval_real_rows` (`probe/experiments.py:78-80`). With `VAL_FRACTION = 0.15` over ~3,000 rows, `eval_real_rows` is ~450 images; `eval_ai_rows` is the selfgen eval partition for one generator/domain — **tens**. This is arithmetic from `config.py` and `0004` §5, not a measurement: nothing is on disk yet.

**A model answering "real" to everything scores ≥90%.** The 0-shot point would look excellent, the curve would be flat, and by `0004` §9's own pre-registered criterion — *"a curve with no knee falsifies the claim"* — the build would falsify its own hypothesis on an artifact of set composition. E2 has it in reverse: ~3,000 Midjourney against ~450 reals, where "always AI" scores ~87%.

This is `2026-08-01-calibration-and-thresholds.md` §1.2 — class proportions changing what a number means — sitting inside the headline experiment.

**Resolution, split by what each metric actually needs:**

- **AUC** — no change. Prevalence-invariant; uses all ~450 reals for maximum power.
- **TPR / FPR** — no change. Each is computed within one class.
- **Anything cut-point-based** — reported as balanced accuracy, which uses all reals and estimates the balanced quantity without discarding data.
- **α = E[z]** — requires a **balanced subsample**, drawn per trial with the trial's seed. Moment-balancing is the whole argument for `α* = E[z]`; on a ≥90%-real batch the mean logit is dragged toward the real class and α is actively wrong. The labels are used to *construct the batch*, not to fit α, so the label-free property survives — but the resulting number is optimistic relative to a deployment batch that cannot be balanced, and is reported as such.

**E1's knee criterion moves to AUC ≥ 0.90**, with the balanced-accuracy knee reported alongside. `probe/experiments.py:222` currently hard-codes accuracy ≥ 0.90.

**Sample-size caveat that outranks every choice above:** FPR sits on ~450 reals (estimable to ~±0.2%); TPR sits on tens of AI images (granularity ~3%, wide interval) — and TPR is the side the claim lives on. `n_ai` is reported next to it. **More self-generated eval images buy more than any metric change in this document.**

## 6. Verdicts, derived rather than inherited

`0004` §8 took its vocabulary — `AUTHENTIC | SUSPICIOUS | SYNTHETIC | PLAUSIBLE | STRIPPED` — from Plurall's product spec (`notes.md:54-57`, `[stated]`). That is the correct source for the P1 drill, which is to implement *their* function (§12). It is the wrong source for our own system, because it describes a different model's observables.

### 6.1 What this model can actually observe

| Signal | What it establishes |
|---|---|
| Head A score | Resemblance to generators seen in training |
| **Mahalanobis distance** | Whether the image resembles *anything* fitted on |
| Head B | Which generator family — meaningful only if the gate passes |
| Spectral score | Independent frequency-domain evidence |
| C2PA / EXIF bytes | A *declaration*, not an inference |

### 6.2 Two axes, not one ramp

The inherited vocabulary is a single confidence ramp. The distinctive property of this build — the thing `0004` §2 is a claim *about* — is that it knows when it is outside its own competence. That is the Mahalanobis gate, and it is **orthogonal** to the score. "Scores 0.9 but I have never seen this source" and "scores 0.1 but I have never seen this source" are different states, both reachable, and Midjourney at 0-shot is the first.

Collapsing them into one middle bucket destroys the demo's point. So the output carries **two fields**:

**`verdict`** — what the evidence says:

| Verdict | Reachable when | Implied action |
|---|---|---|
| `DECLARED_SYNTHETIC` | C2PA `trainedAlgorithmicMedia` present | Accept; no model involved |
| `LIKELY_SYNTHETIC` | Fused score above the 1%-FPR cut | Act |
| `WEAK_EVIDENCE` | Between the 10%- and 1%-FPR cuts | Human review |
| `NO_EVIDENCE` | Below the 10%-FPR cut | Pass — **not** "verified" |

**`reliability`** — whether the verdict is entitled to belief:

| Value | Meaning |
|---|---|
| `IN_DISTRIBUTION` | Resembles fitted classes; Head B's generator name is usable |
| `UNKNOWN_SOURCE` | Outside everything fitted on; the score is not evidence and Head B's name is meaningless |

Five states, each with a genuinely different action — including `UNKNOWN_SOURCE`, whose action is *"collect ~30 labelled examples of this source,"* which is `0004` §2's thesis rendered as a UI element.

**`UNKNOWN_SOURCE` does not suppress the verdict.** It is displayed alongside it. Midjourney at 0-shot then reads `LIKELY_SYNTHETIC / UNKNOWN_SOURCE` — the right answer, correctly disclaimed — which is a stronger and more honest demo than a blank, and it makes the gate visible rather than merely present.

### 6.3 Cards do not carry verdicts

The confusion in `0004` §8 came from cards and fusion sharing one vocabulary. A card does exactly two things: produce a score, or have nothing to say.

```
Card: score: float | None,  silent_because: str | None
```

`STRIPPED`, `NOT_APPLICABLE` and `PLAUSIBLE` were never verdicts — they are **reasons for silence**. Making that explicit removes three words and loses no information: `0004` §8's rule *"absent metadata is absent evidence, excluded not scored 0.5"* survives intact as `silent_because="no EXIF or C2PA manifest"`. `PLAUSIBLE` in particular was overloaded across two unrelated states — "EXIF present but inconclusive" (weak evidence, scored, counts) and "no cards could speak" (no information) — which should not have shared a word on screen.

`ABSTAIN` also disappears as a verdict: it was never a statement about the image, and it is now the `reliability` field.

### 6.4 Two mechanical defects in fusion this exposes

`fuse_cards` takes an unweighted mean (`probe/cards.py:197`). Two consequences, both visible in the demo:

- **`SYNTHETIC` was unreachable whenever the EXIF card fired AUTHENTIC.** Camera metadata pins that card at 0.05, so with three scoreable cards the ceiling is `(1.0 + 1.0 + 0.05)/3 = 0.68` — below the 0.85 cut. Any image carrying camera EXIF capped at `SUSPICIOUS` regardless of the evidence. Defensible as a deliberate choice; not defensible as an accident.
- **The weakest card gets half the vote.** Gemini/GPT-Image API output carries no EXIF → silent → excluded, leaving two cards. Head A at 0.98 with the spectral card at 0.55 fuses to 0.77. Expect this to dominate demo behaviour.

Both are recorded here and left as **open** — §7's FPR-derived cuts change where the boundaries sit but not the mean's behaviour. Weighted fusion is not being introduced the day before the deadline.

## 7. Thresholds from a false-positive budget

`FUSION_SYNTHETIC_THRESHOLD = 0.85` / `FUSION_SUSPICIOUS_THRESHOLD = 0.5` (`config.py:82-83`) are Plurall's `[stated]` numbers, and they were being applied to a Platt scale that no longer exists. Removing calibration moves where they land even though it does not move AUC.

**They are instead fitted from the real val split:** the `LIKELY_SYNTHETIC` cut is the 99th percentile of real fused scores (FPR = 1% by construction) and the `WEAK_EVIDENCE` cut is the 90th (FPR = 10%). The thresholds then carry a stated meaning — *"one in a hundred real photos gets called synthetic"* — rather than being constants someone picked.

They remain **parameters**, per `0004` §8's last rule, which stands: the product exposes them under Detection Settings and so does the code. What changes is the default's provenance.

## 8. Deliberately not built

- **ECE and reliability diagrams.** The research doc asks for them ([§4](../research/2026-08-01-calibration-and-thresholds.md), and `0002` §11 calls it the cheapest open item). They need hundreds of eval points per bin; the AI side of every eval set here is tens of images. ECE would be noise wearing a decimal point. **This is the one place the research doc's ambition exceeds the data**, and saying so is better than shipping the number.
- **F1.** Precision and recall at the eval set's arbitrary prevalence, collapsed into a scalar that moves with set composition and cannot be decomposed afterwards. Strictly worse than its parts.
- **A stored precision or a deployment probability.** Both require a prevalence the project does not have. Precision is quoted with π named (§4.3).
- **`VERIFIED_CAPTURE` as a verdict.** A valid manufacturer C2PA manifest *would* be positive evidence of capture, but `_parse_c2pa` (`probe/cards.py:120-123`) is a byte-substring search that cannot validate a signature. The state is not honestly reachable, so it gets no name. The EXIF card still contributes 0.05.
- **`AUTHENTIC` / `REAL` as a verdict.** Absence of synthesis evidence is not evidence of capture. Retaining the word would contradict the file's own rule three lines above it (`probe/cards.py:12`), and it is the claim an interviewer should poke at.
- **`DEEPFAKE` as a verdict.** This model detects synthesis, not identity manipulation. The face-filtered v1 corpus is retired and nothing in the v2 corpus is a face swap.
- **Multicalibration / per-evidence-pattern calibration.** `2026-08-01-calibration-and-thresholds.md` §3 is right that "which cards fired" is a computationally-identifiable subgroup, and `round2-drills.md` D1/C2 correctly identifies it as the strongest critique available of a mean-based fusion. It stays a **talking point**, not a build item: with three scoreable cards and tens of eval images per generator, per-pattern fitting has no data.

## 9. What this changes about `0004`

| `0004` section | Status under `0005` |
|---|---|
| §3's row restoring calibration to scope | **Superseded.** Calibration is out of scope again; OOD gating and per-dimension explainability remain in. |
| §6.6 (calibration fitted on image-level scores) | **Moot, not violated.** Nothing is fitted, so nothing can be fitted at the wrong level. The row → image → patch nesting still governs every split. |
| §7.1 (Platt/temperature on a held-out split) | **Superseded** — §3. |
| §7.2 (Head B + Mahalanobis gate) | **Stands, promoted.** The gate becomes a first-class output field (§6.2) rather than an override. |
| §8's verdict vocabulary and per-card verdicts | **Superseded** — §6. |
| §8's `STRIPPED`-excluded-not-0.5 rule | **Stands**, re-expressed as card silence (§6.3). |
| §8's "thresholds are parameters" | **Stands.** Only the default's provenance changes (§7). |
| §8.1 (provenance quarantined in its own card) | **Stands, reinforced.** `DECLARED_SYNTHETIC` is a separate verdict precisely so a declaration is never averaged with an inference. |
| §9 E1's success criterion (knee on accuracy ≥0.90) | **Superseded** — knee on AUC ≥0.90, balanced accuracy reported alongside (§5). |
| §9 E5 ("AUC, not just accuracy") | **Stands, absorbed.** AUC stops being one experiment and becomes the primary metric of all of them. E5 remains as the val-split measurement. |
| §9 E1/E2 eval composition | **Corrected** — §5. This is a defect fix, not a change of intent. |
| Everything else | **Stands.** The claim, the data constraints, the patch design and the frozen backbone are untouched. |

## 10. Discarded alternatives

**Keep Platt and add ECE to verify it.** The research-doc-faithful option. Rejected on data: ECE on tens of AI images is uninterpretable, so verification would be theatre, and the thing being verified is fitted on the wrong distribution regardless.

**Keep Platt and refit it per N-shot draw on target-generator images.** This would put the calibrator on the right distribution — and would spend labelled target images outside the stated N, corrupting the one number the demo exists to produce. Rejected outright.

**Balance E1's eval set by subsampling reals for all metrics.** Simpler to narrate — one eval set, every metric on it. Rejected because it discards ~420 of ~450 real images and inflates the variance of AUC and FPR, which do not need balance. Balance is applied only where the estimator requires it (§5).

**Suppress the verdict under `UNKNOWN_SOURCE`.** More conservative, and briefly preferred. Rejected: it hides the gate's most informative output, and the Midjourney case — right answer, correctly disclaimed — is a better demonstration of the system's self-knowledge than a blank field (§6.2).

**Weighted or evidence-count-aware fusion.** The correct fix for §6.4's two defects, and the improvement `round2-drills.md` names. Rejected for this build on time: it is new behaviour the day before the deadline, and the defects are now documented rather than latent.

**Adopt Plurall's verdict vocabulary anyway, for surface familiarity.** Rejected — §12.

## 11. Risks accepted

| Risk | Status |
|---|---|
| No calibrated probability is emitted, and their product ships one | **Deliberate.** The honest position is that we measured what we could support and named it accurately. §8's list is the answer to "why not," and it is a stronger answer than a number fitted on COCO_AI. |
| TPR rests on tens of AI images per generator | **Real, and the binding limit.** Reported with `n_ai` attached. More generated eval images is the highest-value remaining data action. |
| `α` is reported without E3's re-encode control if S9 is cut | Real. [`plan-c-source-verification.md`](../reference/2026-08-02-plan-c-source-verification.md) §3.2 makes it a precondition; the caveat travels with the number rather than being dropped. |
| A new verdict vocabulary diverges from the spec the interviewer wrote | **Deliberate** — §12. |
| §6.4's fusion defects ship unfixed | Accepted and documented. Known-and-stated beats fixed-at-midnight. |
| The balanced subsample for `α` makes it optimistic vs. a real deployment batch | Real, stated at the point of reporting (§5). |

## 12. Relationship to Plurall's spec

Their vocabulary and thresholds are `[stated]` facts about *their* product (`notes.md:54-57`). Nothing here revises them, and nothing here changes the P1 drill: `notes.md:134` and [`../dev/round2-drills.md`](../dev/round2-drills.md) D1 specify implementing **their** fusion/verdict function from **their** spec, twice, second time from a blank file. That drill is unchanged and remains the highest-EV hour in the prep plan.

What changes is that our own system stops borrowing a vocabulary derived from a different set of observables. That divergence is an asset in the room, not a liability: being able to say *"your spec has five verdicts; my model can only honestly support four plus a reliability flag, and here is which of your five it cannot reach and why"* is a specific, defensible critique of a product surface — which is what `round2-drills.md` §D1 identifies as the strongest available contribution on their own product.

The three concrete versions of that critique, all now on record here:

1. **`AUTHENTIC` is unreachable** for any detector of this class. Absence of synthesis evidence is not evidence of capture (§8).
2. **A missing card is not a neutral card**, and a mean-based fusion cannot express the difference (§6.3, §6.4) — already identified in `round2-drills.md` D1.
3. **The score and the model's entitlement to it are orthogonal**, and a single verdict field cannot carry both (§6.2).
