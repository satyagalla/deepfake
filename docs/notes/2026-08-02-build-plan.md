# Build Plan — 2026-08-02 → Monday 2026-08-03

Execution plan for [`../decisions/0004-adaptation-hypothesis-demo-build.md`](../decisions/0004-adaptation-hypothesis-demo-build.md). Edited in place as stages complete.

**Environment:** A100 40GB on Colab Pro. Not local. Data on Colab local disk (`/content`), features persisted to Drive via `DEEPFAKE_DATA_ROOT`.

---

## 0. The cap, and why it is written first

`notes.md` §Prep plan is explicit: **the live-coding round is the actual round.** P0 (own-repo fluency) and P1 (live-coding drills) are the highest-EV hours available, and P3 — this build — is hard-capped because *"polish beyond that is negative-value time — it comes straight out of P0/P1."*

**Rules:**

- **Hard cap: 7 hours.** When it is spent, the build stops wherever it is. A working small thing beats an unfinished large one (`0003` §7).
- **Stages are cut from the bottom, never the middle.** The cut order is fixed in §3 in advance so it is not renegotiated at hour six under pressure.
- **S1–S5 are the demo.** Everything after S5 is upside.
- **After every stage that touches data: open the images and look at them.** Round 1's root cause was not the dataset choice — it was *not inspecting the data after processing* (`notes.md` line 79). This is the one rule with no exceptions.

**Dual-purpose stage:** S7 (fusion → verdict → confidence) is simultaneously the build's card layer **and** P1 drill #1 from `notes.md`, which specifies exactly that function from Plurall's spec. Building it is practising it. It is the best hour in the plan.

---

## 1. Invariants

Assert these in code. Do not verify them by eye once and assume they hold.

| # | Invariant | Where | Why |
|---|---|---|---|
| I1 | **Exactly N=16 patches per image, every image** | extraction loop | Bag size correlates with resolution correlates with label — leaks through mean, max and top-k alike (`0004` §6.3) |
| I2 | **Split at the COCO_AI row level** | split builder | One row = 1 real + 6 fakes. Splitting within a row leaks the pairing, which is the corpus's most valuable property (`bottlenecks.md` §7) |
| I3 | **No image's patches straddle train/val** | split builder | Nesting is row → image → patch |
| I4 | **All classes re-saved JPEG q95** | `data/download.py:80-82` (already implemented) | Container↔label shortcut priced at ~11 points (§2.5) |
| I5 | **CLIP normalization constants, not ImageNet's** | extraction | `model/demo.py:36` uses ImageNet's. Carrying that over is a silent, plausible-looking bug |
| I6 | **L2-normalize features before any fit** | extraction | SSAFE does; required for Mahalanobis conditioning (`0004` §7.2) |
| I7 | **Midjourney never enters training at any N** | split builder | It is the honesty anchor. Losing it costs the only clean 0-shot number |
| ~~I8~~ | ~~**Calibration fitted on image-level aggregated scores**~~ | ~~Head A fit~~ | **Retired 2026-08-02** ([`0005`](../decisions/0005-measurement-and-verdict-semantics.md) §3): nothing is fitted post-hoc, so nothing can be fitted at the wrong level. The row → image → patch nesting it protected is already carried by I2/I3. Replaced by the two below. |
| **I8** | **No post-hoc fitted stage between Head A's score and the verdict** | Head A fit, card scoring | A calibrator fitted on COCO_AI val and applied to a live 1024px Gemini upload is a plausible-looking wrong number, not a probability (`0005` §3). Card scores come from the classifier's own `predict_proba` |
| **I9** | **Every cut-point metric is reported on a class-balanced basis, and `α = E[z]` is computed on a balanced subsample** | `evaluate_head_a` | E1's eval set is ~450 real against *tens* of AI, so "always real" scores ≥90% and a flat curve would falsify the claim by artifact (`0005` §5). AUC/TPR/FPR need no balancing and get the full set |

---

## 2. Stages

### S1 — Data (~45 min) · **highest risk, do first**

1. **Fix `data/download.py:68`.** Currently `ds.select_columns(["caption", "coco_image", "dalle_image"])` — discards five of six generators. Take all seven columns.
2. **Drop the person-caption filter.** `PERSON_KEYWORDS` / `caption_has_person()` existed to serve the face crop, which `0003` §4.1 removed.
3. Download ~3,000 rows → ~3,000 real + ~18,000 fake.
4. **Measure the native resolution histogram per column.** This is `bottlenecks.md` §9.1, still `[second-hand]`, and every downscale figure in `0004` §6.1 inherits its status. It also **determines whether N=16 is right** — if the corpus is nearer 270px than 480px, patches and whole images nearly coincide and N should drop.
5. Write to Colab local disk, not mounted Drive. 21k small files over Drive FUSE is the classic Colab time sink.

**Checkpoint:** open 10 images per column and look at them. Confirm resolutions match the histogram. Confirm the real/fake pairing is intact for a sampled row.

**Abort condition:** if the download is not working at 60 min, fall back to `--n-pairs 1000`. The N-shot claim does not need 3,000 rows.

### S2 — Self-generated N-shot pool (~45 min, start in background during S1)

- ~150–300 images across Gemini + GPT-Image, prompted with COCO captions (keeps content domain matched — the standing pairing constraint).
- ~20/generator on **off-domain** prompts (E4).
- ~20/generator through the **web UI** rather than the API (container control — API PNG vs web WebP is the discriminating test `bottlenecks.md` §4.2 asks for, since SynthID passes the naive container control by design).
- Re-save everything at JPEG q95 alongside the rest (I4).

**Checkpoint:** confirm the two containers actually differ. If the web UI returns the same format as the API, the control is void and should be recorded as such rather than reported.

### S3 — Feature extraction (~45 min)

Frozen CLIP ViT-L/14 @224, fp16, batch 256–512.

- 16 native-resolution 224 patches per image (I1) **+ 1 whole-image resized view**.
- CLIP's own normalization (I5); L2-normalize outputs (I6).
- Cache per-patch features — the aggregator ablation (E6) needs them. ~345k × 768 × fp16 ≈ **530 MB**.
- **Second arm: the standard resize-and-centre-crop pipeline** (E7). One extra pass, ~5 min. This is UniversalFakeDetect's exact setup and serves as the reproducible control.
- Persist features to Drive (small, fast). Leave images on local disk.

Expected GPU time is a few minutes; JPEG decode dominates. If it is slow, the bottleneck is I/O, not the A100.

**Checkpoint:** assert I1 across the whole array. Confirm feature norms are 1.0 after L2.

### S4 — Heads (~30 min)

- **Head A:** logistic regression, real vs AI, on concatenated patch + whole features. **No calibration stage** (I8, [`0005`](../decisions/0005-measurement-and-verdict-semantics.md) §3) — raw `decision_function` for metrics, the classifier's own `predict_proba` where a `[0,1]` card score is needed.
- **Head B:** multiclass over `{real, sd21, sdxl, sd3, sd35, dalle, gemini, gptimage}`.
- **Mahalanobis gate:** per-class means, **one pooled covariance with Ledoit-Wolf shrinkage** (never per-class — singular at 768-d against ~150 samples).

Seconds to fit. This is the point at which the frozen-backbone decision pays for itself.

### S5 — E1, the headline (~30 min)

N-shot adaptation curve: **AUC** on a held-out generator vs N ∈ {0, 5, 10, 20, 30, 50, 100} images from it. Multiple random draws per N; plot the spread, not just the mean.

**The knee is the finding, wherever it is.** A curve with no knee falsifies the claim (`0004` §9) and gets reported as such.

**Metric set** ([`0005`](../decisions/0005-measurement-and-verdict-semantics.md) §4): AUC and balanced accuracy plotted; TPR, FPR, `n_ai`, `n_real`, `alpha`, `accuracy_at_alpha` and the oracle ceiling in the JSON. Knee on **AUC ≥ 0.90**. Precision is *derived* at a named prevalence, never stored (§4.3).

**Fix before running:** `run_e1`'s eval set is ~450 real val images against *tens* of selfgen eval images (I9). Unfixed, "always real" scores ≥90%, the curve goes flat, and the build falsifies its own claim on set composition. Balance applies to cut-point metrics and to `α` only — AUC, TPR and FPR keep the full set.

**S1–S5 complete = the demo exists.** Everything below is upside.

### S6 — Free experiments (~30 min)

All come off the S3 cache at zero marginal GPU cost:

- **E5 — AUC.** `bottlenecks.md` §4.1: AUC has never been computed on this project. First measurement, and it separates "lost separability" from "misaligned threshold" — the open question in `plan-c-source-verification.md`. Now on a genuinely held-out val split: the calibrator that was fitted on it is gone (`0005` §3). Read against `plan-c` §5's rule — **AUC ≳0.75 with accuracy ≈50% → threshold problem; AUC ≈0.5 → Chameleon regime.**
- **E2 — Midjourney 0-shot.** The clean held-out number.
- **E6 — aggregator ablation.** mean / max / top-k / trimmed, plus the flat-patch filter.
- **E7 — preprocessing arms.** Native patches vs standard resize.

### S7 — Cards, fusion, verdict (~60 min) · **also P1 drill #1**

**Two different artifacts, and they must not be conflated** — [`0005`](../decisions/0005-measurement-and-verdict-semantics.md) §12:

**(a) The drill.** Plurall's schema `{dimension, label, score, verdict, detail}` → fused score → verdict (`SYNTHETIC` ≥0.85, `SUSPICIOUS` ≥0.5, else `AUTHENTIC`) → confidence, handling a `STRIPPED` card and disagreeing dimensions. **Their spec, unchanged** (`notes.md:134`). Type it from scratch rather than generating it. That is the drill, and it is still the best hour in the plan.

**(b) Our system's card layer**, which uses a vocabulary derived from what *this* model can observe (`0005` §6):

- **AI Model** (Head B), **Spectral** (radial FFT), **EXIF** (EXIF + C2PA, deterministic) — built.
- **Diffusion**, **Temporal**, **Web Intelligence** — honest stubs with declared scope.
- **Cards carry no verdicts.** A card has a `score` or a `silent_because`. `STRIPPED` / `NOT_APPLICABLE` / `PLAUSIBLE` were reasons for silence, not verdicts. The exclusion rule is unchanged: absent metadata is excluded from the fused score, never scored 0.5.
- **Two output fields:** `verdict ∈ {DECLARED_SYNTHETIC, LIKELY_SYNTHETIC, WEAK_EVIDENCE, NO_EVIDENCE}` and `reliability ∈ {IN_DISTRIBUTION, UNKNOWN_SOURCE}`. `AUTHENTIC` is dropped as unreachable.
- **Mahalanobis abstention accompanies the verdict, it does not suppress it.** Midjourney at 0-shot should read `LIKELY_SYNTHETIC / UNKNOWN_SOURCE`.
- Thresholds stay **parameters** (Detection Settings), but the defaults are **fitted from the real val split** — 99th percentile = 1% FPR for `LIKELY_SYNTHETIC`, 90th = 10% FPR for `WEAK_EVIDENCE` (`0005` §7).
- **Known open defects** (`0005` §6.4), documented not fixed: the unweighted mean makes the top verdict unreachable when the EXIF card fires on camera metadata, and gives the weakest card half the vote when API output silences EXIF.

### S8 — Gradio demo (~45 min)

New path, not `model/demo.py`. Note what the old one does: MTCNN crop, and **`return (None, None, None, None)` when no face is detected** — a faceless upload produces no prediction at all. The new path has no face crop, so this fails by construction rather than by fix.

Must display: fused score, **`verdict` and `reliability` side by side** (`0005` §6.2 — the gate is shown, not used to blank the answer), all six cards with their score or their `silent_because` reason, and **E1's curve on screen** so a live result lands on a chart that was already visible.

### S9 — E3, degradation ladder (~30 min)

Score each card across re-encode / rescale / re-render. This is the **watermark measurement**: a learned score that degrades identically to the provenance card is reading SynthID rather than synthesis. Gemini vs GPT-Image differ in watermarking policy and provide the natural experiment for free.

Highest product relevance of anything in the plan, which is why it hurts to have it this low. It is here because it is not needed for the headline claim.

### S10 — E4, off-domain prompts

Is the model reading synthesis or COCO's content distribution? Accuracy drop at equal N.

---

## 3. Cut order, fixed in advance

Cut from the bottom: **S10 → S9 → S8 → S7 → S6.**

- **S8 cut** → demo from a notebook. Less impressive, costs nothing evidential.
- **S7 cut** → the cards story becomes verbal, backed by the spec. Painful, because it is the product-surface answer *and* the drill — but it is recoverable in conversation in a way a missing headline number is not.
- **S6 cut** → should never happen; it is minutes off a cache that already exists.
- **S1–S5 are never cut.** Without them there is no claim.

**If S1 fails outright** (data will not land): fall back to a plan built on `sd21 → sdxl` N-shot transfer within whatever columns did download. The adaptation claim is generator-agnostic — it does not require Gemini to be *demonstrated*, only to be *stated*. Then say so, and let the live upload be the 0-shot point.

---

## 4. Status

| Stage | Status |
|---|---|
| S1 Data | Code written (`data/download.py`, `data/resolution_stats.py`) -- not yet run |
| S2 Generation | Code written (`data/selfgen_prompts.py`, `data/selfgen_organize.py`) -- generation itself is manual, not yet run |
| S3 Extraction | Code written (`probe/patches.py`, `probe/extract.py`) -- not yet run |
| S4 Heads | Code written (`probe/heads.py`, `probe/fit_production.py`) -- not yet run |
| S5 E1 curve | Code written (`probe/experiments.py::run_e1`) -- not yet run |
| S6 Free experiments | Code written (E2/E4/E5/E6/E7 in `probe/experiments.py`) -- not yet run |
| S7 Cards/fusion | Code written (`probe/cards.py`, `probe/spectral.py`) -- not yet run |
| S8 Gradio | Code written (`probe/demo.py`) -- not yet run |
| S9 Degradation ladder | Code written (`probe/degradation.py`) -- not yet run |
| S10 Off-domain | Code written (E4, folded into `probe/experiments.py`); notebook wiring done (`adaptation_probe.ipynb`) |

### `0005` rework — partially applied, tree does not currently run

Spec: [`2026-08-02-0005-rework-spec.md`](2026-08-02-0005-rework-spec.md). **Start there, not here** — it carries the exact remaining diff.

| File | State |
|---|---|
| `config.py` | ✅ `VERDICT_*` thresholds + FPR budgets replace `FUSION_*` (§7) |
| `probe/heads.py` | ✅ Calibrator gone; `score_head_a` / `prob_head_a` replace `predict_proba_head_a` (I8) |
| `probe/thresholds.py` | ✅ New — the §4 metric set, `α = E[z]`, oracle ceiling, `precision_at_prevalence`, FPR-budget cut points |
| `probe/spectral.py` | ✅ Calibration-free fit; val stays held out |
| `probe/cards.py` | ✅ Rewritten — `verdict` + `reliability`, cards carry `score`-or-`silent_because` (§6) |
| `probe/experiments.py` | ✅ New metric set, `α` on a balanced subsample (I9), two-line `plot_e1`, knee on AUC ≥ 0.90 |
| `probe/fit_production.py` | ⚠️ **Half-applied — `NameError` on `val_rows`** |
| `probe/degradation.py` | ❌ **`ImportError`** — still imports `predict_proba_head_a` |
| `probe/demo.py` | ❌ **`AttributeError`** — still reads `card.verdict` / `fusion.abstained` |
| `adaptation_probe.ipynb` | ❌ Prose only, non-blocking |

Nothing is committed (`probe/` is untracked), so there is no clean revert and nothing is at risk from finishing forward.

**The smoke-test note below predates the `0005` rework and no longer holds for the three files above.** The six completed files parse; the tree as a whole does not import. Re-run the smoke test after the spec is applied.

All modules pass syntax checks and were smoke-tested locally against a synthetic feature cache
matching `extract.py`'s exact on-disk schema (patch sampling invariants, pooling, Head A/B fit +
predict, Mahalanobis gate abstention, EXIF/C2PA card, fusion, degradation-ladder transforms, E1-E6
end to end) -- see the session that authored this update. **Nothing has run against real data or
the real CLIP backbone yet** -- that requires Colab (`adaptation_probe.ipynb`), per the Environment
note at the top of this doc. Nothing is on disk as of 2026-08-02: `checkpoints/`, `dataset/`,
`data_raw/`, `outputs/`, `runs/`, and the new `probe/`-namespaced data root are all absent `[measured]`.
