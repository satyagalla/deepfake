# Investigation: Resolution-Swap Counterfactual Probe — Findings on the Upscale-Artifact Shortcut Hypothesis

Follow-up to [`2026-07-26-upscale-artifact.md`](2026-07-26-upscale-artifact.md) and
[`2026-07-27-fft-srm-template-swap-probe.md`](2026-07-27-fft-srm-template-swap-probe.md). The
latter built a counterfactual probe that never varied `rgb`, so it couldn't reach the spatial
branch or the core upscale-artifact hypothesis. This doc implements
[`docs/reference/resolution_swap_probe.md`](../reference/resolution_swap_probe.md)'s spec
(`model/resolution_swap_probe.py`, `debug.ipynb`), which fixes that gap by manipulating `rgb`
pixels directly, and records what it found against the trained checkpoint.

## What was built / run

`model/resolution_swap_probe.py`: for a held-out image, downsample-then-upsample its `rgb` via
plain `PIL.BILINEAR` (no anti-aliasing, matching MTCNN's own `crop_resize`) to simulate a higher
effective face-crop upscale factor, matched to another class's typical factor; recompute
`fft_mag`/`srm_residual` from the modified `rgb` so every model input stays internally consistent;
compare against a matched `r=1` no-op resize round-trip (isolates the deliberate factor increase
from generic resize-roundtrip noise).

Run config, held constant across every result below: checkpoint `best_model.pt`, `split="val"`,
`PROBE_N=30` held-out images per achievable source class (`real`, `deepfake` — `edited` is never a
source since the round-trip can only add blur and it has the highest median factor). Indices are
deterministic (`_class_indices()` + a fixed prefix slice), so **the same 30 images per source class
were used in every run below** — only the injected `MEDIAN_UPSCALE_FACTOR` values differed. That
turns the comparison across runs into a paired one, not independent resampling.

Two runs, differing only in where `MEDIAN_UPSCALE_FACTOR` came from:

- **Run A**: the original ad hoc estimate from `2026-07-26-upscale-artifact.md` (n=5/8/19 detected
  faces/class, real=10.40x, deepfake=11.05x, edited=11.83x).
- **Run B**: remeasured at n=200 detected faces/class (this investigation, `debug.ipynb`'s "Measure
  `MEDIAN_UPSCALE_FACTOR` with a larger sample size" section) — same proxy methodology
  (`MTCNN.detect()` with `select_largest=True`, matching `data/face_filter.py`'s config, so
  `boxes[0]` is exactly the face `crop_resize()` would use), reusing `caption_has_person()` and
  `find_casia_tampered()` rather than reimplementing them.

## Findings

### Remeasured upscale factors (n=200/class) vs. the original n=5/8/19 estimate

| class | original median (n=5/8/19) | remeasured median (n=200) | remeasured range |
|---|---|---|---|
| real | 10.40x | 8.18x | 0.87x – 39.59x |
| deepfake | 11.05x | 8.59x | 1.10x – 30.84x |
| edited | 11.83x | 12.43x | 2.45x – 32.90x |

The `real < deepfake < edited` ordering that `ACHIEVABLE_TARGETS` depends on held at the larger
sample (checked with an explicit assertion before reuse), but the absolute values shifted
substantially — real/deepfake's medians dropped ~17-22%, edited's rose ~5%. The remeasured range
extending below 1x (a detected box already larger than 380px, i.e. would be *downscaled*, not
upscaled) shows the original n=5/8/19 estimate wasn't just imprecise, it landed on an
unrepresentative slice of a much wider distribution.

### Resulting resize ratios, both measurements

| pair | ratio r (original medians) | ratio r (remeasured, n=200) |
|---|---|---|
| real→deepfake | 1.063 | 1.051 |
| deepfake→edited | 1.071 | 1.447 |
| real→edited | 1.138 | 1.520 |

### Probe results, both runs (same 30 images per source class each time)

| pair | mean ΔP(target), Run A | flips, Run A | mean ΔP(target), Run B | flips, Run B |
|---|---|---|---|---|
| real→deepfake | −0.0275 | 0/30 | −0.0257 | 0/30 |
| deepfake→edited | +0.0001 | 0/30 | +0.0003 | 0/30 |
| real→edited | +0.0721 | 2/30 | +0.0613 | 1/30 |

(mean ΔP(target) = swap prediction's P(target_class) minus its matched `r=1` control's
P(target_class); flips = images where `pred` switched to `target_class` under the real swap but
not the control.)

## Interpretation

Two of the three achievable pairs are clean, stable nulls in both runs: `real→deepfake` stays
negative (the wrong direction for the shortcut hypothesis) and `deepfake→edited` stays pinned at
~0, regardless of which median estimate was used to compute the injected ratio.

The third pair, `real→edited`, is the only one with a non-trivial effect in either run — but the
remeasurement is what actually stress-tests whether that effect is a genuine target-specific
interaction or just "this pair happened to get the biggest injected blur." It weighs against the
latter, on two independent grounds:

1. **Cross-pair, at matched perturbation size.** Under the remeasured medians, `deepfake→edited`'s
   ratio (1.447) sits close to `real→edited`'s (1.520) — within ~5% of each other, a big change
   from Run A where `real→edited`'s ratio (1.138) was clearly the largest of the three. If the
   `real→edited` effect were simply "biggest injected blur gets the biggest shift," `deepfake→edited`
   should now show a comparable effect. It doesn't — it stayed at +0.0003, roughly two orders of
   magnitude smaller than `real→edited`'s +0.0613.
2. **Within-pair, dose-response.** `real→edited`'s own ratio grew from 1.138 to 1.520 (+34%)
   between Run A and Run B — more blur injected into the *same* 30 images — yet its measured effect
   shrank slightly (0.0721 → 0.0613, 2 flips → 1 flip) instead of growing. A purely
   magnitude-driven effect would predict the opposite.

Both observations point the same direction: whatever produces the `real→edited` shift is not
simply proportional to how much blur was added. That weighs against the "biggest-perturbation
confound" explanation raised for this pair in the prior debugging session, and is more consistent
with the *interaction* hypothesis `2026-07-26-upscale-artifact.md` revised toward — some
real-content-specific interaction between the added blur and that class's pixel statistics, not a
generic shortcut or pure magnitude effect.

That said, this remains a small effect in absolute terms in every run: mean ΔP(target) of 0.06-0.07
and 1-2 flipped predictions out of 30 (93-97% of images kept their originally predicted class). It
is not evidence the model broadly or reliably exploits this cue.

## Status

**Rules out** (replicated across two independent upscale-factor measurements spanning a >30% swing
in injected perturbation size): a general, uniform resolution-magnitude shortcut operating across
all three achievable directions. Two of three pairs show no meaningful gap between the real swap
and its matched `r=1` control in either run.

**Also rules out** (this investigation's main new finding, not just a replication): the explanation
that `real→edited`'s effect is an artifact of it happening to receive the largest injected
perturbation. Both the cross-pair and within-pair comparisons above are inconsistent with a
magnitude-driven account of that one non-null result.

**Does not confirm**: that the `real→edited` effect reflects the spatial branch relying on this cue
as a load-bearing shortcut. The effect is small, isolated to one of three directions, and this
remains a diagnostic probe against one checkpoint — no retraining or targeted ablation of the cue
itself has been performed.

## Limitations

- `PROBE_N=30` per source class in every run — modest, though the paired same-images design
  (identical 30 images across both runs, only the injected ratio differs) makes the within-pair and
  cross-pair comparisons above more reliable than the raw N alone would suggest.
- The n=200 upscale-factor remeasurement uses the same proxy as the original n=5/8/19 measurement
  (`MTCNN.detect()` box coordinates only, not the full `extract_face`/JPEG-resave path) — more
  samples, same underlying approximation.
- `edited`'s remeasurement is still CASIA-only (`find_casia_tampered()` against `EDITED_SRC`), no
  PS-Battles, consistent with the original investigation's scope.
- The double-bilinear-pass resize round-trip (per `resolution_swap_probe.md`) approximates but does
  not pixel-exactly reproduce a genuinely different original face-crop size.
- Only two ratio points were compared for `real→edited` (1.138 and 1.520), not a full dose-response
  curve — enough to make a monotonic magnitude-driven explanation look unlikely, not enough to rule
  out a non-monotonic one.
- Still checkpoint-diagnostic only, consistent with every probe in this line of investigation so
  far — no retraining or targeted ablation of the resampling cue has been performed.

## Recommended next steps

1. If the `real→edited` residual is worth pinning down further: a finer dose-response sweep within
   just that pair (3-4 ratio points between 1.0 and ~1.5) would show whether the effect stays flat
   across ratio (supports "not the magnitude, something else content-specific") or moves in some
   other non-monotonic pattern.
2. Otherwise, since no general shortcut was found, the tiered ablation plan's remaining item
   (resolution-matched retrain) is lower priority than originally scoped — reasonable to move to the
   already-known cross-generator generalization failure (README: confidently wrong on
   Gemini/gpt-image-1 images) as the next investigation, per `CLAUDE.md`'s "Next" list.
