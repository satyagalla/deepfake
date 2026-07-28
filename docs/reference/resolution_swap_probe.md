# Resolution-Swap Counterfactual Probe (spec)

Originally written as instructions-only (no code), to be implemented as a follow-up coding task,
the same way `docs/reference/data_download.md` and `docs/reference/model_code.md` preceded
`data/*.py` and `model/*.py`. **Implemented**: `model/resolution_swap_probe.py`, with an
interactive companion section in `debug.ipynb` mirroring the existing
`model/counterfactual_probe.py` / `debug.ipynb` pairing. Findings:
`docs/investigations/2026-07-27-resolution-swap-probe.md`.

## What this tests

The hypothesis in `docs/investigations/2026-07-26-upscale-artifact.md`: does the trained model use
MTCNN's bilinear-resize blur artifact (present in all three classes, at differing typical
magnitudes) as a classification shortcut? `docs/investigations/2026-07-27-fft-srm-template-swap-probe.md`
found the existing probe can't answer this — it never varies `rgb`, only derived `fft_mag`/
`srm_residual` channels. This probe fixes that by manipulating `rgb` pixels directly, so the
spatial branch is actually exercised.

Chosen design: **targeted swap to another class's measured typical upscale factor** (not a
continuous severity sweep — rejected as unpredictable/risky to interpret without more setup).

## Why this doesn't need the original raw source images

`manifest.csv` only stores `path, class, split, source_dataset` — no face-box coordinates, no
link back to the raw pre-crop source image (`data/face_filter.py:287`). Re-detecting the exact
same face box on the original raw file and re-cropping at a different size is possible in
principle but fragile (depends on `data_raw/` still being populated, and MTCNN isn't guaranteed
to reproduce the same box). Instead, simulate the effect directly on the already-processed
380x380 image already in the dataset:

1. Take the image's existing `rgb` pixels (already one bilinear upscale pass, at its class's
   typical magnitude).
2. Downsample to a smaller side length using plain `PIL.BILINEAR` — matching MTCNN's own
   `crop_resize` method exactly (bilinear, unconditional, **no anti-aliasing** — a
   higher-quality/antialiased downsample would introduce a different artifact than the one under
   study, defeating the point).
3. Upsample back to 380x380, again with `PIL.BILINEAR`.

This compounds an extra blur pass on top of the image's existing blur, approximating "as if this
image's original detected face box had been smaller." It is an approximation, not a pixel-exact
reproduction of a genuinely different original crop size (see Limitations).

**One-directional constraint**: this can only push an image toward a *higher* effective upscale
factor (more blur) — it cannot undo blur already baked into the pixels to simulate a *lower*
factor. This constrains which swap directions are achievable.

## Computing the resize ratio

Per-image true upscale factor isn't recoverable (box coordinates aren't stored). Use each class's
*measured median* upscale factor from `2026-07-26-upscale-artifact.md`'s table as a stand-in for
"this image's own" factor — real: 10.40x, deepfake: 11.05x, edited: 11.83x. This is itself an
approximation built on a small, noisy sample (n=5/8/19 in the original measurement); carry that
uncertainty forward explicitly in any write-up rather than treating these numbers as precise.

Given the one-directional constraint, only these source→target pairs are achievable (source
median < target median): real→deepfake, real→edited, deepfake→edited. The reverse directions
(deepfake→real, edited→real, edited→deepfake) are not achievable this way and should not be
attempted.

For an achievable pair, ratio `r = target_median_factor / source_median_factor` (e.g.
real→edited: 11.83/10.40 ≈ 1.14). Intermediate size = `round(380 / r)`. Downsample 380→that size,
then upsample back to 380, both with `PIL.BILINEAR`.

## Recompute derived channels — don't inject foreign ones

After producing the blurred `rgb` tensor, recompute `fft_mag` via
`ForgeryDataset._fft_magnitude()` and `srm_residual` via `model.branches.SRMFilter()` — reuse
those exact functions from `model/dataset.py`, don't reimplement them. Also apply the same
`rgb` normalization (`IMAGENET_MEAN`/`IMAGENET_STD`) `ForgeryDataset.__getitem__` uses. Everything
fed to the model should stay internally consistent with how it was trained/evaluated, unlike the
prior probe's cross-channel template injection.

## Include a no-op control

Alongside the real factor-change condition, also run the *same* downsample/upsample round-trip
with `r = 1` (i.e., resize to 380 and back to 380, no actual size change) on the same images. This
isolates whether merely re-encoding through an extra bilinear round-trip shifts predictions on its
own (e.g. via subtle resampling/quantization noise), separate from the deliberate factor increase.
Without this control, any shift under the real condition can't be cleanly attributed to the
factor change itself.

## What to record per probed image

Mirror the existing probe's record shape for consistency: `path`, `true_class`, `baseline`
(unmodified `rgb`/`fft_mag`/`srm_residual` → pred + full probs + gate), then a
`resolution_swaps` dict keyed by achievable target class, each holding both the `r=1` control
prediction and the real factor-swap prediction (pred + full probs + gate for each).

## Aggregation

Per target class, mean change in `P(target_class)` between the real factor-swap and its matched
`r=1` control (not vs. raw baseline — the control is the correct reference point here, since it
isolates the resize-roundtrip-only effect). Also report the fraction of images where `pred`
actually flips toward the target class under the real condition but not the control.

## Interpretation

Evidence *for* a resolution-magnitude shortcut: `P(target_class)` rises meaningfully more under
the real factor-swap than under the matched `r=1` control, and/or `pred` flips toward the target
class for a non-trivial fraction of images. Because `rgb` is genuinely varied here, this would
implicate the spatial branch specifically — the thing the prior probe couldn't reach.

Evidence *against*: no meaningful gap between the real condition and the `r=1` control. Since this
version actually varies `rgb`, a null result here is a materially stronger "no shortcut" claim
than the prior probe's.

## Sample scope

Reuse `_class_indices()` and the val split as the existing probe does. Per achievable source
class, pull `PROBE_N` held-out images (disjoint from indices already used by
`model/counterfactual_probe.py`'s template/probe split isn't required for correctness here since
there's no shared template — just keep it for traceability/comparability across the two probes).

## Limitations to carry into the resulting investigation write-up

- The double-resize simulation approximates but does not pixel-exactly reproduce what a genuinely
  different original crop size would have produced (two compounded bilinear passes vs. one).
- Uses small-sample (n=5/8/19) class-median upscale factors as a per-image stand-in, not each
  image's actual native factor (unrecoverable from the processed dataset).
- Only tests the upward-achievable direction pairs (real→deepfake, real→edited,
  deepfake→edited) — says nothing about the unreachable reverse directions.
- No model was retrained here — this is still a diagnostic probe against the existing checkpoint,
  same as the prior one.
