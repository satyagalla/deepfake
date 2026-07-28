# Investigation: FFT/SRM Class-Template Swap Probe — Findings and Scope Gap

Follow-up to [`2026-07-26-upscale-artifact.md`](2026-07-26-upscale-artifact.md), whose tier-2
next step read: *"counterfactual probe on the trained checkpoint — swap which class gets which
upscale factor (e.g. take a real photo, force it through deepfake-typical resampling parameters)
and see if the prediction follows content or resampling."* This doc records what was actually
built against that description (`model/counterfactual_probe.py`, `debug.ipynb`), what it found,
and — the main point of this report — why it does not actually settle the hypothesis it was meant
to test.

## What was built

`model/counterfactual_probe.py`:

- `extract_templates()` — averages `fft_mag` and `srm_residual` (the spectral/noise-residual
  branch inputs, both computed from an image's own RGB pixels in `ForgeryDataset.__getitem__`)
  over `TEMPLATE_N` samples per class, producing one "artifact template" per class.
- `run_probe()` — for held-out images, keeps the image's true `rgb` fixed and substitutes its
  `fft_mag`/`srm_residual` with another class's averaged template (`fft`, `srm`, or `both`
  variants), records the model's prediction under every swap alongside the unmodified baseline.
- `summarize()` — mean change in `P(swap_class)` vs. baseline, split into `same_class` (swap
  target equals the image's true class — a noise floor from using an averaged template instead of
  the image's own exact values) vs. `cross_class` (swap target differs from true class). `cross -
  same` was treated as the shortcut-signal readout.
- `swap_target_spread()` (added this session) — a second check comparing, per image, the three
  swap-target predictions (real-template vs. edited-template vs. deepfake-template) *against each
  other* rather than against baseline. Small spread means which class's template was used barely
  matters.

## Findings (N=10 probe images/class, N=50 template images/class)

| variant | same_class (noise floor) | cross_class | cross − same | mean spread across swap targets |
|---|---|---|---|---|
| fft | 0.0013 | -0.0002 | -0.0015 | 0.0008 |
| srm | -0.0512 | 0.0277 | 0.0790 | 0.0029 |
| both | -0.0446 | 0.0248 | 0.0694 | 0.0034 |

- `fft`: inert on both measures — no evidence the spectral-magnitude branch carries a
  swappable class signal at all.
- `srm`/`both`: a real, non-trivial effect on `cross - same` (~0.07-0.08), but the spread across
  *which* class's template was substituted is 25-90x smaller than that effect. Per-image spot
  checks confirm this directly: for a given image, swapping in `real`'s, `edited`'s, or
  `deepfake`'s SRM template produces nearly identical output, regardless of source class. The
  effect is real but not class-targeted — it reads as "any averaged/smoothed SRM residual,
  replacing this image's own raw one, shifts the prediction (usually toward `real`)," not "the
  model recognizes class X's specific artifact fingerprint."
- This was run at `PROBE_N=10`; a larger run (`PROBE_N=30`) was intended but never actually landed
  in the saved notebook — the numbers above are still the small debug configuration, not a
  higher-confidence version of it.

## Why this doesn't reach the original hypothesis

Two structural gaps, not just a sample-size issue:

1. **`rgb` is never modified, in any condition.** `fft_mag` and `srm_residual` are both
   deterministic functions of an image's own `rgb` pixels (`ForgeryDataset.__getitem__`,
   `model/dataset.py:36-54`). The blur artifact under investigation in
   `2026-07-26-upscale-artifact.md` is a *pixel-domain* phenomenon — it's baked into `rgb` itself.
   Every condition tested here (baseline and all 9 swaps) feeds the model the image's own true,
   unmodified pixels; only the derived channels change. The spatial branch (EfficientNet-B4 on
   `rgb`), which carries real, non-negligible gate weight in every sampled prediction (~0.26-0.37,
   roughly on par with the other two branches), is completely unprobed. If the model is
   shortcutting on blur magnitude/pattern through the spatial branch directly, this test cannot
   detect it.
2. **Averaging over `TEMPLATE_N` images can hide a real effect, not just rule one out.** The
   revised hypothesis in `2026-07-26-upscale-artifact.md` is specifically about an *interaction*
   between the resampling artifact and each image's own content statistics — not a fixed additive
   per-class fingerprint. If that interaction is content-dependent, averaging 50 images into one
   template could wash it out before the swap is ever applied. A null result here is consistent
   with "no shortcut," but equally consistent with "a shortcut exists but doesn't survive being
   averaged into a template."

## Status

**Rules out** (moderate confidence, small N): a targeted per-class shortcut specifically in the
*averaged* `fft_mag`/`srm_residual` channels, transferable across images independent of content.

**Does not confirm or rule out**: the core upscale-artifact shortcut-learning hypothesis from
`2026-07-26-upscale-artifact.md`, particularly any version of it operating through the spatial/RGB
branch, since `rgb` was never varied by this probe.

## Next step

See `docs/reference/resolution_swap_probe.md` for the spec of a properly-scoped follow-up that
manipulates `rgb` pixels directly (re-simulating a different effective upscale factor) instead of
swapping derived-channel templates.
