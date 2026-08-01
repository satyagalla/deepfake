# Investigation: CASIA Authentic-Image Fingerprint Probe — Measuring the Corpus/Compression Fingerprint Behind `edited`

Follow-up to [`../decisions/0002-frozen-backbone-generalization.md`](../decisions/0002-frozen-backbone-generalization.md) §6.4 and §11. That doc argued the `edited` class's unusually strong performance (highest ROC-AUC of any class, 0.990; `edited↔deepfake` confusions near zero) is *consistent with* a learned corpus/compression fingerprint rather than a genuine manipulation signature — CASIA is a different corpus from COCO_AI (different cameras, JPEG histories, native resolutions), and nothing in the pipeline equalizes that beyond framing/output-resolution (MTCNN crop) and final compression (q95 re-save). §6.4 called this "not proof" and §11 listed measuring it, via a held-out second `edited` corpus, as an open item. This doc runs that measurement.

## What was built / run

`model/casia_authentic_probe.py` + its interactive companion in `debug.ipynb`: CASIA v2.0's own `Au_` (authentic) images — excluded from the `edited` class by `data/face_filter.py`'s `find_casia_tampered()`, so never downloaded into the manifest and never seen by training or eval — face-cropped with the identical MTCNN config used to build the training set (`image_size`/`margin`/`select_largest`/`post_process=False`), then re-encoded at quality=95 (matching `face_filter.py`'s save step) before building `rgb`/`fft_mag`/`srm_residual` exactly as `ForgeryDataset.__getitem__` does, and run through the trained checkpoint (`best_model.pt`, macro-F1 0.9079).

Because these images are genuinely unmanipulated but share CASIA's corpus/compression/sensor identity with `edited`'s CASIA half, any `edited` prediction on them cannot reflect a manipulation cue — it can only be the fingerprint.

Run config: `CASIA_N=1000` source images sampled (`seed=42`, per `config.SEED`) from the 7,491 images `find_casia_authentic` identified as `Au_`. MTCNN detected faces in 161 of them (839 rejected, 83.9%) — a far higher rejection rate than typical for this pipeline, likely because CASIA's authentic set skews toward non-portrait content; not independently confirmed (see Limitations).

## Findings

`n=161` probed authentic CASIA face crops:

| class | pred_count | pred_fraction | mean_prob |
|---|---|---|---|
| real | 126 | 0.783 | 0.746 |
| edited | 28 | 0.174 | 0.199 |
| deepfake | 7 | 0.043 | 0.055 |

Baseline comparison, from the val confusion matrix already recorded in `0002` §5.1:

| comparison | rate |
|---|---|
| true `real` (COCO_AI, val) → predicted `edited` | 8/291 = 2.7% |
| CASIA authentic (this probe, never trained on) → predicted `edited` | 28/161 = **17.4%** |
| true `edited` (val, CASIA+PS-Battles mix) → predicted `edited` (recall) | 145/162 = 89.5% |

## Interpretation

17.4% is roughly **6.4x** the 2.7% rate at which the model false-positives `edited` on genuine COCO_AI real images. CASIA authentic images share no manipulation signal with `edited` — the only thing they share with `edited`'s CASIA half is corpus/compression/sensor identity. A gap this size, on images the model has never seen in either direction, is attributable to that shared identity, not to sampling noise or a generic tendency to call unfamiliar images `edited`.

That rules out the more optimistic reading that the fingerprint effect might be negligible. But it also does not support the pessimistic reading that `edited`'s performance is *mostly* fingerprint: 17.4% sits far below `edited`'s own 89.5% val recall. If the model's `edited` detection were driven mainly by corpus identity rather than manipulation content, held-out authentic images carrying that same corpus identity should be called `edited` at a rate approaching 89.5%, not 17.4%. The majority of these authentic images (78.3%) are still correctly called `real`, which is the model actually discriminating on something beyond provenance for most of this sample.

Net read: the fingerprint is real, measurable, and non-negligible — a meaningful partial confound — but it is not the primary explanation for `edited`'s 0.990 ROC-AUC / 89.5% recall on the true val split. It is one ingredient, quantified here for the first time, not the whole story.

## Status

**Rules out** (moderate confidence, n=161): a zero-fingerprint null. 17.4% vs. the 2.7% same-model baseline on true-real images is a clear, non-trivial gap that a matched real-corpus comparison doesn't show.

**Rules out** (same evidence): `edited`'s val performance being explained *mostly* by corpus fingerprint. 78.3% of never-trained-on, same-corpus authentic images are still correctly classified `real`, which a mostly-fingerprint account would not predict.

**Does not confirm**: the exact fraction of `edited`'s 89.5% recall / 0.990 AUC on the true val split attributable to fingerprint vs. manipulation signal. This probe measures a ceiling from the authentic-image side (how often fingerprint alone triggers `edited`), not a decomposition of the true-positive side.

## Limitations

- `n=161` is modest — normal-approximation 95% CI on 17.4% is roughly [11.6%, 23.2%]. It clears the 2.7% baseline comfortably, but a tighter estimate would help before leaning on the exact number.
- Only CASIA's authentic half was probed. PS-Battles' "original" half (also excluded by `classify_ps_battles_path`, per `data/face_filter.py`) covers `edited`'s other source corpus and is untested here.
- The 83.9% face-detection rejection rate on this sample is unusually high; whether the surviving 161 crops are representative of what actually got trained on for `edited`'s CASIA half (e.g. biased toward larger/closer portrait framing) was not checked.
- Single checkpoint, no retraining or ablation — diagnostic only, consistent with every other probe in this project's investigation line.

## Recommended next steps

1. If a tighter estimate matters before committing to a data-sourcing decision: bump `CASIA_N` (this run used 1000 source images for 161 detected faces; ~2000-2500 would put the detected-face count in the 300-400 range).
2. Run the same probe against PS-Battles' "original" images once downloaded, to cover `edited`'s other source corpus.
3. Feed this measured number into `0002-frozen-backbone-generalization.md` §6.4/§8.3/§11 — the fingerprint claim there was evidence-consistent but unmeasured; it can now be tagged `[measured, partial]` with this number attached.
