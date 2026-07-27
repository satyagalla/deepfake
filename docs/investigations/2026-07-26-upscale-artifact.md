# Investigation: Face-Crop Upscale Artifact Across All Three Classes

Started from an observation during a live demo review: `docs/screenshots/demo_real.png`, `demo_edited.png`, and `demo_deepfake.png` all show heavy blur/blockiness, not just the `deepfake` class. This doc records what was measured to explain that, how the working hypothesis changed as data came in, and what's still open.

## Background

`model/demo.py` and `data/face_filter.py` both run every image through the same `facenet_pytorch.MTCNN` config (`image_size=380, margin=40, post_process=False`, from `config.py`'s `IMAGE_SIZE`/`FACE_MARGIN`) before it ever reaches the 3-branch fusion model. `MTCNN`'s internal `crop_resize()` (`facenet_pytorch/models/utils/detect_face.py:309-325`) crops the detected face box and resizes it to 380x380 via **`PIL.Image.BILINEAR`, unconditionally** — no anti-aliasing, no scale-aware resampling choice, regardless of how small the source box is.

That means the amount of blur in the final 380x380 crop is a direct function of one thing: **how small the detected face box was relative to 380px** (the "upscale factor"). This doc measures that factor for all three classes.

## Methodology

- Face boxes were measured with `MTCNN.detect()` (box coordinates only, not the full save-to-disk path) using the project's exact `IMAGE_SIZE=380` / `FACE_MARGIN=40` config.
- Upscale factor is computed as `380 / max(box_width, box_height)`.
- **real / deepfake**: pulled live from `NasrinImp/COCO_AI` (`HF_DATASET` in `config.py`) via `datasets` streaming, using the same `caption_has_person()` keyword filter `data/download.py` uses, so the sample matches what the real pipeline would keep.
- **edited**: pulled live from Kaggle (`divg07/casia-20-image-tampering-detection-dataset`, `KAGGLE_DATASET` in `config.py`). CASIA's tampered (`Tp_`) files are listed alphabetically by scene-category code (`ani`=animal, `arc`=architecture, `art`, `cha`=people/characters, `ind`=indoor, `nat`=nature, `pla`=plant, `sec`, `txt`=texture); only `_cha_` filenames were pulled (31 files), since that's the category most likely to contain a detectable face, matching the same face-driven filtering `find_casia_tampered()` implicitly does across the full (unfiltered-by-category) class in the real pipeline.
- **No download/processing was cached** — every number here came from a small, one-off pull. Sample sizes are small (12-31 candidates per class, further reduced by MTCNN's no-face rejections). Treat this as a directional signal, not a statistically confident measurement.

## Findings

| Class | Native resolution | Sample (faces found / candidates) | Face box size range | Upscale factor (median) |
|---|---|---|---|---|
| `real` (`coco_image`) | ~640x480 (COCO standard, varies) | 5 / 12 | 14x18 – 57x74 | **10.40x** |
| `deepfake` (`dalle_image`) | uniformly 270x270 | 8 / 12 | 15x23 – 77x87 | **11.05x** |
| `edited` (CASIA `Tp_*`, `cha` category) | varies widely: 256x384, 384x256, 518x800, 700x469, 720x440, 756x513, 800x531 | 19 / 31 | 9x12 – 43x37 | **11.83x** |

**All three classes land in the same ~10x-12x median upscale range.** This directly contradicts the initial hypothesis (formed before `edited` was measured, and before `real` was actually checked rather than assumed): that `deepfake`'s tiny 270x270 native canvas would make it uniquely blurrier than `real`'s larger 640x480 canvas. It doesn't — `real`'s faces are just as small a fraction of its (larger) frame, because both sources use full-scene, non-portrait framing (COCO-style captions like "a man with a bike at a marina," not close-up portraits). CASIA's `edited` class, despite having the most variable native resolution of the three, ends up in the same range for the same underlying reason: tampered-object scenes, not face-centric shots.

## Hypothesis — revised

**What's ruled out (by the measurement above):** a clean, single-scalar "upscale magnitude correlates with class" shortcut. If that were the dominant issue, real vs. deepfake vs. edited would show clearly separated blur-severity distributions. They don't, at this sample size.

**What's still a live concern, reframed:** the *magnitude* of upscaling is similar across classes, but the underlying pixel content going into that same bilinear operation is not — a natural camera-sensor photo (`real`), a DALL-E 3 diffusion output (`deepfake`), and a spliced/tampered photo (`edited`) each have different native frequency-domain and noise characteristics before the crop. The literature below establishes two things relevant here:

1. Bilinear resampling itself imprints a specific, detectable periodic signature into an image, independent of content (Popescu & Farid, 2005). This is why the spectral/noise-residual branches are sensitive to this pipeline step at all.
2. Detectors in this exact research area (CNN/diffusion-generated image forensics) have been shown, empirically, to key off resizing-pipeline artifacts that were mistaken for genuine generator fingerprints (Chandrasegaran et al., CVPR 2021) — and forensic detector performance is known to be sensitive to resizing/compression choices specifically in the diffusion-model setting (Corvi et al., ICASSP 2023).

So the open question is no longer "does one class get uniquely more blurred" (measured, not the case) but: **does the model's spectral/noise-residual branches separate real/edited/deepfake using the interaction between bilinear-resampling artifact and each class's underlying pixel statistics, or using genuine content-level manipulation signatures?** Both are technically "real, learnable signal" in this dataset; only the latter is the generalizable thing the architecture was designed to detect. This is the same shortcut-learning risk as before, just no longer framed as a simple blur-magnitude confound.

Supporting precedent for why a model can't be assumed to distinguish these two on its own, even in principle:

- Geirhos et al., *"Shortcut Learning in Deep Neural Networks,"* Nature Machine Intelligence 2, 665–673 (2020) — https://www.nature.com/articles/s42256-020-00257-z — a network cannot distinguish, from the training signal alone, a decision rule that reflects the intended task from one that merely correlates with the label in this dataset; both minimize loss equally well until tested where the correlation breaks.
- Zech et al., *"Variable Generalization Performance of a Deep Learning Model to Detect Pneumonia in Chest Radiographs,"* PLOS Medicine 15(11):e1002683 (2018) — https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683 — the closest cross-domain precedent for "a dataset-construction artifact, present in every image of a class, gets learned as if it were the target concept": a pneumonia-detection CNN instead learned to separate hospital scanner/department (visible via an inverted color scheme and embedded laterality text), because that correlated with disease prevalence in the training data. In-distribution accuracy was strong; cross-hospital generalization collapsed.

Sources used for the earlier (partially revised) hypothesis, still relevant to the resampling-artifact mechanism itself:

- Popescu & Farid, *"Exposing Digital Forgeries by Detecting Traces of Resampling,"* IEEE Trans. Signal Processing 53(2), 758–767 (2005) — https://ui.adsabs.harvard.edu/abs/2005ITSP...53..758P/abstract
- Chandrasegaran, Tran & Cheung, *"A Closer Look at Fourier Spectrum Discrepancies for CNN-Generated Images Detection,"* CVPR 2021 — https://openaccess.thecvf.com/content/CVPR2021/html/Chandrasegaran_A_Closer_Look_at_Fourier_Spectrum_Discrepancies_for_CNN-Generated_Images_CVPR_2021_paper.html
- Corvi, Cozzolino, Zingarini, Poggi, Nagano & Verdoliva, *"On the Detection of Synthetic Images Generated by Diffusion Models,"* ICASSP 2023 — https://arxiv.org/abs/2211.00680
- Frank et al., *"Leveraging Frequency Analysis for Deep Fake Image Recognition,"* ICML 2020 — https://arxiv.org/abs/2003.08685 (establishes that GAN upsampling layers leave a genuine, generator-side frequency fingerprint — the counterpoint showing frequency-domain differences *can* be legitimate signal, not just pipeline artifact; the risk is conflating the two, not that frequency-domain evidence is inherently suspect)

## Limitations of this investigation

- Sample sizes are small (n=5/8/19 detected faces per class) and were not saved to a fixed, reproducible dataset — re-running the pull would sample different rows.
- `edited` was only measured on the `cha` (people) category of CASIA, not the full class `data/face_filter.py` actually processes (`ani`, `arc`, `art`, `ind`, `nat`, `pla`, `sec`, `txt` also contribute images, filtered down by whichever ones happen to contain a detectable face). The true class-wide upscale-factor distribution for `edited` could differ from this people-only slice.
- Measurement used `MTCNN.detect()` for box coordinates only, not the full `extract_face`/`crop_resize`/JPEG-quality-95-resave path the actual pipeline runs. The upscale-factor arithmetic is identical either way, but the final on-disk pixel values weren't reproduced pixel-for-pixel here.
- No model was trained or re-evaluated as part of this investigation — everything here is a data/pipeline measurement, not evidence about what the already-trained model actually learned.

## Recommended next steps

Still the tiered ablation plan from the original discussion, now correctly scoped to the revised hypothesis (interaction effect, not magnitude confound):

1. **Cheapest:** a diagnostic classifier using only a resampling-artifact proxy (e.g. Laplacian variance, or high-frequency FFT energy) per class — check whether it's linearly separable at all now that magnitude alone isn't the story.
2. **Moderate:** counterfactual probe on the trained checkpoint — swap which class gets which upscale factor (e.g. take a `real` photo, force it through deepfake-typical resampling parameters) and see if the prediction follows content or resampling.
3. **Expensive:** retrain after normalizing native resolution/framing across all three sources, compare metrics to the current run.
