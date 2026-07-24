# Data Download & Preprocessing — Instructions

Prep-phase only — must finish **before** the 1.5hr training clock starts (per `architecture_decisions.md`). Goal: three class folders of face-cropped images plus a manifest, ready for the `Dataset`/`DataLoader` described in `model_code.md`.

## File layout

- `data/download.py` — fetches the three raw sources into local/Drive folders.
- `data/face_filter.py` — runs face detection/cropping over all three sources uniformly, writes the final `train/val` folder structure and `manifest.csv`.
- Colab notebook section "Data": calls these two scripts (or their functions) and prints the class-balance check before moving to the model section.

## 1. Download

Fetch sources into separate raw folders (`deepfake_src`, `real_src`, `edited_src`, `edited_src_psbattles`):
- **Deepfake:** the DALL-E 3 slice from COCO_AI/SynthBuster.
- **Real:** the pristine images paired 1:1 with that same DALL-E 3 slice (not an unrelated curated set — see `architecture_decisions.md`'s Dataset section for why).
- **Edited:** CASIA v2.0's *tampered* images, plus PS-Battles' *derivative* (Photoshopped) images for manipulation-technique diversity (CASIA alone is splice/copy-move/removal only). Both datasets also ship their own authentic/original sets — ignore both; the real class stays single-sourced from `real_src` to avoid mixing real-image pools. PS-Battles' Kaggle mirror layout hasn't been independently inspected — `face_filter.py`'s `find_ps_battles_derivatives()` looks for a folder-name signal (`photoshop`/`derivative`/`manipulat` vs. `original`) and raises loudly instead of guessing if it can't find one; if it raises, inspect the unzipped structure and adjust the heuristic.

**COCO_AI pairs are sampled from all of COCO, not just person-containing images** — most of COCO has no face in frame at all, which is the actual cause of face-filtering's low survival rate (not the detector). `data/download.py` filters on the row's `caption` field for person-indicating words before saving a pair, and keeps collecting from the (shuffled) stream until `--n-pairs` *matching* pairs are found, not just the first N raw rows.

**Do not guess or hardcode a download URL/Kaggle slug for the COCO_AI/SynthBuster source or CASIA v2.0** — confirm the exact identifier first (Kaggle dataset slug, GitHub release, or direct link), since a wrong guess here silently pulls the wrong data. If using Kaggle, the standard pattern is: upload `kaggle.json`, point `KAGGLE_CONFIG_DIR` at it, then use the Kaggle CLI/API to download and unzip into the raw folder — same pattern for both Kaggle sources if both end up hosted there.

## 2. Face detect + crop — uniform across all three classes

Use MTCNN via `facenet_pytorch` (already present in the local `.venv`; install it explicitly in Colab, it isn't preinstalled there).

Requirements for the detector call:
- Output crop size should match EfficientNet-B4's native input resolution (380x380) — crop once here, don't add a separate resize step downstream.
- Add a margin around the detected face (roughly 40px at this crop size) rather than cropping tight to landmarks — the edited/deepfake branches rely on boundary and context cues near the face edge, not just the interior.
- One crop per source image: if multiple faces are detected, keep only the largest and discard the rest. Never emit multiple crops from a single source image — that risks near-duplicate leakage across the train/val split if the split isn't done at the source-image level (it should be).
- Batch the detector calls rather than looping one image at a time — at this dataset scale (roughly 1-3k images per class) this should take seconds to low minutes on a GPU runtime, not be a bottleneck worth optimizing further.

**Logging requirement, not optional:** for every source folder, log total images seen vs. faces detected vs. rejected. Pay specific attention to the `deepfake_src` rejection rate — generated faces with anatomical artifacts can fail detection more often than real faces, which means face-filtering risks systematically discarding the hardest, most informative examples. If the surviving deepfake count looks too small to train on (rough floor: a few hundred images), stop before building the rest of the pipeline and flag it — that's a signal the source dataset choice needs revisiting, not something to push through silently.

## 3. Output structure

Produce `train/` and `val/` folders, each with `real/`, `edited/`, `deepfake/` subfolders of cropped images, plus a `manifest.csv` with one row per image recording its path, class, split, and source dataset.

- Split at the **source-image level** (before any per-image duplication risk), stratified by class, roughly 85/15 train/val, fixed random seed.
- Write the manifest incrementally as crops are produced — it's the single source of truth the model code reads from, and it's also where the class-count-based loss weights (`model_code.md`) get computed from.

## 4. Class balance check — do this before starting model work

Load the manifest and print per-class, per-split counts. Expect `edited` and `deepfake` to be far smaller than `real` (CASIA's tampered set and a ~1k-image diffusion slice, both further reduced by face-filtering). These counts must feed directly into the class-weighted loss in `model_code.md` — don't hardcode assumed proportions.

## Time estimate (prep phase — not counted against the 1.5hr training clock)

| Step | Estimate |
|---|---|
| Downloads (network-bound, host-dependent) | variable — start this first, do everything else while it runs |
| Detector install/setup | ~2 min |
| Face detect + crop, all three sources (~5-10k images total) | ~5-10 min on GPU |
| Survival-rate check + manifest/split | ~5 min |
| **Total active work** | **~15-20 min**, plus download wait time |
