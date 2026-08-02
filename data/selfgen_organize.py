"""Organize the manually-generated N-shot pool (build-plan S2).

Generation itself is manual (Gemini/GPT-Image, API and web UI, per §5's
container control and E4's off-domain control) -- this script only takes
whatever landed in the intake drop zone and turns it into the standard
raw layout + manifest, re-saved at JPEG q95 (I4).

Drop generated images into:

    data_raw/selfgen_intake/<generator>/<domain>/<container>/*

    <generator>  in {gemini, gptimage}
    <domain>     in {indomain, offdomain}   -- indomain = COCO-caption prompted
    <container>  in {api, web}              -- which path the image came from

e.g. data_raw/selfgen_intake/gemini/indomain/web/img017.webp

Refuses to guess an unrecognized folder name rather than silently
dropping or mislabeling images -- the same policy face_filter.py uses for
the PS-Battles layout.

Usage:
    python data/selfgen_organize.py
"""
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    JPEG_QUALITY,
    SELFGEN_CONTAINERS,
    SELFGEN_DOMAINS,
    SELFGEN_GENERATORS,
    SELFGEN_INTAKE_DIR,
    SELFGEN_MANIFEST_PATH,
    SELFGEN_RAW_DIR,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    if not SELFGEN_INTAKE_DIR.exists():
        raise SystemExit(
            f"{SELFGEN_INTAKE_DIR} does not exist -- drop generated images into "
            f"{SELFGEN_INTAKE_DIR}/<generator>/<domain>/<container>/ first "
            "(see this script's docstring for the layout)."
        )

    rows = []
    written = skipped_bad_layout = skipped_unreadable = 0
    ext_counts: dict[tuple[str, str], Counter] = {}

    for generator_dir in sorted(SELFGEN_INTAKE_DIR.iterdir()):
        if not generator_dir.is_dir():
            continue
        if generator_dir.name not in SELFGEN_GENERATORS:
            print(f"WARNING: skipping unrecognized generator folder '{generator_dir.name}' "
                  f"(expected one of {SELFGEN_GENERATORS}) -- refusing to guess.")
            continue
        generator = generator_dir.name

        for domain_dir in sorted(generator_dir.iterdir()):
            if not domain_dir.is_dir() or domain_dir.name not in SELFGEN_DOMAINS:
                if domain_dir.is_dir():
                    print(f"WARNING: skipping unrecognized domain folder '{domain_dir.name}' under {generator_dir}")
                continue
            domain = domain_dir.name

            for container_dir in sorted(domain_dir.iterdir()):
                if not container_dir.is_dir() or container_dir.name not in SELFGEN_CONTAINERS:
                    if container_dir.is_dir():
                        print(f"WARNING: skipping unrecognized container folder '{container_dir.name}' under {domain_dir}")
                    continue
                container = container_dir.name

                out_dir = SELFGEN_RAW_DIR / generator / domain / container
                out_dir.mkdir(parents=True, exist_ok=True)
                key = (generator, container)
                ext_counts.setdefault(key, Counter())

                files = sorted(p for p in container_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
                for i, src_path in enumerate(files):
                    ext_counts[key][src_path.suffix.lower()] += 1
                    out_name = f"{i:04d}.jpg"
                    out_path = out_dir / out_name
                    try:
                        with Image.open(src_path) as im:
                            im.convert("RGB").save(out_path, quality=JPEG_QUALITY)
                    except Exception as e:
                        print(f"  skip unreadable {src_path}: {e}")
                        skipped_unreadable += 1
                        continue
                    rows.append(
                        {
                            "path": out_path.relative_to(SELFGEN_RAW_DIR).as_posix(),
                            "generator": generator,
                            "domain": domain,
                            "container": container,
                            "orig_ext": src_path.suffix.lower(),
                        }
                    )
                    written += 1

    if not rows:
        raise SystemExit(
            f"No images organized from {SELFGEN_INTAKE_DIR} -- check the folder layout matches "
            "<generator>/<domain>/<container>/ (see this script's docstring)."
        )

    SELFGEN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SELFGEN_MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "generator", "domain", "container", "orig_ext"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOrganized {written} image(s) into {SELFGEN_RAW_DIR}, manifest at {SELFGEN_MANIFEST_PATH}")

    # S2's checkpoint: "confirm the two containers actually differ." SynthID
    # (and any watermarking) passes a naive container control by design
    # (bottlenecks.md §4.2) -- the only thing this can check for free is
    # whether web vs API actually shipped different file formats at all.
    print("\n=== container control check (original file extensions, before JPEG re-save) ===")
    void_pairs = []
    for generator in SELFGEN_GENERATORS:
        api_exts = ext_counts.get((generator, "api"), Counter())
        web_exts = ext_counts.get((generator, "web"), Counter())
        if not api_exts and not web_exts:
            continue
        print(f"  {generator}: api={dict(api_exts)}  web={dict(web_exts)}")
        if api_exts and web_exts and set(api_exts) == set(web_exts):
            void_pairs.append(generator)
    if void_pairs:
        print(
            f"\nWARNING: {void_pairs} -- API and web UI produced the SAME file extension(s). "
            "The container control is void for these generators (0004 §5); record that rather "
            "than reporting the control as run."
        )
    else:
        print("\nContainers differ by extension for every generator with both arms populated -- control is live.")


if __name__ == "__main__":
    main()
