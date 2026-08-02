"""Shared resumable-generation driver for the S2 self-gen API path (Gemini +
GPT-Image).

Not run directly -- see selfgen_gemini_generate.py / selfgen_gptimage_generate.py.
Both hit external APIs from a Colab runtime that can disconnect, or hand back
a transient/rate-limit error, at any point during a ~150-image run, so this
factors out the two things that must be identical for both: a single shared
prompt list (content-domain match between generators is the point, and a
disconnect must not re-sample a different set on resume) and a loop that
skips whatever's already on disk, retries transient failures, and never lets
one bad prompt kill the rest of the run.

Only covers the api container. The web UI container (needed for 0004 S5's
container control) stays manual -- it needs a human at a browser.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SELFGEN_INTAKE_DIR

PROMPTS_MANIFEST_PATH = SELFGEN_INTAKE_DIR / "_prompts_manifest.json"


def load_or_create_prompts(n_indomain: int, n_offdomain: int) -> dict[str, list[str]]:
    """Both generators must see the same prompts, and a resumed run (any
    generator, after any disconnect) must not re-sample a different set --
    so the list is written once on first use and every later call just reads
    it back, ignoring n_indomain/n_offdomain."""
    if PROMPTS_MANIFEST_PATH.exists():
        data = json.loads(PROMPTS_MANIFEST_PATH.read_text(encoding="utf-8"))
        print(
            f"[prompts] reusing existing manifest at {PROMPTS_MANIFEST_PATH} "
            f"({len(data['indomain'])} in-domain, {len(data['offdomain'])} off-domain)"
        )
        return data

    from data.selfgen_prompts import OFFDOMAIN_PROMPTS, sample_indomain_prompts

    indomain = sample_indomain_prompts(n_indomain)
    offdomain = OFFDOMAIN_PROMPTS[:n_offdomain]
    data = {"indomain": indomain, "offdomain": offdomain}

    PROMPTS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(
        f"[prompts] wrote new manifest to {PROMPTS_MANIFEST_PATH} "
        f"({len(indomain)} in-domain, {len(offdomain)} off-domain)"
    )
    return data


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write-then-rename so a run killed mid-write (Colab disconnect) never
    leaves a truncated file that the skip-if-exists check would mistake for
    a completed image."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_generation(
    generator: str,
    generate_one,
    n_indomain: int,
    n_offdomain: int,
    sleep_s: float = 1.0,
    max_retries: int = 5,
) -> None:
    """Drives one generator (gemini or gptimage) across both domains, api
    container only, writing to the layout selfgen_organize.py expects:

        data_raw/selfgen_intake/<generator>/<indomain|offdomain>/api/<idx>.png

    generate_one(prompt: str, out_path: Path) -> None must write the image
    to out_path (use atomic_write_bytes) and raise on failure. Transient
    failures are retried with exponential backoff; a prompt that still fails
    after max_retries is logged to <generator>/_failures.log and skipped --
    one bad prompt never aborts the other ~150.

    Resumability: an image already on disk at out_path is treated as done
    and skipped, so re-running this after a disconnect/timeout/crash picks
    up exactly where it left off (including retrying only what previously
    failed permanently, since those left no output file) at the cost of a
    directory listing.
    """
    prompts = load_or_create_prompts(n_indomain, n_offdomain)
    failures_path = SELFGEN_INTAKE_DIR / generator / "_failures.log"
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    total_done = total_skipped = total_failed = 0
    for domain, plist in prompts.items():
        out_dir = SELFGEN_INTAKE_DIR / generator / domain / "api"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(plist):
            out_path = out_dir / f"{i:04d}.png"
            if out_path.exists() and out_path.stat().st_size > 0:
                total_skipped += 1
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    generate_one(prompt, out_path)
                    total_done += 1
                    print(f"[{generator}/{domain}] {i:04d} OK: {prompt[:60]!r}")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait = sleep_s * (2 ** (attempt - 1))
                        print(
                            f"[{generator}/{domain}] {i:04d} attempt {attempt}/{max_retries} "
                            f"failed: {e} -- retrying in {wait:.0f}s"
                        )
                        time.sleep(wait)
                    else:
                        print(f"[{generator}/{domain}] {i:04d} FAILED after {max_retries} attempts: {e}")
                        total_failed += 1
                        with open(failures_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{domain}\t{i:04d}\t{prompt}\t{e}\n"
                                f"{traceback.format_exc()}\n"
                            )
            time.sleep(sleep_s)  # politeness gap between successful calls too

    print(
        f"\n[{generator}] done. generated={total_done} "
        f"skipped(already done)={total_skipped} failed={total_failed}"
    )
    if total_failed:
        print(
            f"[{generator}] {total_failed} prompt(s) failed permanently -- see {failures_path}. "
            "Re-run this script to retry just those; everything already written is skipped."
        )
