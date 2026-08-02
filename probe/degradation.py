"""E3: degradation ladder (`0004` §9, §8.1, build plan S9).

Scores each card across re-encode / rescale / re-render and tracks how
each card's score moves. This is the watermark measurement: a learned
score (AI Model) that degrades identically to the provenance card (EXIF)
is reading a robust watermark like SynthID rather than a synthesis
artifact -- and since EXIF/C2PA metadata does not survive a plain
re-encode (this module's own re-save strips it, same as any real
pipeline would), EXIF predictably flips to STRIPPED at the first rung.
The interesting question is whether **AI Model** and **Spectral** hold
steady or also collapse. Gemini vs GPT-Image differ in watermarking
policy and are the natural experiment (§8.1).

Lowest priority in the cut order (build plan §3) -- it is not needed for
the headline claim, but is the closest thing in this build to the
product's actual robustness spec.

Usage:
    python -m probe.degradation
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_POOL_METHOD, HEAD_A_NAME, PROBE_OUTPUTS_DIR, SELFGEN_GENERATORS, SELFGEN_RAW_DIR, SPECTRAL_HEAD_NAME
from probe.extract import build_clip, featurize_single_image
from probe.heads import load, prob_head_a
from probe.spectral import spectral_score

LADDER = {
    "reencode_q75": lambda im: _reencode(im, quality=75),
    "reencode_q50": lambda im: _reencode(im, quality=50),
    "reencode_q30": lambda im: _reencode(im, quality=30),
    "rescale_0.75": lambda im: _rescale(im, scale=0.75),
    "rescale_0.5": lambda im: _rescale(im, scale=0.5),
    "rescale_0.25": lambda im: _rescale(im, scale=0.25),
    "rerender_0.5_q50": lambda im: _rescale(_reencode(im, quality=50), scale=0.5),
    "rerender_0.25_q30": lambda im: _rescale(_reencode(im, quality=30), scale=0.25),
}


def _reencode(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _rescale(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def sample_generator_images(generator: str, domain: str = "indomain", n: int = 10) -> list[Path]:
    root = SELFGEN_RAW_DIR / generator / domain
    if not root.exists():
        return []
    paths = sorted(p for container_dir in root.iterdir() if container_dir.is_dir() for p in container_dir.glob("*.jpg"))
    return paths[:n]


def run_e3(pool_method: str = DEFAULT_POOL_METHOD, n_per_generator: int = 10) -> dict:
    model, preprocess_val = build_clip()
    head_a = load(HEAD_A_NAME)
    spectral_fitted = load(SPECTRAL_HEAD_NAME)

    results: dict = {}
    for generator in SELFGEN_GENERATORS:
        paths = sample_generator_images(generator, n=n_per_generator)
        if not paths:
            print(f"E3[{generator}]: no self-generated images found -- skipping.")
            continue

        per_step = {"baseline": {"ai_model": [], "spectral": []}}
        for step_name in LADDER:
            per_step[step_name] = {"ai_model": [], "spectral": []}

        for path in paths:
            base_image = Image.open(path).convert("RGB")
            x_row = featurize_single_image(base_image, model, preprocess_val, pool_method=pool_method, key=path.stem)
            per_step["baseline"]["ai_model"].append(float(prob_head_a(head_a, x_row.reshape(1, -1))[0]))
            per_step["baseline"]["spectral"].append(spectral_score(spectral_fitted, base_image))

            for step_name, fn in LADDER.items():
                degraded = fn(base_image)
                x_deg = featurize_single_image(degraded, model, preprocess_val, pool_method=pool_method, key=f"{path.stem}:{step_name}")
                per_step[step_name]["ai_model"].append(float(prob_head_a(head_a, x_deg.reshape(1, -1))[0]))
                per_step[step_name]["spectral"].append(spectral_score(spectral_fitted, degraded))

        results[generator] = {
            step: {card: float(np.mean(scores)) for card, scores in cards.items()}
            for step, cards in per_step.items()
        }
        print(f"E3[{generator}]: {json.dumps(results[generator], indent=2)}")

    return results


def plot_e3(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = ["baseline"] + list(LADDER.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, card in zip(axes, ["ai_model", "spectral"]):
        for generator, per_step in results.items():
            ys = [per_step.get(s, {}).get(card, float("nan")) for s in steps]
            ax.plot(range(len(steps)), ys, marker="o", label=generator)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=7)
        ax.set_title(card)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("P(AI-generated)")
    fig.suptitle("E3: degradation ladder -- per-card score by generator")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-method", default=DEFAULT_POOL_METHOD)
    parser.add_argument("--n-per-generator", type=int, default=10)
    args = parser.parse_args()

    results = run_e3(pool_method=args.pool_method, n_per_generator=args.n_per_generator)
    if not results:
        raise SystemExit("E3 produced no results -- run data/selfgen_organize.py first.")

    PROBE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_OUTPUTS_DIR / "e3_degradation.json").write_text(json.dumps(results, indent=2))
    plot_e3(results, PROBE_OUTPUTS_DIR / "e3_degradation.png")
    print(f"\nResults written to {PROBE_OUTPUTS_DIR / 'e3_degradation.json'} and .png")


if __name__ == "__main__":
    main()
