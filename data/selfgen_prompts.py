"""Prompt lists for the self-generated N-shot pool (build-plan S2).

This does not call any generator API -- `0004` §5 needs the pool built
through a mix of the API *and* the web UI (container control, §4.2) and
through prompts a human reads and pastes, so generation itself stays a
manual step. This script's job is only to produce the two prompt sets
consistently:

- **in-domain**: real COCO captions, so image *content* stays matched to
  the COCO_AI corpus (the standing pairing constraint -- a domain-shifted
  N-shot pool would confound "new generator" with "new content").
- **off-domain**: fixed prompts describing scenes/subjects outside COCO's
  everyday-photo distribution, for E4 (`0004` §9).

Usage:
    python data/selfgen_prompts.py --indomain 130 --offdomain 20 --out prompts.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HF_DATASET, SEED

# Fixed, not sampled: off-domain by construction means "not drawn from
# COCO's distribution," so these are hand-picked to span subject matter
# (fantasy, abstract, macro, architectural render, sci-fi) that COCO's
# everyday-scene captions essentially never produce.
OFFDOMAIN_PROMPTS = [
    "a bioluminescent jellyfish forest deep underwater, glowing blue and purple",
    "an abstract fractal pattern of interlocking geometric crystals",
    "a steampunk airship docked above a cloud city at sunset",
    "a macro photograph of frost crystals forming on a spider web",
    "a futuristic space station interior with holographic control panels",
    "a surreal melting clock draped over a desert dune, Dali-inspired",
    "an ornate Gothic cathedral interior rendered in stained-glass light",
    "a dragon perched on a volcanic cliff at dawn, wings spread",
    "a cyberpunk alley with neon signs reflected in rain puddles",
    "an alien landscape with twin suns and floating rock formations",
    "a close-up of a hummingbird's wing mid-flap, iridescent feathers",
    "an underwater coral reef city with glass domes and kelp gardens",
    "a minimalist architectural render of a concrete brutalist museum",
    "a swirling nebula of orange and violet gas clouds in deep space",
    "a robot blacksmith forging a glowing sword in a stone forge",
    "a crystal cave illuminated by bioluminescent fungi",
    "a wireframe blueprint of a mechanical clockwork heart",
    "a Norse longship sailing through an aurora-lit fjord at night",
    "a topographic map rendered as a 3D isometric miniature diorama",
    "a giant tortoise carrying a floating garden city on its shell",
    "an art deco skyscraper lobby with brass fixtures and marble floors",
    "a school of bioluminescent fish forming a spiral in dark water",
    "a mechanical owl made of brass gears perched on a dead tree",
    "a volcanic lava flow meeting the ocean at night, steam rising",
    "an isometric pixel-art rendering of a floating island monastery",
    "a glass greenhouse filled with alien carnivorous plants",
    "a snow leopard silhouette against the aurora borealis",
    "a labyrinth of mirrors reflecting infinite geometric patterns",
    "a colossal stone golem awakening in an overgrown ruin",
    "a wind-swept salt flat reflecting a sky full of stars",
]


def sample_indomain_prompts(n: int, seed: int = SEED) -> list[str]:
    """Streams captions only (no images) from COCO_AI -- cheap, since this
    just needs the text to hand-paste into a generator."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.select_columns(["caption"])
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    prompts, seen_captions = [], set()
    for row in ds:
        cap = (row.get("caption") or "").strip()
        if not cap or cap in seen_captions:
            continue
        seen_captions.add(cap)
        prompts.append(cap)
        if len(prompts) >= n:
            break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indomain", type=int, default=130, help="number of COCO-caption prompts to sample")
    parser.add_argument(
        "--offdomain", type=int, default=len(OFFDOMAIN_PROMPTS),
        help=f"number of off-domain prompts to take (max {len(OFFDOMAIN_PROMPTS)})",
    )
    parser.add_argument("--out", type=Path, default=None, help="write prompts here instead of stdout")
    args = parser.parse_args()

    indomain = sample_indomain_prompts(args.indomain)
    offdomain = OFFDOMAIN_PROMPTS[: args.offdomain]

    lines = ["# in-domain (COCO captions)"] + indomain + ["", "# off-domain"] + offdomain
    text = "\n".join(lines)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(indomain)} in-domain + {len(offdomain)} off-domain prompts to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
