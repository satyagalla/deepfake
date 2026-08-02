"""Smoke test for the GPT image generator (gpt-image-2) before doing any real
S2 generation by hand.

Not part of the pipeline -- selfgen_organize.py still expects generation to
happen manually (API *and* web UI, for the container control). This just
checks the API path works and shows what its output looks like, using two
throwaway prompts. Output goes to a scratch folder, not data_raw/selfgen_intake.

Mirrors selfgen_gemini_test.py's setup and resolution reasoning.

Usage:
    set OPENAI_API_KEY=...   (or export on non-Windows)
    python data/selfgen_gptimage_test.py
"""
import base64
import sys
from pathlib import Path
from google.colab import userdata

# __file__ isn't defined when this runs as a notebook cell (Colab) -- fall
# back to cwd, which Colab sets to /content by default.
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_THIS_DIR.parent))

from openai import OpenAI

MODEL = "gpt-image-2"

TEST_PROMPTS = [
    # in-domain style: an ordinary COCO-caption-like scene
    "a golden retriever catching a frisbee in a sunny park",
    # off-domain style: matches selfgen_prompts.py's OFFDOMAIN_PROMPTS flavor
    "a bioluminescent jellyfish forest deep underwater, glowing blue and purple",
]

OUT_DIR = _THIS_DIR.parent / "data_raw" / "selfgen_test"


def main() -> None:
    api_key = userdata.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first.")

    client = OpenAI(api_key=api_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i}] generating: {prompt!r}")
        response = client.images.generate(
            model=MODEL,
            prompt=prompt,
            # 1024x1024, same reasoning as selfgen_gemini_test.py -- matches
            # sdxl/sd3/sd35 (most of COCO_AI's fake columns) and yields
            # exactly N_PATCHES=16 non-overlapping 224px tiles, no upsampling.
            size="1024x1024",
            quality="high",
            n=1,
        )

        out_path = OUT_DIR / f"gptimage_test_{i:02d}.png"
        out_path.write_bytes(base64.b64decode(response.data[0].b64_json))
        print(f"    saved: {out_path}")

    print(f"\nDone. Check {OUT_DIR}")


if __name__ == "__main__":
    main()
