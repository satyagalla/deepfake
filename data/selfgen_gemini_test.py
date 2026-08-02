"""Smoke test for the Gemini image generator ("nano banana" / gemini-2.5-flash-image)
before doing any real S2 generation by hand.

Not part of the pipeline -- selfgen_organize.py still expects generation to
happen manually (API *and* web UI, for the container control). This just
checks the API path works and shows what its output looks like, using two
throwaway prompts. Output goes to a scratch folder, not data_raw/selfgen_intake.

Usage:
    set GEMINI_API_KEY=...   (or export on non-Windows)
    python data/selfgen_gemini_test.py
"""
import sys
from pathlib import Path
from google.colab import userdata

# __file__ isn't defined when this runs as a notebook cell (Colab) -- fall
# back to cwd, which Colab sets to /content by default.
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_THIS_DIR.parent))

from google import genai

MODEL = "gemini-3.1-flash-image"

TEST_PROMPTS = [
    # in-domain style: an ordinary COCO-caption-like scene
    "a golden retriever catching a frisbee in a sunny park",
    # off-domain style: matches selfgen_prompts.py's OFFDOMAIN_PROMPTS flavor
    "a bioluminescent jellyfish forest deep underwater, glowing blue and purple",
]

OUT_DIR = _THIS_DIR.parent / "data_raw" / "selfgen_test"


def main() -> None:
    api_key = userdata.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY first.")

    client = genai.Client(api_key=api_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Explicit, not left to the API default -- 1024x1024 matches sdxl/sd3/sd35
    # (most of COCO_AI's fake columns) and yields exactly N_PATCHES=16
    # non-overlapping 224px tiles with no upsampling. 4K would reopen the
    # corpus resolution mismatch (bottlenecks.md 2.1); some preview flash
    # variants have also been reported to silently ignore image_size when
    # only set implicitly, so pin it rather than relying on the default.
    image_config = genai.types.ImageConfig(image_size="1K", aspect_ratio="1:1")

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i}] generating: {prompt!r}")
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(image_config=image_config),
        )

        saved = False
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"    text: {part.text}")
            elif part.inline_data is not None:
                out_path = OUT_DIR / f"test_{i:02d}.png"
                out_path.write_bytes(part.inline_data.data)
                print(f"    saved: {out_path}")
                saved = True

        if not saved:
            print("    WARNING: no image returned for this prompt")

    print(f"\nDone. Check {OUT_DIR}")


if __name__ == "__main__":
    main()
