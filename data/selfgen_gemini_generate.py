"""Automated S2 generation: Gemini, api container, resumable.

Generates one image per prompt in the shared prompts manifest (built on
first use by selfgen_generate.load_or_create_prompts) via the Gemini API and
writes straight into selfgen_organize.py's intake layout:

    data_raw/selfgen_intake/gemini/<indomain|offdomain>/api/<idx>.png

The web UI container (needed for 0004 S5's container control) stays manual.
Safe to re-run after a Colab disconnect or an API error: already-written
files are skipped and only what previously failed is retried (see
selfgen_generate.run_generation's docstring) -- nothing already generated is
redone or lost.

Mirrors selfgen_gemini_test.py's client setup and resolution reasoning
(1024x1024 / "1K" -- matches sdxl/sd3/sd35, exactly N_PATCHES=16 tiles, no
upsampling); run that test script first to confirm the API path works.

Usage (Colab):
    Set the GEMINI_API_KEY secret, then:
    !python data/selfgen_gemini_generate.py --indomain 130 --offdomain 20
"""
import argparse
import sys
from pathlib import Path
from google.colab import userdata

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_THIS_DIR.parent))

from google import genai

from data.selfgen_generate import atomic_write_bytes, run_generation

MODEL = "gemini-3.1-flash-image"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indomain", type=int, default=130)
    parser.add_argument("--offdomain", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between successful calls")
    parser.add_argument("--max-retries", type=int, default=5, help="retries per prompt before logging a permanent failure")
    args = parser.parse_args()

    api_key = userdata.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY first.")

    client = genai.Client(api_key=api_key)
    image_config = genai.types.ImageConfig(image_size="1K", aspect_ratio="1:1")

    def generate_one(prompt: str, out_path: Path) -> None:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(image_config=image_config),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                atomic_write_bytes(out_path, part.inline_data.data)
                return
        text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
        raise RuntimeError(f"no image returned (text response: {text_parts[:1]})")

    run_generation(
        "gemini",
        generate_one,
        n_indomain=args.indomain,
        n_offdomain=args.offdomain,
        sleep_s=args.sleep,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
