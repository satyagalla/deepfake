"""Automated S2 generation: GPT-Image, api container, resumable.

Generates one image per prompt in the shared prompts manifest (built on
first use by selfgen_generate.load_or_create_prompts) via the OpenAI API and
writes straight into selfgen_organize.py's intake layout:

    data_raw/selfgen_intake/gptimage/<indomain|offdomain>/api/<idx>.png

The web UI container (needed for 0004 S5's container control) stays manual.
Safe to re-run after a Colab disconnect or an API error: already-written
files are skipped and only what previously failed is retried (see
selfgen_generate.run_generation's docstring) -- nothing already generated is
redone or lost.

Mirrors selfgen_gptimage_test.py's client setup and resolution reasoning
(1024x1024 -- matches sdxl/sd3/sd35, exactly N_PATCHES=16 tiles, no
upsampling); run that test script first to confirm the API path works.

Usage (Colab):
    Set the OPENAI_API_KEY secret, then:
    !python data/selfgen_gptimage_generate.py --indomain 130 --offdomain 20
"""
import argparse
import base64
import sys
from pathlib import Path
from google.colab import userdata

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_THIS_DIR.parent))

from openai import OpenAI

from data.selfgen_generate import atomic_write_bytes, run_generation

MODEL = "gpt-image-2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indomain", type=int, default=130)
    parser.add_argument("--offdomain", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between successful calls")
    parser.add_argument("--max-retries", type=int, default=5, help="retries per prompt before logging a permanent failure")
    args = parser.parse_args()

    api_key = userdata.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first.")

    client = OpenAI(api_key=api_key)

    def generate_one(prompt: str, out_path: Path) -> None:
        response = client.images.generate(
            model=MODEL,
            prompt=prompt,
            size="1024x1024",
            quality="high",
            n=1,
        )
        atomic_write_bytes(out_path, base64.b64decode(response.data[0].b64_json))

    run_generation(
        "gptimage",
        generate_one,
        n_indomain=args.indomain,
        n_offdomain=args.offdomain,
        sleep_s=args.sleep,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
