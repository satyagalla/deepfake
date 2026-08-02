"""New-path Gradio demo (`0004` §8, build plan S8) -- no face crop, no
MTCNN. Scores the whole upload through the six evidence cards
(`probe/cards.py`) and shows the fused verdict alongside the E1 N-shot
adaptation curve for whichever generator Head B names, with this upload's
score marked on it -- so a live result lands on a chart that was already
on screen (`0004` §12's Monday sequence, step 4).

Requires `probe/fit_production.py` to have been run first (saves Head A,
Head B, the Mahalanobis gate, and the spectral head to
config.PROBE_CHECKPOINT_DIR), and ideally `probe/experiments.py` (for the
E1 curve backdrop -- the demo still works without it, just without the
curve).

Usage:
    python -m probe.demo
    python -m probe.demo --share
"""
import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_POOL_METHOD, GATE_NAME, HEAD_A_NAME, HEAD_B_NAME, PROBE_OUTPUTS_DIR, SPECTRAL_HEAD_NAME
from probe.cards import run_all_cards
from probe.extract import build_clip, featurize_single_image
from probe.heads import load


class Predictor:
    """Loads the CLIP backbone + fitted heads once; call .predict(path) per request."""

    def __init__(self, pool_method: str = DEFAULT_POOL_METHOD):
        self.pool_method = pool_method
        print("Loading CLIP backbone...")
        self.model, self.preprocess_val = build_clip()
        print("Loading fitted heads (run `python -m probe.fit_production` first if this fails)...")
        self.head_a = load(HEAD_A_NAME)
        self.head_b = load(HEAD_B_NAME)
        self.gate = load(GATE_NAME)
        self.spectral_fitted = load(SPECTRAL_HEAD_NAME)

        from config import VERDICT_LIKELY_SYNTHETIC_THRESHOLD, VERDICT_WEAK_EVIDENCE_THRESHOLD, VERDICT_THRESHOLDS_NAME
        try:
            cuts = load(VERDICT_THRESHOLDS_NAME)
            self.likely_threshold = cuts["likely_synthetic"]
            self.weak_threshold = cuts["weak_evidence"]
            print(f"Loaded FPR-derived verdict cut points: {cuts}")
        except Exception:
            self.likely_threshold = VERDICT_LIKELY_SYNTHETIC_THRESHOLD
            self.weak_threshold = VERDICT_WEAK_EVIDENCE_THRESHOLD
            print("No fitted verdict thresholds found -- falling back to config defaults (0.85 / 0.5).")

        self.e1_results = self._load_e1_results()

    @staticmethod
    def _load_e1_results() -> dict:
        path = PROBE_OUTPUTS_DIR / "experiments.json"
        if not path.exists():
            print(f"NOTE: {path} not found -- E1 curve panel will be blank. Run `python -m probe.experiments` first.")
            return {}
        return json.loads(path.read_text())

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        x_row = featurize_single_image(image, self.model, self.preprocess_val, pool_method=self.pool_method)
        cards, fusion = run_all_cards(
            image, Path(image_path), x_row, self.head_a, self.head_b, self.gate, self.spectral_fitted,
            likely_threshold=self.likely_threshold, weak_threshold=self.weak_threshold,
        )
        return image, cards, fusion


def render_curve_with_point(e1_results: list[dict], generator: str, live_score: float | None) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if e1_results:
        ns = sorted(set(r["n"] for r in e1_results))
        means = [float(np.mean([r["auc"] for r in e1_results if r["n"] == n])) for n in ns]
        stds = [float(np.std([r["auc"] for r in e1_results if r["n"] == n])) for n in ns]
        for r in e1_results:
            ax.scatter(r["n"], r["auc"], alpha=0.3, color="gray", s=15, zorder=1)
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4, color="C0", zorder=2, label="pre-registered E1 curve (AUC)")
    else:
        ax.text(0.5, 0.5, f"No pre-computed E1 curve for '{generator}'\n(run probe.experiments first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")

    if live_score is not None:
        ax.axhline(live_score, color="red", linestyle="--", alpha=0.7, zorder=3,
                    label=f"this upload: AI Model P(AI)={live_score:.2f}")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel("N images from target generator")
    ax.set_ylabel("AUC / P(AI)")
    ax.set_title(f"E1: N-shot adaptation curve -- {generator}")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def cards_to_rows(cards) -> list[list]:
    return [
        [c.dimension, c.label,
         f"{c.score:.3f}" if c.score is not None else "--",
         "--" if c.speaks else c.silent_because,
         c.detail]
        for c in cards
    ]


def build_app(pool_method: str = DEFAULT_POOL_METHOD) -> gr.Blocks:
    predictor = Predictor(pool_method)

    def run(image_path):
        if image_path is None:
            return None, "", [], None

        image, cards, fusion = predictor.predict(image_path)
        ai_card = next(c for c in cards if c.dimension == "AI Model")
        predicted_generator = ai_card.label.split("=", 1)[-1]

        if fusion.fused_score is not None:
            verdict_str = (f"## {fusion.verdict}\n\n"
                           f"**{fusion.reliability}** · fused score {fusion.fused_score:.2f} · "
                           f"{fusion.n_cards_used} card(s) used, {fusion.n_cards_silent} silent")
        else:
            verdict_str = (f"## {fusion.verdict}\n\n**{fusion.reliability}** · "
                           "no scoreable evidence -- every card was silent")
        if fusion.reliability == "UNKNOWN_SOURCE":
            verdict_str += (
                "\n\n> **UNKNOWN_SOURCE** -- outside everything the heads were fitted on. The verdict above "
                "is shown rather than blanked, but it is not entitled to belief and Head B's generator name "
                "is meaningless. **~30 labelled examples of this source would fix it** -- that is the price "
                "this build exists to measure."
            )

        e1_key = f"e1_{predicted_generator}"
        curve_img = render_curve_with_point(predictor.e1_results.get(e1_key, []), predicted_generator, ai_card.score)
        return image, verdict_str, cards_to_rows(cards), curve_img

    with gr.Blocks(title="Adaptation Probe Demo") as demo:
        gr.Markdown(
            "# Adaptation Hypothesis Demo (0004)\n"
            "Upload any image -- no face crop. Six independent evidence cards are fused into one verdict. "
            "The chart on the right shows the pre-registered N-shot adaptation curve for whichever generator "
            "the AI Model card names, with this upload's score marked on it."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="filepath", label="Upload image")
                btn = gr.Button("Analyze", variant="primary")
                out_image = gr.Image(label="Input")
            with gr.Column():
                out_verdict = gr.Markdown()
                out_curve = gr.Image(label="E1: N-shot adaptation curve")
        out_cards = gr.Dataframe(
            headers=["dimension", "label", "score", "silent because", "detail"],
            label="Evidence cards",
            wrap=True,
        )
        btn.click(run, inputs=inp, outputs=[out_image, out_verdict, out_cards, out_curve])
        inp.change(run, inputs=inp, outputs=[out_image, out_verdict, out_cards, out_curve])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-method", default=DEFAULT_POOL_METHOD)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    args = parser.parse_args()

    demo = build_app(args.pool_method)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
