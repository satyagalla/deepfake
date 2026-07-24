"""Standalone Gradio demo: upload any image, get the 3-way prediction with
class probabilities, gate contribution weights, and a Grad-CAM overlay --
the same explainability pair reported in eval.py, on arbitrary user images
instead of the val split.

Reuses the exact face-crop settings from data/face_filter.py and the exact
per-item input construction from model/dataset.py's ForgeryDataset, so a
demo prediction is built the same way a training/eval sample is.

Usage:
    python model/demo.py --checkpoint checkpoints/best_model.pt
    python model/demo.py --checkpoint checkpoints/best_model.pt --share
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, DEVICE, EMBED_DIM, FACE_MARGIN, IMAGE_SIZE
from model.branches import SRMFilter
from model.eval import BRANCH_NAMES, load_model
from model.fusion import ForgeryClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Predictor:
    """Loads the model + face detector once; call .predict(image) per request."""

    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        self.device = device
        self.model: ForgeryClassifier = load_model(checkpoint_path, device=device)
        self.mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=FACE_MARGIN,
            select_largest=True,
            keep_all=False,
            post_process=False,
            device=device,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.srm = SRMFilter()

    @staticmethod
    def _fft_magnitude(raw01: torch.Tensor) -> torch.Tensor:
        gray = raw01.mean(dim=0, keepdim=True)
        mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(gray)))
        log_mag = torch.log1p(mag)
        lo, hi = log_mag.min(), log_mag.max()
        return (log_mag - lo) / (hi - lo + 1e-8)

    def _crop_face(self, image: Image.Image) -> Image.Image | None:
        face = self.mtcnn(image.convert("RGB"))
        if face is None:
            return None
        arr = face.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    def predict(self, image: Image.Image):
        """Returns (face_crop, probs dict, gate dict, gradcam_overlay ndarray)
        or (None, None, None, None) if no face was detected."""
        face = self._crop_face(image)
        if face is None:
            return None, None, None, None

        raw01 = self.to_tensor(face)
        rgb = self.normalize(raw01.clone())
        fft_mag = self._fft_magnitude(raw01)
        srm_residual = self.srm(raw01 * 255.0)

        rgb_b = rgb.unsqueeze(0).to(self.device)
        fft_b = fft_mag.unsqueeze(0).to(self.device)
        srm_b = srm_residual.unsqueeze(0).to(self.device)

        self.model.enable_gradcam(True)
        logits, gate_weights = self.model(rgb_b, fft_b, srm_b)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

        self.model.zero_grad(set_to_none=True)
        logits[0, pred_idx].backward()
        feat_map = self.model.spatial.last_feature_map
        weights = feat_map.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=rgb_b.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        self.model.enable_gradcam(False)

        probs_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
        gate_dict = {name: float(gate_weights[0, i]) for i, name in enumerate(BRANCH_NAMES)}
        overlay = self._render_overlay(face, cam.numpy(), probs_dict, gate_dict)
        return face, probs_dict, gate_dict, overlay

    @staticmethod
    def _render_overlay(face: Image.Image, cam: np.ndarray, probs: dict, gate: dict) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(face)
        ax.imshow(cam, cmap="jet", alpha=0.4)
        pred_cls = max(probs, key=probs.get)
        gate_str = ", ".join(f"{k}:{v:.2f}" for k, v in gate.items())
        ax.set_title(f"pred={pred_cls} ({probs[pred_cls]:.2f})\n{gate_str}", fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf


def build_app(checkpoint_path: str, device: str = DEVICE) -> gr.Blocks:
    predictor = Predictor(checkpoint_path, device=device)

    def run(image: Image.Image):
        if image is None:
            return None, None, None
        face, probs, gate, overlay = predictor.predict(image)
        if face is None:
            gr.Warning("No face detected in this image -- the model expects a face crop, per the dataset design.")
            return None, None, None
        return overlay, probs, gate

    with gr.Blocks(title="Forgery Classifier Demo") as demo:
        gr.Markdown(
            "# Real / Edited / Deepfake Classifier\n"
            "Upload any image with a face. The app auto-detects and crops the face "
            "(same MTCNN settings used in training), then runs the 3-branch fusion "
            "model (spatial + spectral + noise-residual). Shows the class probabilities, "
            "each branch's gate contribution weight, and a Grad-CAM overlay on the "
            "predicted class."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload image")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_overlay = gr.Image(label="Grad-CAM overlay (predicted class)")
                out_probs = gr.Label(label="Class probabilities", num_top_classes=3)
                out_gate = gr.Label(label="Gate contribution weights (per branch)", num_top_classes=3)
        btn.click(run, inputs=inp, outputs=[out_overlay, out_probs, out_gate])
        inp.change(run, inputs=inp, outputs=[out_overlay, out_probs, out_gate])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    args = parser.parse_args()

    demo = build_app(args.checkpoint)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
