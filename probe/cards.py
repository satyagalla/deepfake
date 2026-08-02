"""Evidence cards + fusion (`0004` §8; verdict semantics per `0005` §6).

Six cards. Fusion is **evidence-level**: each card is computed
independently from its own signal (CLIP embedding, FFT spectrum, file
metadata) and the fused score combines card scores -- never a single
feature-level classifier wearing six labels.

**Cards do not carry verdicts** (`0005` §6.3). A card does exactly two
things: produce a `score` (P(synthetic) in [0,1]), or set
`silent_because`. `STRIPPED` / `NOT_APPLICABLE` / `PLAUSIBLE` were never
verdicts -- they were *reasons for silence*, and saying so removes three
words from the vocabulary without losing information. The rule they
encoded is unchanged: **absent metadata is absent evidence**, excluded
from the fused score rather than scored 0.5.

Verdicts are a fusion-level concept, and there are **two orthogonal
fields**, because the score and the model's entitlement to it are
different things (`0005` §6.2):

- `verdict`    -- DECLARED_SYNTHETIC | LIKELY_SYNTHETIC | WEAK_EVIDENCE | NO_EVIDENCE
- `reliability`-- IN_DISTRIBUTION | UNKNOWN_SOURCE

`AUTHENTIC` is deliberately absent: absence of synthesis evidence is not
evidence of capture, and this system cannot establish the latter.
`UNKNOWN_SOURCE` **does not suppress the verdict** -- Midjourney at
0-shot should read `LIKELY_SYNTHETIC / UNKNOWN_SOURCE`, the right answer
correctly disclaimed, which is a better demonstration of the gate than a
blank field.

Cut points come from a false-positive budget on the real val split
(`0005` §7), fitted by `probe/fit_production.py`, with `config.py`'s
constants as the fallback.

**Two known defects, documented not fixed** (`0005` §6.4): the unweighted
mean makes LIKELY_SYNTHETIC unreachable whenever the EXIF card fires on
camera metadata (ceiling `(1.0+1.0+0.05)/3 = 0.68`), and it gives the
weakest card half the vote once metadata-free API output silences EXIF.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VERDICT_LIKELY_SYNTHETIC_THRESHOLD, VERDICT_WEAK_EVIDENCE_THRESHOLD
from probe.heads import HeadA, HeadB, MahalanobisGate, gate_distance, predict_head_b, prob_head_a
from probe.spectral import spectral_score

VERDICTS = ("DECLARED_SYNTHETIC", "LIKELY_SYNTHETIC", "WEAK_EVIDENCE", "NO_EVIDENCE")
RELIABILITIES = ("IN_DISTRIBUTION", "UNKNOWN_SOURCE")


@dataclass
class Card:
    dimension: str
    label: str
    score: float | None  # P(synthetic) in [0,1], or None when silent
    silent_because: str | None  # why this card has nothing to say; None if it scored
    detail: str
    declaration: str | None = None  # "synthetic" iff provenance *declares* it, not infers it

    @property
    def speaks(self) -> bool:
        return self.score is not None and self.silent_because is None


@dataclass
class FusionResult:
    fused_score: float | None
    verdict: str
    reliability: str
    n_cards_used: int
    n_cards_silent: int
    cards: list = field(default_factory=list)


def verdict_from_score(
    score: float,
    likely_threshold: float = VERDICT_LIKELY_SYNTHETIC_THRESHOLD,
    weak_threshold: float = VERDICT_WEAK_EVIDENCE_THRESHOLD,
) -> str:
    """Three-way over the learned evidence. `DECLARED_SYNTHETIC` is not
    reachable here -- a declaration is not an inference, and `fuse_cards`
    promotes it separately so provenance is never averaged into a score
    (`0004` §8.1's quarantine rule)."""
    if score >= likely_threshold:
        return "LIKELY_SYNTHETIC"
    if score >= weak_threshold:
        return "WEAK_EVIDENCE"
    return "NO_EVIDENCE"


# --- AI Model card (Head A + Head B + Mahalanobis gate) ---


def build_ai_model_card(head_a: HeadA, head_b: HeadB, gate: MahalanobisGate, x_row: np.ndarray) -> tuple[Card, str]:
    """x_row: (D,) the pooled+concat CLIP feature vector for one image.
    Returns (card, reliability) -- reliability is a property of the whole
    prediction, not of this card, so it is surfaced separately."""
    X = x_row.reshape(1, -1)
    p_ai = float(prob_head_a(head_a, X)[0])
    gen_labels, gen_confs = predict_head_b(head_b, X)
    gen_label, gen_conf = str(gen_labels[0]), float(gen_confs[0])
    distance = float(gate_distance(gate, X)[0])
    reliability = "UNKNOWN_SOURCE" if distance > gate.threshold else "IN_DISTRIBUTION"

    detail = (
        f"Predicted generator: {gen_label} (Head B confidence {gen_conf:.2f}). "
        f"Mahalanobis distance to nearest known class: {distance:.2f} (gate threshold {gate.threshold:.2f})."
    )
    if reliability == "UNKNOWN_SOURCE":
        detail += (
            " UNKNOWN_SOURCE: this image is outside everything the heads were fitted on, so the generator "
            "name above is meaningless and the score is not entitled to belief. ~30 labelled examples of "
            "this source would fix it -- that is the measurement this build exists to make."
        )

    card = Card(dimension="AI Model", label=f"generator={gen_label}", score=p_ai, silent_because=None, detail=detail)
    return card, reliability


# --- Spectral card (radial FFT, independent of CLIP) ---


def build_spectral_card(image: Image.Image, spectral_fitted: dict) -> Card:
    return Card(
        dimension="Spectral",
        label="radial FFT profile",
        score=spectral_score(spectral_fitted, image),
        silent_because=None,
        detail="Azimuthally-averaged radial FFT log-power profile, logistic-regression scored "
        "independently of the CLIP embedding (a different evidence dimension, not a reuse of the AI Model card's signal).",
    )


# --- EXIF/C2PA card (deterministic, no learning) ---

_CAMERA_EXIF_TAGS = {271: "Make", 272: "Model"}  # PIL Exif tag ids


def _parse_exif(image: Image.Image) -> dict:
    exif = image.getexif()
    make = exif.get(271) if exif else None
    model = exif.get(272) if exif else None
    return {"present": bool(exif) and len(exif) > 0, "make": make, "model": model, "has_camera_tags": bool(make or model)}


def _parse_c2pa(raw_bytes: bytes) -> dict:
    """Best-effort byte-level heuristic, not a full C2PA/JUMBF validator:
    looks for the manifest box signature and, if present, for the
    'trainedAlgorithmicMedia' digitalSourceType claim C2PA-aware
    generators (e.g. Gemini's SynthID pipeline) attach. A miss here is not
    proof of no manifest -- it is why this card is deterministic and
    silent rather than confidently low-scoring when nothing is found.

    Because this cannot *validate* a signature, a manufacturer manifest is
    never treated as positive proof of capture -- there is deliberately no
    VERIFIED_CAPTURE verdict (`0005` §8)."""
    lowered = raw_bytes.lower()
    has_manifest = b"c2pa" in lowered or b"jumb" in lowered
    claims_ai_generated = b"trainedalgorithmicmedia" in lowered.replace(b"_", b"")
    return {"has_manifest": has_manifest, "claims_ai_generated": claims_ai_generated}


def build_exif_card(image_path: Path | None) -> Card:
    if image_path is None or not Path(image_path).exists():
        return Card(
            dimension="EXIF", label="no source file", score=None,
            silent_because="no underlying file to inspect (in-memory or re-encoded image)",
            detail="Absent metadata is absent evidence -- excluded from the fused score rather than scored neutral.",
        )
    image_path = Path(image_path)
    raw = image_path.read_bytes()
    with Image.open(image_path) as image:
        exif_info = _parse_exif(image)
    c2pa_info = _parse_c2pa(raw)

    if c2pa_info["claims_ai_generated"]:
        return Card(
            dimension="EXIF", label="C2PA: AI-generated content declared", score=1.0, silent_because=None,
            declaration="synthetic",
            detail="C2PA manifest declares a 'trainedAlgorithmicMedia' digital source type. This is a "
            "declaration, not an inference -- it is reported as its own verdict rather than averaged "
            "into the learned score (`0004` §8.1).",
        )
    if not exif_info["present"] and not c2pa_info["has_manifest"]:
        return Card(
            dimension="EXIF", label="no metadata", score=None,
            silent_because="no EXIF or C2PA manifest found",
            detail="Absent metadata is absent evidence -- excluded from the fused score, not scored 0.5. "
            "Most images off any platform have EXIF stripped, so this is the common case, not the edge case.",
        )
    if exif_info["has_camera_tags"] and not c2pa_info["has_manifest"]:
        return Card(
            dimension="EXIF", label=f"camera metadata ({exif_info['make']} {exif_info['model']})",
            score=0.05, silent_because=None,
            detail="EXIF Make/Model tags present, no C2PA AI-generation claim found. Weak evidence only: "
            "EXIF is trivially copyable, and this card cannot validate a signature.",
        )
    return Card(
        dimension="EXIF", label="metadata present, inconclusive", score=0.5, silent_because=None,
        detail="Metadata is present but neither a clear camera signature nor an AI-generation claim was found.",
    )


# --- honest stubs (§10) ---


def build_diffusion_card() -> Card:
    return Card(
        dimension="Diffusion", label="not implemented", score=None,
        silent_because="not implemented -- documented stub",
        detail="Demoted to a documented stub (0004 §10): VAE round-trip reconstruction error (AEROBLADE) "
        "is a matched-decoder signal (cross-decoder mAP 0.543-0.623, vs 0.992 matched), and Gemini/GPT-Image "
        "decode through VQ tokenizers, not an SD-family VAE -- least competent exactly where the live test happens.",
    )


def build_temporal_card() -> Card:
    return Card(
        dimension="Temporal", label="not applicable", score=None,
        silent_because="input is a still image -- no temporal signal exists",
        detail="Temporal consistency has nothing to measure on a single frame.",
    )


def build_web_intelligence_card() -> Card:
    return Card(
        dimension="Web Intelligence", label="not implemented", score=None,
        silent_because="no reverse-image-search index available",
        detail="Out of scope for this build.",
    )


# --- fusion ---


def fuse_cards(
    cards: list[Card],
    reliability: str = "IN_DISTRIBUTION",
    likely_threshold: float = VERDICT_LIKELY_SYNTHETIC_THRESHOLD,
    weak_threshold: float = VERDICT_WEAK_EVIDENCE_THRESHOLD,
) -> FusionResult:
    scoreable = [c for c in cards if c.speaks]
    n_silent = len(cards) - len(scoreable)
    fused_score = float(np.mean([c.score for c in scoreable])) if scoreable else None

    if any(c.declaration == "synthetic" for c in cards):
        verdict = "DECLARED_SYNTHETIC"  # a declaration outranks any inference
    elif fused_score is None:
        verdict = "NO_EVIDENCE"
    else:
        verdict = verdict_from_score(fused_score, likely_threshold, weak_threshold)

    return FusionResult(
        fused_score=fused_score, verdict=verdict, reliability=reliability,
        n_cards_used=len(scoreable), n_cards_silent=n_silent, cards=cards,
    )


def run_all_cards(
    image: Image.Image,
    image_path: Path | None,
    x_row: np.ndarray,
    head_a: HeadA,
    head_b: HeadB,
    gate: MahalanobisGate,
    spectral_fitted: dict,
    likely_threshold: float = VERDICT_LIKELY_SYNTHETIC_THRESHOLD,
    weak_threshold: float = VERDICT_WEAK_EVIDENCE_THRESHOLD,
) -> tuple[list[Card], FusionResult]:
    ai_card, reliability = build_ai_model_card(head_a, head_b, gate, x_row)
    cards = [
        ai_card,
        build_spectral_card(image, spectral_fitted),
        build_exif_card(image_path),
        build_diffusion_card(),
        build_temporal_card(),
        build_web_intelligence_card(),
    ]
    fusion = fuse_cards(cards, reliability=reliability, likely_threshold=likely_threshold, weak_threshold=weak_threshold)
    return cards, fusion
