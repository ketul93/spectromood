"""Audio -> (image, state, questions) -> Clef. Shared by the evaluation script and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectromood.audio import frame_levels_db, prepare
from spectromood.clef import ClefClient, ClefResult
from spectromood.features import Prosody, Tracks, compute_tracks, features_text, summarise
from spectromood.prompts import GUIDE_PREAMBLE, build_questions
from spectromood.render import render_jpeg, to_data_uri

_MEL_CONTEXT = (
    "The attached image is a log-mel spectrogram of a short speech recording "
    "(time on the x-axis, frequency on the y-axis, brighter colours mean more energy). "
    "Judge the speaker's emotion from the vocal delivery visible in it: energy, "
    "pitch/harmonic patterns, rhythm and abruptness. The words themselves are not available."
)
_COMPOSITE_CONTEXT = (
    "The attached image visualises a short speech recording in three panels sharing a "
    "time axis: (1) a log-mel spectrogram, brighter = more energy; (2) the pitch (F0) "
    "contour in Hz on a log scale, with gaps where the voice is unvoiced; (3) the loudness "
    "envelope in dB. Judge the speaker's emotion from the vocal delivery: pitch level and "
    "movement, loudness and its dynamics, speaking rhythm and voice quality. The words "
    "themselves are not available."
)
_FEATURES_CONTEXT = (
    "A short speech recording was analysed acoustically; the words are not available. "
    "Judge the speaker's emotion from these vocal delivery measurements."
)


@dataclass(frozen=True)
class Variant:
    image: str | None  # "mel", "composite" or None
    features: bool = False  # measured features as text (relative to the speaker's baseline when given)
    guide: bool = False  # literature cue profiles in the state and the answer options

    @property
    def context(self) -> str:
        return {"mel": _MEL_CONTEXT, "composite": _COMPOSITE_CONTEXT, None: _FEATURES_CONTEXT}[self.image]


# The ablation reported in the README. "plots+features+guide" is the recommended setup.
VARIANTS: dict[str, Variant] = {
    "spectrogram": Variant("mel"),
    "plots": Variant("composite"),
    "features": Variant(None, features=True),
    "plots+features+guide": Variant("composite", features=True, guide=True),
}
DEFAULT_VARIANT = "plots+features+guide"


@dataclass
class Prepared:
    tracks: Tracks | None
    prosody: Prosody
    images: dict[str, bytes]  # representation -> jpeg


def active_level_dbfs(y: np.ndarray) -> float:
    """Level of the loudest speech frames (90th percentile), before any normalisation."""
    return float(np.percentile(frame_levels_db(y), 90))


def prepare_audio(y: np.ndarray, representations: set[str], max_image_bytes: int) -> Prepared:
    level = active_level_dbfs(y)
    y = prepare(y)
    tracks = compute_tracks(y)
    prosody = summarise(y, tracks, level)
    images = {rep: render_jpeg(tracks, rep, max_image_bytes) for rep in sorted(representations)}
    return Prepared(tracks=tracks, prosody=prosody, images=images)


def build_request(
    prepared: Prepared, variant: str, baseline: Prosody | None = None
) -> tuple[str, list[str], dict[str, Any]]:
    """Return (state, image data URIs, questions) for a variant.

    Without a baseline (e.g. before the dashboard user calibrates), features are absolute.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    v = VARIANTS[variant]
    parts = [v.context]
    if v.features:
        parts.append(features_text(prepared.prosody, baseline))
    if v.guide:
        parts.append(GUIDE_PREAMBLE)
    images = [to_data_uri(prepared.images[v.image])] if v.image else []
    return "\n\n".join(parts), images, build_questions(guided=v.guide)


def representations_for(variants: list[str]) -> set[str]:
    return {VARIANTS[v].image for v in variants if VARIANTS[v].image}


def classify(client: ClefClient, prepared: Prepared, variant: str, baseline: Prosody | None = None) -> ClefResult:
    state, images, questions = build_request(prepared, variant, baseline)
    return client.classify(state, images, questions)
