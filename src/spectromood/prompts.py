"""Questions and domain guidance sent to Clef."""

from __future__ import annotations

from typing import Any

from spectromood.config import AROUSAL_LEVELS, EMOTIONS

# Typical acoustic profiles of each emotion relative to the speaker's calm voice, summarised
# from Banse & Scherer (1996) and the Juslin & Laukka (2003) review of vocal expression.
CUE_GUIDE: dict[str, str] = {
    "angry": (
        "Anger: clearly louder than baseline, higher spectral tilt (more energy above 1 kHz, harsh/tense "
        "voice), sharp loudness attacks, raised pitch with a wide range, fast speech rate."
    ),
    "fearful": (
        "Fear: raised pitch (often well above baseline) with high pitch instability or tremor, fast rate, "
        "breathy voice (lower harmonics-to-noise), loudness only moderately raised, pitch range irregular."
    ),
    "happy": (
        "Happiness: raised pitch with a wide, lively pitch range and lots of movement, moderately louder, "
        "moderately fast, clear voice (higher harmonics-to-noise), rising or bouncing contours."
    ),
    "sad": (
        "Sadness: quieter than other emotions, slower with more pauses, lower spectral tilt (dull, little "
        "energy above 1 kHz), soft attacks, falling final pitch. Quiet sadness keeps pitch at or below "
        "baseline with a narrow range; intense, tearful sadness (despair) raises pitch but stays soft and dull."
    ),
    "neutral": (
        "Neutral: close to the speaker's baseline on every measure; moderate pitch range, normal loudness "
        "and rate, no strong deviations in any direction."
    ),
}

GUIDE_PREAMBLE = (
    "How emotions typically change the voice (relative to the speaker's calm voice). Weigh the "
    "measurements against these profiles; no single cue is decisive:\n"
    + "\n".join(f"- {text}" for text in CUE_GUIDE.values())
)


def build_questions(guided: bool = False) -> dict[str, Any]:
    """One 5-way emotion choice plus an arousal score. `guided` swaps the plain option
    descriptions for the acoustic cue profiles above."""
    descriptions = CUE_GUIDE if guided else EMOTIONS
    return {
        "emotion": {
            "type": "choice",
            "instructions": "Which emotion is the speaker expressing in this speech recording?",
            "criteria": {e: descriptions[e] for e in EMOTIONS},
        },
        "arousal": {
            "type": "score",
            "instructions": "How emotionally activated (aroused) does the speaker sound?",
            "criteria": list(AROUSAL_LEVELS),
        },
    }
