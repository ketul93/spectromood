"""Shared constants and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Sample rate every clip is resampled to before analysis.
SAMPLE_RATE = 16_000

# Emotion label -> description given to Clef as the choice criteria.
# Keep keys in sync with the RAVDESS subset chosen by scripts/download_dataset.py.
EMOTIONS: dict[str, str] = {
    "angry": "Anger or rage: forceful, loud, tense, sharp attacks, raised and harsh voice.",
    "fearful": "Fear or anxiety: tense, trembling or breathy, often high and unstable pitch.",
    "happy": "Happiness or joy: bright, energetic, lively pitch movement, upbeat.",
    "sad": "Sadness: low energy, slow, quiet, low and flat pitch, falling intonation.",
    "neutral": "Neutral: calm, matter-of-fact delivery with no notable emotional colouring.",
}

AROUSAL_LEVELS = ["very calm", "calm", "moderate", "excited", "highly agitated"]

# clef-flash (9B) is the only model used: it beat the larger `clef` in every setup we tried,
# at about half the latency and a third of the price.
CLEF_MODEL = "clef-flash"


@dataclass(frozen=True)
class Settings:
    account_id: str
    api_token: str
    model: str
    timeout_s: float
    max_image_bytes: int
    # Optional: call Clef through the worker/ proxy instead of the REST API.
    worker_url: str = ""
    worker_token: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip(),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN", "").strip(),
            model=CLEF_MODEL,
            timeout_s=float(os.getenv("CLEF_TIMEOUT_S", "90")),
            # Image bytes count against Clef's 65k-token context, so keep them small.
            max_image_bytes=int(os.getenv("CLEF_MAX_IMAGE_BYTES", "60000")),
            worker_url=os.getenv("CLEF_WORKER_URL", "").strip(),
            worker_token=os.getenv("CLEF_WORKER_TOKEN", "").strip(),
        )

    def require_credentials(self) -> None:
        if self.worker_url:
            return
        missing = [
            name
            for name, value in (
                ("CLOUDFLARE_ACCOUNT_ID", self.account_id),
                ("CLOUDFLARE_API_TOKEN", self.api_token),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing environment variables: {', '.join(missing)} (see .env.example), or set CLEF_WORKER_URL"
            )
