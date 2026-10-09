"""Audio loading and normalisation."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from spectromood.config import SAMPLE_RATE

# A window is skipped (not sent to Clef) when its loudest frames are below SILENCE_DBFS,
# or when loud and quiet frames differ by less than MIN_CONTRAST_DB. Speech always swings
# between syllables and pauses; steady room noise/hum does not. An absolute threshold
# alone fails because quiet speakers and noisy rooms overlap in level.
SILENCE_DBFS = -70.0
MIN_CONTRAST_DB = 12.0
_FRAME = SAMPLE_RATE // 20  # 50 ms


def _to_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y


def load_file(path: str | Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return _to_mono_16k(y, sr)


def load_bytes(data: bytes) -> np.ndarray:
    """Decode WAV/FLAC/OGG bytes (anything libsndfile reads) to mono 16 kHz float32."""
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return _to_mono_16k(y, sr)


def level_dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


def frame_levels_db(y: np.ndarray) -> np.ndarray:
    n = y.size // _FRAME
    if n == 0:
        return np.array([-200.0])
    frames = y[: n * _FRAME].reshape(n, _FRAME)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-10))


def is_silent(y: np.ndarray) -> bool:
    levels = frame_levels_db(y)
    loud, quiet = np.percentile(levels, 95), np.percentile(levels, 10)
    return bool(loud < SILENCE_DBFS or loud - quiet < MIN_CONTRAST_DB)


def prepare(y: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence and peak-normalise so loudness cues are comparable."""
    trimmed, _ = librosa.effects.trim(y, top_db=35)
    if trimmed.size < SAMPLE_RATE // 4:
        trimmed = y
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    return trimmed / peak * 0.95 if peak > 0 else trimmed
