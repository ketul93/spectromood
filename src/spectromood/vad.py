"""Voice activity detection with Silero VAD (MIT, github.com/snakers4/silero-vad, v5.1.2).

Used to skip windows that contain no human speech (traffic, typing, music, a fan...) so they
are never sent to Clef. The 2 MB ONNX model ships in spectromood/assets/.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from spectromood.config import SAMPLE_RATE

MODEL = Path(__file__).parent / "assets" / "silero_vad.onnx"
CHUNK = 512  # samples per step at 16 kHz (32 ms)
CONTEXT = 64  # samples of the previous chunk the v5 model expects in front of each chunk
THRESHOLD = 0.5  # speech probability above which a chunk counts as speech

# Minimum amount of speech for a window to be analysed / for calibration to be accepted.
MIN_SPEECH_S = float(os.getenv("VAD_MIN_SPEECH_S", "0.5"))
MIN_BASELINE_SPEECH_S = float(os.getenv("VAD_MIN_BASELINE_SPEECH_S", "2.0"))


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])


def speech_probabilities(y: np.ndarray) -> np.ndarray:
    """Speech probability for each 32 ms chunk of a mono 16 kHz signal."""
    session = _session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, CONTEXT), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    y = np.asarray(y, dtype=np.float32)
    n = len(y) // CHUNK
    probs = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = np.concatenate([context, y[None, i * CHUNK : (i + 1) * CHUNK]], axis=1)
        out, state = session.run(None, {"input": x, "state": state, "sr": sr})
        context = x[:, -CONTEXT:]
        probs[i] = float(out[0, 0])
    return probs


def speech_seconds(y: np.ndarray) -> float:
    """Total duration of chunks the model considers speech."""
    return float(np.sum(speech_probabilities(y) > THRESHOLD) * CHUNK / SAMPLE_RATE)
