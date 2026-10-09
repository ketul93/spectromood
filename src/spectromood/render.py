"""Turn audio tracks into images Clef can look at.

Two representations:
  * ``mel``       - a single labelled log-mel spectrogram.
  * ``composite`` - log-mel spectrogram + pitch contour + loudness curve, stacked on a
                    shared time axis. Clef is a general vision-language model rather than
                    a spectrogram specialist, so explicit pitch/loudness plots make the
                    prosodic cues readable instead of leaving them implicit in a heatmap.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spectromood.config import SAMPLE_RATE
from spectromood.features import F0_MAX, F0_MIN, Tracks

REPRESENTATIONS = ("mel", "composite")

_W, _H, _DPI = 9.6, 7.2, 100  # 960x720 px


def _mel_panel(ax: plt.Axes, tracks: Tracks) -> None:
    extent = (0.0, float(tracks.times[-1]) if tracks.times.size else 0.0, 0, tracks.mel_db.shape[0])
    ax.imshow(tracks.mel_db, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=-80, vmax=0)
    # Label a few mel bins with their approximate frequency.
    freqs = librosa.mel_frequencies(n_mels=tracks.mel_db.shape[0], fmax=SAMPLE_RATE / 2)
    ticks = [np.searchsorted(freqs, f) for f in (250, 500, 1000, 2000, 4000)]
    ax.set_yticks(ticks, ["250", "500", "1k", "2k", "4k"])
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (log-mel, brighter = more energy)")


def render_figure(tracks: Tracks, representation: str) -> plt.Figure:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}")
    if representation == "mel":
        fig, ax = plt.subplots(figsize=(_W, _H * 0.6), dpi=_DPI)
        _mel_panel(ax, tracks)
        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    fig, (ax_mel, ax_f0, ax_rms) = plt.subplots(
        3, 1, figsize=(_W, _H), dpi=_DPI, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]}
    )
    _mel_panel(ax_mel, tracks)

    ax_f0.plot(tracks.times, tracks.f0_hz, color="#1f77b4", linewidth=2)
    ax_f0.set_yscale("log")
    ax_f0.set_ylim(F0_MIN, F0_MAX)
    ax_f0.set_yticks([75, 100, 150, 200, 300, 450], ["75", "100", "150", "200", "300", "450"])
    ax_f0.minorticks_off()
    ax_f0.set_ylabel("Pitch (Hz)")
    ax_f0.set_title("Pitch contour (gaps = unvoiced)")
    ax_f0.grid(True, alpha=0.3)

    ax_rms.plot(tracks.times, tracks.rms_db, color="#d62728", linewidth=1.5)
    ax_rms.set_ylim(-60, 0)
    ax_rms.set_ylabel("Loudness (dB)")
    ax_rms.set_xlabel("Time (s)")
    ax_rms.set_title("Loudness envelope")
    ax_rms.grid(True, alpha=0.3)
    if tracks.times.size:
        ax_rms.set_xlim(0, float(tracks.times[-1]))

    fig.tight_layout()
    return fig


def figure_to_jpeg(fig: plt.Figure, max_bytes: int) -> bytes:
    """Encode as JPEG, lowering quality (then size) until it fits within max_bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI)
    plt.close(fig)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    while True:
        for quality in (85, 75, 65, 55, 45):
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            if out.tell() <= max_bytes:
                return out.getvalue()
        if img.width < 320:
            raise ValueError(f"Cannot fit image under {max_bytes} bytes")
        img = img.resize((int(img.width * 0.8), int(img.height * 0.8)), Image.LANCZOS)


def render_jpeg(tracks: Tracks, representation: str, max_bytes: int) -> bytes:
    return figure_to_jpeg(render_figure(tracks, representation), max_bytes)


def to_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
