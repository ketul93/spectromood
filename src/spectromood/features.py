"""Acoustic features: drawn in the composite plots and given to Clef as text.

Besides pitch and loudness, this measures the voice-quality cues the speech-emotion
literature relies on (e.g. Banse & Scherer 1996; Juslin & Laukka 2003; the eGeMAPS set):
spectral tilt, harmonics-to-noise ratio, short-term pitch/loudness instability, contour
slope, pauses and attack sharpness. Absolute values vary a lot between speakers, so
`feature_vector` can express them relative to the same speaker's calm baseline recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import librosa
import numpy as np

from spectromood.config import SAMPLE_RATE

# Bump when feature definitions change: invalidates cached features in data/features/.
FEATURES_VERSION = "2"

HOP = 160  # 10 ms at 16 kHz
N_FFT = 1024
F0_MIN, F0_MAX = 60.0, 600.0


@dataclass
class Tracks:
    """Frame-level curves shared by the plots and the summary features."""

    mel_db: np.ndarray  # (n_mels, frames)
    f0_hz: np.ndarray  # (frames,), NaN where unvoiced
    rms_db: np.ndarray  # (frames,)
    times: np.ndarray  # (frames,) seconds


@dataclass
class Prosody:
    duration_s: float
    voiced_fraction: float
    pause_fraction: float
    pitch_median_hz: float
    pitch_range_st: float
    pitch_variability_st: float
    pitch_slope_st_per_s: float
    final_pitch_change_st: float
    pitch_instability_pct: float
    level_dbfs: float
    loudness_range_db: float
    loudness_instability_db: float
    attack_db_per_s: float
    spectral_tilt_db: float
    brightness_hz: float
    hnr_db: float
    onsets_per_s: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> Prosody:
        """Build from untrusted JSON (e.g. a baseline sent by the browser)."""
        values = {}
        for f in fields(cls):
            v = float(data[f.name])
            if not np.isfinite(v):
                raise ValueError(f"{f.name} must be finite")
            values[f.name] = v
        return cls(**values)


def mean_prosody(items: list[Prosody]) -> Prosody:
    """Average several recordings, e.g. a corpus baseline when speakers are unknown."""
    if not items:
        raise ValueError("need at least one recording")
    return Prosody(**{f.name: float(np.mean([getattr(p, f.name) for p in items])) for f in fields(Prosody)})


def compute_tracks(y: np.ndarray) -> Tracks:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=128, fmax=SAMPLE_RATE / 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    f0, voiced, _ = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE, frame_length=N_FFT, hop_length=HOP)
    f0 = np.where(voiced, f0, np.nan)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-5))
    n = min(mel_db.shape[1], f0.shape[0], rms_db.shape[0])
    times = librosa.frames_to_time(np.arange(n), sr=SAMPLE_RATE, hop_length=HOP)
    return Tracks(mel_db=mel_db[:, :n], f0_hz=f0[:n], rms_db=rms_db[:n], times=times)


def _semitones(f0: np.ndarray, ref: float) -> np.ndarray:
    return 12.0 * np.log2(f0 / ref)


def _hnr_db(y: np.ndarray, f0: np.ndarray, max_frames: int = 150) -> float:
    """Median harmonics-to-noise ratio over voiced frames (Boersma-style autocorrelation).

    The windowed autocorrelation is divided by the window's own autocorrelation, and the
    peak is searched within +/-10% of the pYIN period, so it measures periodicity at the
    known pitch rather than picking up octave errors.
    """
    idx = np.flatnonzero(~np.isnan(f0))
    if idx.size == 0:
        return 0.0
    idx = idx[:: max(1, idx.size // max_frames)]
    padded = np.pad(y, N_FFT // 2)  # pYIN frames are centred
    window = np.hanning(N_FFT)
    n = 2 * N_FFT
    win_ac = np.fft.irfft(np.abs(np.fft.rfft(window, n)) ** 2)[:N_FFT]
    win_ac /= win_ac[0]
    hnrs = []
    for i in idx:
        frame = padded[i * HOP : i * HOP + N_FFT]
        if frame.size < N_FFT:
            continue
        x = (frame - frame.mean()) * window
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2)[:N_FFT]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0] / np.maximum(win_ac, 1e-6)
        lag = SAMPLE_RATE / f0[i]
        lo, hi = int(lag * 0.9), min(int(np.ceil(lag * 1.1)) + 1, N_FFT // 2)
        if lo >= hi:
            continue
        r = float(np.clip(ac[lo:hi].max(), 1e-3, 0.999))
        hnrs.append(10.0 * np.log10(r / (1.0 - r)))
    return float(np.median(hnrs)) if hnrs else 0.0


def summarise(y: np.ndarray, tracks: Tracks, level_dbfs: float = 0.0) -> Prosody:
    """Summarise a prepared (trimmed, peak-normalised) clip.

    level_dbfs is measured on the signal *before* normalisation, since normalising
    removes how loud the speaker actually was.
    """
    f0 = tracks.f0_hz
    voiced_mask = ~np.isnan(f0)
    voiced = f0[voiced_mask]
    duration = y.size / SAMPLE_RATE

    median = pitch_range = pitch_var = slope = final = instability = 0.0
    if voiced.size >= 5:
        median = float(np.median(voiced))
        st = _semitones(voiced, median)
        pitch_range = float(np.percentile(st, 95) - np.percentile(st, 5))
        pitch_var = float(np.std(st))
        slope = float(np.polyfit(tracks.times[voiced_mask], st, 1)[0])
        final = float(np.median(st[-max(3, st.size // 4) :]))
        # Frame-to-frame period perturbation within voiced stretches (a coarse jitter).
        both = voiced_mask[1:] & voiced_mask[:-1]
        if both.any():
            periods = 1.0 / f0
            instability = float(np.mean(np.abs(np.diff(periods))[both]) / np.nanmean(periods) * 100)

    # Frames within 40 dB of the peak count as speech for loudness statistics.
    speech_mask = tracks.rms_db > tracks.rms_db.max() - 40.0
    speech = tracks.rms_db[speech_mask]
    loud_range = float(np.percentile(speech, 95) - np.percentile(speech, 10)) if speech.size else 0.0
    pause = float(np.mean(tracks.rms_db < tracks.rms_db.max() - 30.0))
    both = voiced_mask[1:] & voiced_mask[:-1]
    loud_instability = float(np.mean(np.abs(np.diff(tracks.rms_db))[both])) if both.any() else 0.0
    rises = np.diff(tracks.rms_db) / (HOP / SAMPLE_RATE)
    attack = float(np.percentile(rises[rises > 0], 95)) if (rises > 0).any() else 0.0

    power = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    power = power[:, : f0.size]
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    frames = power[:, voiced_mask] if voiced_mask.any() else power
    low = frames[(freqs >= 50) & (freqs < 1000)].sum()
    high = frames[(freqs >= 1000) & (freqs < 5000)].sum()
    tilt = float(10.0 * np.log10(max(high, 1e-12) / max(low, 1e-12)))
    spectrum = frames.sum(axis=1)
    brightness = float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))

    onsets = librosa.onset.onset_detect(y=y, sr=SAMPLE_RATE, hop_length=HOP, units="time")

    return Prosody(
        duration_s=duration,
        voiced_fraction=float(voiced.size / max(f0.size, 1)),
        pause_fraction=pause,
        pitch_median_hz=median,
        pitch_range_st=pitch_range,
        pitch_variability_st=pitch_var,
        pitch_slope_st_per_s=slope,
        final_pitch_change_st=final,
        pitch_instability_pct=instability,
        level_dbfs=level_dbfs,
        loudness_range_db=loud_range,
        loudness_instability_db=loud_instability,
        attack_db_per_s=attack,
        spectral_tilt_db=tilt,
        brightness_hz=brightness,
        hnr_db=_hnr_db(y, f0),
        onsets_per_s=float(len(onsets) / duration) if duration > 0 else 0.0,
    )


# (key, human label, unit) for what Clef sees. Absolute level is left out when there is
# no baseline: it depends on microphone gain, not on the speaker.
_ABSOLUTE = [
    ("pitch_median_hz", "median pitch", "Hz"),
    ("pitch_range_st", "pitch range", "st"),
    ("pitch_variability_st", "pitch variability", "st"),
    ("pitch_slope_st_per_s", "pitch slope", "st/s"),
    ("final_pitch_change_st", "final pitch vs median", "st"),
    ("pitch_instability_pct", "pitch instability", "%"),
    ("loudness_range_db", "loudness range", "dB"),
    ("loudness_instability_db", "loudness instability", "dB"),
    ("attack_db_per_s", "attack sharpness", "dB/s"),
    ("spectral_tilt_db", "spectral tilt (1-5k vs <1k)", "dB"),
    ("hnr_db", "harmonics-to-noise", "dB"),
    ("onsets_per_s", "speech rate", "onsets/s"),
    ("pause_fraction", "pause fraction", ""),
]
_RELATIVE = [
    ("pitch_vs_baseline_st", "pitch vs baseline", "st"),
    ("pitch_range_vs_baseline", "pitch range vs baseline", "x"),
    ("level_vs_baseline_db", "loudness vs baseline", "dB"),
    ("tilt_vs_baseline_db", "spectral tilt vs baseline", "dB"),
    ("hnr_vs_baseline_db", "harmonics-to-noise vs baseline", "dB"),
    ("rate_vs_baseline", "speech rate vs baseline", "x"),
    ("pitch_instability_vs_baseline", "pitch instability vs baseline", "x"),
]


def _ratio(a: float, b: float) -> float:
    return a / b if b > 1e-6 else 1.0


def feature_vector(p: Prosody, baseline: Prosody | None = None) -> dict[str, float]:
    """The features shown to Clef (and used by the reference classifier), in a fixed order."""
    vec = {key: getattr(p, key) for key, _, _ in _ABSOLUTE}
    if baseline is not None:
        pitch_rel = (
            12.0 * np.log2(p.pitch_median_hz / baseline.pitch_median_hz)
            if p.pitch_median_hz > 0 and baseline.pitch_median_hz > 0
            else 0.0
        )
        vec.update(
            pitch_vs_baseline_st=float(pitch_rel),
            pitch_range_vs_baseline=_ratio(p.pitch_range_st, baseline.pitch_range_st),
            level_vs_baseline_db=p.level_dbfs - baseline.level_dbfs,
            tilt_vs_baseline_db=p.spectral_tilt_db - baseline.spectral_tilt_db,
            hnr_vs_baseline_db=p.hnr_db - baseline.hnr_db,
            rate_vs_baseline=_ratio(p.onsets_per_s, baseline.onsets_per_s),
            pitch_instability_vs_baseline=_ratio(p.pitch_instability_pct, baseline.pitch_instability_pct),
        )
    return vec


def _labels(with_baseline: bool) -> list[tuple[str, str, str]]:
    return _ABSOLUTE + (_RELATIVE if with_baseline else [])


def features_text(p: Prosody, baseline: Prosody | None = None) -> str:
    vec = feature_vector(p, baseline)
    parts = [f"{label} {vec[key]:+.2f}{unit}" if "vs" in key else f"{label} {vec[key]:.2f}{unit}"
             for key, label, unit in _labels(baseline is not None)]  # fmt: skip
    head = "Measured acoustic features of this recording"
    if baseline is not None:
        head += ' ("vs baseline" compares with the same speaker\'s calm reference recording)'
    return head + ": " + "; ".join(parts) + "."
