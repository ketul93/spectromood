"""Realtime dashboard backend: the browser posts short audio windows, we ask Clef.

Run: uvicorn spectromood.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from spectromood.audio import is_silent, level_dbfs, load_bytes
from spectromood.clef import ClefClient, ClefError, ClefQuotaError
from spectromood.config import AROUSAL_LEVELS, EMOTIONS, SAMPLE_RATE, Settings
from spectromood.features import Prosody, feature_vector
from spectromood.pipeline import DEFAULT_VARIANT as PIPELINE_DEFAULT
from spectromood.pipeline import VARIANTS, build_request, prepare_audio
from spectromood.render import to_data_uri
from spectromood.vad import MIN_BASELINE_SPEECH_S, MIN_SPEECH_S, speech_seconds

STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 8 * 1024 * 1024
MIN_SECONDS, MAX_SECONDS = 1.0, 15.0
MIN_BASELINE_SECONDS = 2.0

# The recommended setup (see README); "Calibrate my voice" supplies the speaker baseline.
DEFAULT_VARIANT = os.getenv("DASHBOARD_VARIANT", PIPELINE_DEFAULT)
if DEFAULT_VARIANT not in VARIANTS:
    raise ValueError(f"DASHBOARD_VARIANT must be one of {sorted(VARIANTS)}")


def _warm_up(max_image_bytes: int) -> None:
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 2 * t) ** 2).astype(np.float32)
    prepare_audio(tone, {"composite"}, max_image_bytes)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.client = None
    app.state.config_error = None
    try:
        app.state.client = ClefClient(settings)
    except RuntimeError as exc:
        app.state.config_error = str(exc)
    # Bounds concurrent Clef calls across all browser tabs.
    app.state.clef_slots = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_CLEF", "4")))
    # librosa's numba kernels compile on first use (~5 s); do it now, not on the first request.
    await run_in_threadpool(_warm_up, settings.max_image_bytes)
    yield
    if app.state.client:
        app.state.client.close()


app = FastAPI(title="Speech emotion via Cloudflare Clef", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict:
    s: Settings = request.app.state.settings
    return {
        "ok": request.app.state.client is not None,
        "model": s.model,
        "error": request.app.state.config_error,
        "emotions": list(EMOTIONS),
        "arousal_levels": AROUSAL_LEVELS,
        "variants": list(VARIANTS),
        "default_variant": DEFAULT_VARIANT,
        "baseline_variants": [k for k, v in VARIANTS.items() if v.features],
    }


async def _read_audio(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty body; POST a WAV file")
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, "Audio too large")
    try:
        y = await run_in_threadpool(load_bytes, body)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    seconds = y.size / SAMPLE_RATE
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise HTTPException(400, f"Audio must be {MIN_SECONDS}-{MAX_SECONDS}s, got {seconds:.1f}s")
    return y, seconds


def _parse_baseline(header: str | None) -> Prosody | None:
    if not header:
        return None
    try:
        return Prosody.from_dict(json.loads(header))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, f"Invalid X-Baseline header: {exc}") from exc


@app.post("/api/baseline")
async def baseline(request: Request) -> dict:
    """Measure the speaker's calm reference voice. The browser keeps the result and sends it
    back as the X-Baseline header, so the server stays stateless."""
    y, seconds = await _read_audio(request)
    if seconds < MIN_BASELINE_SECONDS:
        raise HTTPException(400, f"Record at least {MIN_BASELINE_SECONDS:.0f}s of calm speech")
    if is_silent(y):
        raise HTTPException(400, "No speech detected; speak calmly and a little closer to the mic")
    speech_s = await run_in_threadpool(speech_seconds, y)
    if speech_s < MIN_BASELINE_SPEECH_S:
        raise HTTPException(400, f"Only {speech_s:.1f}s of voice detected; keep talking calmly for the whole countdown")
    s: Settings = request.app.state.settings
    prepared = await run_in_threadpool(prepare_audio, y, set(), s.max_image_bytes)
    if prepared.prosody.pitch_median_hz <= 0:
        raise HTTPException(400, "Could not find a voiced pitch; speak normally for a few seconds")
    return {"baseline": prepared.prosody.to_dict(), "seconds": round(seconds, 2), "speech_seconds": round(speech_s, 2)}


@app.post("/api/analyze")
async def analyze(request: Request, variant: str = Query(DEFAULT_VARIANT)) -> dict:
    client: ClefClient | None = request.app.state.client
    if client is None:
        raise HTTPException(503, f"Clef is not configured: {request.app.state.config_error}")
    if variant not in VARIANTS:
        raise HTTPException(400, f"Unknown variant {variant!r}")

    base = _parse_baseline(request.headers.get("x-baseline"))
    y, seconds = await _read_audio(request)
    level = level_dbfs(y)
    if is_silent(y):
        return {"status": "silence", "level_dbfs": round(level, 1), "seconds": round(seconds, 2)}
    # No human voice (noise, music, typing...): never call Clef, so no tokens are spent.
    speech_s = await run_in_threadpool(speech_seconds, y)
    if speech_s < MIN_SPEECH_S:
        return {"status": "no_speech", "speech_seconds": round(speech_s, 2), "seconds": round(seconds, 2)}

    start = time.perf_counter()
    rep = VARIANTS[variant].image
    s: Settings = request.app.state.settings
    prepared = await run_in_threadpool(prepare_audio, y, {rep} if rep else set(), s.max_image_bytes)
    state, images, questions = build_request(prepared, variant, base)
    prep_s = time.perf_counter() - start

    async with request.app.state.clef_slots:
        try:
            result = await run_in_threadpool(client.classify, state, images, questions)
        except ClefQuotaError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ClefError as exc:
            raise HTTPException(502, str(exc)) from exc

    return {
        "status": "ok",
        "variant": variant,
        "model": result.model or s.model,
        "emotion": result.emotion,
        "probabilities": result.probabilities,
        "confidence": result.confidence,
        "arousal": result.arousal,
        "clef_latency_s": round(result.latency_s, 2),
        "prep_s": round(prep_s, 2),
        "level_dbfs": round(level, 1),
        "seconds": round(seconds, 2),
        "speech_seconds": round(speech_s, 2),
        "prosody": prepared.prosody.to_dict(),
        "features": {k: round(v, 2) for k, v in feature_vector(prepared.prosody, base).items()},
        "baseline_used": base is not None and VARIANTS[variant].features,
        "image": to_data_uri(prepared.images[rep]) if rep else None,
    }
