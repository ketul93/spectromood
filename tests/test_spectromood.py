import io
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from spectromood import audio
from spectromood.clef import ClefClient, ClefError, ClefQuotaError, ClefResult, parse_response
from spectromood.config import EMOTIONS, SAMPLE_RATE, Settings
from spectromood.datasets import Benchmark, load_prepared, write_csv
from spectromood.features import Prosody, compute_tracks, feature_vector, features_text, mean_prosody, summarise
from spectromood.metrics import confusion, score
from spectromood.pipeline import DEFAULT_VARIANT, VARIANTS, build_request, prepare_audio
from spectromood.prompts import build_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_data import select_clips

SETTINGS = Settings(account_id="acc", api_token="tok", model="clef-flash", timeout_s=5, max_image_bytes=60_000)


def voiced_tone(f0=200.0, seconds=2.0, sr=SAMPLE_RATE):
    """Harmonic tone in syllable-like bursts separated by near-silent gaps."""
    t = np.arange(int(seconds * sr)) / sr
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    envelope = np.sin(2 * np.pi * 2 * t) ** 2
    return (0.3 * y * envelope / np.max(np.abs(y))).astype(np.float32)


def wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_load_bytes_resamples_to_16k_mono():
    stereo = np.stack([voiced_tone(sr=48_000)] * 2, axis=1)
    y = audio.load_bytes(wav_bytes(stereo, 48_000))
    assert y.ndim == 1
    assert abs(y.size - 2 * SAMPLE_RATE) <= 1


def test_silence_detection():
    rng = np.random.default_rng(0)
    assert audio.is_silent(np.zeros(SAMPLE_RATE, dtype=np.float32))
    # Steady room noise is skipped even when fairly loud...
    assert audio.is_silent(rng.normal(0, 0.01, 2 * SAMPLE_RATE).astype(np.float32))
    assert not audio.is_silent(voiced_tone())
    # ...but quiet speech with pauses (like RAVDESS "sad", ~-50 dBFS peaks) is kept.
    quiet = voiced_tone() * 0.01
    quiet[: SAMPLE_RATE // 2] = 0
    assert not audio.is_silent(quiet + rng.normal(0, 1e-5, quiet.size).astype(np.float32))


def test_pitch_is_recovered():
    y = voiced_tone(f0=220.0)
    prosody = summarise(y, compute_tracks(y))
    assert prosody.pitch_median_hz == pytest.approx(220.0, rel=0.03)
    assert prosody.voiced_fraction > 0.5
    assert "median pitch 22" in features_text(prosody)


def test_voice_quality_features_move_the_right_way():
    rng = np.random.default_rng(0)
    clean = voiced_tone(f0=200.0)
    noisy = clean + rng.normal(0, 0.05, clean.size).astype(np.float32)
    p_clean = summarise(clean, compute_tracks(clean))
    p_noisy = summarise(noisy, compute_tracks(noisy))
    # A clean harmonic tone is strongly periodic; added noise lowers HNR and raises tilt.
    assert p_clean.hnr_db > 15
    assert p_noisy.hnr_db < p_clean.hnr_db - 5
    assert p_noisy.spectral_tilt_db > p_clean.spectral_tilt_db


def test_baseline_relative_features():
    y = voiced_tone(f0=200.0)
    base = summarise(y, compute_tracks(y), level_dbfs=-30.0)
    higher = voiced_tone(f0=400.0)
    p = summarise(higher, compute_tracks(higher), level_dbfs=-20.0)
    vec = feature_vector(p, base)
    assert vec["pitch_vs_baseline_st"] == pytest.approx(12.0, abs=0.5)
    assert vec["level_vs_baseline_db"] == pytest.approx(10.0)
    assert "pitch_vs_baseline_st" not in feature_vector(p)
    assert "pitch vs baseline +1" in features_text(p, base)


def test_mean_prosody_averages_fields():
    a = summarise(voiced_tone(f0=200.0), compute_tracks(voiced_tone(f0=200.0)), level_dbfs=-30.0)
    b = summarise(voiced_tone(f0=300.0), compute_tracks(voiced_tone(f0=300.0)), level_dbfs=-20.0)
    m = mean_prosody([a, b])
    assert m.level_dbfs == pytest.approx(-25.0)
    assert m.pitch_median_hz == pytest.approx((a.pitch_median_hz + b.pitch_median_hz) / 2)


def test_prosody_from_dict_rejects_bad_input():
    good = summarise(voiced_tone(), compute_tracks(voiced_tone())).to_dict()
    assert Prosody.from_dict(good).pitch_median_hz == good["pitch_median_hz"]
    with pytest.raises(KeyError):
        Prosody.from_dict({k: v for k, v in good.items() if k != "hnr_db"})
    with pytest.raises(ValueError):
        Prosody.from_dict({**good, "hnr_db": "nan"})


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_build_request_per_variant(variant):
    v = VARIANTS[variant]
    prepared = prepare_audio(voiced_tone(), {v.image} if v.image else set(), max_image_bytes=40_000)
    base = prepare_audio(voiced_tone(f0=150.0), set(), 40_000).prosody
    state, images, questions = build_request(prepared, variant, base)
    assert ("Measured acoustic features" in state) == v.features
    assert ("vs baseline" in state) == v.features
    assert ("How emotions typically change the voice" in state) == v.guide
    assert set(questions) == {"emotion", "arousal"}
    assert len(images) == (1 if v.image else 0)
    for uri in images:
        assert uri.startswith("data:image/jpeg;base64,")
    for jpeg in prepared.images.values():
        assert len(jpeg) <= 40_000 and jpeg[:2] == b"\xff\xd8"


def test_default_variant_is_the_full_setup():
    v = VARIANTS[DEFAULT_VARIANT]
    assert v.image == "composite" and v.features and v.guide


def clef_payload(choice="angry"):
    probs = {e: 0.05 for e in EMOTIONS}
    probs[choice] = 0.8
    return {
        "success": True,
        "errors": [],
        "result": {
            "model": "clef-flash",
            "answers": {
                "emotion": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.8},
                "arousal": {"type": "score", "score": 3.2, "legend": [], "probabilities": [], "confidence": 0.5},
            },
            "usage": {"input_tokens": 100},
        },
    }


def test_parse_response_envelope_and_bare():
    r = parse_response(clef_payload("sad"), 1.0)
    assert r.emotion == "sad" and r.probabilities["sad"] == 0.8 and r.arousal == 3.2
    bare = parse_response(clef_payload("happy")["result"], 1.0)
    assert bare.emotion == "happy"


def test_guided_questions_use_cue_descriptions():
    plain = build_questions()["emotion"]["criteria"]
    guided = build_questions(guided=True)["emotion"]["criteria"]
    assert set(plain) == set(guided) == set(EMOTIONS)
    assert "louder" in guided["angry"] and plain["angry"] != guided["angry"]


def test_parse_response_list_probabilities_and_errors():
    payload = clef_payload()
    payload["result"]["answers"]["emotion"]["probabilities"] = [
        {"option": "angry", "probability": 0.7},
        {"option": "sad", "probability": 0.3},
    ]
    assert parse_response(payload, 0.1).probabilities == {"angry": 0.7, "sad": 0.3}
    with pytest.raises(ClefError):
        parse_response({"success": False, "errors": [{"message": "bad"}]}, 0.1)


def test_client_request_shape_and_retry(monkeypatch):
    monkeypatch.setattr("spectromood.clef.time.sleep", lambda s: None)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=clef_payload())

    client = ClefClient(SETTINGS, client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.classify("state", ["data:image/jpeg;base64,AAAA"])
    assert result.emotion == "angry"
    assert len(seen) == 2
    req = seen[-1]
    assert req.url.path == "/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
    assert req.headers["Authorization"] == "Bearer tok"
    body = json.loads(req.content)
    assert body["model"] == "clef-flash"
    assert body["images"] == ["data:image/jpeg;base64,AAAA"]
    assert set(body["questions"]["emotion"]["criteria"]) == set(EMOTIONS)


def test_client_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"errors": [{"message": "bad input"}]})

    client = ClefClient(SETTINGS, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ClefError, match="400"):
        client.classify("state")
    assert len(calls) == 1


def test_client_requires_credentials():
    with pytest.raises(RuntimeError, match="CLOUDFLARE_API_TOKEN"):
        ClefClient(Settings(account_id="a", api_token="", model="clef-flash", timeout_s=1, max_image_bytes=1))


def test_score():
    labels = ["angry", "sad"]
    assert confusion(["angry", "angry", "sad"], ["angry", "sad", "sad"], labels).tolist() == [[1, 1], [0, 1]]
    rec = lambda t, p, a: {"clip": {"emotion": t}, "emotion": p, "arousal": a, "latency_s": 0.5}
    s = score([rec("angry", "angry", 3.0), rec("sad", "angry", 1.0), rec("sad", "sad", 0.5)], labels)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["recall"] == {"angry": 1.0, "sad": 0.5}
    assert s["arousal_auc"] == 1.0


def test_select_clips_per_speaker_and_corpus():
    rows = [
        {"row": i, "emotion": e, "speaker": f"s{i % 6}"} for i, e in enumerate([e for e in EMOTIONS for _ in range(30)])
    ]
    clips, baselines = select_clips(rows, per_emotion=4, seed=1)
    assert len(clips) == 20 and {c["emotion"] for c in clips} == set(EMOTIONS)
    assert {b["speaker"] for b in baselines} == {c["speaker"] for c in clips}
    assert not {b["row"] for b in baselines} & {c["row"] for c in clips}
    assert all(b["emotion"] == "neutral" for b in baselines)

    anonymous = [{**r, "speaker": None} for r in rows]
    clips, baselines = select_clips(anonymous, per_emotion=4, seed=1)
    assert len(baselines) == 5 and {b["speaker"] for b in baselines} == {"*"}
    assert not {b["row"] for b in baselines} & {c["row"] for c in clips}


def test_benchmark_corpus_baseline(tmp_path):
    root = tmp_path / "xx"
    (root / "clips").mkdir(parents=True)
    for name, f0 in [("a.wav", 200.0), ("b.wav", 300.0), ("c.wav", 250.0)]:
        sf.write(root / "clips" / name, voiced_tone(f0=f0), SAMPLE_RATE)
    write_csv(root / "manifest.csv", [{"file": "c.wav", "emotion": "angry", "speaker": "*"}])
    write_csv(root / "baselines.csv", [{"file": "a.wav", "speaker": "*"}, {"file": "b.wav", "speaker": "*"}])
    bench = Benchmark.load(tmp_path, "xx")
    assert not bench.per_speaker
    base = bench.baseline_for("*", 40_000)
    assert 200 < base.pitch_median_hz < 300


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    from spectromood.server import app

    sent = []

    def fake_classify(self, state, images=None, questions=None):
        sent.append({"state": state, "images": images, "questions": questions})
        return ClefResult(
            emotion="angry", probabilities={"angry": 0.9, "sad": 0.1}, confidence=0.9,
            arousal=3.5, latency_s=0.01, model="clef-flash",
        )  # fmt: skip

    monkeypatch.setattr(ClefClient, "classify", fake_classify)
    with TestClient(app) as client:
        client.sent = sent
        yield client


def test_api_health(api):
    data = api.get("/api/health").json()
    assert data["ok"] and data["emotions"] == list(EMOTIONS)


def test_api_analyze_ok(api):
    resp = api.post("/api/analyze", content=wav_bytes(voiced_tone(sr=48_000), 48_000))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["emotion"] == "angry" and data["image"].startswith("data:image/jpeg")


def test_api_baseline_roundtrip(api):
    resp = api.post("/api/baseline", content=wav_bytes(voiced_tone(f0=150.0, seconds=3.0), SAMPLE_RATE))
    assert resp.status_code == 200, resp.text
    baseline = resp.json()["baseline"]
    assert baseline["pitch_median_hz"] == pytest.approx(150.0, rel=0.05)

    variant = DEFAULT_VARIANT
    resp = api.post(
        "/api/analyze",
        params={"variant": variant},  # "+" must be URL-encoded
        content=wav_bytes(voiced_tone(f0=300.0), SAMPLE_RATE),
        headers={"X-Baseline": json.dumps(baseline)},
    )
    data = resp.json()
    assert data["baseline_used"] is True
    assert data["features"]["pitch_vs_baseline_st"] == pytest.approx(12.0, abs=0.7)
    assert "vs baseline" in api.sent[-1]["state"]
    assert "louder" in api.sent[-1]["questions"]["emotion"]["criteria"]["angry"]

    # Without a baseline the same variant still works, using absolute features.
    resp = api.post("/api/analyze", params={"variant": variant}, content=wav_bytes(voiced_tone(f0=300.0), SAMPLE_RATE))
    assert resp.json()["baseline_used"] is False and "vs baseline" not in api.sent[-1]["state"]


def test_api_baseline_rejects_bad_input(api):
    assert api.post("/api/baseline", content=wav_bytes(voiced_tone(seconds=1.5), SAMPLE_RATE)).status_code == 400
    silent = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    assert api.post("/api/baseline", content=wav_bytes(silent, SAMPLE_RATE)).status_code == 400
    bad = api.post("/api/analyze", content=wav_bytes(voiced_tone(), SAMPLE_RATE), headers={"X-Baseline": "{}"})
    assert bad.status_code == 400


def test_api_analyze_silence_and_bad_input(api):
    silent = api.post("/api/analyze", content=wav_bytes(np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE))
    assert silent.json()["status"] == "silence"
    assert api.post("/api/analyze", content=b"not audio").status_code == 400
    short = wav_bytes(voiced_tone(seconds=0.5), SAMPLE_RATE)
    assert api.post("/api/analyze", content=short).status_code == 400
    assert api.post("/api/analyze?variant=nope", content=short).status_code == 400


def test_index_served(api):
    assert "SpectroMood" in api.get("/").text
    assert api.get("/static/app.js").status_code == 200


def test_worker_mode_uses_proxy_url_and_bare_response():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=clef_payload("fearful")["result"])

    settings = Settings(
        account_id="", api_token="", model="clef-flash", timeout_s=5, max_image_bytes=1,
        worker_url="http://localhost:8787/", worker_token="secret",
    )  # fmt: skip
    client = ClefClient(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.classify("state").emotion == "fearful"
    assert str(seen[0].url) == "http://localhost:8787/"
    assert seen[0].headers["Authorization"] == "Bearer secret"


def test_quota_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            502, json={"error": "4006: you have used up your daily free allocation of 10,000 neurons"}
        )

    client = ClefClient(SETTINGS, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ClefQuotaError, match="00:00 UTC"):
        client.classify("state")
    assert len(calls) == 1


def test_load_prepared_fresh_equals_cached(tmp_path):
    (tmp_path / "clips").mkdir()
    sf.write(tmp_path / "clips" / "a.wav", voiced_tone(f0=210.0), SAMPLE_RATE)
    fresh = load_prepared(tmp_path, "a.wav", set(), 40_000).prosody
    cached = load_prepared(tmp_path, "a.wav", set(), 40_000).prosody
    assert fresh == cached


def bursty_noise(seconds=3.0):
    """Noise that passes the silence gate (loud bursts with quiet gaps) but is not a voice."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.2, int(seconds * SAMPLE_RATE)).astype(np.float32)
    y *= (np.sin(2 * np.pi * 1.5 * np.arange(y.size) / SAMPLE_RATE) > 0).astype(np.float32)
    return y


def test_vad_rejects_noise_and_accepts_voiced_sound():
    from spectromood.vad import speech_seconds

    assert speech_seconds(np.zeros(SAMPLE_RATE * 2, dtype=np.float32)) == 0.0
    assert speech_seconds(bursty_noise()) < 0.5
    assert speech_seconds(voiced_tone(seconds=3.0)) > 1.0


def test_api_noise_is_never_sent_to_clef(api):
    assert not audio.is_silent(bursty_noise())  # it gets past the cheap silence gate...
    before = len(api.sent)
    resp = api.post("/api/analyze", content=wav_bytes(bursty_noise(), SAMPLE_RATE)).json()
    assert resp["status"] == "no_speech"  # ...but the VAD stops it
    assert len(api.sent) == before


def test_api_baseline_requires_voice(api):
    resp = api.post("/api/baseline", content=wav_bytes(bursty_noise(4.0), SAMPLE_RATE))
    assert resp.status_code == 400 and "voice" in resp.json()["detail"]
