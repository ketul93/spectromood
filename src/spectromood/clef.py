"""Minimal client for Cloudflare's Clef decision model on Workers AI (REST API)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from spectromood.config import Settings
from spectromood.prompts import build_questions

API = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"

# Server-side failures worth retrying (rate limit / transient upstream errors).
_RETRY_STATUS = {429, 500, 502, 503, 504}


class ClefError(RuntimeError):
    pass


class ClefQuotaError(ClefError):
    """The account's Workers AI allocation is used up; retrying won't help until it resets."""


# Workers AI error 4006: daily free allocation (10,000 neurons) exhausted on the Free plan.
_QUOTA_MARKERS = ("4006", "daily free allocation")


@dataclass
class ClefResult:
    emotion: str
    probabilities: dict[str, float]
    confidence: float | None
    arousal: float | None
    latency_s: float
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _probabilities(answer: dict[str, Any]) -> dict[str, float]:
    probs = answer.get("probabilities")
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    if isinstance(probs, list) and all(isinstance(p, dict) for p in probs):
        # Accept [{"option"/"label"/"name": ..., "probability"/"p": ...}, ...]
        out = {}
        for p in probs:
            key = p.get("option", p.get("label", p.get("name")))
            val = p.get("probability", p.get("p"))
            if key is None or val is None:
                raise ClefError(f"Unrecognised probability entry: {p}")
            out[str(key)] = float(val)
        return out
    raise ClefError(f"Unrecognised probabilities format: {probs!r}")


def parse_response(payload: dict[str, Any], latency_s: float) -> ClefResult:
    """Parse a REST response (with Cloudflare envelope) or a bare binding response."""
    if "result" in payload or "success" in payload:
        if not payload.get("success", True):
            raise ClefError(f"Clef returned errors: {payload.get('errors')}")
        body = payload.get("result") or {}
    else:
        body = payload
    answers = body.get("answers") or {}
    emo = answers.get("emotion")
    if not isinstance(emo, dict):
        raise ClefError(f"No emotion answer in response: {payload}")
    probs = _probabilities(emo)
    choice = emo.get("choice") or max(probs, key=probs.get)
    confidence = emo.get("confidence")
    aro = answers.get("arousal") or {}
    arousal = aro.get("score")
    return ClefResult(
        emotion=str(choice),
        probabilities=probs,
        confidence=confidence,
        arousal=float(arousal) if arousal is not None else None,
        latency_s=latency_s,
        model=str(body.get("model", "")),
        usage=body.get("usage") or {},
        raw=payload,
    )


class ClefClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None, retries: int = 3):
        settings.require_credentials()
        self.settings = settings
        self.retries = retries
        self._client = client or httpx.Client(timeout=settings.timeout_s)

    @property
    def url(self) -> str:
        if self.settings.worker_url:
            return self.settings.worker_url
        return API.format(account=self.settings.account_id, model=self.settings.model)

    def build_body(self, state: str, images: list[str], questions: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "state": state,
            "questions": questions or build_questions(),
        }
        if images:
            body["images"] = images
        return body

    def classify(
        self, state: str, images: list[str] | None = None, questions: dict[str, Any] | None = None
    ) -> ClefResult:
        body = self.build_body(state, images or [], questions)
        token = self.settings.worker_token if self.settings.worker_url else self.settings.api_token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            start = time.perf_counter()
            try:
                resp = self._client.post(self.url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                latency = time.perf_counter() - start
                if resp.status_code >= 400 and any(m in resp.text for m in _QUOTA_MARKERS):
                    raise ClefQuotaError(
                        "Workers AI allocation used up (error 4006). The Free plan allows 10,000 neurons/day, "
                        "resetting at 00:00 UTC; upgrade to Workers Paid to continue now."
                    )
                if resp.status_code in _RETRY_STATUS:
                    last_error = ClefError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                elif resp.status_code >= 400:
                    raise ClefError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
                else:
                    return parse_response(resp.json(), latency)
            time.sleep(2**attempt)
        raise ClefError(f"Clef request failed after {self.retries} attempts: {last_error}")

    def close(self) -> None:
        self._client.close()
