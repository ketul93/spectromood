"""Download the benchmark clips for one or more languages into data/<lang>/clips/.

The clip lists (data/<lang>/manifest.csv and baselines.csv) are committed, so everyone evaluates
exactly the same recordings. Use --select to (re)create the lists for a language from its source.

    python scripts/prepare_data.py --lang en,it,ru,es        # fetch audio for the committed lists
    python scripts/prepare_data.py --lang it --select        # rebuild a clip list, then fetch

Sources and licences:
  en  RAVDESS (Livingstone & Russo 2018), CC BY-NC-SA 4.0, zenodo.org/records/1188976
  it  Emozionalmente via CAMEO (amu-cai/CAMEO), CC BY 4.0
  ru  RESD via CAMEO, MIT
  es  MESD via CAMEO, CC BY 4.0
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import soundfile as sf

from spectromood.config import EMOTIONS
from spectromood.datasets import CORPUS, read_csv, write_csv

ROOT = Path(__file__).resolve().parent.parent
ROWS_API = "https://datasets-server.huggingface.co/rows"

RAVDESS_RECORD = "https://zenodo.org/api/records/1188976"
RAVDESS_ARCHIVE = "Audio_Speech_Actors_01-24.zip"

# Source label -> our label. Anything unmapped (disgust, surprise, ...) is ignored.
LABELS = {
    "anger": "angry", "angry": "angry",
    "fear": "fearful", "fearful": "fearful",
    "happiness": "happy", "happy": "happy",
    "sadness": "sad", "sad": "sad",
    "neutral": "neutral",
}  # fmt: skip


@dataclass(frozen=True)
class HFSource:
    dataset: str
    config: str
    split: str
    licence: str
    emotion_field: str = "emotion"
    speaker_field: str | None = "speaker_id"
    audio_field: str = "audio"
    gated: bool = False


HF_SOURCES: dict[str, HFSource] = {
    "it": HFSource("amu-cai/CAMEO", "default", "emozionalmente", "CC BY 4.0"),
    "ru": HFSource("amu-cai/CAMEO", "default", "resd", "MIT", speaker_field=None),
    "es": HFSource("amu-cai/CAMEO", "default", "mesd", "CC BY 4.0", speaker_field=None),
}


# ---------------------------------------------------------------- Hugging Face datasets server
def _headers(source: HFSource) -> dict[str, str]:
    token = os.getenv("HF_TOKEN", "").strip()
    if source.gated and not token:
        sys.exit(f"{source.dataset} is gated: accept its terms on huggingface.co and set HF_TOKEN in .env")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _rows(client: httpx.Client, source: HFSource, offset: int, length: int) -> dict:
    params = {"dataset": source.dataset, "config": source.config, "split": source.split}
    for attempt in range(8):
        r = client.get(ROWS_API, params={**params, "offset": offset, "length": length}, headers=_headers(source))
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            # Rate limited or busy: honour Retry-After, otherwise back off exponentially (max ~1 min).
            wait = float(r.headers.get("retry-after") or min(60, 2 ** (attempt + 1)))
            print(f"  datasets-server {r.status_code}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        sys.exit(f"datasets-server error {r.status_code}: {r.text[:300]}")
    sys.exit("datasets-server kept failing; completed clips are kept, re-run later to resume")


def list_hf_rows(client: httpx.Client, source: HFSource) -> list[dict]:
    """All rows' metadata (no audio download): row index, emotion, speaker."""
    out, offset = [], 0
    while True:
        page = _rows(client, source, offset, 100)
        for item in page["rows"]:
            row = item["row"]
            label = LABELS.get(str(row.get(source.emotion_field, "")).strip().lower())
            if label is None:
                continue
            speaker = row.get(source.speaker_field) if source.speaker_field else None
            if speaker in (None, "", "<UNK>"):
                speaker = None
            out.append({"row": item["row_idx"], "emotion": label, "speaker": None if speaker is None else str(speaker)})
        offset += len(page["rows"])
        if not page["rows"] or offset >= page.get("num_rows_total", offset):
            return out
        time.sleep(0.2)


def audio_urls(client: httpx.Client, source: HFSource, rows: list[int]) -> dict[int, str]:
    """Signed audio URLs for the wanted rows, one datasets-server call per page of 100 rows."""
    urls: dict[int, str] = {}
    for page in sorted({row // 100 * 100 for row in rows}):
        for item in _rows(client, source, page, 100)["rows"]:
            if item["row_idx"] in rows:
                audio = item["row"][source.audio_field]
                urls[item["row_idx"]] = audio[0]["src"] if isinstance(audio, list) else audio["src"]
        time.sleep(0.5)
    return urls


def download(client: httpx.Client, url: str) -> bytes:
    for attempt in range(6):
        r = client.get(url, follow_redirects=True, timeout=120)
        if r.status_code == 200:
            return r.content
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("retry-after") or min(60, 2 ** (attempt + 1))))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Could not download {url[:80]}")


# ---------------------------------------------------------------- clip selection
def select_clips(rows: list[dict], per_emotion: int, seed: int, corpus_baselines: int = 5):
    """Balanced clip list plus calm baselines.

    With speaker ids: clips are spread round-robin over speakers that also have a spare neutral
    clip, which becomes that speaker's baseline. Without speaker ids: random clips, and a corpus
    baseline averaged over `corpus_baselines` held-out neutral clips.
    """
    rng = random.Random(seed)
    emotions = list(EMOTIONS)
    if all(r["speaker"] is not None for r in rows):
        by_speaker: dict[str, list[dict]] = {}
        for r in rows:
            by_speaker.setdefault(r["speaker"], []).append(r)
        eligible = sorted(s for s, rs in by_speaker.items() if sum(x["emotion"] == "neutral" for x in rs) >= 2)
        rng.shuffle(eligible)
        clips: list[dict] = []
        for i, emotion in enumerate(emotions):
            order = eligible[i:] + eligible[:i]  # rotate so emotions draw on different speakers
            pools = {s: [r for r in by_speaker[s] if r["emotion"] == emotion and r not in clips] for s in order}
            picked: list[dict] = []
            while len(picked) < per_emotion and any(pools.values()):
                for s in order:
                    if pools[s] and len(picked) < per_emotion:
                        picked.append(pools[s].pop(rng.randrange(len(pools[s]))))
            if len(picked) < per_emotion:
                sys.exit(f"Only {len(picked)} '{emotion}' clips available")
            clips += picked
        baselines = []
        for s in sorted({c["speaker"] for c in clips}):
            spare = [r for r in by_speaker[s] if r["emotion"] == "neutral" and r not in clips]
            baselines.append({**rng.choice(spare), "speaker": s})
        return clips, baselines

    clips = []
    for emotion in emotions:
        pool = [r for r in rows if r["emotion"] == emotion]
        if len(pool) < per_emotion:
            sys.exit(f"Only {len(pool)} '{emotion}' clips available")
        clips += rng.sample(pool, per_emotion)
    spare = [r for r in rows if r["emotion"] == "neutral" and r not in clips]
    baselines = [{**r, "speaker": CORPUS} for r in rng.sample(spare, corpus_baselines)]
    return clips, baselines


def prepare_hf(lang: str, select: bool, per_emotion: int, seed: int) -> None:
    source = HF_SOURCES[lang]
    root = ROOT / "data" / lang
    with httpx.Client(timeout=60) as client:
        if select or not (root / "manifest.csv").exists():
            print(f"[{lang}] listing {source.dataset}/{source.split} ...", flush=True)
            clips, baselines = select_clips(list_hf_rows(client, source), per_emotion, seed)
            name = lambda r: f"{lang}_{r['row']:06d}.wav"
            src = f"{source.dataset}:{source.split}"
            write_csv(root / "manifest.csv", [
                {"file": name(c), "emotion": c["emotion"], "speaker": c["speaker"] or CORPUS,
                 "source": src, "row": c["row"], "licence": source.licence} for c in clips
            ])  # fmt: skip
            write_csv(root / "baselines.csv", [
                {"file": name(b), "speaker": b["speaker"], "source": src, "row": b["row"]} for b in baselines
            ])  # fmt: skip
        wanted = read_csv(root / "manifest.csv") + read_csv(root / "baselines.csv")
        missing = [w for w in wanted if not (root / "clips" / w["file"]).exists()]
        (root / "clips").mkdir(parents=True, exist_ok=True)
        urls = audio_urls(client, source, [int(w["row"]) for w in missing]) if missing else {}
        for i, item in enumerate(missing, 1):
            audio = download(client, urls[int(item["row"])])
            y, sr = sf.read(io.BytesIO(audio), always_2d=False)  # validates the download
            sf.write(root / "clips" / item["file"], y, sr)
            if i % 10 == 0:
                print(f"[{lang}] {i}/{len(missing)} clips", flush=True)
    print(f"[{lang}] ready: {len(read_csv(root / 'manifest.csv'))} clips in {root / 'clips'}")


# ---------------------------------------------------------------- RAVDESS (English)
def prepare_ravdess() -> None:
    root = ROOT / "data" / "en"
    archive = ROOT / "data" / "_download" / RAVDESS_ARCHIVE
    meta = httpx.get(RAVDESS_RECORD, timeout=30, follow_redirects=True).json()
    entry = next(f for f in meta["files"] if f["key"] == RAVDESS_ARCHIVE)
    if not archive.exists() or archive.stat().st_size != entry["size"]:
        print(f"[en] downloading {RAVDESS_ARCHIVE} ({entry['size'] / 1e6:.0f} MB) ...")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", entry["links"]["self"], timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            with archive.with_suffix(".part").open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        archive.with_suffix(".part").replace(archive)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != entry["checksum"].removeprefix("md5:"):
        sys.exit(f"Checksum mismatch for {archive}")
    wanted = [r["file"] for r in read_csv(root / "manifest.csv") + read_csv(root / "baselines.csv")]
    (root / "clips").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in wanted:
            actor = name.removesuffix(".wav").split("-")[-1]
            # Write to a path we construct ourselves; never trust archive paths.
            (root / "clips" / name).write_bytes(zf.read(f"Actor_{actor}/{name}"))
    print(f"[en] ready: {len(wanted)} files in {root / 'clips'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en,it,ru,es", help="comma-separated: en,it,ru,es")
    ap.add_argument("--select", action="store_true", help="rebuild the clip lists (changes the benchmark)")
    ap.add_argument("--per-emotion", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for lang in [x.strip() for x in args.lang.split(",") if x.strip()]:
        if lang == "en":
            prepare_ravdess()
        elif lang in HF_SOURCES:
            prepare_hf(lang, args.select, args.per_emotion, args.seed)
        else:
            sys.exit(f"Unknown language {lang!r}")


if __name__ == "__main__":
    main()
