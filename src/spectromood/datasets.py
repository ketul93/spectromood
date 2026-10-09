"""Benchmark layout and cached per-clip features.

Each language lives in data/<lang>/:
  manifest.csv    evaluated clips: file, emotion, speaker (+ source columns)
  baselines.csv   calm reference clips: file, speaker. speaker "*" = corpus baseline, used when
                  a dataset has no speaker ids (the average of several held-out neutral clips)
  clips/          audio (downloaded by scripts/prepare_data.py, not committed)
  cache/          computed features/images and Clef responses (not committed)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from spectromood.audio import load_file
from spectromood.features import FEATURES_VERSION, Prosody, mean_prosody
from spectromood.pipeline import Prepared, prepare_audio

CORPUS = "*"


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Benchmark:
    lang: str
    root: Path
    clips: list[dict]
    baselines: list[dict]

    @classmethod
    def load(cls, data_dir: Path, lang: str) -> Benchmark:
        root = data_dir / lang
        return cls(lang, root, read_csv(root / "manifest.csv"), read_csv(root / "baselines.csv"))

    @property
    def per_speaker(self) -> bool:
        return all(b["speaker"] != CORPUS for b in self.baselines)

    def prepared(self, clip_file: str, reps: set[str], max_image_bytes: int) -> Prepared:
        return load_prepared(self.root, clip_file, reps, max_image_bytes)

    def baseline_for(self, speaker: str, max_image_bytes: int) -> Prosody:
        """The speaker's calm reference, or the corpus baseline when speakers are unknown."""
        rows = [b for b in self.baselines if b["speaker"] == speaker] or [
            b for b in self.baselines if b["speaker"] == CORPUS
        ]
        if not rows:
            raise KeyError(f"No baseline for speaker {speaker!r} in {self.lang}")
        prosodies = [self.prepared(b["file"], set(), max_image_bytes).prosody for b in rows]
        # A single reference clip is used as-is, so requests match its cached 2-decimal values.
        return prosodies[0] if len(prosodies) == 1 else mean_prosody(prosodies)


def load_prepared(root: Path, clip_file: str, reps: set[str], max_image_bytes: int) -> Prepared:
    """Features + images for a clip, cached on disk (they take ~1 s per clip to compute)."""
    stem = Path(clip_file).stem
    cache = root / "cache" / "features" / f"{stem}.json"
    image_paths = {rep: root / "cache" / "images" / rep / f"{stem}.jpg" for rep in reps}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("version") == FEATURES_VERSION and all(p.exists() for p in image_paths.values()):
            images = {rep: p.read_bytes() for rep, p in image_paths.items()}
            return Prepared(tracks=None, prosody=Prosody.from_dict(cached["prosody"]), images=images)
    prepared = prepare_audio(load_file(root / "clips" / clip_file), reps, max_image_bytes)
    for rep, jpeg in prepared.images.items():
        image_paths[rep].parent.mkdir(parents=True, exist_ok=True)
        image_paths[rep].write_bytes(jpeg)
    rounded = prepared.prosody.to_dict()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"version": FEATURES_VERSION, "prosody": rounded}))
    # Return the same 2-decimal values the cache holds, so fresh and cached runs send identical
    # requests (derived baseline-relative numbers would otherwise drift in the last digit).
    prepared.prosody = Prosody.from_dict(rounded)
    return prepared
