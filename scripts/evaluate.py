"""Evaluate Clef on the benchmark clips of one or more languages.

    python scripts/evaluate.py --lang en,it,ru,es          # recommended setup + ablation
    python scripts/evaluate.py --lang it --variants plots+features+guide

Every Clef answer is cached in data/<lang>/cache/clef/ under a hash of the exact request, so a
re-run only pays for new or changed requests. Results go to reports/results.{md,json} (merged
with earlier languages) and the figures to docs/images/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from spectromood.clef import ClefClient, ClefError, ClefQuotaError
from spectromood.config import CLEF_MODEL, EMOTIONS, Settings
from spectromood.datasets import Benchmark
from spectromood.metrics import plot_confusion, plot_results, score
from spectromood.pipeline import DEFAULT_VARIANT, VARIANTS, build_request, representations_for

ROOT = Path(__file__).resolve().parent.parent
LABELS = list(EMOTIONS)
LANGUAGES = {
    "en": ("English", "RAVDESS"),
    "it": ("Italian", "Emozionalmente"),
    "ru": ("Russian", "RESD"),
    "es": ("Spanish", "MESD"),
}


def evaluate_language(lang: str, variants: list[str], models: list[str], settings: Settings, concurrency: int):
    bench = Benchmark.load(ROOT / "data", lang)
    reps = representations_for(variants)
    prepared = {c["file"]: bench.prepared(c["file"], reps, settings.max_image_bytes) for c in bench.clips}
    baselines = {s: bench.baseline_for(s, settings.max_image_bytes) for s in {c["speaker"] for c in bench.clips}}
    print(f"[{lang}] {len(bench.clips)} clips, {'per-speaker' if bench.per_speaker else 'corpus'} baseline")

    jobs, records = [], []
    for model in models:
        for variant in variants:
            for clip in bench.clips:
                state, images, questions = build_request(prepared[clip["file"]], variant, baselines[clip["speaker"]])
                sha = hashlib.sha256(json.dumps([model, state, images, questions]).encode()).hexdigest()
                path = bench.root / "cache" / "clef" / model / variant / f"{Path(clip['file']).stem}.json"
                if path.exists() and (cached := json.loads(path.read_text())).get("request_sha") == sha:
                    records.append(cached)
                else:
                    jobs.append((model, variant, clip, path, state, images, questions, sha))
    print(f"[{lang}] {len(jobs)} Clef calls to make ({len(records)} cached)")

    # Only needs credentials when something is not cached yet.
    clients = {m: ClefClient(replace(settings, model=m)) for m in {job[0] for job in jobs}}

    def run(model, variant, clip, path, state, images, questions, sha):
        result = clients[model].classify(state, images, questions)
        record = {"clip": clip, "variant": variant, "model": model, "request_sha": sha, **asdict(result)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        return record

    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                records.append(fut.result())
            except ClefQuotaError as exc:
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(f"Stopped: {exc}\nCompleted calls are cached; re-run to resume.")
            except ClefError as exc:
                failures += 1
                print(f"[{lang}] call failed: {exc}")
            if n % 25 == 0:
                print(f"[{lang}] {n}/{len(jobs)} calls done")
    for c in clients.values():
        c.close()
    if failures:
        sys.exit(f"[{lang}] {failures} calls failed; re-run to retry only those")

    language, dataset = LANGUAGES[lang]
    rows = []
    for model in models:
        for variant in variants:
            run_records = [r for r in records if r["model"] == model and r["variant"] == variant]
            rows.append({
                "lang": lang, "language": language, "dataset": dataset,
                "baseline": "per speaker" if bench.per_speaker else "corpus",
                "model": model, "variant": variant, **score(run_records, LABELS),
            })  # fmt: skip
    return rows


def write_report(rows: list[dict]) -> None:
    rows = sorted(
        rows, key=lambda r: (list(LANGUAGES).index(r["lang"]), r["model"], list(VARIANTS).index(r["variant"]))
    )
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps(rows, indent=1))

    main = [r for r in rows if r["model"] == "clef-flash" and r["variant"] == DEFAULT_VARIANT]
    lines = [
        "# Results",
        "",
        f"clef-flash, `{DEFAULT_VARIANT}` setup, 50 clips per language (10 per emotion). Chance = 20%.",
        "",
        "| language | dataset | baseline | accuracy | macro-F1 | arousal AUC | "
        + " | ".join(LABELS) + " |",
        "|---|---|---|---|---|---|" + "---|" * len(LABELS),
    ]  # fmt: skip
    for r in main:
        lines.append(
            f"| {r['language']} | {r['dataset']} | {r['baseline']} | **{r['accuracy']:.0%}** | {r['macro_f1']:.2f} | "
            f"{r['arousal_auc']:.2f} | " + " | ".join(f"{r['recall'][lab]:.0%}" for lab in LABELS) + " |"
        )
    lines += ["", "## Ablation (accuracy)", "", "| language | model | " + " | ".join(VARIANTS) + " |",
              "|---|---|" + "---|" * len(VARIANTS)]  # fmt: skip
    for lang in dict.fromkeys(r["lang"] for r in rows):
        for model in dict.fromkeys(r["model"] for r in rows if r["lang"] == lang):
            cells = []
            for v in VARIANTS:
                hit = next((r for r in rows if r["lang"] == lang and r["model"] == model and r["variant"] == v), None)
                cells.append(f"{hit['accuracy']:.0%}" if hit else "–")
            lines.append(f"| {LANGUAGES[lang][0]} | {model} | " + " | ".join(cells) + " |")
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n")

    images = ROOT / "docs" / "images"
    flash = [r for r in rows if r["model"] == "clef-flash"]
    plot_results(flash, [v for v in VARIANTS if any(r["variant"] == v for r in flash)], images / "results.png", 0.2)
    for r in main:
        plot_confusion(
            r["confusion"], LABELS, f"{r['language']} ({r['dataset']})", images / f"confusion_{r['lang']}.png"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en,it,ru,es")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    models = [CLEF_MODEL]
    if bad := (set(langs) - set(LANGUAGES)) | (set(variants) - set(VARIANTS)):
        sys.exit(f"Unknown language/variant: {sorted(bad)}")

    settings = Settings.from_env()
    results_path = ROOT / "reports" / "results.json"
    previous = json.loads(results_path.read_text()) if results_path.exists() else []
    fresh = [row for lang in langs for row in evaluate_language(lang, variants, models, settings, args.concurrency)]
    keys = {(r["lang"], r["model"], r["variant"]) for r in fresh}
    write_report([r for r in previous if (r["lang"], r["model"], r["variant"]) not in keys] + fresh)
    print("Wrote reports/results.md, reports/results.json and docs/images/")


if __name__ == "__main__":
    main()
