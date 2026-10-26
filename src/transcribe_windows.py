"""Transcribe chosen windows of tapes with one Parakeet model, keeping per-word confidence and the audio.

    candidates  <corpus root> --jobs jobs.jsonl --exclude ids.txt --out cands.jsonl   (CPU; v1 word counts only)
    run         cands.jsonl --jobs jobs.jsonl --model m.nemo --out DIR                (GPU)

`candidates` lists dense 3-minute windows (at least MIN_WORDS words in the existing parakeet.json),
at most PER_TAPE per tape, from a random sample of room tapes. `run` downloads each tape once (a few
in parallel), cuts its windows, and writes DIR/<conv>_<start>.json (words with confidence, segments)
next to DIR/<conv>_<start>.flac, ready for select_windows.py and the correction page.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

WIN_S, MIN_WORDS, PER_TAPE = 180.0, 250, 4
ROOMS = {"OVAL", "EOB", "CAB"}


def cmd_candidates(root: Path, jobs_path: Path, exclude: list[Path], out: Path, tapes: int, seed: int) -> None:
    ex = set()
    for f in exclude:
        ex |= set(re.findall(r"\d{3}-\d{3}", f.read_text()))
    jobs = [json.loads(l) for l in jobs_path.read_text().splitlines() if l.strip()]
    jobs = [j for j in jobs if j["location"] in ROOMS and j["conv"] not in ex]
    rng = random.Random(seed)
    rng.shuffle(jobs)
    rows, n_tapes = [], 0
    for j in jobs:
        f = root / "out" / j["conv"] / "parakeet.json"
        if not f.exists():
            continue
        starts = [w["s"] for w in json.loads(f.read_text())["words"]]
        dense = [t * WIN_S for t in range(int(starts[-1] // WIN_S) + 1 if starts else 0)
                 if sum(t * WIN_S <= s < (t + 1) * WIN_S for s in starts) >= MIN_WORDS]
        if not dense:
            continue
        for s in sorted(rng.sample(dense, min(PER_TAPE, len(dense)))):
            rows.append({"conv": j["conv"], "location": j["location"], "start": s, "end": s + WIN_S})
        n_tapes += 1
        if n_tapes == tapes:
            break
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} windows from {n_tapes} tapes -> {out}")


def cmd_run(cands: Path, jobs_path: Path, model_path: str, out: Path) -> None:
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict
    import worker
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in jobs_path.read_text().splitlines() if l.strip()}
    by_conv = defaultdict(list)
    for l in cands.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            by_conv[r["conv"]].append(r)
    out.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    model.eval()
    cfg = model.cfg.decoding
    with open_dict(cfg):  # same confidence settings as worker.py --confidence
        cfg.confidence_cfg = OmegaConf.create({
            "preserve_word_confidence": True, "preserve_token_confidence": True, "exclude_blank": True,
            "aggregation": "min", "method_cfg": {"name": "entropy", "entropy_type": "tsallis", "alpha": 0.5,
                                                 "entropy_norm": "exp"}})
    model.change_decoding_strategy(cfg)

    def fetch(conv):
        try:
            return conv, worker.fetch_wav(jobs[conv])
        except Exception as e:  # unfetchable tape: skip its windows
            print(f"  {conv}: {type(e).__name__}", flush=True)
            return conv, None

    done = 0
    with ThreadPoolExecutor(4) as pool:
        for conv, wav in pool.map(fetch, list(by_conv)):
            if wav is None:
                continue
            audio = worker.read_audio(wav)
            for r in by_conv[conv]:
                stem = out / f"{conv}_{int(r['start']):05d}"
                if stem.with_suffix(".json").exists():
                    continue
                a, b = int(r["start"] * worker.SR), int(r["end"] * worker.SR)
                sf.write(str(stem.with_suffix(".flac")), audio[a:b], worker.SR)
                try:
                    hyp = model.transcribe([str(stem.with_suffix(".flac"))], timestamps=True, verbose=False)[0]
                except RuntimeError as e:  # NeMo's word-confidence aggregation fails on some audio: skip the window
                    print(f"  {stem.name}: {str(e).splitlines()[0][:80]}", flush=True)
                    stem.with_suffix(".flac").unlink(missing_ok=True)
                    continue
                ws = hyp.timestamp.get("word", [])
                conf = [float(c) for c in (getattr(hyp, "word_confidence", None) or [])]
                words = [{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2),
                          **({"c": round(conf[k], 3)} if len(conf) == len(ws) else {})} for k, w in enumerate(ws)]
                segs = [{"text": s["segment"].strip(), "start": round(s["start"], 2), "end": round(s["end"], 2)}
                        for s in hyp.timestamp.get("segment", [])]
                tmp = stem.with_suffix(".tmp")
                tmp.write_text(json.dumps({**r, "model": model_path, "words": words, "segments": segs}))
                tmp.replace(stem.with_suffix(".json"))
                done += 1
            (worker.CACHE / f"{conv}.wav").unlink(missing_ok=True)
            print(f"{conv}: {len(by_conv[conv])} windows ({done} total)", flush=True)
    print("DONE", done)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["candidates", "run"])
    ap.add_argument("path", type=Path, help="candidates: corpus root; run: cands.jsonl")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model")
    ap.add_argument("--tapes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1974)
    a = ap.parse_args()
    if a.cmd == "candidates":
        cmd_candidates(a.path, a.jobs, a.exclude, a.out, a.tapes, a.seed)
    else:
        cmd_run(a.path, a.jobs, a.model, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
