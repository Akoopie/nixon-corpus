"""Self-training data for Parakeet: stretches where the fine-tuned Parakeet and Whisper heard the same words.

Two independent models agreeing word for word is strong evidence the words are right. For each tape,
the normalized word sequences are aligned (merge.py's alignment); runs of agreed words are cut into
clips of MIN_S-MAX_S seconds with no long pause inside. A clip keeps Parakeet's own (cased,
punctuated) words as its text, matching the training transcripts' style. Clips are capped per tape
so no single tape dominates, and the selection is balanced across locations.

  select  <root> --jobs jobs.jsonl --out picks.jsonl     (needs only the JSON outputs)
  cut     picks.jsonl --audio-dir DIR --manifest out.jsonl  (downloads each tape's audio once, cuts clips)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from score import norm_words

MIN_S, MAX_S, MAX_GAP = 3.0, 15.0, 1.0   # clip length, and the longest pause allowed inside one
MIN_WORDS, MIN_WPS, MAX_WPS = 6, 1.2, 5.5
PER_TAPE_S = 240                         # at most 4 minutes of clips from one tape
PHONE_SHARE = 0.15                       # telephone tapes' share of the selection


def agreed_runs(pk: dict, wh: dict) -> list[list[dict]]:
    """Runs of consecutive Parakeet words whose normalized tokens Whisper matched exactly, in order."""
    pw = [w for w in pk["words"] if norm_words(w["w"])]
    p_tok = [(i, t) for i, w in enumerate(pw) for t in norm_words(w["w"])]
    w_tok = [t for w in wh["words"] for t in norm_words(w["w"])]
    sm = SequenceMatcher(None, [t for _, t in p_tok], w_tok, autojunk=False)
    agreed = set()
    for b in sm.get_matching_blocks():
        idx = {p_tok[k][0] for k in range(b.a, b.a + b.size)}
        # a Parakeet word counts as agreed only if all of its tokens fell inside the block
        agreed |= {i for i in idx if all(b.a <= k < b.a + b.size for k, (j, _) in enumerate(p_tok) if j == i)}
    runs, cur = [], []
    for i, w in enumerate(pw):
        if i in agreed and (not cur or w["s"] - cur[-1]["e"] <= MAX_GAP):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = [w] if i in agreed else []
    if cur:
        runs.append(cur)
    return runs


def clips_from_run(run: list[dict]) -> list[list[dict]]:
    """Split one agreed run into clips of MIN_S-MAX_S seconds."""
    out, cur = [], []
    for w in run:
        if cur and w["e"] - cur[0]["s"] > MAX_S:
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    keep = []
    for c in out:
        dur = c[-1]["e"] - c[0]["s"]
        if dur >= MIN_S and len(c) >= MIN_WORDS and MIN_WPS <= len(c) / dur <= MAX_WPS:
            keep.append(c)
    return keep


def cmd_select(root: Path, jobs_path: Path, out: Path, exclude: set[str], seed: int, hours: float) -> None:
    jobs = [json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(jobs)  # a random spread of tapes across the whole period
    share = defaultdict(float)
    for j in jobs:
        share[j["location"]] += 1 / len(jobs)
    # each location's share of the corpus, except that phone calls (WHT), already the easy case, are capped
    # at PHONE_SHARE; their surplus goes to the rooms, which are the point of the project
    rooms = sum(f for loc, f in share.items() if loc != "WHT")
    share = {loc: (PHONE_SHARE if loc == "WHT" else f / rooms * (1 - PHONE_SHARE)) for loc, f in share.items()}
    quota = {loc: hours * 3600 * f for loc, f in share.items()}
    picks, by_loc = [], defaultdict(float)
    for job in jobs:
        if all(by_loc[l] >= q for l, q in quota.items()):
            break
        d = root / "out" / job["conv"]
        if (job["conv"] in exclude or by_loc[job["location"]] >= quota[job["location"]]
                or not ((d / "parakeet.json").exists() and (d / "whisper.json").exists())):
            continue
        pk = json.loads((d / "parakeet.json").read_text(encoding="utf-8"))
        wh = json.loads((d / "whisper.json").read_text(encoding="utf-8"))
        cands = [c for r in agreed_runs(pk, wh) for c in clips_from_run(r)]
        rng.shuffle(cands)
        budget = PER_TAPE_S
        for c in cands:
            dur = c[-1]["e"] - c[0]["s"]
            if dur > budget:
                continue
            budget -= dur
            picks.append({"conv": job["conv"], "location": job["location"], "start": round(c[0]["s"] - 0.15, 2),
                          "end": round(c[-1]["e"] + 0.15, 2), "text": " ".join(w["w"] for w in c).strip()})
            by_loc[job["location"]] += dur
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picks), encoding="utf-8", newline="\n")
    print(f"{len(picks)} clips, {sum(by_loc.values()) / 3600:.1f} h -> {out}; by location (h): "
          + ", ".join(f"{k} {v / 3600:.1f}" for k, v in sorted(by_loc.items(), key=lambda x: -x[1])))


def _cut_tape(args) -> tuple[list[dict], bool]:
    conv, job, ps, audio_dir = args
    import soundfile as sf
    import worker
    done = [audio_dir / conv / f"{k:04d}.wav" for k in range(len(ps))]
    if (audio_dir / conv / "DONE").exists() and all(f.exists() for f in done):  # resumed run: already cut
        return [{"audio_filepath": str(f), "duration": round(p["end"] - max(0.0, p["start"]), 2), "text": p["text"],
                 "source": "agreement"} for f, p in zip(done, ps)], True
    try:
        audio = worker.read_audio(worker.fetch_wav(job))
    except Exception as e:  # unfetchable tape: skip its clips
        print(f"  {conv}: {type(e).__name__}", flush=True)
        return [], False
    (audio_dir / conv).mkdir(parents=True, exist_ok=True)
    rows = []
    for k, p in enumerate(ps):
        f = audio_dir / conv / f"{k:04d}.wav"
        a = max(0.0, p["start"])
        sf.write(str(f), audio[int(a * worker.SR):int(p["end"] * worker.SR)], worker.SR)
        rows.append({"audio_filepath": str(f), "duration": round(p["end"] - a, 2), "text": p["text"], "source": "agreement"})
    (audio_dir / conv / "DONE").write_text(str(len(rows)))  # written last: a crash mid-tape leaves no marker
    (worker.CACHE / f"{conv}.wav").unlink(missing_ok=True)
    return rows, True


def cmd_cut(picks_path: Path, jobs_path: Path, audio_dir: Path, manifest: Path, workers: int) -> None:
    from multiprocessing import Pool
    os.environ.setdefault("WORKER_CACHE", str(audio_dir / "_cache"))
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    by_conv = defaultdict(list)
    for l in picks_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            p = json.loads(l)
            by_conv[p["conv"]].append(p)
    tasks = [(c, jobs[c], ps, audio_dir) for c, ps in by_conv.items()]
    rows, failed = [], 0
    with Pool(workers) as pool:
        for r, ok in pool.imap_unordered(_cut_tape, tasks):
            rows += r
            failed += not ok
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    print(f"{len(rows)} clips ({sum(r['duration'] for r in rows) / 3600:.1f} h) from {len(tasks) - failed} tapes "
          f"-> {manifest}; {failed} tapes could not be fetched")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "cut"])
    ap.add_argument("path", type=Path, help="select: corpus root; cut: picks.jsonl")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="select: picks.jsonl to write")
    ap.add_argument("--exclude", type=Path, help="file of conversation ids to leave out (validation, held-out tests)")
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--seed", type=int, default=1974)
    ap.add_argument("--hours", type=float, default=80.0, help="select: stop at about this many hours, split by location share")
    ap.add_argument("--workers", type=int, default=8, help="cut: tapes downloaded and cut in parallel")
    args = ap.parse_args()
    if args.cmd == "select":
        ex = set(args.exclude.read_text().split()) if args.exclude else set()
        cmd_select(args.path, args.jobs, args.out, ex, args.seed, args.hours)
    else:
        cmd_cut(args.path, args.jobs, args.audio_dir, args.manifest, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
