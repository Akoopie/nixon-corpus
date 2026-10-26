"""Pick listening-session clips where Parakeet and Whisper disagree, and cut them from the tape audio.

For each tape (out/<conv>/{parakeet,whisper,speakers}.json), merged disagreement spans of 3-8 s are
ranked by how much the two models differ; up to PER_TAPE spans per tape are kept. The tape's audio
is downloaded and converted with the batch worker's own code, and each clip is cut with 0.4 s of
padding. Writes <root>/session/clips.json and <root>/session/audio/<id>.wav.

    python src/listening_clips.py <root> --jobs jobs.jsonl [--n 100] [--pad 3] [--diff 0.3 0.75] [--append]

--diff keeps only spans whose word-level difference between the models lies in that range
(lower = easier); --append adds to an existing session instead of replacing it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

PER_TAPE, MIN_S, MAX_S = 5, 3.0, 8.0


def words_in(words, a, b):
    return " ".join(w["w"] for w in words if a <= w["s"] < b).strip()


def spans(merged: list[dict]) -> list[tuple[float, float]]:
    """Disagreement spans (consecutive words carrying the same Whisper alternative), widened to MIN_S."""
    out, k = [], 0
    while k < len(merged):
        if "alt" not in merged[k]:
            k += 1
            continue
        j = k
        while j < len(merged) and merged[j].get("alt") == merged[k]["alt"]:
            j += 1
        a, b = merged[k]["s"], merged[j - 1]["e"]
        if b - a < MIN_S:
            mid = (a + b) / 2
            a, b = mid - MIN_S / 2, mid + MIN_S / 2
        if b - a <= MAX_S:
            out.append((max(0.0, a), b))
        k = j
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pad", type=float, default=3.0, help="seconds of context before and after the disputed part")
    ap.add_argument("--diff", type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("WORKER_CACHE", str(args.root / "cache"))
    import soundfile as sf
    from merge import merge
    import worker  # reuses fetch_wav / read_audio, so clips come from the same audio the models heard

    session = args.root / "session"
    (session / "audio").mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in args.jobs.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_tape = max(1, -(-args.n // len(jobs)))
    clips = []
    for job in jobs:
        d = args.root / "out" / job["conv"]
        pk = json.loads((d / "parakeet.json").read_text(encoding="utf-8"))
        wh = json.loads((d / "whisper.json").read_text(encoding="utf-8"))
        named = json.loads((d / "speakers.json").read_text(encoding="utf-8")) if (d / "speakers.json").exists() else []
        cands = []
        for a, b in spans(merge(pk, wh)):
            p, w = words_in(pk["words"], a, b), words_in(wh["words"], a, b)
            diff = 1 - SequenceMatcher(None, p.lower().split(), w.lower().split()).ratio()
            if (p or w) and args.diff[0] <= diff <= args.diff[1]:
                cands.append((diff, a, b, p, w))
        cands.sort(reverse=True)
        chosen = []
        for c in cands:  # most-different first, no overlaps, at most per_tape
            if all(c[2] + 1 < x[1] or c[1] > x[2] + 1 for x in chosen):
                chosen.append(c)
            if len(chosen) == min(per_tape, PER_TAPE):
                break
        if not chosen:
            continue
        audio = worker.read_audio(worker.fetch_wav(job))
        people = [p.split(",")[0] for p in job["participants"] if "operator" not in p.lower() and "Unknown" not in p]
        for diff, a, b, p, w in sorted(chosen, key=lambda x: x[1]):
            cid = f"{job['conv']}_{int(a * 10):06d}"
            lo, hi = max(0, int((a - args.pad) * worker.SR)), int((b + args.pad) * worker.SR)
            sf.write(str(session / "audio" / f"{cid}.wav"), audio[lo:hi], worker.SR)
            guess = sorted({s["speaker"] for s in named if s["start"] < b and s["end"] > a})
            clips.append({"id": cid, "conv": job["conv"], "date": job["date"], "location": job["location"],
                          "participants": people, "start": round(a, 2), "end": round(b, 2),
                          "clip_offset": round(a - lo / worker.SR, 2), "clip_len": round(b - a, 2),
                          "parakeet": p, "whisper": w, "speaker_guess": guess,
                          "before": words_in(pk["words"], a - 12, a), "after": words_in(pk["words"], b, b + 12)})
        (worker.CACHE / f"{job['conv']}.wav").unlink(missing_ok=True)
        print(f"{job['conv']}: {len(chosen)} clips", flush=True)
    old = json.loads((session / "clips.json").read_text(encoding="utf-8")) if args.append and (session / "clips.json").exists() else []
    seen = {c["id"] for c in old}
    new = [c for c in clips if c["id"] not in seen][:args.n]
    (session / "clips.json").write_text(json.dumps(old + new, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(new)} new clips ({len(old) + len(new)} in the session) -> {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
