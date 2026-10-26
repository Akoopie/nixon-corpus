"""Pick 3-minute room-recording windows for the listener to correct, graded by how unsure Parakeet was.

Difficulty is the window's mean per-word confidence (worker.py --confidence: entropy-based, the word's
least confident token), lower meaning harder; the share of words below UNSURE is reported alongside. Windows must hold dense speech
(at least MIN_WORDS words), and come one per tape from tapes outside validation, dev and the listener's
earlier sessions. Three bands are drawn from the distribution's percentiles; the very top is skipped
because it is mostly noise and crosstalk nobody can transcribe.

    python3 select_windows.py <corpus root> --jobs jobs.jsonl --exclude ids.txt --out windows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WIN_S = 180.0
MIN_WORDS = 250                      # about 85 words a minute: a real conversation, not a silent room
ROOMS = {"OVAL", "EOB", "CAB"}
BANDS = {"light": (0.20, 0.35), "medium": (0.50, 0.65), "heavy": (0.80, 0.92)}  # percentiles, easiest first


UNSURE = 0.5


def windows(pk: dict) -> list[dict]:
    ws = [w for w in pk["words"] if "c" in w]
    if not ws:
        return []
    out, t = [], 0.0
    while t + WIN_S <= ws[-1]["e"] + 1:
        cs = [w["c"] for w in ws if t <= w["s"] < t + WIN_S]
        if len(cs) >= MIN_WORDS:
            out.append({"start": t, "end": t + WIN_S, "pk_words": len(cs),
                        "unsure": round(sum(c < UNSURE for c in cs) / len(cs), 3),
                        "mean_conf": round(sum(cs) / len(cs), 3)})
        t += WIN_S
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tapes", type=int, default=1500, help="room tapes to grade (random sample)")
    ap.add_argument("--input", default="parakeet_r3.json", help="Parakeet output with per-word confidence")
    ap.add_argument("--drafts", type=Path, help="grade transcribe_windows.py drafts instead of whole-tape outputs")
    ap.add_argument("--per-band", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1974)
    a = ap.parse_args()
    ex = set()
    for f in a.exclude:
        ex |= set(re.findall(r"\d{3}-\d{3}", f.read_text()))
    if a.drafts:  # windows already transcribed one per file by transcribe_windows.py run
        cands = []
        for f in sorted(a.drafts.glob("*.json")):
            d = json.loads(f.read_text())
            cs = [w["c"] for w in d["words"] if "c" in w]
            if d["conv"] in ex or len(cs) < MIN_WORDS:
                continue
            cands.append({"conv": d["conv"], "location": d["location"], "start": d["start"], "end": d["end"],
                          "pk_words": len(cs), "unsure": round(sum(c < UNSURE for c in cs) / len(cs), 3),
                          "mean_conf": round(sum(cs) / len(cs), 3)})
        return pick(cands, a)
    jobs = [json.loads(l) for l in a.jobs.read_text().splitlines() if l.strip()]
    jobs = [j for j in jobs if j["location"] in ROOMS and j["conv"] not in ex]
    rng = random.Random(a.seed)
    rng.shuffle(jobs)
    cands = []
    for j in jobs[:a.tapes]:
        d = a.root / "out" / j["conv"]
        if not (d / a.input).exists():
            continue
        for w in windows(json.loads((d / a.input).read_text())):
            cands.append({"conv": j["conv"], "location": j["location"], "date": j.get("date", ""), **w})
    return pick(cands, a)


def pick(cands: list[dict], a) -> int:
    rng = random.Random(a.seed)
    cands.sort(key=lambda c: -c["mean_conf"])  # most confident (easiest) first
    n = len(cands)
    if not n:
        print("no dense windows found")
        return 1
    print(f"{n} dense windows from {len({c['conv'] for c in cands})} tapes; mean confidence "
          + ", ".join(f"p{q}={cands[int(q / 100 * (n - 1))]['mean_conf']:.2f}" for q in (10, 25, 50, 75, 90)))
    picks, used = [], set()
    for band, (lo, hi) in BANDS.items():
        pool = cands[int(lo * n):int(hi * n)]
        rng.shuffle(pool)
        k = 0
        for c in pool:
            if c["conv"] in used:
                continue
            used.add(c["conv"])
            picks.append({**c, "band": band})
            k += 1
            if k == a.per_band:
                break
    a.out.write_text("".join(json.dumps(p) + "\n" for p in picks))
    for p in picks:
        print(f"  {p['band']:6} {p['conv']} {p['location']:4} {p['start'] / 60:5.1f} min  {p['pk_words']} words  "
              f"unsure {p['unsure']:.0%}  mean confidence {p['mean_conf']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
