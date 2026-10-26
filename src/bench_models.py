"""Benchmark ASR model sizes (small/medium/large-v3) on the same clip.

For a *searchable research corpus* the metric that matters most is whether the
key proper nouns survive transcription (a mis-rendered name = an unfindable
passage). So this reports:

  1. Named-entity scorecard  - which canonical Watergate names each model recovers
  2. Anchor WER              - WER on a few passages whose wording we're confident of
  3. Side-by-side            - the same time windows rendered by each model

Usage: python src/bench_models.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jiwer

from paths import BENCH_DIR

# model label -> transcript json. The original `small` run was overwritten by
# the full-tape large-v3 transcript, so it is skipped (load() returns None).
MODELS = {
    "small":    BENCH_DIR / "bench-886-008-small.json",
    "medium":   BENCH_DIR / "bench-886-008-medium.json",
    "large-v3": BENCH_DIR / "bench-886-008-largev3.json",
}

# Candidate names. We do NOT assume all are spoken in this 8-min, 2-speaker
# clip — many won't be. large-v3 (most accurate) is treated as the reference
# for which names are ACTUALLY present; small/medium are judged on recovering
# only those, not the whole list.
ENTITIES = ["Dean", "Gray", "Liddy", "Gordon", "Magruder", "Mitchell",
            "Ehrlichman", "Haldeman", "Hunt", "Colson", "Segretti",
            "Kalmbach", "Watergate", "FBI", "CIA"]

# Exact spots where the `small` model produced a garbled/wrong name. The test:
# do bigger models recover the right word here, or is the audio just muddy?
ERROR_SPOTS = [
    ("small said 'Urban' (Ehrlichman?)",     112, 124),
    ("small said 'Bob Baldwin' (Magruder?)", 276, 286),
    ("small said 'Jack...plan' (Liddy?)",     307, 318),
    ("small said 'Gordon...FBI'",             350, 360),
]

# Anchors: (label, reference_text, t_start, t_end) — windows in seconds.
# Wording from the widely-published NARA/Kutler transcripts; tapes are muddy so
# treat absolute WER as approximate, the cross-model comparison as the signal.
ANCHORS = [
    ("cancer", "we have a cancer within close to the presidency that's growing", 218, 250),
]


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def load(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["segments"]


def window_text(segs, t0, t1):
    return " ".join(s["text"] for s in segs if s["start"] < t1 and s["end"] > t0).strip()


def full_text(segs):
    return " ".join(s["text"] for s in segs)


def count_entities(text):
    low = text.lower()
    return {e: len(re.findall(rf"\b{re.escape(e.lower())}\b", low)) for e in ENTITIES}


def main():
    loaded = {m: load(p) for m, p in MODELS.items()}
    present = {m: s for m, s in loaded.items() if s}
    print("Models available:", ", ".join(present) or "(none yet)")
    if not present:
        return

    found = {m: count_entities(full_text(s)) for m, s in present.items()}
    # Ground-truth set of names actually spoken = those the reference model finds.
    ref = "large-v3" if "large-v3" in present else ("medium" if "medium" in present else None)
    present_names = [e for e in ENTITIES if ref and found[ref][e]] if ref else ENTITIES
    print("\n=== 1. NAMED-ENTITY SCORECARD ===")
    print(f"(reference for 'actually spoken' = {ref}; only those names are scored)")
    hdr = "name".ljust(12) + "".join(m.ljust(10) for m in present)
    print(hdr)
    recovered = {m: 0 for m in present}
    for e in present_names:
        row = e.ljust(12)
        for m in present:
            n = found[m][e]
            if n:
                recovered[m] += 1
            row += (str(n) if n else "MISS").ljust(10)
        print(row)
    print("-" * len(hdr))
    tot = len(present_names)
    print("RECOVERED".ljust(12) + "".join(f"{recovered[m]}/{tot}".ljust(10) for m in present))

    print("\n=== 1b. ERROR-CORRECTION TEST (the real question) ===")
    print("At spots where `small` garbled a name, what does each model say?")
    for label, t0, t1 in ERROR_SPOTS:
        print(f"--- {label}  ({t0}-{t1}s) ---")
        for m, s in present.items():
            print(f"   {m:9}: {window_text(s, t0, t1)[:130]}")

    print("\n=== 2. ANCHOR CHECK (canonical phrase recovered verbatim?) ===")
    for label, ref, t0, t1 in ANCHORS:
        nref = _norm_text(ref)
        print(f"[{label}] ref: \"{ref}\"")
        for m, s in present.items():
            hyp = window_text(s, t0, t1)
            nhyp = _norm_text(hyp)
            exact = "EXACT MATCH" if nref in nhyp else "not verbatim"
            try:
                wer = jiwer.wer(nref, nhyp)  # whole-window WER (inflated by context)
            except Exception:
                wer = float("nan")
            print(f"   {m:9} [{exact}] window-WER~{wer:4.2f}")

    print("\n=== 3. SIDE-BY-SIDE key windows ===")
    for label, _, t0, t1 in ANCHORS + [("origin", "", 300, 360)]:
        print(f"--- {label} ({t0}-{t1}s) ---")
        for m, s in present.items():
            print(f"   {m:9}: {window_text(s, t0, t1)[:140]}")


if __name__ == "__main__":
    main()
