"""Quality checks for ASR transcripts: catch the ways Whisper fails on degraded tape.

Failure modes seen on 886-008 (large-v3), each handled here:
  * junk segments    - text with no letters or digits (e.g. ".")        -> dropped
  * repetition loops - the same sentence repeated many times in a row   -> kept once, window flagged
  * looping text     - one segment that repeats itself ("the the the")  -> flagged
  * gaps             - long stretches with no transcript at all         -> window flagged

Flagged windows are where the decoder probably lost real speech: the batch
worker re-decodes them, and anything still flagged is marked unreliable.

Usage:
    python src/qc.py <transcript.json>            # report only
    python src/qc.py <transcript.json> --write    # save cleaned segments + qc report into the JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

# A sentence of >= LONG_WORDS words repeated >= LONG_RUN times in a row is a
# loop; short back-channel ("That's right.") legitimately repeats, so it needs
# a longer run before we call it one.
LONG_WORDS, LONG_RUN, SHORT_RUN = 4, 3, 5
# Whisper's own decode-failure heuristic: text that compresses this well is looping.
COMPRESSION_RATIO_MAX = 2.4
MIN_GAP_S = 20.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b))


def check(segments: list[dict], min_gap: float = MIN_GAP_S) -> tuple[list[dict], dict]:
    """Return (cleaned segments, report). Input segments are not modified."""
    report = {"junk_dropped": [], "repeat_loops": [], "looping_text": [], "gaps": []}

    kept = []
    for s in segments:
        if _norm(s.get("text") or ""):
            kept.append(dict(s))
        else:
            report["junk_dropped"].append({"start": s["start"], "end": s["end"], "text": s.get("text")})

    clean = []
    i = 0
    while i < len(kept):
        key = _norm(kept[i]["text"])
        j = i
        while j + 1 < len(kept) and _norm(kept[j + 1]["text"]) == key:
            j += 1
        run = j - i + 1
        is_loop = run >= (LONG_RUN if len(key.split()) >= LONG_WORDS else SHORT_RUN)
        if is_loop:
            first = kept[i]
            first["end"] = kept[j]["end"]
            first["qc"] = f"repeat_x{run}"
            clean.append(first)
            report["repeat_loops"].append({"start": kept[i]["start"], "end": kept[j]["end"],
                                           "count": run, "text": kept[i]["text"]})
        else:
            clean.extend(kept[i:j + 1])
        i = j + 1

    for s in clean:
        if len(s["text"]) >= 50 and _compression_ratio(s["text"]) > COMPRESSION_RATIO_MAX:
            s["qc"] = (s.get("qc", "") + " looping_text").strip()
            report["looping_text"].append({"start": s["start"], "end": s["end"], "text": s["text"][:120]})

    for a, b in zip(clean, clean[1:]):
        if b["start"] - a["end"] >= min_gap:
            report["gaps"].append({"start": a["end"], "end": b["start"],
                                   "seconds": round(b["start"] - a["end"], 1)})

    windows = report["repeat_loops"] + report["gaps"]
    report["suspect_seconds"] = round(sum(w["end"] - w["start"] for w in windows), 1)
    return clean, report


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def print_report(name: str, n_in: int, clean: list[dict], r: dict) -> None:
    print(f"{name}: {n_in} segments in -> {len(clean)} after cleaning")
    print(f"  junk segments dropped: {len(r['junk_dropped'])}")
    print(f"  repetition loops: {len(r['repeat_loops'])}")
    for w in r["repeat_loops"]:
        print(f"    {_mmss(w['start'])}-{_mmss(w['end'])}  x{w['count']}  {w['text'][:70]!r}")
    print(f"  self-repeating segments: {len(r['looping_text'])}")
    for w in r["looping_text"]:
        print(f"    {_mmss(w['start'])}  {w['text'][:70]!r}")
    print(f"  gaps >= {MIN_GAP_S:.0f}s with no transcript: {len(r['gaps'])}")
    for w in r["gaps"]:
        print(f"    {_mmss(w['start'])}-{_mmss(w['end'])}  ({w['seconds']}s)")
    print(f"  suspect audio total: {r['suspect_seconds'] / 60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--write", action="store_true", help="save cleaned segments + qc report into the JSON")
    args = ap.parse_args()

    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    segs = data["segments"]
    clean, report = check(segs)
    print_report(args.transcript.name, len(segs), clean, report)

    if args.write:
        data["segments"] = clean
        data["qc"] = report
        args.transcript.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote cleaned transcript -> {args.transcript}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
