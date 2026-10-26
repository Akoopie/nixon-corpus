"""Map anonymous diarization clusters (SPEAKER_00, SPEAKER_01, ...) to real names.

Diarization tells us "speaker A vs speaker B" but not *who* they are. We resolve
that using two signals:
  - the NARA manifest participant list for this conversation (the candidate names)
  - linguistic cues in each cluster's speech (e.g. someone addressed as
    "Mr. President" / "sir" is being spoken TO; the President rarely defers)

Workflow:
  1. python src/label_speakers.py <transcript.json> --analyze
       → prints each cluster's stats + sample lines + the manifest participants,
         so a human (or an LLM) can assign names.
  2. python src/label_speakers.py <transcript.json> --apply '{"SPEAKER_00":"Nixon, Richard M.","SPEAKER_01":"Dean, John W."}'
       → rewrites the `speaker` field on every segment and saves.

The --analyze step also prints a heuristic guess for who the President is.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import connect

DEFER_CUES = re.compile(r"\b(mr\.?\s+president|sir)\b", re.I)


def manifest_participants(conv_id: str) -> str:
    conn = connect()
    row = conn.execute(
        "SELECT participants FROM conversations WHERE conversation_id=?", (conv_id,)
    ).fetchone()
    conn.close()
    return row["participants"] if row and row["participants"] else "(not in manifest)"


def analyze(segs: list[dict], conv_id: str) -> None:
    by_spk = defaultdict(list)
    for s in segs:
        by_spk[s.get("speaker", "UNKNOWN")].append(s)

    print(f"Conversation: {conv_id}")
    print(f"Manifest participants: {manifest_participants(conv_id)}")
    print(f"Clusters found: {len(by_spk)}\n")

    # Heuristic: the person OTHERS address with 'sir'/'Mr. President' is the
    # President. We approximate by flagging clusters whose speech contains the
    # deferential cue — those speakers are deferring TO someone, so they are the
    # subordinates; the remaining dominant cluster is likely the President.
    defers = {}
    for spk, items in by_spk.items():
        dur = sum(s.get("end", 0) - s.get("start", 0) for s in items)
        defer_hits = sum(1 for s in items if DEFER_CUES.search(s.get("text", "")))
        defers[spk] = defer_hits
        print(f"[{spk}] segments={len(items)}  speaking_time={dur/60:.1f}min  "
              f"deferential_cues={defer_hits}")
        for s in items[:4]:
            print(f"    {s.get('start',0):6.1f}s: {s.get('text','').strip()[:80]}")
        print()

    # Whoever uses the fewest deferential cues but holds substantial time is the
    # most-likely President (others defer to them, they defer to no one).
    ranked = sorted(by_spk, key=lambda k: (defers[k],
                    -sum(s.get("end",0)-s.get("start",0) for s in by_spk[k])))
    print(f"Heuristic guess — likely President (least deferential): {ranked[0]}")
    print("(Verify against the sample lines before applying.)")


def apply(segs: list[dict], mapping: dict, path: Path) -> None:
    n = 0
    for s in segs:
        spk = s.get("speaker")
        if spk in mapping:
            s["speaker"] = mapping[spk]
            n += 1
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"] = segs
    data["speaker_map"] = mapping
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Applied mapping to {n} segments -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--analyze", action="store_true", help="print clusters + participants for assignment")
    ap.add_argument("--apply", metavar="JSON", help='mapping like \'{"SPEAKER_00":"Nixon, Richard M."}\'')
    args = ap.parse_args()

    data = json.loads(args.transcript.read_text(encoding="utf-8"))
    segs = data["segments"]
    conv_id = data.get("conversation_id", args.transcript.stem)

    if args.apply:
        apply(segs, json.loads(args.apply), args.transcript)
    else:
        analyze(segs, conv_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
