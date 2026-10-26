"""Summarize a fleet run from its logs and outputs (standard library only, so it runs on a bare CPU pod).

  * throughput and cost: audio hours per stage, pod-hours from each pod's boot-to-exit log span
  * failures
  * what Whisper dropped: stretches of >= 20 s with no Whisper segment (qc.py's gap check) where
    the fine-tuned Parakeet heard words, i.e. speech one model kept and the other lost

    python3 src/fleet_report.py --jobs /workspace/corpus/jobs_trial.jsonl --root /workspace/corpus [--price 0.89]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from qc import check

LINE = re.compile(r"^\[(parakeet|whisper)\] (\S+): +([\d.]+) min audio in +(\d+) s")
STAMP = re.compile(r"^(\d\d:\d\d:\d\d) ")


def pod_hours(log: Path) -> float:
    """Boot to the terminate line, or to now for a pod still running (pods log UTC); handles midnight."""
    text = log.read_text(errors="replace")
    stamps = [datetime.strptime(m.group(1), "%H:%M:%S") for l in text.splitlines() if (m := STAMP.match(l))]
    if not stamps:
        return 0.0
    end = stamps[-1] if "terminating" in text else datetime.strptime(datetime.utcnow().strftime("%H:%M:%S"), "%H:%M:%S")
    span = end - stamps[0]
    return (span + timedelta(days=1) if span.total_seconds() < 0 else span).total_seconds() / 3600


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--price", type=float, default=0.89, help="$ per pod-hour")
    args = ap.parse_args()
    jobs = [json.loads(l) for l in args.jobs.read_text(encoding="utf-8").splitlines() if l.strip()]

    audio = {"parakeet": 0.0, "whisper": 0.0}
    busy = {"parakeet": 0.0, "whisper": 0.0}
    hours, finished = 0.0, 0
    for log in sorted((args.root / "logs").glob("*.log")):
        text = log.read_text(errors="replace")
        for m in map(LINE.match, text.splitlines()):
            if m:
                audio[m.group(1)] += float(m.group(3)) / 60
                busy[m.group(1)] += int(m.group(4)) / 3600
        hours += pod_hours(log)
        finished += "job list exhausted" in text
    logs = len(list((args.root / "logs").glob("*.log")))
    print(f"pods: {logs} ({finished} finished), {hours:.2f} pod-hours from logs = ${hours * args.price:.2f} "
          f"(excludes ~1 min per pod before boot)")
    for stage in audio:
        rate = audio[stage] / busy[stage] if busy[stage] else 0
        print(f"{stage:9} {audio[stage]:7.1f} audio h, {busy[stage]:5.2f} busy h, {rate:5.0f}x realtime")
    if audio["whisper"] and hours:
        per_h = hours * args.price / audio["whisper"]
        print(f"cost per audio hour: ${per_h:.4f}  (x 2,077 h -> ${per_h * 2077:.0f} for the whole corpus)")
    failed = sorted(p.name for p in (args.root / "failed").glob("*.txt")) if (args.root / "failed").exists() else []
    print(f"failed: {len(failed)} {failed[:12]}")

    both = gaps = gap_s = lost_words = pk_words = wh_words = loops = 0
    for j in jobs:
        d = args.root / "out" / j["conv"]
        if not ((d / "whisper.json").exists() and (d / "parakeet.json").exists()):
            continue
        wh = json.loads((d / "whisper.json").read_text(encoding="utf-8"))
        if "words" not in wh:
            continue
        pk = json.loads((d / "parakeet.json").read_text(encoding="utf-8"))
        both += 1
        wh_words += len(wh["words"])
        pk_words += len(pk["words"])
        _, report = check(wh["segments"])
        loops += len(report["repeat_loops"])
        for g in report["gaps"]:
            heard = sum(g["start"] <= w["s"] < g["end"] for w in pk["words"])
            if heard >= 10:  # Parakeet heard real speech where Whisper has nothing
                gaps += 1
                gap_s += g["end"] - g["start"]
                lost_words += heard
    if both:
        print(f"tapes with both stages: {both}; words: Whisper {wh_words:,}, Parakeet {pk_words:,}")
        print(f"Whisper silent >= 20 s where Parakeet heard >= 10 words: {gaps} stretches, "
              f"{gap_s / 3600:.2f} h, {lost_words:,} Parakeet words ({lost_words / max(pk_words, 1):.1%}); "
              f"Whisper repetition loops: {loops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
