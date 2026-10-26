"""Training data for fine-tuning ASR on the Nixon tapes themselves.

Source: the Watergate Special Prosecution Force transcripts the Nixon Library
posts (US-government work, public domain), minus the 8 validation
conversations, which must never be trained on.

    python src/finetune_data.py fetch     # transcript PDFs (+ their per-conversation audio, when available)
    python src/finetune_data.py parse     # turns JSON per conversation -> training/transcripts/

Later steps (alignment + clip cutting) run on a GPU pod.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from acquire import USER_AGENT, fetch, load_ledger
from paths import DATA_DIR, MANIFEST_DIR
from validation import PDF, WSPF, parse_pdf

TRAIN_DIR = DATA_DIR / "training"
PDF_DIR = TRAIN_DIR / "sources"
TURNS_DIR = TRAIN_DIR / "transcripts"
WSPF_INDEX = MANIFEST_DIR / "sources" / "wspf_list.jsonl"
WSPF_PROBE = MANIFEST_DIR / "sources" / "wspf_probe.txt"
# Held out for evaluation: never fetched into, or trained from, this directory.
VALIDATION = {"886-008", "421-018", "038-034", "051-001", "488-015", "741-002", "779-002", "037-205"}


def training_conversations() -> list[dict]:
    online = {l.split()[0] for l in WSPF_PROBE.read_text().splitlines() if l.split()[1:2] == ["200"]}
    rows = [json.loads(l) for l in WSPF_INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["conv"] in online and r["conv"] not in VALIDATION]


def cmd_fetch() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ledger, failed = load_ledger(), 0
    convs = training_conversations()
    for r in convs:
        if not fetch(session, WSPF + f"{r['conv']}.pdf", PDF_DIR / f"wspf_{r['conv']}.pdf", ledger, accept=PDF):
            failed += 1
    print(f"{len(convs) - failed}/{len(convs)} training transcripts in {PDF_DIR}")
    return 1 if failed else 0


def cmd_parse() -> int:
    TURNS_DIR.mkdir(parents=True, exist_ok=True)
    total_words = 0
    for r in training_conversations():
        pdf = PDF_DIR / f"wspf_{r['conv']}.pdf"
        if not pdf.exists():
            continue
        turns = parse_pdf(str(pdf))
        words = sum(len(t["text"].split()) for t in turns)
        total_words += words
        minutes = int(m.group(1)) if (m := re.search(r"\((\d+)\s*min", r.get("cassette", ""))) else None
        (TURNS_DIR / f"{r['conv']}.json").write_text(json.dumps(
            {"conversation_id": "37-wht-conversation-" + r["conv"], "date": r["date"], "location": r["loc"],
             "listed_minutes": minutes, "source_url": WSPF + f"{r['conv']}.pdf", "turns": turns},
            indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Parsed {len(list(TURNS_DIR.glob('*.json')))} transcripts, {total_words:,} words -> {TURNS_DIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "parse"])
    args = ap.parse_args()
    return cmd_fetch() if args.cmd == "fetch" else cmd_parse()


if __name__ == "__main__":
    sys.exit(main())
