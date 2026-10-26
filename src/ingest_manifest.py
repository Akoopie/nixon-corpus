"""Download the NARA Nixon conversation-list manifest and load it into the corpus.

The manifest (CC0 / public domain, ~33 MB, 22,723 conversations) is the spine of
the whole project: every transcript segment is later attached to one of these
conversation rows so we inherit date / participants / location metadata for free.

Usage:
    python src/ingest_manifest.py            # download + load
    python src/ingest_manifest.py --skip-download   # reuse local copy
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

from db import connect, init_schema
from paths import MANIFEST_DIR

CSV_URL = "https://www.archives.gov/open/data/37-wht-dataset-conversationlist-2015-09-22.csv"
DICT_URL = "https://www.archives.gov/files/open/nixon/37-wht-dataset-conversationlist-csv-datadictionary-2015-08-28.pdf"

CSV_PATH = MANIFEST_DIR / "conversationlist.csv"
DICT_PATH = MANIFEST_DIR / "datadictionary.pdf"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n       -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB ({pct:4.1f}%)", end="")
        print()


def _pick(row: dict, *candidates: str) -> str:
    """Return the first non-empty value among candidate column names (case-insensitive)."""
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for c in candidates:
        v = lower.get(c.lower())
        if v and v.strip():
            return v.strip()
    return ""


def load_csv(conn) -> int:
    """Load the manifest CSV into the conversations table. Returns row count.

    NARA column names are mapped defensively because the exact header spelling
    is confirmed against the data dictionary on first real run.
    """
    init_schema(conn)
    n = 0
    with open(CSV_PATH, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        print("CSV columns:", reader.fieldnames)
        rows = []
        for row in reader:
            # Actual NARA columns: identifier (stable, e.g. 37-wht-conversation-001-001),
            # conversationTitle (e.g. 001-001), conversationNumber, startDateTime, startDate,
            # startTime, endTime, recordingDevice, locationCode, participants, description.
            conv_id = _pick(row, "identifier", "conversationTitle", "conversationNumber", "id")
            if not conv_id:
                continue
            rows.append((
                conv_id,
                _pick(row, "conversationTitle", "title"),
                _pick(row, "startDateTime", "startDate", "date"),
                _pick(row, "startTime", "beginTime"),
                _pick(row, "endTime", "end_time"),
                _pick(row, "recordingDevice", "device"),
                _pick(row, "locationCode", "location"),
                _pick(row, "participants", "persons"),
                _pick(row, "description", "summary", "generalDescription"),
            ))
            n += 1
        conn.executemany(
            """INSERT OR REPLACE INTO conversations
               (conversation_id, title, date, start_time, end_time, device, location, participants, summary)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    conn.commit()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true", help="reuse local CSV if present")
    args = ap.parse_args()

    if not args.skip_download or not CSV_PATH.exists():
        download(CSV_URL, CSV_PATH)
        try:
            download(DICT_URL, DICT_PATH)
        except Exception as e:  # data dictionary is nice-to-have, not required
            print(f"  (data dictionary download skipped: {e})")

    conn = connect()
    n = load_csv(conn)
    total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    print(f"Loaded {n} rows this run; conversations table now holds {total} rows.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
