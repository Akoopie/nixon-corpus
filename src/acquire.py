"""Download source audio and NARA finding aids into the local archive.

Driven by the audio manifest (manifest/audio_manifest.csv), which maps every
NARA conversation to its best download URL plus other parts and fallbacks.

Modes:
    python src/acquire.py nixontapes        # whole-tape audio that exists ONLY on nixontapes.org
                                            # (site closes 2027-07-11) + NARA time-code sheets
    python src/acquire.py subject-logs      # NARA tape subject logs (one ~240 MB ZIP)
    python src/acquire.py conversations 886-008 421-018 ...   # per-conversation files
    python src/acquire.py conversations --all                 # every per-conversation file (~250 GB)

Polite by design: one connection, a pause between files, resumable via HTTP
Range, and each file is verified (content type, expected size) before it
counts. Finished files are recorded with their sha256 in manifest/downloads.jsonl,
so re-running skips what is already done.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from paths import AUDIO_DIR, MANIFEST_DIR

AUDIO_MANIFEST = MANIFEST_DIR / "audio_manifest.csv"
LEDGER = MANIFEST_DIR / "downloads.jsonl"
SUBJECT_LOGS_URL = ("https://www.nixonlibrary.gov/sites/default/files/2021-04/"
                    "Nixon%20White%20House%20Tapes%2C%20updated%202021-04.zip")

USER_AGENT = "nixon-corpus-archiver/0.1 (non-commercial research archive; single connection)"
AUDIO_TYPES = ("audio/", "application/octet-stream")
# Whole-tape parts on nixontapes.org: rmn_e001a.mp3, rmn_e462a_tr1.mp3, cab/051b_i.mp3 ...
# but rmn_e779_02.mp3 (chron4) is a single conversation.
WHOLE_TAPE = re.compile(r"/(rmn_e\d{3}[a-z]?(_[a-z0-9_]+)?|cab/\d{3}[a-z]?(_[a-z0-9]+)?)\.mp3$", re.I)
NT_PER_CONV = re.compile(r"/rmn_e\d{3}_\d{2,3}\.mp3$", re.I)
RETRY_WAITS = [5, 15, 45, 120]
MIN_FREE_GB = 40  # stop downloading before the disk gets tight
MAX_RATE = 0.0    # bytes/s cap per process (--max-rate); 0 = unlimited. Also caps disk writes.


class NotWanted(Exception):
    """The server answered, but with something other than the file (e.g. an HTML page)."""


def load_ledger() -> dict[str, dict]:
    done = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            try:  # several downloaders may append at once; tolerate a mangled line
                e = json.loads(line)
                done[e["url"]] = e
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def record(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _get_once(session: requests.Session, url: str, part: Path,
              accept: tuple[str, ...]) -> tuple[int, str | None]:
    """Download url into `part`, resuming if it already holds bytes.
    Returns (total size, sha256 if the whole file streamed through this call)."""
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
        if r.status_code == 416:  # range past the end: the .part is already complete
            return have, None
        if r.status_code in (429, 503):
            raise requests.HTTPError(f"{r.status_code} retry-after={r.headers.get('Retry-After')}", response=r)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        if not ctype.startswith(accept):
            raise NotWanted(f"unexpected content type {ctype!r}")
        if r.status_code == 200 and have:  # server ignored Range; start over
            have = 0
        total = (int(r.headers["Content-Range"].rsplit("/", 1)[1]) if r.status_code == 206
                 else int(r.headers.get("Content-Length") or 0))
        hasher = hashlib.sha256() if not have else None  # hash while streaming: no re-read
        start, done = time.monotonic(), 0
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                if hasher:
                    hasher.update(chunk)
                done += len(chunk)
                if MAX_RATE and (ahead := done / MAX_RATE - (time.monotonic() - start)) > 0:
                    time.sleep(ahead)
    size = part.stat().st_size
    if total and size != total:
        raise IOError(f"incomplete: {size} of {total} bytes")
    return size, hasher.hexdigest() if hasher else None


def fetch(session: requests.Session, url: str, dest: Path, ledger: dict,
          accept: tuple[str, ...] = AUDIO_TYPES) -> dict | None:
    """Download one file with retries. Returns its ledger entry, or None if unobtainable."""
    prev = ledger.get(url)
    if prev and dest.exists() and dest.stat().st_size == prev["bytes"]:
        return prev
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            size, digest = _get_once(session, url, part, accept)
            break
        except NotWanted as e:
            print(f"    skip {url}: {e}")
            part.unlink(missing_ok=True)
            return None
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (403, 404, 410):
                print(f"    skip {url}: HTTP {code}")
                return None
            print(f"    attempt {attempt + 1}/{len(RETRY_WAITS) + 1} failed {url}: {e}")
        except (requests.RequestException, IOError) as e:
            print(f"    attempt {attempt + 1}/{len(RETRY_WAITS) + 1} failed {url}: {e}")
    else:
        print(f"    FAILED {url} (partial kept for resume)")
        return None
    part.replace(dest)
    entry = {"url": url, "path": str(dest), "bytes": size, "sha256": digest or sha256(dest),
             "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record(entry)
    ledger[url] = entry
    return entry


def nixontapes_dest(url: str) -> Path:
    """http://nixontapeaudio.org/chron1/rmn_e001a.mp3 -> audio/nixontapes/chron1/rmn_e001a.mp3"""
    p = urlparse(url).path.strip("/").split("/")
    return AUDIO_DIR / "nixontapes" / p[-2] / unquote(p[-1])


def read_manifest() -> list[dict]:
    with open(AUDIO_MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def conversation_options(row: dict) -> list[list[str]]:
    """Ordered ways to get one conversation's audio; each option is a list of URLs
    that together make up the recording (a multi-part file is one option)."""
    urls = [row["primary_url"]] + row["alt_urls"].split()
    # NARA's catalog lists ~250 files as "...-%20Pa_0.mp3", which its media server can't serve;
    # the real file is "...-pa_0.mp3" (verified 2026-10-18), so try that first.
    urls = [u.replace("-%20Pa_0.mp3", "-pa_0.mp3") for u in urls if "-%20Pa_0.mp3" in u] + urls
    per_conv = [u for u in urls if u and (NT_PER_CONV.search(u) or not WHOLE_TAPE.search(u))]
    options, used = [], set()
    for u in per_conv:
        if u in used:
            continue
        m = re.search(r"(\d{3}-\d{3})[a-z]\.mp3$", u)
        if m:  # 886-008a.mp3 + 886-008b.mp3 are halves of one recording
            sib = [v for v in per_conv if re.search(re.escape(m.group(1)) + r"[a-z]\.mp3$", v)
                   and v.rsplit("/", 1)[0] == u.rsplit("/", 1)[0]]
            options.append(sorted(sib))
            used.update(sib)
        else:
            options.append([u])
            used.add(u)
    return options


def cmd_nixontapes(session, ledger, delay: float) -> int:
    rows = [r for r in read_manifest() if r["tier"].startswith("D")]
    sheets = sorted({r["time_code_xls"] for r in rows if r["time_code_xls"]})
    audio = sorted({u for r in rows for u in [r["primary_url"]] + r["alt_urls"].split() if u})
    print(f"nixontapes.org-only audio: {len(rows)} conversations, {len(audio)} tape files, "
          f"{len(sheets)} time-code sheets")
    sheet_failed = 0
    for url in sheets:
        cached = url in ledger
        if not fetch(session, url, MANIFEST_DIR / "timecodes" / url.rsplit("/", 1)[1], ledger,
                     accept=("application/vnd.ms-excel", "application/octet-stream", "application/x-msexcel")):
            sheet_failed += 1
        if not cached:
            time.sleep(delay / 4)
    print(f"  time-code sheets: {len(sheets) - sheet_failed}/{len(sheets)} present")
    failed, t0, got = 0, time.time(), 0
    for i, url in enumerate(audio, 1):
        cached = url in ledger
        e = fetch(session, url, nixontapes_dest(url), ledger)
        if not e:
            failed += 1
            continue
        if not cached:
            got += e["bytes"]
            rate = got / max(time.time() - t0, 1) / 1e6
            print(f"  [{i}/{len(audio)}] {e['bytes'] / 1e6:6.1f} MB  {url.rsplit('/', 1)[1]}  ({rate:.1f} MB/s)")
            time.sleep(delay)
    print(f"Done: {len(audio) - failed}/{len(audio)} audio files present, {got / 1e9:.1f} GB new this run.")
    return 1 if failed or sheet_failed else 0


def cmd_subject_logs(session, ledger, delay: float) -> int:
    e = fetch(session, SUBJECT_LOGS_URL, MANIFEST_DIR / "subject_logs" / "nixon-white-house-tapes-2021-04.zip",
              ledger, accept=("application/zip", "application/octet-stream", "application/x-zip"))
    print(f"Subject logs: {e['path']} ({e['bytes'] / 1e6:.0f} MB)" if e else "Subject logs download FAILED")
    return 0 if e else 1


def free_gb() -> float:
    import shutil
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(AUDIO_DIR).free / 1e9  # the disk the audio lands on, not the system disk


def cmd_conversations(session, ledger, delay: float, ids: list[str]) -> int:
    by_id = {r["conv_id"]: r for r in read_manifest()}
    failed, got, t0 = 0, 0, time.time()
    for n, cid in enumerate(ids, 1):
        cid = cid.removeprefix("37-wht-conversation-")
        row = by_id.get(cid)
        if not row:
            print(f"{cid}: not in manifest")
            failed += 1
            continue
        if free_gb() < MIN_FREE_GB:
            print(f"Stopping: only {free_gb():.0f} GB free (limit {MIN_FREE_GB} GB). Re-run to resume.")
            return 1
        for option in conversation_options(row):
            dest_dir = AUDIO_DIR / "conversations" / cid
            cached = all(u in ledger for u in option)
            entries = [fetch(session, u, dest_dir / unquote(u.rsplit("/", 1)[1]), ledger) for u in option]
            if all(entries):
                if not cached:
                    size = sum(e["bytes"] for e in entries)
                    got += size
                    print(f"[{n}/{len(ids)}] {cid}: {len(entries)} file(s), {size / 1e6:.1f} MB from "
                          f"{urlparse(option[0]).netloc}  ({got / 1e9:.1f} GB, "
                          f"{got / max(time.time() - t0, 1) / 1e6:.1f} MB/s)", flush=True)
                    time.sleep(delay)
                break
            time.sleep(delay)
        else:
            print(f"[{n}/{len(ids)}] {cid}: no per-conversation source worked "
                  f"(tier {row['tier']}; needs whole-tape split)")
            failed += 1
    print(f"Done: {len(ids) - failed}/{len(ids)} conversations present, {got / 1e9:.1f} GB new this run.")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["nixontapes", "subject-logs", "conversations"])
    ap.add_argument("ids", nargs="*", help="conversation ids for 'conversations' mode, e.g. 886-008")
    ap.add_argument("--all", action="store_true", help="conversations mode: every conversation that has "
                    "a per-conversation file (tiers A and B)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to pause between files (default: 2)")
    ap.add_argument("--max-rate", type=float, default=0, help="download speed cap in MB/s (default: none)")
    args = ap.parse_args()
    global MAX_RATE
    MAX_RATE = args.max_rate * 1e6

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ledger = load_ledger()
    if args.mode == "nixontapes":
        return cmd_nixontapes(session, ledger, args.delay)
    if args.mode == "subject-logs":
        return cmd_subject_logs(session, ledger, args.delay)
    if args.all:
        args.ids = [r["conv_id"] for r in read_manifest() if r["tier"][0] in "AB"]
    if not args.ids:
        ap.error("conversations mode needs ids or --all")
    return cmd_conversations(session, ledger, args.delay, args.ids)


if __name__ == "__main__":
    sys.exit(main())
