"""NARA tape subject logs: the archivists' per-conversation topic outlines.

    python src/subject_logs.py extract    # unpack the Library's bulk ZIP (works on a truncated copy)

The Library serves its 2021-04 bulk ZIP (240.8 MB) truncated at ~207.5 MB, so
the central directory at the end never arrives. Files are recovered by walking
the local file headers instead, which works for every entry stored before the
cut; each is checked against its CRC-32. What's lost is only the trailing
Acrobat search index, which we don't need.
"""
from __future__ import annotations

import argparse
import mmap
import struct
import sys
import zlib
from pathlib import Path

from paths import MANIFEST_DIR

LOGS_DIR = MANIFEST_DIR / "subject_logs"
ZIP_NAME = "nixon-white-house-tapes-2021-04.zip"
EXTRACTED = LOGS_DIR / "extracted"
PREFIX = "Nixon White House Tapes, updated 2021-04/"


def walk_local_headers(data) -> list[tuple[str, bytes | None]]:
    """(name, file bytes or None if incomplete/corrupt) for each local file header."""
    out, pos = [], 0
    while (i := data.find(b"PK\x03\x04", pos)) >= 0 and i + 30 <= len(data):
        _, flags, method, _, _, crc, csize, _, nlen, xlen = struct.unpack("<HHHHHIIIHH", data[i + 4:i + 30])
        name = data[i + 30:i + 30 + nlen].decode("utf-8" if flags & 0x800 else "cp437", "replace")
        start = i + 30 + nlen + xlen
        body = None
        if start + csize <= len(data) and not flags & 8:  # sizes are in the header
            raw = data[start:start + csize]
            try:
                body = zlib.decompress(raw, -15) if method == 8 else raw if method == 0 else None
            except zlib.error:
                body = None
            if body is not None and zlib.crc32(body) != crc:
                body = None
        out.append((name, body))
        pos = start + max(csize, 1)
    return out


def cmd_extract() -> int:
    src = LOGS_DIR / ZIP_NAME
    if not src.exists():
        src = LOGS_DIR / (ZIP_NAME + ".part")
    if not src.exists():
        print(f"No ZIP found in {LOGS_DIR}; run: python src/acquire.py subject-logs")
        return 1
    with open(src, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        entries = walk_local_headers(data)
    ok = bad = 0
    for name, body in entries:
        if name.endswith("/"):
            continue
        rel = name.removeprefix(PREFIX)
        if body is None:
            print(f"  incomplete: {rel}")
            bad += 1
            continue
        dest = EXTRACTED / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        ok += 1
    print(f"Extracted {ok} files to {EXTRACTED} ({bad} incomplete) from {src.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract"])
    ap.parse_args()
    return cmd_extract()


if __name__ == "__main__":
    sys.exit(main())
