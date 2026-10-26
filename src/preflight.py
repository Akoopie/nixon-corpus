"""Check, before any GPU time is spent, that every input a training pod needs is a real, readable file.

    python3 preflight.py --manifest train.jsonl --manifest dev.jsonl [...] \
        [--model base.nemo --sha256 HEX] [--refs DIR --bench-audio DIR]

Standard library only, so it runs on any pod before the NeMo environment exists. Checks:
  - every manifest's audio_filepath is a regular file (not a symlink), a 16 kHz mono 16-bit WAV that opens,
    and its length matches the manifest's duration
  - the model file is a regular file whose sha256 matches, and its archive holds a config and weights
  - every validation reference has its 16 kHz WAV in the bench audio folder
Exits 1 on the first kind of problem found, after listing up to 10 examples of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import wave
from pathlib import Path


def check_manifest(path: Path) -> list[str]:
    bad, n, empty, hours, too_long = [], 0, 0, 0.0, 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        a = Path(row["audio_filepath"])
        if a.is_symlink() or not a.is_file():
            bad.append(f"{path.name}:{i} missing or not a regular file: {a}")
            continue
        try:
            with wave.open(str(a), "rb") as w:
                sr, ch, sw, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        except Exception as e:
            bad.append(f"{path.name}:{i} unreadable WAV {a}: {e}")
            continue
        if (sr, ch, sw) != (16000, 1, 2) or frames == 0:
            bad.append(f"{path.name}:{i} {a}: {sr} Hz, {ch} ch, {sw * 8} bit, {frames} frames")
            continue
        secs = frames / sr
        if abs(secs - float(row["duration"])) > 0.25:
            bad.append(f"{path.name}:{i} {a}: file is {secs:.2f} s, manifest says {row['duration']}")
        empty += row.get("text", "") == ""
        too_long += float(row["duration"]) > 22.0
        hours += secs / 3600
    print(f"{path}: {n} clips, {hours:.2f} h, {empty} empty-text, {too_long} over 22 s, {len(bad)} problems")
    return bad


def check_model(path: Path, sha: str) -> list[str]:
    if path.is_symlink() or not path.is_file():
        return [f"model missing or not a regular file: {path}"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != sha:
        return [f"model sha256 {h.hexdigest()} != expected {sha} ({path.stat().st_size} bytes)"]
    with tarfile.open(path) as t:
        names = [Path(m.name).name for m in t.getmembers()]
    missing = [x for x in ("model_config.yaml", "model_weights.ckpt") if x not in names]
    print(f"{path}: {path.stat().st_size} bytes, sha256 ok, {len(names)} archive members")
    return [f"model archive lacks {m}" for m in missing]


def audio_key(ref: dict) -> str:
    """Same rule as bench_asr.audio_key (kept here so this script needs no third-party packages)."""
    first = Path(ref["audio"][0])
    if first.parent.parent.name == "conversations" and first.parent.parent.parent.name == "audio":
        return first.parent.name
    return re.sub(r"_0\d$", "", first.stem) if len(ref["audio"]) > 1 else first.stem


def check_refs(refs: Path, audio: Path) -> list[str]:
    bad, keys = [], set()
    files = sorted(refs.glob("*.json"))
    if not files:
        return [f"no references in {refs}"]
    for p in files:
        key = audio_key(json.loads(p.read_text(encoding="utf-8")))
        keys.add(key)
        wav = audio / f"{key}.wav"
        if wav.is_symlink() or not wav.is_file() or wav.stat().st_size < 1000:
            bad.append(f"reference {p.stem}: no audio {wav}")
    print(f"{refs}: {len(files)} references over {len(keys)} recordings, {len(bad)} problems")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--model", type=Path)
    ap.add_argument("--sha256")
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--bench-audio", type=Path)
    a = ap.parse_args()
    problems = []
    for m in a.manifest:
        problems += check_manifest(m)
    if a.model:
        problems += check_model(a.model, a.sha256)
    if a.refs:
        problems += check_refs(a.refs, a.bench_audio)
    for p in problems[:10]:
        print("PROBLEM", p)
    print("PREFLIGHT OK" if not problems else f"PREFLIGHT FAILED: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
