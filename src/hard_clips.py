"""Pick short hard stretches for quick triage by the listener: nothing said / several people at once / one person.

    python hard_clips.py <drafts dir> --out DIR [--n 220] [--exclude-tapes ids.txt]

From transcribe_windows.py drafts (words with confidence + window audio), every CLIP_S-second step of a window
is a candidate. Two kinds are taken: stretches where the model wrote words but was very unsure of them (mean
confidence in the session's bottom LOW_SHARE: likely noise or crosstalk turned into words), and stretches where
it wrote nothing although the audio is loud (possible missed speech). At most PER_WINDOW per window, so the
set spreads over many tapes. Writes DIR/audio/<id>.wav and DIR/audio_clean/<id>.wav plus DIR/session.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import wave
from array import array
from pathlib import Path

CLIP_S, LOW_SHARE, PER_WINDOW, SR = 5.0, 0.10, 3, 16000
QUIET_SHARE = 0.25  # share of the set taken from loud stretches with no words
CLEAN = ("highpass=f=90,equalizer=f=120:t=q:w=8:g=-30,equalizer=f=180:t=q:w=8:g=-25,equalizer=f=240:t=q:w=8:g=-30,"
         "equalizer=f=300:t=q:w=8:g=-20,equalizer=f=360:t=q:w=8:g=-20,equalizer=f=420:t=q:w=8:g=-15,lowpass=f=7000,"
         "afftdn=nr=12:nf=-35,dynaudnorm=f=150:g=15:p=0.9")


def pcm(path: Path, extra: list[str]) -> array:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *extra, "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return array("h", raw)


def rms(a: array) -> float:
    return (sum(x * x for x in a) / max(len(a), 1)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drafts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--exclude-tapes", type=Path)
    ap.add_argument("--seed", type=int, default=1974)
    a = ap.parse_args()
    ex = set(re.findall(r"\d{3}-\d{3}", a.exclude_tapes.read_text())) if a.exclude_tapes else set()
    worded, empty = [], []
    for f in sorted(a.drafts.glob("*.json")):
        d = json.loads(f.read_text())
        if d["conv"] in ex or not f.with_suffix(".flac").exists():
            continue
        ws = [w for w in d["words"] if "c" in w]
        audio = pcm(f.with_suffix(".flac"), [])
        t = 0.0
        while t + CLIP_S <= d["end"] - d["start"]:
            inside = [w for w in ws if t <= (w["s"] + w["e"]) / 2 < t + CLIP_S]
            seg = {"window": f.stem, "conv": d["conv"], "start": t, "end": t + CLIP_S,
                   "draft": " ".join(w["w"] for w in inside)}
            if inside:
                worded.append({**seg, "conf": sum(w["c"] for w in inside) / len(inside)})
            else:
                empty.append({**seg, "rms": rms(audio[int(t * SR):int((t + CLIP_S) * SR)])})
            t += CLIP_S
    worded.sort(key=lambda s: s["conf"])
    empty.sort(key=lambda s: -s["rms"])
    rng = random.Random(a.seed)
    low = worded[:int(LOW_SHARE * len(worded))]
    loud = empty[:max(1, len(empty) // 4)]
    rng.shuffle(low)
    rng.shuffle(loud)
    picks, per = [], {}
    for pool, quota in ((low, round(a.n * (1 - QUIET_SHARE))), (loud, round(a.n * QUIET_SHARE))):
        k = 0
        for s in pool:
            if k == quota:
                break
            if per.get(s["window"], 0) >= PER_WINDOW:
                continue
            per[s["window"]] = per.get(s["window"], 0) + 1
            picks.append({**s, "kind": "unsure words" if "conf" in s else "loud, no words"})
            k += 1
    rng.shuffle(picks)
    for d in ("audio", "audio_clean"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    for p in picks:
        p["id"] = f"{p['window']}_{int(p['start'] * 10):05d}"
        src = a.drafts / f"{p['window']}.flac"
        for d, extra in (("audio", []), ("audio_clean", ["-af", CLEAN])):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(p["start"]), "-t", str(CLIP_S), "-i", str(src), *extra,
                            "-ar", str(SR), "-ac", "1", str(a.out / d / f"{p['id']}.wav")], check=True)
    (a.out / "session.json").write_text(json.dumps(picks, indent=1))
    print(f"{len(worded)} worded and {len(empty)} empty {CLIP_S:.0f}-s stretches; picked {len(picks)} "
          f"({sum(p['kind'] == 'unsure words' for p in picks)} unsure words, {sum(p['kind'] != 'unsure words' for p in picks)} loud with no words) "
          f"from {len(per)} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
