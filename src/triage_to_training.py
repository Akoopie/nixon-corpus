"""Turn the listener's triage answers (hard_clips.py + triage_page.html) into empty-text Parakeet training clips.

    python src/triage_to_training.py <triage dir> --out DIR [--audio-prefix /workspace/correct1/v5data/clips]

"Nothing said" clips are used whole with empty text, teaching the model to write nothing over tape noise.
For "mixed" clips, each run of at least MIN_RUN seconds that the listener left unmarked (no voices) is cut out
and used the same way. Crosstalk and one-person clips are not training text: they are listed in
DIR/crosstalk.jsonl and DIR/speech.jsonl for later use. Original (not hum-filtered) audio is used.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

MIN_RUN = 1.0  # NeMo's training loader drops clips under 1 s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("triage", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio-prefix", default="/workspace/correct1/v5data/clips")
    a = ap.parse_args()
    labels = {}
    for l in (a.triage / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            labels[x["id"]] = x  # a later answer for the same clip wins
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    rows, other = [], {"crosstalk": [], "speech": []}
    for cid, x in labels.items():
        if x["answer"] in other:
            other[x["answer"]].append(x)
            continue
        with wave.open(str(a.triage / "audio" / f"{cid}.wav"), "rb") as r:
            sr, frames = r.getframerate(), r.readframes(r.getnframes())
        length = len(frames) / 2 / sr
        if x["answer"] == "nothing":
            runs = [(0.0, length)]
        else:  # mixed: unmarked seconds are noise; join neighbours into runs
            noise = [s for s in range(5) if s not in set(x.get("voiced_seconds") or [])]
            runs, cur = [], None
            for s in noise:
                cur = (cur[0], s + 1) if cur and cur[1] == s else (cur and runs.append(cur)) or (s, s + 1)
            if cur:
                runs.append(cur)
            runs = [(float(s), min(float(e), length)) for s, e in runs]
        for k, (s, e) in enumerate(runs):
            if e - s < MIN_RUN:
                continue
            name = f"{cid}_n{k}"
            with wave.open(str(a.out / "clips" / f"{name}.wav"), "wb") as o:
                o.setnchannels(1), o.setsampwidth(2), o.setframerate(sr)
                o.writeframes(frames[int(s * sr) * 2:int(e * sr) * 2])
            rows.append({"audio_filepath": f"{a.audio_prefix}/{name}.wav", "duration": round(e - s, 2), "text": "",
                         "source": "triage-" + x["answer"], "conv": x["conv"]})
    (a.out / "train_silence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    for k, v in other.items():
        (a.out / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v), encoding="utf-8", newline="\n")
    print(f"{len(rows)} empty-text clips ({sum(r['duration'] for r in rows):.0f} s) from "
          f"{sum(x['answer'] == 'nothing' for x in labels.values())} 'nothing' and {sum(x['answer'] == 'mixed' for x in labels.values())} "
          f"'mixed' answers; {len(other['crosstalk'])} crosstalk and {len(other['speech'])} one-person clips listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
