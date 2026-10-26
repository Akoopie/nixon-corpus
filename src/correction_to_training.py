"""Turn the listener's corrected chunks (correction_session.py) into Parakeet training and test manifests.

    python src/correction_to_training.py <session dir> --out DIR [--holdout-per-band 2] [--audio-prefix /workspace/correct1/clips]

Uses the original (not hum-filtered) chunk audio, since the model hears raw tape audio. A chunk is left out
when it has no speech, or contains [?] (a stretch the listener could not make out: training on it would teach
the model to drop words). Chunks marked "some of this is a guess" are kept for training and flagged; the test
set uses only chunks without guesses. Whole windows are held out, HOLDOUT per difficulty band, chosen at
random (seed 1974) among complete windows, so a test window's tape never appears in training.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import wave
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-per-band", type=int, default=2)
    ap.add_argument("--audio-prefix", default="/workspace/correct1/clips", help="where the clips will live on the pod")
    ap.add_argument("--seed", type=int, default=1974)
    a = ap.parse_args()
    session = json.loads((a.session / "session.json").read_text(encoding="utf-8"))
    labels = {}
    for l in (a.session / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            labels[x["id"]] = x  # a later answer for the same chunk wins
    complete = [w for w in session if all(c["id"] in labels for c in w["chunks"])]
    rng = random.Random(a.seed)
    held = set()
    for band in ("light", "medium", "heavy"):
        pool = [w["window"] for w in complete if w["band"] == band]
        held |= set(rng.sample(pool, min(a.holdout_per_band, len(pool))))
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "test": []}
    skipped = {"no speech": 0, "[?]": 0, "guess (test)": 0, "unsaved": 0}
    for w in session:
        split = "test" if w["window"] in held else "train"
        for c in w["chunks"]:
            x = labels.get(c["id"])
            if x is None:
                skipped["unsaved"] += 1
                continue
            if x["no_speech"] or not x["text"].strip():
                skipped["no speech"] += 1
                continue
            if x["has_unknown"] or "[?]" in x["text"]:
                skipped["[?]"] += 1
                continue
            if split == "test" and x["guess"]:
                skipped["guess (test)"] += 1
                continue
            shutil.copy(a.session / "audio" / f"{c['id']}.wav", a.out / "clips" / f"{c['id']}.wav")
            # the file's own length: a window's last chunk is shorter than end - start when the conversation ends first
            with wave.open(str(a.out / "clips" / f"{c['id']}.wav"), "rb") as wf:
                secs = wf.getnframes() / wf.getframerate()
            rows[split].append({"audio_filepath": f"{a.audio_prefix}/{c['id']}.wav", "duration": round(secs, 2),
                                "text": x["text"], "source": "correction", "window": w["window"], "band": w["band"],
                                "guess": x["guess"], "edited": x["edited"]})
    for split, r in rows.items():
        (a.out / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r),
                                              encoding="utf-8", newline="\n")
    (a.out / "heldout_windows.txt").write_text("\n".join(sorted(held)) + "\n")
    hours = lambda r: sum(x["duration"] for x in r) / 3600
    print(f"{len(complete)} of {len(session)} windows complete; held out: {sorted(held)}")
    print(f"train {len(rows['train'])} chunks ({hours(rows['train']):.2f} h), test {len(rows['test'])} chunks "
          f"({hours(rows['test']):.2f} h); left out: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
