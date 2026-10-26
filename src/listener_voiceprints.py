"""Voiceprints from the listener's speaker labels (who said which words), for the speaker classifier.

For each answered clip with word-level speaker marks and a name chosen for A or B, the typed words are
force-aligned to the clip's original audio; contiguous words marked as the same named person become a
span, and each span of at least MIN_S seconds is embedded with the corpus's voiceprint model. Marks of
"both", "not sure", "(unknown)" or "Not Nixon" are not used as labels.

    python src/listener_voiceprints.py <listening root> --out DIR     # writes segments.jsonl + embeddings.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MIN_S = 1.0
ROOM_OF = {"OVAL": "oval", "EOB": "eob", "WHT": "phone"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import soundfile as sf
    import torch
    import whisperx
    from pyannote.audio import Inference, Model

    s = args.root / "session"
    labels = {}
    for l in (s / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            labels[x["id"]] = x
    clips = {c["id"]: c for c in json.loads((s / "clips.json").read_text(encoding="utf-8"))}
    token = os.environ.get("HF_TOKEN") or (Path(__file__).resolve().parents[1] / ".secrets" / "hf_token.txt").read_text().strip()
    embed = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=token), window="whole")
    align_model, meta = whisperx.load_align_model(language_code="en", device="cpu")
    rows, embs = [], []
    for x in labels.values():
        names, marks, words = x.get("names", {}), x.get("speakers") or [], x["text"].split()
        if not any(marks) or len(marks) != len(words):
            continue
        audio, sr = sf.read(str(s / "audio" / f"{x['id']}.wav"), dtype="float32")
        res = whisperx.align([{"text": x["text"], "start": 0.0, "end": len(audio) / sr}], align_model, meta, audio, "cpu",
                             return_char_alignments=False)
        aw = [w for seg in res["segments"] for w in seg.get("words", [])]
        if len(aw) != len(words):
            continue
        k = 0
        while k < len(words):
            who = marks[k]
            j = k
            while j < len(words) and marks[j] == who:
                j += 1
            name = names.get(who, "") if who in ("A", "B") else ""
            timed = [w for w in aw[k:j] if "start" in w]
            if name and not name.startswith(("(", "Not ")) and timed:
                a, b = timed[0]["start"], timed[-1]["end"]
                if b - a >= MIN_S:
                    v = embed({"waveform": torch.from_numpy(audio[int(a * sr):int(b * sr)]).unsqueeze(0), "sample_rate": sr})
                    clip = clips[x["id"]]
                    rows.append({"conv": x["conv"], "speaker": name.split(",")[0], "start": round(clip["start"] - clip.get("clip_offset", 3) + a, 2),
                                 "end": round(clip["start"] - clip.get("clip_offset", 3) + b, 2),
                                 "room": ROOM_OF.get(clip["location"], "other"), "source": "listener"})
                    embs.append(np.asarray(v, np.float32))
            k = j
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "segments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    np.save(args.out / "embeddings.npy", np.asarray(embs, np.float32).reshape(len(embs), -1))
    from collections import Counter
    print(f"{len(rows)} labeled spans -> {args.out}: {dict(Counter(r['speaker'] for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
