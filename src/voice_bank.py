"""Build a reference voice bank for the listening session: a few clean samples per person.

Samples come from the voiceprint training clips, whose speakers are confirmed by the WSPF human
transcripts. Per person, the classifier ranks clips by how typical they are (probability of the
labeled speaker after room normalization); the top 5-9.5 s clips from different conversations are
kept, a telephone clip first where one exists. Kissinger, who has no labeled clips, is taken from
798-011, where NARA lists only Nixon and Kissinger and the classifier marks the non-Nixon voice.

    python src/voice_bank.py <listening root> [--per 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path(os.environ.get("NIXON_DATA", Path.home() / "nixon-data"))
VP = DATA / "voiceprints_gpu"
ROOM_OF = {"OVAL": "oval", "EOB": "eob", "WHT": "phone"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--per", type=int, default=3)
    args = ap.parse_args()
    os.environ.setdefault("WORKER_CACHE", str(args.root / "cache"))
    import soundfile as sf
    import worker
    from name_speakers import load_classifier

    jobs = {json.loads(l)["conv"]: json.loads(l) for l in (DATA / "jobs_all_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    segs = [json.loads(l) for l in (VP / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    emb = np.load(VP / "embeddings.npy").astype(np.float32)
    head, classes, means = load_classifier(VP / "speaker_classifier.pt")
    X = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    rooms = [ROOM_OF.get(jobs.get(s["conv"], {}).get("location", ""), "_avg") for s in segs]
    X = X - np.stack([means.get(r, means["_avg"]) for r in rooms])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    with torch.no_grad():
        prob = head(torch.tensor(X)).softmax(1).numpy()

    loaded, broken = {}, set()

    def audio_of(conv):
        """The conversation's audio, or None if no source works (some NARA files can't be fetched)."""
        if conv not in loaded and conv not in broken:
            try:
                loaded[conv] = worker.read_audio(worker.fetch_wav(jobs[conv]))
            except RuntimeError:
                broken.add(conv)
        return loaded.get(conv)

    picks = {}
    for spk in [c for c in classes if c != "other"]:
        k = classes.index(spk)
        cand = sorted((i for i, s in enumerate(segs) if s["speaker"] == spk and 5 <= s["end"] - s["start"] <= 9.5
                       and s["conv"] in jobs), key=lambda i: -prob[i, k])
        chosen, convs = [], set()
        phone = [i for i in cand if rooms[i] == "phone"][:1]
        for i in phone + cand:
            if segs[i]["conv"] not in convs and audio_of(segs[i]["conv"]) is not None:
                chosen.append(i)
                convs.add(segs[i]["conv"])
            if len(chosen) == args.per:
                break
        picks[spk] = [(segs[i]["conv"], segs[i]["start"], segs[i]["end"], float(prob[i, k])) for i in chosen]

    # Kissinger by elimination on 798-011 (NARA: Nixon + Kissinger only)
    named = json.loads((DATA / "demo" / "out" / "798-011" / "speakers.json").read_text(encoding="utf-8"))
    kis = sorted((s for s in named if s["speaker"] == "other" and 5 <= s["end"] - s["start"] <= 9.5),
                 key=lambda s: -s["confidence"])[:args.per]
    picks["Kissinger"] = [("798-011", s["start"], s["end"], s["confidence"]) for s in kis]

    out = args.root / "session" / "voices"
    out.mkdir(parents=True, exist_ok=True)
    bank = {}
    by_conv = {}
    for spk, rows in picks.items():
        for r in rows:
            by_conv.setdefault(r[0], []).append((spk, r))
    for conv, rows in by_conv.items():
        audio = audio_of(conv)
        for spk, (c, a, b, p) in rows:
            name = f"{spk}_{c}_{int(a)}.wav"
            sf.write(str(out / name), audio[int(a * worker.SR):int(b * worker.SR)], worker.SR)
            bank.setdefault(spk, []).append({"file": name, "conv": c, "location": jobs[c]["location"],
                                             "date": jobs[c]["date"], "seconds": round(b - a, 1), "typicality": round(p, 3)})
        (worker.CACHE / f"{conv}.wav").unlink(missing_ok=True)
        print(conv, [s for s, _ in rows], flush=True)
    (args.root / "session" / "voices.json").write_text(json.dumps(bank, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in bank.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
