"""Turn listening-session answers into training and test clips for Parakeet.

Answers with typed words (sure or best guess) are split by tape (a tape's clips all go to one side).
The listener may type only the words they could make out, so each answer is force-aligned to the
clip's ORIGINAL audio with the wav2vec2 aligner, and the clip is cut to the stretch those words
cover. That way training never sees audio holding words the label leaves out. Answers that align
poorly are dropped. Writes listen_train.jsonl / listen_test.jsonl (NeMo manifests) and clips to
<root>/session/train_audio/.

    python src/listening_to_training.py <listening root> [--test-share 0.3]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PAD, MIN_SCORE, MIN_S = 0.15, 0.30, 0.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--test-tapes", nargs="*", help="fix the held-out tapes (keeps an earlier test set); new answers on them are left out")
    args = ap.parse_args()
    import soundfile as sf
    import whisperx

    s = args.root / "session"
    labels = {}
    for l in (s / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            x = json.loads(l)
            labels[x["id"]] = x  # the latest answer for a clip wins
    usable = [x for x in labels.values() if x["text"].strip() and x["confidence"] in ("sure", "guess")]
    tapes = sorted({x["conv"] for x in usable})
    random.Random(1974).shuffle(tapes)
    test_tapes = set(args.test_tapes) if args.test_tapes else set(tapes[:max(1, round(len(tapes) * args.test_share))])
    keep_test = ({json.loads(l)["id"] for l in (s / "listen_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
                 if args.test_tapes and (s / "listen_test.jsonl").exists() else None)
    if keep_test is not None:  # the earlier test set stays exactly as it was
        usable = [x for x in usable if x["conv"] not in test_tapes or x["id"] in keep_test]

    align_model, meta = whisperx.load_align_model(language_code="en", device="cpu")
    (s / "train_audio").mkdir(exist_ok=True)
    rows = {"train": [], "test": []}
    dropped = 0
    for x in usable:
        audio, sr = sf.read(str(s / "audio" / f"{x['id']}.wav"), dtype="float32")
        res = whisperx.align([{"text": x["text"], "start": 0.0, "end": len(audio) / sr}], align_model, meta,
                             audio, "cpu", return_char_alignments=False)
        words = [w for seg in res["segments"] for w in seg.get("words", []) if "start" in w]
        n_typed = len(x["text"].split())
        score = sum(w.get("score", 0) for w in words) / max(len(words), 1)
        if len(words) < 0.8 * n_typed or score < MIN_SCORE:
            dropped += 1
            continue
        a, b = max(0.0, words[0]["start"] - PAD), min(len(audio) / sr, words[-1]["end"] + PAD)
        if b - a < MIN_S:
            dropped += 1
            continue
        f = s / "train_audio" / f"{x['id']}.wav"
        sf.write(str(f), audio[int(a * sr):int(b * sr)], sr)
        side = "test" if x["conv"] in test_tapes else "train"
        rows[side].append({"audio_filepath": f.name, "duration": round(b - a, 2), "text": x["text"].strip(),
                           "id": x["id"], "conv": x["conv"], "confidence": x["confidence"], "align_score": round(score, 3)})
    for side, r in rows.items():
        (s / f"listen_{side}.jsonl").write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in r),
                                                encoding="utf-8", newline="\n")
    print(f"{len(usable)} usable answers on {len(tapes)} tapes -> train {len(rows['train'])} clips "
          f"({sum(z['duration'] for z in rows['train']):.0f} s), test {len(rows['test'])} clips "
          f"({sum(z['duration'] for z in rows['test']):.0f} s) from tapes {sorted(test_tapes)}; dropped {dropped} (poor alignment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
