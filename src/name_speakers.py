"""Put names on the voices in worker output, using the voiceprint classifier and NARA's guest list.

For each Whisper segment that has a voiceprint: subtract the tape's room voiceprint, score it
with the classifier, keep only people NARA lists as present (plus "other"), and name the
speaker when the classifier is at least MIN_CONF sure; otherwise "Unknown (possibly X)".

    python src/name_speakers.py <corpus_root> --jobs jobs.jsonl --classifier speaker_classifier.pt
    python src/name_speakers.py <corpus_root> ... --evaluate     # score against human transcripts (validation tapes)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from torch import nn

from score import REFS, norm_words

MIN_CONF = 0.7  # v4 classifier: 95.7% correct on 64% of validation segments (0.6: 93.4% on 73%)
ROOM_OF = {"OVAL": "oval", "EOB": "eob", "WHT": "phone"}  # others fall back to the average room


def load_classifier(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    head = nn.Linear(len(next(iter(ck["room_means"].values()))), len(classes)) if ck["hidden"] == 0 else None
    head.load_state_dict({k.removeprefix("0."): v for k, v in ck["state"].items()})
    head.eval()
    means = {k: np.asarray(v, np.float32) for k, v in ck["room_means"].items()}
    means["_avg"] = np.mean(list(means.values()), 0)
    return head, classes, means


def surnames(participants: list[str]) -> set[str]:
    return {p.split(",")[0].strip().lower() for p in participants if p.strip()}


def name_segments(out: dict, job: dict, head, classes, means) -> list[dict]:
    prints = np.asarray(out["voiceprints"]["segments"], np.float32)
    if not len(prints):
        return []
    X = prints / np.linalg.norm(prints, axis=1, keepdims=True)
    X = X - means.get(ROOM_OF.get(job["location"], ""), means["_avg"])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    present = surnames(job.get("participants", []))
    allowed = torch.tensor([c == "other" or c.lower() in present for c in classes])
    with torch.no_grad():
        logits = head(torch.tensor(X))
        logits[:, ~allowed] = -1e9  # NARA's guest list: nobody absent can be named
        prob = logits.softmax(1).numpy()
    named = []
    for seg in out["segments"]:
        if "vp" not in seg:
            continue
        p = prob[seg["vp"]]
        best = int(p.argmax())
        label = classes[best] if p[best] >= MIN_CONF else f"Unknown (possibly {classes[best]})"
        named.append({**{k: seg[k] for k in ("start", "end", "text")}, "speaker": label,
                      "confidence": round(float(p[best]), 3)})
    return named


def truth_by_segment(named: list[dict], conv: str, whisper_out: dict) -> list[str | None]:
    """Who really spoke each segment, per the human transcript: align the reference words to
    Whisper's words and take the majority reference speaker of each segment's matched words."""
    # Prefer the WSPF / NARA transcripts; the OCR'd 1974 House print has unreliable speaker labels.
    refs = sorted(REFS.glob(f"{conv}_*.json"), key=lambda p: ("wspf" not in p.name, "nara" not in p.name))
    if not refs:
        return [None] * len(named)
    turns = json.loads(refs[0].read_text(encoding="utf-8"))["turns"]
    ref_words, ref_spk = [], []
    for t in turns:
        w = norm_words(t["text"])
        ref_words += w
        ref_spk += [t["speaker"]] * len(w)
    hyp = [(n, w["s"]) for w in whisper_out["words"] for n in norm_words(w["w"])]
    sm = SequenceMatcher(None, ref_words, [h[0] for h in hyp], autojunk=False)
    time_spk = []
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            time_spk.append((hyp[b.b + k][1], ref_spk[b.a + k]))
    truths = []
    for seg in named:
        votes = Counter(s for t, s in time_spk if seg["start"] <= t < seg["end"])
        truths.append(votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1] >= 2 else None)
    return truths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="corpus root holding out/<conv>/whisper.json")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    head, classes, means = load_classifier(args.classifier)
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in args.jobs.read_text(encoding="utf-8").splitlines() if l.strip()}
    total = Counter()
    for conv, job in jobs.items():
        path = args.root / "out" / conv / "whisper.json"
        if not path.exists():
            continue
        out = json.loads(path.read_text(encoding="utf-8"))
        named = name_segments(out, job, head, classes, means)
        (args.root / "out" / conv / "speakers.json").write_text(json.dumps(named, ensure_ascii=False), encoding="utf-8")
        labeled = [n for n in named if not n["speaker"].startswith("Unknown")]
        line = f"{conv} ({job['location']}): {len(named)} segments, {len(labeled)} named " \
               f"({len(labeled) / max(len(named), 1):.0%}): {dict(Counter(n['speaker'] for n in labeled).most_common(5))}"
        if args.evaluate:
            truths = truth_by_segment(named, conv, out)
            # people the classifier has no class for (Kissinger, Iacocca, ...) count as "other"
            truths = [t if t is None or t in classes else "other" for t in truths]
            pairs = [(n["speaker"], t) for n, t in zip(named, truths) if t]
            named_pairs = [(p, t) for p, t in pairs if not p.startswith("Unknown")]
            right = sum(p == t for p, t in named_pairs)
            total.update(segments=len(pairs), named=len(named_pairs), right=right)
            line += f" | vs human transcript: {right}/{len(named_pairs)} named correctly"
        print(line)
    if args.evaluate and total["named"]:
        print(f"OVERALL: named {total['named']}/{total['segments']} segments with a known speaker "
              f"({total['named'] / total['segments']:.0%}); {total['right'] / total['named']:.1%} of those correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
