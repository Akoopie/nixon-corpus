"""Merge the two transcripts of a tape (fine-tuned Parakeet + Whisper) into one word sequence.

Parakeet is the more accurate model, so its words are the backbone. Whisper fills the stretches
where Parakeet went silent: an inserted run of Whisper words is kept only if no Parakeet word lies
within GAP_PAD seconds of it and Whisper was reasonably sure (mean word probability >= MIN_P).
Where both models heard something different, Parakeet's words stand and Whisper's are kept as an
alternative for the correction pass.

    python src/merge.py <root> --jobs jobs.jsonl                 # writes out/<conv>/merged.json
    python src/merge.py <root> --jobs jobs.jsonl --evaluate      # room score vs the human transcripts
    python src/merge.py <root> --jobs jobs.jsonl --parakeet parakeet_v5.json   # the v5 re-run's words
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

from score import REFS, norm_words, score_one

GAP_PAD = 0.5   # s: a Whisper run this close to a Parakeet word is a disagreement, not a gap
MIN_P = 0.5     # mean Whisper word probability needed to fill a gap

# validation references per tape (the room / phone sets of bench_asr.py that have worker output)
REF_OF = {"886-008": "886-008_wspf", "779-002": "779-002_wspf", "488-015": "488-015_nara",
          "421-018": "421-018_wspf", "051-001": "051-001_wspf", "038-034": "038-034_wspf"}
ROOMS = {"886-008", "779-002", "488-015", "421-018", "051-001"}


def tokens(words: list[dict]) -> list[tuple[str, dict]]:
    """(normalised token, source word) pairs; one ASR word can normalise to 0..n tokens."""
    return [(t, w) for w in words for t in norm_words(w["w"])]


def merge(pk: dict, wh: dict, min_p: float = MIN_P, gap_pad: float = GAP_PAD,
          prefer_p: float | None = None) -> list[dict]:
    p, w = tokens(pk["words"]), tokens(wh["words"])
    p_times = sorted(x["s"] for _, x in p)
    sm = SequenceMatcher(None, [t for t, _ in p], [t for t, _ in w], autojunk=False)
    out = []

    def near_parakeet(a: float, b: float) -> bool:
        import bisect
        i = bisect.bisect_left(p_times, a - gap_pad)
        return i < len(p_times) and p_times[i] <= b + gap_pad

    def c(x: dict) -> dict:  # Parakeet's per-word confidence travels with its tokens (v5 re-run onward)
        return {"c": x["c"]} if "c" in x else {}

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += [{"w": t, "s": x["s"], "e": x["e"], "src": "both", **c(x)} for t, x in p[i1:i2]]
        elif op == "delete":
            out += [{"w": t, "s": x["s"], "e": x["e"], "src": "parakeet", **c(x)} for t, x in p[i1:i2]]
        elif op == "replace":
            probs = [x.get("p") for _, x in w[j1:j2] if x.get("p") is not None]
            if prefer_p is not None and probs and min(probs) >= prefer_p:  # Whisper is sure of every word
                alt = " ".join(t for t, _ in p[i1:i2])
                out += [{"w": t, "s": x["s"], "e": x["e"], "src": "whisper", "alt": alt} for t, x in w[j1:j2]]
            else:
                alt = " ".join(t for t, _ in w[j1:j2])
                out += [{"w": t, "s": x["s"], "e": x["e"], "src": "parakeet", "alt": alt, **c(x)} for t, x in p[i1:i2]]
        else:  # insert: Whisper heard words Parakeet did not
            run = w[j1:j2]
            a, b = run[0][1]["s"], run[-1][1]["e"]
            probs = [x.get("p") for _, x in run if x.get("p") is not None]
            sure = not probs or sum(probs) / len(probs) >= min_p
            if not near_parakeet(a, b) and sure:
                out += [{"w": t, "s": x["s"], "e": x["e"], "src": "whisper"} for t, x in run]
    out.sort(key=lambda x: x["s"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--min-p", type=float, default=MIN_P)
    ap.add_argument("--gap-pad", type=float, default=GAP_PAD)
    ap.add_argument("--parakeet", default="parakeet.json", help="Parakeet file name in out/<conv>/ (v5 re-run: parakeet_v5.json)")
    args = ap.parse_args()
    convs = [json.loads(l)["conv"] for l in args.jobs.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, skipped = [], []
    for conv in convs:
        d = args.root / "out" / conv
        if not ((d / args.parakeet).exists() and (d / "whisper.json").exists()):
            skipped.append(conv)
            continue
        pk = json.loads((d / args.parakeet).read_text(encoding="utf-8"))
        wh = json.loads((d / "whisper.json").read_text(encoding="utf-8"))
        merged = merge(pk, wh, args.min_p, args.gap_pad)
        if not args.evaluate:
            (d / "merged.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        src = {k: sum(x["src"] == k for x in merged) for k in ("both", "parakeet", "whisper")}
        line = f"{conv}: {len(merged)} words ({src['both']} both, {src['parakeet']} Parakeet only, {src['whisper']} filled from Whisper)"
        if args.evaluate and conv in REF_OF:
            ref = REFS / f"{REF_OF[conv]}.json"
            scores = {name: score_one(ref, [t for x in hyp for t in norm_words(x["w"])])
                      for name, hyp in (("parakeet", pk["words"]), ("whisper", wh["words"]), ("merged", merged))}
            rows.append((conv, scores))
            line += " | words right: " + ", ".join(f"{k} {v['complete']:.1%}" for k, v in scores.items())
        print(line)
    if args.evaluate and rows:
        for label, sel in (("ROOM", [r for r in rows if r[0] in ROOMS]), ("PHONE", [r for r in rows if r[0] not in ROOMS])):
            if not sel:
                continue
            tot = {k: sum(s[k]["complete"] * s[k]["N"] for _, s in sel) / sum(s[k]["N"] for _, s in sel)
                   for k in ("parakeet", "whisper", "merged")}
            print(f"{label} ({len(sel)} tapes, word-weighted): " + ", ".join(f"{k} {v:.1%}" for k, v in tot.items()))
    print(f"{len(convs) - len(skipped)} merged, {len(skipped)} skipped (no {args.parakeet} or whisper.json)"
          + (f": {' '.join(skipped[:20])}{' ...' if len(skipped) > 20 else ''}" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
