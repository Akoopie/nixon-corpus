"""Score a transcript against the human reference transcripts (validation set).

For each reference this reports:
  WER          word error rate on the stretch of the transcript the reference covers
  sub/del/ins  its parts; deletions = speech the model missed, insertions = invented words
  complete     share of reference words the transcript got exactly right (1 - (S+D)/N)
  names        recall of proper names from the reference (capitalised mid-sentence words)

References usually cover only part of a recording (and the transcript may come
from a different transfer of the tape), so the matching stretch is found by
aligning words, not by timestamps. Text on both sides goes through Whisper's
English normaliser (the Open ASR Leaderboard standard): case, punctuation,
numbers and spellings are unified and fillers (uh, um) dropped.

Usage:
    python src/score.py <transcript.json> 886-008_wspf 886-008_hjc
    python src/score.py <transcript.json> --conv 886-008      # every reference for that conversation
    python src/score.py validation/refs/886-008_hjc.json 886-008_wspf   # human vs human
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import jiwer

from paths import VALIDATION_DIR

REFS = VALIDATION_DIR / "refs"
# Capitalised words that are not names (sentence starts are excluded separately).
NOT_NAMES = {"i", "i'm", "i'll", "i've", "i'd", "mr", "mrs", "ms", "president", "god", "oh", "ok", "okay",
             "senator", "judge", "general", "secretary", "attorney", "house", "white", "committee",
             "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
             "january", "february", "march", "april", "may", "june", "july", "august", "september",
             "october", "november", "december", "american", "americans", "christ", "jesus"}


@lru_cache(maxsize=1)
def normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    try:
        from huggingface_hub import hf_hub_download
        mapping = json.loads(Path(hf_hub_download("openai/whisper-large-v3", "normalizer.json"))
                             .read_text(encoding="utf-8"))
    except Exception:  # offline: spelling unification is a minor refinement
        mapping = {}
    return EnglishTextNormalizer(mapping)


def norm_words(text: str) -> list[str]:
    """Normalised words of ONE turn/segment. Short annotations ([unintelligible],
    (inaudible)) are removed first and stray brackets blanked: the normaliser
    deletes everything between brackets, so one unmatched '(' in OCR text would
    otherwise swallow pages."""
    text = re.sub(r"\[[^\]]{0,80}\]|\([^)]{0,80}\)", " ", text)
    text = re.sub(r"[\[\]()<>]", " ", text)
    return normalizer()(text).split()


def load_units(path: Path) -> list[tuple[int, str, str]]:
    """(segment, text, speaker) units from either a transcript JSON or a reference JSON."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if "turns" in d:
        return [(t.get("segment", 0), t["text"], t.get("speaker", "")) for t in d["turns"]]
    return [(0, s["text"], s.get("speaker", "")) for s in d["segments"] if "repeat" not in (s.get("qc") or "")]


SHORT_TURN = 4  # words; short replies ("Right.", "That's fine.") are what models drop first


def ref_name_set(texts: list[str]) -> set[str]:
    """Proper names in a reference: words capitalised mid-sentence at least twice
    and in >= 80% of their mid-sentence uses (so 'Dean' counts, 'That' doesn't)."""
    cap, low = {}, {}
    for text in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\s+--\s*|\s+—\s*", text):
            for w in re.findall(r"[A-Za-z][A-Za-z']*", sent)[1:]:  # skip sentence-initial word
                k = w.lower().removesuffix("'s")
                (cap if w[0].isupper() else low)[k] = (cap if w[0].isupper() else low).get(k, 0) + 1
    return {k for k, n in cap.items()
            if n >= 2 and n / (n + low.get(k, 0)) >= 0.8 and k not in NOT_NAMES and len(k) > 2}


EDGE_SLACK = 20  # words of unmatched text tolerated past the outermost matches


def locate(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """The stretch both texts cover, as (ref_start, ref_end, hyp_start, hyp_end).

    Either side may extend beyond the other: a reference often covers only part
    of a recording, and an exhibit clip may cover only part of a reference. The
    shared stretch runs between the first and last solid word matches (ignoring
    stray matches far outside the main run). Each side may extend past them by
    the other side's leftover length plus EDGE_SLACK, so a model that drops the
    first sentence still gets charged for it, but reference text with no audio
    behind it is not counted as missed."""
    sm = SequenceMatcher(None, ref, hyp, autojunk=True)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
    if not blocks:
        return 0, len(ref), 0, 0
    # Drop leading/trailing blocks whose hyp gap is wildly out of proportion to the ref gap.
    def plausible(b1, b2):
        return (b2.b - (b1.b + b1.size)) <= 3 * max(b2.a - (b1.a + b1.size), 0) + 50
    while len(blocks) > 1 and not plausible(blocks[0], blocks[1]):
        blocks.pop(0)
    while len(blocks) > 1 and not plausible(blocks[-2], blocks[-1]):
        blocks.pop()
    first, last = blocks[0], blocks[-1]
    ref_tail, hyp_tail = len(ref) - (last.a + last.size), len(hyp) - (last.b + last.size)
    ra = first.a - min(first.a, first.b + EDGE_SLACK)
    ha = first.b - min(first.b, first.a + EDGE_SLACK)
    rb = last.a + last.size + min(ref_tail, hyp_tail + EDGE_SLACK)
    hb = last.b + last.size + min(hyp_tail, ref_tail + EDGE_SLACK)
    return ra, rb, ha, hb


def score_one(ref_path: Path, hyp_words: list[str]) -> dict:
    units = load_units(ref_path)
    totals = dict(N=0, S=0, D=0, I=0, names=0, names_hit=0, ref_total=0)
    missed, by_speaker, short = {}, {}, [0, 0]
    names = ref_name_set([t for _, t, _ in units])
    for seg in sorted({s for s, _, _ in units}):
        full_ref, owner = [], []  # owner[i] = (speaker, is_short_turn) of reference word i
        for s, t, spk in units:
            if s == seg:
                w = norm_words(t)
                full_ref += w
                owner += [(spk, len(w) <= SHORT_TURN)] * len(w)
        if not full_ref:
            continue
        ra, rb, a, b = locate(full_ref, hyp_words)
        ref = full_ref[ra:rb]
        totals["ref_total"] += len(full_ref)
        if not ref:
            continue
        out = jiwer.process_words(" ".join(ref), " ".join(hyp_words[a:b]) or "<empty>")
        totals["N"] += len(ref)
        totals["S"] += out.substitutions
        totals["D"] += out.deletions
        totals["I"] += out.insertions
        # Name recall: a reference name counts as recovered if it was aligned as 'equal'.
        hit_idx = {i for chunk in out.alignments[0] if chunk.type == "equal"
                   for i in range(chunk.ref_start_idx, chunk.ref_end_idx)}
        for i, w in enumerate(ref):
            spk, is_short = owner[ra + i]
            hit = i in hit_idx
            stat = by_speaker.setdefault(spk, [0, 0])
            stat[0] += hit
            stat[1] += 1
            if is_short:
                short[0] += hit
                short[1] += 1
            if w in names:
                totals["names"] += 1
                if hit:
                    totals["names_hit"] += 1
                else:
                    missed[w] = missed.get(w, 0) + 1
    n = max(totals["N"], 1)
    return {"N": totals["N"], "wer": (totals["S"] + totals["D"] + totals["I"]) / n,
            "sub": totals["S"] / n, "del": totals["D"] / n, "ins": totals["I"] / n,
            "complete": 1 - (totals["S"] + totals["D"]) / n,
            "name_recall": totals["names_hit"] / max(totals["names"], 1), "names": totals["names"],
            "ref_used": totals["N"] / max(totals["ref_total"], 1),
            "nixon_recall": by_speaker.get("Nixon", [0, 0])[0] / max(by_speaker.get("Nixon", [0, 0])[1], 1),
            "short_recall": short[0] / max(short[1], 1),
            "speaker_recall": {k: round(h / n, 3) for k, (h, n) in by_speaker.items() if n >= 50},
            "missed_names": sorted(missed.items(), key=lambda x: -x[1])[:10]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hypothesis", type=Path, help="transcript JSON (or a reference JSON, for human-vs-human)")
    ap.add_argument("refs", nargs="*", help="reference names, e.g. 886-008_wspf")
    ap.add_argument("--conv", help="score against every reference for this conversation id")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    args = ap.parse_args()

    names = list(args.refs)
    if args.conv:
        names += sorted(p.stem for p in REFS.glob(f"{args.conv}_*.json"))
    if not names:
        ap.error("give reference names or --conv")
    hyp_words = [w for _, t, _ in load_units(args.hypothesis) for w in norm_words(t)]

    results = {}
    for name in names:
        ref_path = REFS / f"{name}.json"
        if ref_path.resolve() == args.hypothesis.resolve():
            continue
        results[name] = score_one(ref_path, hyp_words)
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    print(f"hypothesis: {args.hypothesis.name} ({len(hyp_words)} words)")
    print(f"{'reference':16} {'words':>6} {'used':>5} {'WER':>6} {'sub':>6} {'del':>6} {'ins':>6} "
          f"{'complete':>9} {'names':>11} {'nixon':>6} {'short':>6}")
    for name, r in results.items():
        print(f"{name:16} {r['N']:6} {r['ref_used']:5.0%} {r['wer']:6.1%} {r['sub']:6.1%} {r['del']:6.1%} "
              f"{r['ins']:6.1%} {r['complete']:9.1%} {r['name_recall']:6.0%} of {r['names']} "
              f"{r['nixon_recall']:6.0%} {r['short_recall']:6.0%}")
        if r["missed_names"]:
            print(f"{'':16} missed: " + ", ".join(f"{w}x{c}" for w, c in r["missed_names"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
