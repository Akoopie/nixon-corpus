"""Score the era-LM sweep (lm_sweep.sh): every setting on the dev conversations, then validation.

For each setting (out/<conv>/parakeet_a<weight>_<b|nb>.json) and tape set, word-weighted over tapes:
  right     share of reference words transcribed exactly (score.score_one "complete")
  blank     share of reference words with nothing transcribed (deletions)
  small     share of reference words replaced by a small word (the / a, he / they)
  content   share of reference words replaced by a content word (names, numbers, verbs: can mislead)
  shown     share of the transcript's own words that are wrong content words (substituted or inserted)
  names     name recall
Fillers are dropped on both sides by the normaliser, as everywhere else.

The setting is chosen on the DEV tapes only: the most words right among the settings whose wrong-content
share of shown words is no higher than weight 0's (v5 as in the corpus). Validation is reported for that
setting next to weight 0. The full validation table follows, for learning only; it was not used to choose.

    python src/score_lmsweep.py <sweep root> [--corpus-v5 <dir with out/<conv>/parakeet_v5.json>] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer

from score import REFS, load_units, locate, norm_words, score_one
from paths import DATA_DIR

DEV = ["429-018", "538-015", "856-004", "862-006", "903-006"]  # fine-tuning dev conversations (METHODS 4)
DEV_REFS = DATA_DIR / "training" / "transcripts"               # their WSPF transcripts (turns)
VAL = {"886-008": "886-008_wspf", "488-015": "488-015_nara", "779-002": "779-002_wspf",
       "421-018": "421-018_wspf", "051-001": "051-001_wspf"}  # room validation tapes
PHONE = {"038-034": "038-034_wspf"}
TAGS = [f"a{a}_{b}" for b in ("nb", "b") for a in ("0.0", "0.1", "0.2", "0.3", "0.5")]
SMALL = set("""a an the and or but if so as than then that this these those there here what which who whom whose
when where why how i me my mine we us our you your he him his she her it its they them their is am are was were be
been being do does did have has had having will would shall should can could may might must to of in on at by for
with from about into over up down out off again just very too also all any some each both more most other such only
own same well oh yeah yes yep ok okay right now let lets gonna got get go going say said says know think mean like
one ones thing things way kind sort really there's it's that's he's i'm you're we're they're don't didn't doesn't
isn't wasn't aren't won't can't couldn't wouldn't shouldn't hm huh mr""".split())


def kinds(ref_path: Path, hyp: list[str]) -> dict:
    """Reference words right / blank / replaced by a small or content word, and wrong content words shown."""
    out = dict(N=0, right=0, blank=0, small=0, content=0, shown=0, shown_bad=0)
    units = load_units(ref_path)
    for seg in sorted({s for s, _, _ in units}):
        full = [w for s, t, _ in units if s == seg for w in norm_words(t)]
        if not full:
            continue
        ra, rb, a, b = locate(full, hyp)
        rw, hw = full[ra:rb], hyp[a:b]
        if not rw:
            continue
        al = jiwer.process_words(" ".join(rw), " ".join(hw) or "<empty>")
        out["N"] += len(rw)
        out["shown"] += len(hw)
        for ch in al.alignments[0]:
            n = ch.ref_end_idx - ch.ref_start_idx
            hs = hw[ch.hyp_start_idx:ch.hyp_end_idx]
            if ch.type == "equal":
                out["right"] += n
            elif ch.type == "delete":
                out["blank"] += n
            else:
                bad = [w for w in hs if w not in SMALL]
                out["shown_bad"] += len(bad)
                if ch.type == "substitute":
                    out["content"] += len(bad)
                    out["small"] += len(hs) - len(bad)
    return out


def hyp_words(path: Path) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [t for w in d["words"] for t in norm_words(w["w"])]


def score_set(root: Path, tag: str, refs: dict[str, Path]) -> dict | None:
    tot = dict(N=0, right=0, blank=0, small=0, content=0, shown=0, shown_bad=0, names=0, names_hit=0)
    for conv, ref in refs.items():
        f = root / "out" / conv / f"parakeet_{tag}.json"
        if not f.exists():
            return None
        hyp = hyp_words(f)
        for k, v in kinds(ref, hyp).items():
            tot[k] += v
        s = score_one(ref, hyp)
        tot["names"] += s["names"]
        tot["names_hit"] += round(s["name_recall"] * s["names"])
    n = max(tot["N"], 1)
    return {"right": tot["right"] / n, "blank": tot["blank"] / n, "small": tot["small"] / n,
            "content": tot["content"] / n, "shown": tot["shown_bad"] / max(tot["shown"], 1),
            "names": tot["names_hit"] / max(tot["names"], 1), "N": tot["N"]}


def row(tag: str, r: dict | None) -> str:
    if r is None:
        return f"{tag:9s}  (not finished)"
    return (f"{tag:9s} right {r['right']:6.1%}  blank {r['blank']:5.1%}  small {r['small']:5.1%}  content {r['content']:5.1%}"
            f"  | wrong content shown {r['shown']:5.1%}  | names {r['names']:5.1%}  ({r['N']} ref words)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--corpus-v5", type=Path, help="check a0.0_nb against the corpus's parakeet_v5.json")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    dev_refs = {c: DEV_REFS / f"{c}.json" for c in DEV}
    val_refs = {c: REFS / f"{r}.json" for c, r in VAL.items()}
    phone_refs = {c: REFS / f"{r}.json" for c, r in PHONE.items()}

    if args.corpus_v5:
        for conv in list(VAL) + list(PHONE):
            a, b = args.root / "out" / conv / "parakeet_a0.0_nb.json", args.corpus_v5 / "out" / conv / "parakeet_v5.json"
            if a.exists() and b.exists():
                wa, wb = [w["w"] for w in json.loads(a.read_text(encoding="utf-8"))["words"]], \
                         [w["w"] for w in json.loads(b.read_text(encoding="utf-8"))["words"]]
                print(f"sanity {conv}: a0.0_nb {len(wa)} words, corpus v5 {len(wb)} words, identical: {wa == wb}")

    print("\nDEV (5 conversations, WSPF transcripts; used to choose)")
    dev = {t: score_set(args.root, t, dev_refs) for t in TAGS}
    for t in TAGS:
        print(row(t, dev[t]))
    base = dev.get("a0.0_nb")
    ok = [t for t in TAGS if dev[t] and base and dev[t]["shown"] <= base["shown"] + 1e-9]
    pick = max(ok, key=lambda t: dev[t]["right"]) if ok else None
    print(f"\nchosen on dev: {pick} (most words right with wrong content shown <= weight 0's {base['shown']:.1%})" if base else "")

    val = {t: score_set(args.root, t, val_refs) for t in TAGS}
    phone = {t: score_set(args.root, t, phone_refs) for t in TAGS}
    if pick:
        print("\nVALIDATION, chosen setting vs weight 0 (5 room tapes)")
        for t in ("a0.0_nb", pick):
            print(row(t, val[t]))
        print("phone (038-034)")
        for t in ("a0.0_nb", pick):
            print(row(t, phone[t]))
    print("\nVALIDATION, every setting (for learning; not used to choose)")
    for t in TAGS:
        print(row(t, val[t]))
    if args.json:
        args.json.write_text(json.dumps({"dev": dev, "val": val, "phone": phone, "pick": pick}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
