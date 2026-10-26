"""Two measures added at review (METHODS 7r): name precision of each layer, and the two-recognizer ceiling with v5.

    python src/paper_extra.py            -> ~/nixon-data/paper/extra_measures.json

Uses the alignment of src/frus_test.py (score.py's normaliser and locate(), jiwer word alignment) on the 165 FRUS test
conversations and the five development room recordings.
- Name precision: among hypothesis words, inside the scored stretches, that belong to the reference's name set
  (score.ref_name_set), the share aligned to the identical reference word. It counts only names that occur in the
  reference, so a wrong name the reference never mentions is not seen.
- Ceiling: the share of reference words that Parakeet v5 or the corpus's Whisper output gets right, over the reference
  positions inside either layer's scored stretch, with each layer alone on the same denominator.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jiwer

from frus_test import CORPUS, LAYERS, OUT, VAL_ROOM, layer_words, locate, norm_words, ref_name_set
from paths import DATA_DIR, VALIDATION_DIR


def measure(args):
    conv, ref_path = args
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    units = [(t["segment"], t["text"]) for t in ref["turns"]]
    names = ref_name_set([t for _, t in units])
    res = {"conv": conv, "names": {}, "ceiling": {}}
    hits, ranges = {}, {}
    for layer, fname in LAYERS.items():
        hyp, _, _ = layer_words(CORPUS / conv / fname)
        shown = right = 0
        hit_pos, in_range = set(), set()
        offset = 0
        for seg in sorted({s for s, _ in units}):
            full = [w for s, t in units if s == seg for w in norm_words(t)]
            if not full:
                continue
            ra, rb, a, b = locate(full, hyp)
            rw, hw = full[ra:rb], hyp[a:b]
            if rw:
                al = jiwer.process_words(" ".join(rw), " ".join(hw) or "<empty>")
                for ch in al.alignments[0]:
                    hs = hw[ch.hyp_start_idx:ch.hyp_end_idx]
                    named = sum(1 for w in hs if w in names)
                    shown += named
                    if ch.type == "equal":
                        right += named
                        hit_pos.update(offset + ra + ch.ref_start_idx + k for k in range(ch.ref_end_idx - ch.ref_start_idx))
                in_range.update(range(offset + ra, offset + rb))
            offset += len(full)
        res["names"][layer] = [shown, right]
        hits[layer], ranges[layer] = hit_pos, in_range
    den = ranges["parakeet_v5"] | ranges["whisper"]
    res["ceiling"] = {"den": len(den), "parakeet_v5": len(hits["parakeet_v5"] & den), "whisper": len(hits["whisper"] & den),
                      "either": len((hits["parakeet_v5"] | hits["whisper"]) & den)}
    return res


def summarize(rows):
    out = {"name_precision": {}, "ceiling": {}}
    for layer in LAYERS:
        s = sum(r["names"][layer][0] for r in rows)
        k = sum(r["names"][layer][1] for r in rows)
        out["name_precision"][layer] = {"shown": s, "right": k, "precision": k / max(s, 1)}
    den = sum(r["ceiling"]["den"] for r in rows)
    out["ceiling"] = {"reference_words": den, **{k: sum(r["ceiling"][k] for r in rows) / max(den, 1)
                                                 for k in ("parakeet_v5", "whisper", "either")}}
    return out


def main():
    test = [l.split("\t")[0] for l in (OUT / "per_conv.tsv").read_text(encoding="utf-8").split("\n")[1:] if l.strip()]
    jobs = [(c, str(OUT / "refs" / f"{c}_frus.json")) for c in test]
    with ProcessPoolExecutor(max_workers=12) as ex:
        test_rows = list(ex.map(measure, jobs, chunksize=4))
    dev_rows = [measure((c, str(VALIDATION_DIR / "refs" / f"{r}.json"))) for c, r in VAL_ROOM.items()]
    res = {"test": summarize(test_rows), "development": summarize(dev_rows), "n_test": len(test_rows), "n_dev": len(dev_rows)}
    dst = DATA_DIR / "paper" / "extra_measures.json"
    dst.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
