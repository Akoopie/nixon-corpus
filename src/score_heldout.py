"""Score held-out transcripts (transcribe_tests.py output) against the listener's text.

    python src/score_heldout.py --corrections test.jsonl --listener listen_test.jsonl r4_tests.json r5_tests.json ...

Corrections (the listener's corrected chunks in held-out windows): WER with substitutions / deletions / insertions,
per difficulty band and overall. Listener clips: words of the listener's answer matched in order (answers can be
partial, so WER is not meaningful there). Text is normalized as in score.py. A model name seen in several files
is reported once per file, labelled file:model.
"""
from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
from score import norm_words


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", type=Path, nargs="+")
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--listener", type=Path, required=True)
    a = ap.parse_args()
    corr = {Path(r["audio_filepath"]).stem: r for r in rows(a.corrections)}
    lis = {Path(r["audio_filepath"]).stem: r["text"] for r in rows(a.listener)}
    bands = sorted({r["band"] for r in corr.values()}, key=["light", "medium", "heavy"].index)
    print(f"{'model':22}" + "".join(f"{b + ' (' + str(sum(len(norm_words(r['text'])) for r in corr.values() if r['band'] == b)) + ')':>22}" for b in bands)
          + f"{'all (' + str(sum(len(norm_words(r['text'])) for r in corr.values())) + ')':>12}{'listener':>16}")
    for f in a.tests:
        res = json.loads(f.read_text(encoding="utf-8"))
        for model in res.get("corrections", {}):
            hyp = res["corrections"][model]
            cells = []
            for group in bands + ["all"]:
                keys = [k for k, r in corr.items() if group in ("all", r["band"])]
                ref = [" ".join(norm_words(corr[k]["text"])) for k in keys]
                h = [" ".join(norm_words(hyp.get(k, ""))) for k in keys]
                o = jiwer.process_words(ref, h)
                n = o.hits + o.substitutions + o.deletions
                wer = (o.substitutions + o.deletions + o.insertions) / n
                cells.append(f"{wer:6.1%} ({o.substitutions}/{o.deletions}/{o.insertions})" if group != "all" else f"{wer:6.1%}")
            lh = res.get("listener", {}).get(model, {})
            hit = n_l = 0
            for k, t in lis.items():
                r, h = norm_words(t), norm_words(lh.get(k, ""))
                hit += sum(b.size for b in SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks())
                n_l += len(r)
            print(f"{f.stem + ':' + model:22}" + "".join(f"{c:>22}" for c in cells[:-1]) + f"{cells[-1]:>12}"
                  + f"{f'{hit}/{n_l}':>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
