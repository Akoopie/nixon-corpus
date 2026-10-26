"""Standard WER and bootstrap confidence intervals on the validation set (METHODS §7p).

Re-scores every system the paper reports with score.py's alignment (locate + jiwer, Whisper normalizer), but keeps
the outcome of each reference word (right / substituted / deleted / outside the scored stretch) and the insertions
next to it. That gives the published pooled figures exactly and lets them be resampled:

  tape bootstrap    resample whole references (tapes) with replacement, 10,000 times
  block bootstrap   within each tape, resample blocks of 150 consecutive reference words (about a minute of speech;
                    the references carry no timestamps), keeping every tape in; 10,000 times
  paired            the same resamples applied to two systems at once, for the differences the paper states

Pooling follows bench_asr.summarize: words right, WER and name recall are pooled over reference words; Nixon and
short-turn recall are per-reference rates averaged with reference-word weights, as published.

    python src/val_ci.py            -> ~/nixon-data/paper/val_ci.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_asr import audio_key  # noqa: E402
from correct import apply_edits, propagate  # noqa: E402
from paths import DATA_DIR  # noqa: E402
from score import REFS, SHORT_TURN, load_units, locate, norm_words, ref_name_set  # noqa: E402

OUT = DATA_DIR / "paper" / "val_ci.json"
ROOM6 = ["886-008_wspf", "741-002_trial", "779-002_wspf", "488-015_nara", "421-018_wspf", "051-001_wspf"]
ROOM5 = ["886-008_wspf", "488-015_nara", "779-002_wspf", "421-018_wspf", "051-001_wspf"]
PHONE = ["038-034_wspf", "037-205_trial"]
BLOCK = 150
B = 10_000
SEED = 1974

BENCH = {  # bake-off and round-5 outputs (METHODS §3, §7g)
    "stock_parakeet": DATA_DIR / "bench" / "gpu" / "parakeet-v3-win",
    "parakeet_r1": DATA_DIR / "bench" / "gpu" / "parakeet-nixon",
    "parakeet_v5": DATA_DIR / "round5k" / "bench" / "parakeet-r5-lr6e-5",
    "whisperx_batched": DATA_DIR / "bench" / "gpu" / "whisperx-batched",
    "whisper_sequential": DATA_DIR / "bench" / "gpu" / "fw-sequential",
}


def bench_hyp(folder: Path, ref: str) -> list[str] | None:
    key = audio_key(json.loads((REFS / f"{ref}.json").read_text(encoding="utf-8")))
    f = folder / f"{key}.json"
    if not f.exists():
        return None
    return [w for _, t, _ in load_units(f) for w in norm_words(t)]


def words_of(path: Path) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [t for w in d["words"] for t in norm_words(w["w"])]


def edited(root: Path, conv: str, edits_name: str, prop: bool) -> list[str]:
    """Edit list applied to the merge, as compare_argon.py (METHODS §7i, §7l)."""
    d = root / "out" / conv
    merged = json.loads((d / "merged.json").read_text(encoding="utf-8"))
    edits = json.loads((d / edits_name).read_text(encoding="utf-8"))
    for e in edits:
        e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
    starts = [int(a) * 60 + int(b) for a, b in
              re.findall(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
    words, _, _ = apply_edits(merged, edits, starts)
    if prop:
        words, _ = propagate(words, edits)
    return [x["w"] for x in words]


def plain(path: Path) -> list[str]:
    return [x["w"] for x in json.loads(path.read_text(encoding="utf-8"))]


PIPE = {  # five-tape pipeline layers; conv -> hypothesis words
    "pipe_parakeet_v5": lambda c: words_of(DATA_DIR / "valtest_v5" / "out" / c / "parakeet_v5.json"),
    "pipe_whisper": lambda c: words_of(DATA_DIR / "valtest_v5" / "out" / c / "whisper.json"),
    "merge": lambda c: plain(DATA_DIR / "valtest_v5" / "out" / c / "merged.json"),
    "corrected_check": lambda c: edited(DATA_DIR / "argon_test" / "codes", c, "edits_argon.json", False),
    "corrected_check_prop": lambda c: edited(DATA_DIR / "argon_test" / "codes", c, "edits_argon.json", True),
    "corrected_confmarks": lambda c: edited(DATA_DIR / "valtest_v5_unc", c, "edits_unc.json", False),
    "released_merge": lambda c: plain(DATA_DIR / "corpus_v5" / "out" / c / "merged.json"),
    "released_corrected": lambda c: plain(DATA_DIR / "corpus_v5" / "out" / c / "corrected.json"),
}


def per_word(ref: str, hyp: list[str]) -> dict:
    """score.score_one, keeping each reference word's outcome: 0 outside the scored stretch, 1 right, 2 substituted,
    3 deleted; ins[i] = insertions charged next to word i."""
    units = load_units(REFS / f"{ref}.json")
    names = ref_name_set([t for _, t, _ in units])
    out, ins, is_name, is_nixon, is_short = [], [], [], [], []
    for seg in sorted({s for s, _, _ in units}):
        full, owner = [], []
        for s, t, spk in units:
            if s == seg:
                w = norm_words(t)
                full += w
                owner += [(spk, len(w) <= SHORT_TURN)] * len(w)
        if not full:
            continue
        ra, rb, a, b = locate(full, hyp)
        o, n_ins = [0] * len(full), [0] * len(full)
        if rb > ra:
            al = jiwer.process_words(" ".join(full[ra:rb]), " ".join(hyp[a:b]) or "<empty>")
            for ch in al.alignments[0]:
                code = {"equal": 1, "substitute": 2, "delete": 3}.get(ch.type)
                if code:
                    for i in range(ch.ref_start_idx, ch.ref_end_idx):
                        o[ra + i] = code
                else:  # insertion: charge it to the reference word before it (or the first one)
                    n_ins[ra + max(min(ch.ref_start_idx, rb - ra) - 1, 0)] += ch.hyp_end_idx - ch.hyp_start_idx
        out += o
        ins += n_ins
        is_name += [w in names for w in full]
        is_nixon += [spk == "Nixon" for spk, _ in owner]
        is_short += [sh for _, sh in owner]
    return {"o": np.array(out, np.int8), "ins": np.array(ins, np.int32), "name": np.array(is_name),
            "nixon": np.array(is_nixon), "short": np.array(is_short)}


def counts(pw: dict) -> np.ndarray:
    """Per-block counts: [N, hits, S, D, I, names, name_hits, nixon, nixon_hits, short, short_hits]."""
    o, n = pw["o"], len(pw["o"])
    rows = []
    for s in range(0, n, BLOCK):
        sl = slice(s, s + BLOCK)
        ob, sc = o[sl], o[sl] > 0
        hit = ob == 1
        rows.append([sc.sum(), hit.sum(), (ob == 2).sum(), (ob == 3).sum(), pw["ins"][sl].sum(),
                     (pw["name"][sl] & sc).sum(), (pw["name"][sl] & hit).sum(),
                     (pw["nixon"][sl] & sc).sum(), (pw["nixon"][sl] & hit).sum(),
                     (pw["short"][sl] & sc).sum(), (pw["short"][sl] & hit).sum()])
    return np.array(rows, np.float64)


def stats(per_ref: list[np.ndarray]) -> np.ndarray:
    """per_ref: one summed count vector per reference -> [recall, wer, sub, del, ins, names, nixon, short]."""
    t = np.sum(per_ref, axis=0)
    N = max(t[0], 1)
    w = lambda hits, tot: sum(r[0] * (r[hits] / r[tot] if r[tot] else 0) for r in per_ref) / N
    return np.array([t[1] / N, (t[2] + t[3] + t[4]) / N, t[2] / N, t[3] / N, t[4] / N,
                     t[6] / max(t[5], 1), w(8, 7), w(10, 9)])


KEYS = ["word_recall", "wer", "sub", "del", "ins", "name_recall", "nixon_recall", "short_recall"]


def ci(samples: np.ndarray) -> list:
    return [[round(float(x), 4) for x in np.percentile(samples[:, k], [2.5, 97.5])] for k in range(samples.shape[1])]


def bootstrap(blocks: dict[str, dict[str, np.ndarray]], refs: list[str], systems: list[str], rng) -> dict:
    """blocks[system][ref] = block counts. Returns point estimates and both CIs per system."""
    res = {}
    tape_idx = rng.integers(0, len(refs), size=(B, len(refs)))
    blk_idx = {r: rng.integers(0, len(blocks[systems[0]][r]), size=(B, len(blocks[systems[0]][r]))) for r in refs}
    draws = {}
    for s in systems:
        sums = {r: blocks[s][r].sum(axis=0) for r in refs}
        point = stats([sums[r] for r in refs])
        tape = np.array([stats([sums[refs[j]] for j in row]) for row in tape_idx])
        blk = np.array([stats([blocks[s][r][blk_idx[r][i]].sum(axis=0) for r in refs]) for i in range(B)])
        draws[s] = (point, tape, blk)
        res[s] = {"point": dict(zip(KEYS, map(lambda x: round(float(x), 4), point))),
                  "ci_tape": dict(zip(KEYS, ci(tape))), "ci_block": dict(zip(KEYS, ci(blk))),
                  "N": int(sum(sums[r][0] for r in refs)), "names": int(sum(sums[r][5] for r in refs)),
                  "per_ref": {r: {"N": int(sums[r][0]), "word_recall": round(float(sums[r][1] / max(sums[r][0], 1)), 4),
                                  "wer": round(float((sums[r][2] + sums[r][3] + sums[r][4]) / max(sums[r][0], 1)), 4)}
                              for r in refs}}
    return res, draws


def diff(draws: dict, a: str, b: str) -> dict:
    pa, ta, ba = draws[a]
    pb, tb, bb = draws[b]
    return {"point": dict(zip(KEYS, map(lambda x: round(float(x), 4), pa - pb))),
            "ci_tape": dict(zip(KEYS, ci(ta - tb))), "ci_block": dict(zip(KEYS, ci(ba - bb)))}


def main() -> int:
    rng = np.random.default_rng(SEED)
    result = {"settings": {"block_words": BLOCK, "resamples": B, "seed": SEED,
                           "pooling": "word recall, WER, S/D/I and names pooled over reference words; Nixon and short-turn "
                                      "recall = per-reference rates weighted by reference words (bench_asr.summarize)"}}
    # (i) six room references and the phone check: bench systems
    for set_name, refs in (("room6", ROOM6), ("room5_bench", ROOM5), ("phone", PHONE)):
        blocks = {}
        for s, folder in BENCH.items():
            blocks[s] = {}
            for r in refs:
                hyp = bench_hyp(folder, r)
                if hyp is None:
                    raise SystemExit(f"missing {s} {r}")
                blocks[s][r] = counts(per_word(r, hyp))
        res, draws = bootstrap(blocks, refs, list(BENCH), rng)
        result[set_name] = {"refs": refs, "systems": res,
                            "diffs": {"v5_minus_stock": diff(draws, "parakeet_v5", "stock_parakeet"),
                                      "v5_minus_whisper_sequential": diff(draws, "parakeet_v5", "whisper_sequential"),
                                      "v5_minus_whisperx_batched": diff(draws, "parakeet_v5", "whisperx_batched")}}
        print(set_name, {s: (r["point"]["word_recall"], r["point"]["wer"]) for s, r in res.items()}, flush=True)
    # (ii) five-tape pipeline layers
    blocks = {s: {} for s in PIPE}
    for s, fn in PIPE.items():
        for r in ROOM5:
            blocks[s][r] = counts(per_word(r, fn(r.split("_")[0])))
    res, draws = bootstrap(blocks, ROOM5, list(PIPE), rng)
    result["room5_pipeline"] = {"refs": ROOM5, "systems": res, "diffs": {
        "corrected_minus_merge": diff(draws, "corrected_check", "merge"),
        "confmarks_minus_merge": diff(draws, "corrected_confmarks", "merge"),
        "released_corrected_minus_released_merge": diff(draws, "released_corrected", "released_merge"),
        "merge_minus_parakeet_v5": diff(draws, "merge", "pipe_parakeet_v5")}}
    print("room5_pipeline", {s: (r["point"]["word_recall"], r["point"]["wer"], r["point"]["name_recall"])
                             for s, r in res.items()}, flush=True)
    # phone check for the pipeline layers (038-034 only: the corpus has no copy of the 037-205 exhibit)
    ph = {s: {"038-034_wspf": counts(per_word("038-034_wspf", fn("038-034")))}
          for s, fn in (("released_merge", PIPE["released_merge"]), ("released_corrected", PIPE["released_corrected"]),
                        ("released_parakeet_v5", lambda c: words_of(DATA_DIR / "corpus_v5" / "out" / c / "parakeet_v5.json")),
                        ("released_whisper", lambda c: words_of(DATA_DIR / "corpus_v5" / "out" / c / "whisper.json")))}
    res, _ = bootstrap(ph, ["038-034_wspf"], list(ph), rng)
    result["phone_pipeline_038-034"] = {"systems": {s: {k: v for k, v in r.items() if k != "ci_tape"} for s, r in res.items()}}
    print("phone 038-034", {s: (r["point"]["word_recall"], r["point"]["wer"]) for s, r in res.items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
