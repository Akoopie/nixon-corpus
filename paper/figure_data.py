"""Numbers for the paper's figures -> paper/data/figure_data.json. Every value comes from a measured output
(file named beside it) or is copied from docs/METHODS.md with its section, so every figure traces to a source.

    python paper/figure_data.py

Inputs: ~/nixon-data/frus_test/results.json (METHODS 7o: test set, and the development room tapes scored the same
way), ~/nixon-data/paper/val_ci.json and whisper_artifacts.json (METHODS 7p), the explorer database and the corpus
manifest (Fig 7), voiceprints_v4 (Fig 6).
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import DATA_DIR  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
out = {}

# Fig 2: layers of the released corpus on the development room tapes and the FRUS test set (METHODS 7o)
frus = json.loads((DATA_DIR / "frus_test/results.json").read_text(encoding="utf-8"))
LAYERS = ("whisper", "parakeet_v5", "merged", "corrected")
KEEP = ("N", "recall", "wer", "sub", "del", "ins", "wrong_content_shown", "names", "names_n", "nixon", "short")
out["accuracy"] = {
    "dev": {L: {k: frus["validation_room"][L][k] for k in KEEP} for L in LAYERS},
    "test": {L: {k: frus["pooled"][L][k] for k in KEEP} for L in LAYERS},
    "test_ci": {L: frus["bootstrap"]["layers"][L] for L in LAYERS},
    "test_n_convs": frus["n_scored"],
    "test_ref_words": frus["ref_words_scored"],
}
# Development intervals (METHODS 7p): block bootstrap of the released layers on the five room tapes
vci = json.loads((DATA_DIR / "paper/val_ci.json").read_text(encoding="utf-8"))
_p = vci["room5_pipeline"]["systems"]
out["accuracy"]["dev_ci_block"] = {"merged": _p["released_merge"]["ci_block"], "corrected": _p["released_corrected"]["ci_block"]}

# Per-conversation test results (METHODS 7o, frus_test/per_conv.tsv) and the corrector's evidence for name fixes
_rows = open(DATA_DIR / "frus_test/per_conv.tsv", encoding="utf-8").read().split("\n")
_hdr = _rows[0].split("\t")
out["test_per_conv"] = [dict(zip(_hdr, r.split("\t"))) for r in _rows[1:] if r.strip()]
out["test_by_location"] = {loc: {L: {k: v[L][k] for k in ("recall", "names") if k in v[L]} for L in LAYERS}
                           | {"n": v["n"], "N": v["corrected"]["N"]}
                           for loc, v in frus["by_location"].items()}
out["name_fix_reasons"] = {k: v.get("in_name_matches", 0) for k, v in frus["llm"]["by_reason"].items()}
out["test_speakers"] = frus["speakers"]["per_speaker"]

# Per-recording development scores by fine-tuning round (METHODS 3, 7e-7g, 7p): six room references
ROOM6 = ["886-008_wspf", "741-002_trial", "779-002_wspf", "488-015_nara", "421-018_wspf", "051-001_wspf"]
_sys = vci["room6"]["systems"]
per_ref = {"Base": _sys["stock_parakeet"]["per_ref"], "v1": _sys["parakeet_r1"]["per_ref"],
           "v5": _sys["parakeet_v5"]["per_ref"], "WhisperX": _sys["whisperx_batched"]["per_ref"],
           "Whisper seq.": _sys["whisper_sequential"]["per_ref"]}
for tag, cfg in (("v3", "parakeet-r3"), ("v4", "parakeet-r4")):  # round5k/score_all.txt, same scorer (METHODS 7g)
    per_ref[tag] = {}
    for line in open(DATA_DIR / "round5k/score_all.txt", encoding="utf-8").read().split("\n")[1:]:
        f = line.split()
        if len(f) > 6 and f[0] == cfg and f[1] in ROOM6:
            per_ref[tag][f[1]] = {"word_recall": float(f[6].rstrip("%")) / 100}
out["round_per_ref"] = {k: {r: v[r]["word_recall"] for r in ROOM6 if r in v} for k, v in per_ref.items()}
out["error_parts_dev"] = {k: {p: _sys[s]["point"][p] for p in ("sub", "del", "ins")}
                          for k, s in (("Base", "stock_parakeet"), ("v5", "parakeet_v5"),
                                       ("Whisper seq.", "whisper_sequential"), ("WhisperX", "whisperx_batched"))}

# Fig 3: Whisper-only artifacts in our own layers (METHODS 7p)
art = json.loads((DATA_DIR / "paper/whisper_artifacts.json").read_text(encoding="utf-8"))
bm = art["counts"]["by_mode"]
out["artifacts"] = {
    "examples": art["examples"],
    "loops": {"whisper": bm["batched"]["whisper_loops"] + bm["sequential"]["whisper_loops"],
              "whisper_words": bm["batched"]["whisper_loop_words"] + bm["sequential"]["whisper_loop_words"],
              "per_1000_batched": round(1000 * bm["batched"]["whisper_loop_words"] / bm["batched"]["whisper_words"], 1),
              "per_1000_sequential": round(1000 * bm["sequential"]["whisper_loop_words"] / bm["sequential"]["whisper_words"], 1),
              "parakeet": bm["batched"]["parakeet_v5_loops"] + bm["sequential"]["parakeet_v5_loops"],
              "parakeet_words": bm["batched"]["parakeet_v5_loop_words"] + bm["sequential"]["parakeet_v5_loop_words"]},
    "dropped": {"batched_share": bm["batched"]["whisper_silent_share_of_parakeet_words"],
                "sequential_share": bm["sequential"]["whisper_silent_share_of_parakeet_words"]},
    "parakeet_all": {
        "loop_per_1000": round(1000 * (bm["batched"]["parakeet_v5_loop_words"] + bm["sequential"]["parakeet_v5_loop_words"])
                               / (bm["batched"]["parakeet_words"] + bm["sequential"]["parakeet_words"]), 3),
        "silent_share": (bm["batched"]["parakeet_silent_whisper_words"] + bm["sequential"]["parakeet_silent_whisper_words"])
                        / (bm["batched"]["whisper_words"] + bm["sequential"]["whisper_words"]),
        "tone_words_per_hour": round((bm["batched"]["parakeet_words_in_tones"] + bm["sequential"]["parakeet_words_in_tones"])
                                     / (bm["batched"]["tone_hours"] + bm["sequential"]["tone_hours"]), 1)},
    "by_mode": {m: {"loop_per_1000": round(1000 * bm[m]["whisper_loop_words"] / bm[m]["whisper_words"], 1),
                    "parakeet_loop_per_1000": round(1000 * bm[m]["parakeet_v5_loop_words"] / bm[m]["parakeet_words"], 3),
                    "dropped_share": bm[m]["whisper_silent_share_of_parakeet_words"],
                    "parakeet_silent_share": bm[m]["parakeet_silent_share_of_whisper_words"],
                    "tone_words_per_hour": round(bm[m]["whisper_words_in_tones"] / bm[m]["tone_hours"], 1),
                    "parakeet_tone_words_per_hour": round(bm[m]["parakeet_words_in_tones"] / bm[m]["tone_hours"], 1)}
                for m in ("batched", "sequential")},
    "silence": {"whisper_words_in_tones": bm["batched"]["whisper_words_in_tones"] + bm["sequential"]["whisper_words_in_tones"],
                "whisper_convs_in_tones": bm["batched"]["whisper_convs_text_in_tones"] + bm["sequential"]["whisper_convs_text_in_tones"],
                "parakeet_words_in_tones": bm["batched"]["parakeet_words_in_tones"] + bm["sequential"]["parakeet_words_in_tones"],
                "whisper_stock": bm["batched"]["whisper_stock_phrases"] + bm["sequential"]["whisper_stock_phrases"],
                "parakeet_stock": bm["batched"]["parakeet_stock_phrases"] + bm["sequential"]["parakeet_stock_phrases"]},
}

# Fig 7: audio hours per month, by NARA location code (explorer database dates and places, corpus manifest durations)
edb = sqlite3.connect(DATA_DIR / "explorer/explorer.db")
meta = {r[0]: (r[1], r[2]) for r in edb.execute("select id, date, place from conversations")}
GROUP = {"OVAL": "Oval Office", "EOB": "Executive Office Building", "WHT": "White House telephone",
         "CDHW": "Camp David", "CDST": "Camp David", "CDSD": "Camp David", "CAB": "Cabinet Room"}
month = defaultdict(lambda: defaultdict(float))
for line in open(DATA_DIR / "corpus_v5/manifest.tsv", encoding="utf-8").read().split("\n")[1:]:
    if not line.strip():
        continue
    c, a, _w, _ = line.split("\t")
    if meta.get(c) and meta[c][0]:
        month[meta[c][0][:7]][GROUP.get(meta[c][1], "Other")] += float(a) / 3600
out["by_month_location"] = {m: dict(v) for m, v in sorted(month.items())}

# Copied from METHODS.md (section given)
out["finetune"] = {  # METHODS 3, 7e, 7f, 7g: room words / room Nixon / room short replies / phone, % of words
    "WhisperX large-v3, batched": [57.0, 44.2, 28.6, 79.6],
    "Whisper large-v3, sequential": [64.6, 52.1, 32.8, 80.4],
    "Parakeet v3, stock": [63.8, 49.3, 32.2, 82.0],
    "v1": [68.5, 58.7, 41.3, 86.6],
    "v3": [70.0, 61.2, 43.8, 86.6],
    "v4": [70.7, 61.9, 44.8, 86.7],
    "v5": [72.3, 64.9, 46.7, 87.3],
}
# METHODS 5, classifier v4 (the model behind every corpus label): 6-fold cross-validation per clip, room-normalized linear
_v4 = json.loads((DATA_DIR / "voiceprints_v4/sweep.json").read_text(encoding="utf-8"))
_v4c = next(r for r in _v4["results"] if r["norm"] == "room" and r["hidden"] == 0)
_v4n = Counter(json.loads(l)["speaker"] for l in open(DATA_DIR / "voiceprints_v4/segments.jsonl", encoding="utf-8") if l.strip())
out["speakers"] = {
    "version": "v4", "per_clip": {k: int(v * 100 + 0.5) for k, v in _v4c["per_speaker"].items()},  # half up, as METHODS
    "balanced_acc": round(_v4c["balanced_acc"] * 100, 1), "piles": _v4c["piles"],
    "clips_by_class": {k: _v4n[k] for k in _v4c["per_speaker"]},
    "clips": 24656, "convs": 902, "validation_named_share": 64, "validation_correct": 95.7,
}
out["correction"] = {  # METHODS 7i: v5 merge, 5 room tapes
    "arms": {"Parakeet words only": {"gained": 405, "helped": 258, "hurt": 20},
             "+ Whisper alternatives": {"gained": 564, "helped": 318, "hurt": 23},
             "+ confidence marks": {"gained": 712, "helped": 387, "hurt": 29}},
    "names_by_tape": {"886-008": [85.4, 88.7], "488-015": [61.8, 85.4], "779-002": [58.9, 78.5],
                      "421-018": [55.9, 56.8], "051-001": [30.0, 41.4]},
    "whisper_edits": {"helped": 179, "hurt": 10},
}
(ROOT / "paper/data").mkdir(parents=True, exist_ok=True)
(ROOT / "paper/data/figure_data.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for s in ("dev", "test"):
    for L, v in out["accuracy"][s].items():
        print(f"{s:4} {L:12} recall {v['recall']:.1%} WER {v['wer']:.1%} blank {v['del']:.1%} "
              f"wrong content shown {v['wrong_content_shown']:.1%} names {v['names']:.1%}")
print(json.dumps({k: v for k, v in out["artifacts"].items() if k != "examples"}, indent=0))
