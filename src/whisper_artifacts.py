"""Whisper-only artifacts in our own corpus layers (METHODS §7p).

Every conversation's Whisper layer (batched WhisperX for the trial fifth, sequential faster-whisper for the rest, both
with language forced to English) is set against Parakeet v5 on the same audio and against the released layer:

  loops      a word repeated 8+ times in a row, or a 2-6 word phrase 5+ times (loops() below)
  dropped    stretches of 20 s or more where one recognizer has no word but the other has at least 10 (as METHODS §9)
  tones      words placed inside NARA's withdrawal tones, which were muted to silence before both recognizers ran
  stock      Whisper's stock phrases from subtitle training data ("thanks for watching", "subtitles by" ...)
  foreign    12-word windows with under 40% of tokens in the corpus vocabulary (vocabulary test below)
  blanked    conversations whose released layer has no words although the merge had words: who wrote those words

    python src/whisper_artifacts.py scan       -> ~/nixon-data/paper/whisper_artifacts_scan.json (per conversation)
    python src/whisper_artifacts.py report     -> totals by Whisper mode into ~/nixon-data/paper/whisper_artifacts.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402



def loops(toks: list[str]) -> list[tuple[int, int, int, int]]:
    """Repetition loops: a word repeated 8+ times in a row, or a 2-6-word phrase repeated 5+ times.
    Returns (start, end, phrase length, repeats) per loop, as token indexes; the longest loop at a position wins."""
    out, i = [], 0
    while i < len(toks):
        best = None
        for n in range(1, 7):
            unit, k = toks[i:i + n], 1
            if len(unit) < n:
                break
            while toks[i + k * n:i + (k + 1) * n] == unit:
                k += 1
            if ((n == 1 and k >= 8) or (n > 1 and k >= 5)) and (best is None or k * n > best[1]):
                best = (n, k * n, k)
        if best:
            out.append((i, i + best[1], best[0], best[2]))
            i += best[1]
        else:
            i += 1
    return out


CORPUS = DATA_DIR / "corpus_v5" / "out"
PAPER = DATA_DIR / "paper"
SCAN = PAPER / "whisper_artifacts_scan.json"
OUT = PAPER / "whisper_artifacts.json"
TOK = re.compile(r"[a-z']+")
GAP_S, GAP_WORDS = 20.0, 10
STOCK = re.compile(r"thanks? (you )?for watching|subtitles? (by|provided)|amara\.org|please subscribe|"
                   r"like and subscribe|transcribed by|transcription by|captions? by|www\.|\.com\b|"
                   r"see you (in the )?next (video|time)|the end\b\.?$", re.I)


def load(conv: str):
    d = CORPUS / conv
    wj, pj = d / "whisper.json", d / "parakeet_v5.json"
    if not (wj.exists() and pj.exists()):
        return None
    w = json.loads(wj.read_text(encoding="utf-8"))
    p = json.loads(pj.read_text(encoding="utf-8"))
    layer = None
    for n in ("corrected.json", "merged.json"):
        if (d / n).exists():
            layer = json.loads((d / n).read_text(encoding="utf-8"))
            break
    merged = json.loads((d / "merged.json").read_text(encoding="utf-8")) if (d / "merged.json").exists() else None
    return w, p, layer, merged


def toks(words: list[dict]) -> list[str]:
    return [t for x in words for t in TOK.findall(str(x.get("w", "")).lower())]


def timed(words: list[dict]) -> list[tuple[float, float, str]]:
    return sorted((x["s"], x["e"], x["w"]) for x in words if x.get("s") is not None and x.get("e") is not None)


def silent_stretches(a: list[tuple], b: list[tuple], dur: float) -> list[list]:
    """Stretches >= GAP_S where `a` has no word and `b` has >= GAP_WORDS words: [start, end, b_words]."""
    out, edges = [], [0.0] + [t for s, e, _ in a for t in (s, e)] + [dur]
    starts = [s for s, _, _ in b]
    import bisect
    for i in range(0, len(edges) - 1, 2):
        g0, g1 = edges[i], edges[i + 1]
        if g1 - g0 >= GAP_S:
            n = bisect.bisect_left(starts, g1) - bisect.bisect_left(starts, g0)
            if n >= GAP_WORDS:
                out.append([round(g0, 2), round(g1, 2), n])
    return out


def in_spans(words: list[tuple], spans: list) -> list[tuple]:
    return [x for x in words if any(a <= (x[0] + x[1]) / 2 <= b for a, b in spans)]


def scan_one(conv: str):
    got = load(conv)
    if got is None:
        return conv, None
    w, p, layer, merged = got
    ww, pw = w["words"], p["words"]
    dur = float(p.get("audio_seconds") or w.get("audio_seconds") or 0)
    tones = p.get("tones") or []
    wt, pt = timed(ww), timed(pw)
    wtok, ptok = toks(ww), toks(pw)
    ltok = [x["w"] for x in layer] if layer else []
    r = {"dur": dur, "w_words": len(wtok), "p_words": len(ptok), "layer_words": len(ltok),
         "merged_words": len(merged) if merged else 0,
         "w_p_median": sorted(x["p"] for x in ww if x.get("p") is not None)[len(ww) // 2] if ww else None}
    for name, tk in (("w", wtok), ("p", ptok), ("l", ltok)):
        ls = loops(tk)
        r[f"{name}_loops"] = len(ls)
        r[f"{name}_loop_words"] = sum(b - a for a, b, _, _ in ls)
    # where the longest Whisper loops sit in time (token index -> word time via a parallel list)
    wtimes = [x["s"] for x in ww for _ in TOK.findall(str(x.get("w", "")).lower())]
    r["w_loop_spans"] = sorted(([round(wtimes[a], 2), b - a, " ".join(wtok[a:a + n])]
                                for a, b, n, _ in loops(wtok)), key=lambda x: -x[1])[:3]
    r["w_silent"] = silent_stretches(wt, pt, dur)       # Whisper silent, Parakeet heard speech
    r["p_silent"] = silent_stretches(pt, wt, dur)       # Parakeet silent, Whisper heard speech
    r["tone_s"] = round(sum(b - a for a, b in tones), 2)
    r["w_in_tone"] = [[round(s, 2), x] for s, _, x in in_spans(wt, tones)]
    r["p_in_tone"] = [[round(s, 2), x] for s, _, x in in_spans(pt, tones)]
    wtext = " ".join(str(x["w"]) for x in ww)
    r["w_stock"] = [m.group(0) for m in STOCK.finditer(wtext)]
    r["p_stock"] = [m.group(0) for m in STOCK.finditer(" ".join(str(x["w"]) for x in pw))]
    if merged and not ltok:
        r["blanked_src"] = dict(Counter(x.get("src", "?") for x in merged))
    return conv, r


def foreign_count(args):
    conv, vocab = args
    got = load(conv)
    if got is None:
        return conv, 0
    tk = toks(got[0]["words"])
    return conv, sum(12 for i in range(0, len(tk) - 11, 12) if sum(t in vocab for t in tk[i:i + 12]) / 12 < 0.4)


def cmd_scan(workers: int = 16) -> None:
    convs = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    with Pool(workers) as pool:
        res = dict(r for r in pool.imap_unordered(scan_one, convs, chunksize=40) if r[1] is not None)
    SCAN.write_text(json.dumps(res), encoding="utf-8")
    print(f"scanned {len(res)} of {len(convs)} conversations -> {SCAN}")


def trial_set() -> set[str]:
    return {json.loads(l)["conv"] for l in (DATA_DIR / "jobs_trial.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}


def cmd_report() -> None:
    res = json.loads(SCAN.read_text(encoding="utf-8"))
    trial = trial_set()
    modes = {c: ("batched" if c in trial else "sequential") for c in res}
    out = {"conversations": len(res), "mode_counts": dict(Counter(modes.values())), "by_mode": {}}
    for mode in ("batched", "sequential", "all"):
        rs = [r for c, r in res.items() if mode == "all" or modes[c] == mode]
        t = {"convs": len(rs), "audio_h": round(sum(r["dur"] for r in rs) / 3600, 1),
             "whisper_words": sum(r["w_words"] for r in rs), "parakeet_words": sum(r["p_words"] for r in rs),
             "layer_words": sum(r["layer_words"] for r in rs)}
        for k, lab in (("w", "whisper"), ("p", "parakeet_v5"), ("l", "released")):
            t[f"{lab}_loops"] = sum(r[f"{k}_loops"] for r in rs)
            t[f"{lab}_loop_words"] = sum(r[f"{k}_loop_words"] for r in rs)
            t[f"{lab}_convs_with_loop"] = sum(1 for r in rs if r[f"{k}_loops"])
        t["whisper_silent_stretches"] = sum(len(r["w_silent"]) for r in rs)
        t["whisper_silent_hours"] = round(sum(b - a for r in rs for a, b, _ in r["w_silent"]) / 3600, 1)
        t["whisper_silent_parakeet_words"] = sum(n for r in rs for _, _, n in r["w_silent"])
        t["whisper_silent_share_of_parakeet_words"] = round(t["whisper_silent_parakeet_words"] / max(t["parakeet_words"], 1), 4)
        t["whisper_silent_convs"] = sum(1 for r in rs if r["w_silent"])
        t["parakeet_silent_stretches"] = sum(len(r["p_silent"]) for r in rs)
        t["parakeet_silent_hours"] = round(sum(b - a for r in rs for a, b, _ in r["p_silent"]) / 3600, 1)
        t["parakeet_silent_whisper_words"] = sum(n for r in rs for _, _, n in r["p_silent"])
        t["parakeet_silent_share_of_whisper_words"] = round(t["parakeet_silent_whisper_words"] / max(t["whisper_words"], 1), 4)
        t["tone_convs"] = sum(1 for r in rs if r["tone_s"])
        t["tone_hours"] = round(sum(r["tone_s"] for r in rs) / 3600, 2)
        t["whisper_words_in_tones"] = sum(len(r["w_in_tone"]) for r in rs)
        t["whisper_convs_text_in_tones"] = sum(1 for r in rs if r["w_in_tone"])
        t["parakeet_words_in_tones"] = sum(len(r["p_in_tone"]) for r in rs)
        t["parakeet_convs_text_in_tones"] = sum(1 for r in rs if r["p_in_tone"])
        t["whisper_stock_phrases"] = sum(len(r["w_stock"]) for r in rs)
        t["whisper_stock_top"] = Counter(s.lower() for r in rs for s in r["w_stock"]).most_common(8)
        t["parakeet_stock_phrases"] = sum(len(r["p_stock"]) for r in rs)
        bl = [r for r in rs if "blanked_src" in r]
        t["blanked_convs"] = len(bl)
        t["blanked_merged_words_by_src"] = dict(sum((Counter(r["blanked_src"]) for r in bl), Counter()))
        t["blanked_whisper_words"] = sum(r["w_words"] for r in bl)
        t["blanked_parakeet_words"] = sum(r["p_words"] for r in bl)
        t["blanked_median_dur_s"] = sorted(r["dur"] for r in bl)[len(bl) // 2] if bl else None
        out["by_mode"][mode] = t
    # Whisper word scores: batched WhisperX stores alignment scores, sequential stores token probabilities
    med = {m: sorted(r["w_p_median"] for c, r in res.items() if modes[c] == m and r["w_p_median"] is not None)
           for m in ("batched", "sequential")}
    out["whisper_score_median_by_mode"] = {m: v[len(v) // 2] for m, v in med.items() if v}
    # wrong-language windows by mode (vocabulary = words seen 3+ times in the merged layer)
    vc = Counter()
    for c in res:
        f = CORPUS / c / "merged.json"
        if f.exists():
            vc.update(x["w"] for x in json.loads(f.read_text(encoding="utf-8")))
    vocab = {w for w, n in vc.items() if n >= 3}
    with Pool(16) as pool:
        fr = dict(pool.map(foreign_count, [(c, vocab) for c in res], chunksize=50))
    for mode in ("batched", "sequential", "all"):
        sel = [c for c in res if mode == "all" or modes[c] == mode]
        out["by_mode"][mode]["foreign_words"] = sum(fr[c] for c in sel)
        out["by_mode"][mode]["foreign_convs"] = sum(1 for c in sel if fr[c])
    out["vocab_size"] = len(vocab)
    out["settings"] = {"whisper_language": "en (forced in both modes, src/worker.py)",
                       "sequential": "faster-whisper large-v3, beam 5, condition_on_previous_text=False, temperature "
                                     "fallback 0-1.0, no VAD, word timestamps",
                       "batched": "WhisperX large-v3, batch 16 (its VAD segmentation), wav2vec2 word alignment",
                       "gap_rule": f"stretch >= {GAP_S:.0f} s with no word from one recognizer and >= {GAP_WORDS} from the other",
                       "mode_assignment": "trial job list (jobs_trial.jsonl) = batched; up to 47 trial tapes that failed "
                                          "in the trial were redone sequentially and are counted as batched"}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev["counts"] = out
    OUT.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


# Figure examples, chosen from the scan on tapes with a human transcript (validation references; FRUS for 099-010's
# meeting) and free of offensive language. Reference text is quoted from the source; ref_clean only repairs OCR.
EXAMPLES = [
    {"panel": "a", "kind": "repetition loop", "conv": "421-018", "date": "March 21, 1973", "place": "Executive Office Building",
     "speakers": "Nixon, Ehrlichman, Dean, Haldeman", "t0": 3925.0, "t1": 3990.0, "focus": [3935.7, 3959.3],
     "reference_source": "WSPF transcript (421-018_wspf)",
     "reference": "Nixon: Well, John, you, uh, you, uh, you must feel that's, uh, is enough. Dean: No [unintelligible] "
                  "Ehrlichman: Now, let's, let's try another, let's try another concommitant to that. Supposing Mitchell "
                  "were to step out on that same day",
     "whisper": "As for that concept, John, you were, you were, you were, you were, you were, you were, … ('you were' 16 "
                "times in all) … I don't know what he would say",
     "ours": "but john you you want to do that well i do not think it is good now let us let us try another let us try "
             "another comment for that supposing mitchell were to step out on that same day",
     "note": "Whisper (batched) repeats 'you were' 16 times over 24 s (3,935.7-3,959.3 s) and loses Ehrlichman's turn."},
    {"panel": "b", "kind": "dropped speech", "conv": "886-008", "date": "March 21, 1973", "place": "Oval Office",
     "speakers": "Nixon, Dean (Haldeman, Ehrlichman)", "t0": 1030.0, "t1": 1095.0, "focus": [1058.25, 1078.4],
     "reference_source": "WSPF transcript (886-008_wspf); HJC has the same words",
     "reference": "Nixon: Why [unintelligible] I wonder? I am just trying to think as to why then. We'd just finished the "
                  "Moscow trip. I mean, we were Dean: That's right. Nixon: The Democrats had just nominated McG-- McGovern.",
     "whisper": "… on other occasions where things have been [no words for this passage] I don't know, other than the fact "
                "that they might have been looking for information about the conventions.",
     "ours": "wonder i am just trying to think as to why that we just finished the moscow trip i mean we were . that is "
             "right democrats were not doing that magruder",
     "note": "Whisper (batched) has no words for about 20 s (1,058.3-1,078.4 s on its timeline) where Parakeet heard 62; "
             "the passage (about 60 reference words) is missing from its text, and the words around it are misplaced in "
             "time by about 20 s."},
    {"panel": "c", "kind": "invented text on silence", "conv": "099-010", "date": "May 1, 1972", "place": "Cabinet Room",
     "speakers": "Nixon, Smith, Rogers, Laird, Moorer, Helms, Allison, Kissinger, Ziegler", "t0": 0.0, "t1": 3516.28,
     "focus": [0.0, 3342.0],
     "reference_source": "NARA withdrawal tone (0:00-55:42), muted to silence before transcription; FRUS 1969-76 "
                         "vol. 32 doc. 270 notes the recording of this meeting and transcribes the Oval Office "
                         "conversation that followed (716-002)",
     "reference": "[withdrawn by NARA: tone for the first 55 min 42 s; nothing audible after muting]",
     "whisper": "you … you … you … (one word every 30 s, 111 times, from 0:00 to 55:00)",
     "ours": "(no words)",
     "note": "Sequential Whisper (no VAD) emits 'you' at the start of each 30-s window of silence; Parakeet v5 writes "
             "nothing there."},
]
ALTERNATES = [
    {"kind": "repetition loop", "conv": "779-002", "date": "September 15, 1972", "place": "Oval Office", "t": 337.82,
     "whisper": "That's it. I think that's a good thought. (x11)",
     "reference_wspf_normalized": "that is do you think that is a good development i think that is a good development "
                                  "because it makes it look so god damned phony",
     "why_not_primary": "mild profanity in the reference line"},
    {"kind": "stock subtitle phrase", "conv": "827-010", "date": "December 20, 1972", "place": "Oval Office", "t": 2610.12,
     "whisper": "Then we'll know how to get to the next week. It's going to be a lot of changes. Thank you for watching.",
     "ours": "then we will know how because when will you get the report the next waves why do not they change his mind",
     "why_not_primary": "the FRUS transcript (vol. 9 doc. 209) is an excerpt that does not cover the last seconds"},
]


def per_minute(words: list[dict], t0: float, t1: float, step: float) -> list[int]:
    n = int((t1 - t0) // step) + 1
    out = [0] * n
    for x in words:
        s = x.get("s")
        if s is not None and t0 <= s < t1 and not str(x.get("w", "")).startswith("["):
            out[int((s - t0) // step)] += len(TOK.findall(str(x["w"]).lower())) or 0
    return out


def cmd_examples() -> None:
    for ex in EXAMPLES:
        d = CORPUS / ex["conv"]
        w = json.loads((d / "whisper.json").read_text(encoding="utf-8"))["words"]
        p = json.loads((d / "parakeet_v5.json").read_text(encoding="utf-8"))
        lay = json.loads((d / "corrected.json").read_text(encoding="utf-8"))
        step = 60.0 if ex["t1"] - ex["t0"] > 600 else 5.0
        ex["series"] = {"step_s": step, "t0": ex["t0"],
                        "whisper": per_minute(w, ex["t0"], ex["t1"], step),
                        "parakeet_v5": per_minute(p["words"], ex["t0"], ex["t1"], step),
                        "released": per_minute(lay, ex["t0"], ex["t1"], step)}
        a, b = ex["focus"]
        ex["focus_counts"] = {"whisper_words": sum(1 for x in w if a <= x["s"] < b),
                              "parakeet_words": sum(1 for x in p["words"] if a <= x["s"] < b),
                              "released_words": sum(1 for x in lay if x.get("s") is not None and a <= x["s"] < b)}
        ex["tones"] = p.get("tones") or []
        for k in ("reference", "whisper", "ours"):
            ex[f"{k}_words"] = len(ex[k].split())
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev["examples"] = EXAMPLES
    prev["alternates"] = ALTERNATES
    OUT.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    for ex in EXAMPLES:
        print(ex["panel"], ex["conv"], ex["focus_counts"], {k: ex[f"{k}_words"] for k in ("reference", "whisper", "ours")})


def cmd_extra() -> None:
    """Silent stretches split by how densely the other recognizer wrote words there (speech runs at about 150 words
    a minute; a word every 30 s is a silence hallucination), and released-layer words left inside muted tones."""
    res = json.loads(SCAN.read_text(encoding="utf-8"))
    trial = trial_set()
    extra = {"density_bands_wpm": ["<20", "20-60", ">=60"]}
    for mode in ("batched", "sequential"):
        for key, lab in (("p_silent", "parakeet_silent"), ("w_silent", "whisper_silent")):
            bands = [[0, 0.0, 0] for _ in range(3)]
            for c, r in res.items():
                if (c in trial) != (mode == "batched"):
                    continue
                for a, b, n in r[key]:
                    wpm = n / ((b - a) / 60)
                    k = 0 if wpm < 20 else 1 if wpm < 60 else 2
                    bands[k][0] += 1
                    bands[k][1] += (b - a) / 3600
                    bands[k][2] += n
            extra[f"{mode}_{lab}_by_density"] = [[n, round(h, 1), w] for n, h, w in bands]
    n = convs = 0
    src = Counter()
    for c, r in res.items():
        f = CORPUS / c / "corrected.json"
        if not r["tone_s"] or not f.exists():
            continue
        tones = json.loads((CORPUS / c / "parakeet_v5.json").read_text(encoding="utf-8")).get("tones") or []
        k = [x for x in json.loads(f.read_text(encoding="utf-8"))
             if x.get("s") is not None and any(a <= (x["s"] + x["e"]) / 2 <= b for a, b in tones)]
        if k:
            convs += 1
            n += len(k)
            src.update(x.get("src", "?") for x in k)
    extra["released_words_in_tones"] = {"words": n, "convs": convs, "by_src": dict(src)}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev["counts_extra"] = extra
    OUT.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    {"scan": cmd_scan, "report": cmd_report, "examples": cmd_examples, "extra": cmd_extra}[sys.argv[1]]()
