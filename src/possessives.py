"""Possessives in model edits (release v1.2): the corrected words are stored normalized, so a possessive "'s" written by
the correction model or a Whisper fill reads as "is" ("agnew is phone", "god is sake").

    python src/possessives.py find      # candidates + Whisper's raw reading -> ~/nixon-data/possessives/candidates.json
    python src/possessives.py ask       # the undecided ones to Gemini (standard API, threads) -> answers.json
    python src/possessives.py rule      # instead of ask, when no model is available: a grammar rule -> answers.json
    python src/possessives.py write     # -> docs/possessive_fixes.json, applied by listener_fixes.apply_possessives

A candidate is an "is" from a lowercase source (model edit, propagated edit, Whisper fill) after a word that can take
a possessive. Decision: Whisper's raw output has "X's" within 1.5 s (the speaker said "X's", possessive or
contraction) -> "X's"; Whisper has "X is" -> keep; otherwise Gemini decides possessive (-> "X's") or not (keep).
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper"))
from paths import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "possessives"
CORPUS = DATA_DIR / "corpus_v5" / "out"
FIXES = ROOT / "docs" / "possessive_fixes.json"
LOWER = {"llm", "whisper", "listener"}
# words whose "'s" is never a possessive in this text ("it's", "that's", "he's", ...): the expansion "is" is right
NOT_POSSESSIVE = set("""it that there what he she this who where here how why when which i you we they let one
everything nothing something anything everybody nobody somebody anybody everyone someone anyone this these those
is was are be not no yes oh uh um well so and but or if then now just also only still""".split())
MODEL = "gemini-4-argon"
API = "https://generativelanguage.googleapis.com"


def final_words(conv: str) -> list[dict]:
    import check_quotes  # the released word layer: merged + model edits + propagation + tones dropped + fixes by ear
    return check_quotes.final_words(conv)


def find_one(conv: str) -> list[dict]:
    try:
        words = final_words(conv)
    except Exception:  # noqa: BLE001  (conversations without words or layers)
        return []
    wh = None
    out = []
    for i in range(1, len(words)):
        w, x = words[i], words[i - 1]
        if w["w"] != "is" or w.get("src") not in LOWER:
            continue
        if not re.fullmatch(r"[a-z]+", x["w"]) or x["w"] in NOT_POSSESSIVE:
            continue
        if wh is None:
            p = CORPUS / conv / "whisper.json"
            wh = json.loads(p.read_text(encoding="utf-8"))["words"] if p.exists() else []
        near = [re.sub(r"[^a-z'’]", "", y["w"].lower()).replace("’", "'") for y in wh
                if x["s"] - 1.5 <= y["s"] <= w["e"] + 1.5]
        whisper = "s" if f"{x['w']}'s" in near else "is" if any(
            a == x["w"] and b == "is" for a, b in zip(near, near[1:])) else None
        ctx_l = " ".join(y["w"] for y in words[max(0, i - 9):i - 1])
        ctx_r = " ".join(y["w"] for y in words[i + 1:i + 7])
        out.append({"conv": conv, "at": w["s"], "word": x["w"], "whisper": whisper,
                    "text": f"{ctx_l} [{x['w']} is] {ctx_r}"})
    return out


def find() -> int:
    convs = sorted(p.name for p in CORPUS.iterdir() if (p / "merged.json").exists())
    cands = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for got in ex.map(find_one, convs, chunksize=64):
            cands.extend(got)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=0), encoding="utf-8")
    n = {k: sum(1 for c in cands if c["whisper"] == k) for k in ("s", "is", None)}
    print(f"{len(cands)} candidates in {len(set(c['conv'] for c in cands))} conversations; Whisper 'X's' {n['s']}, "
          f"Whisper 'X is' {n['is']}, undecided {n[None]}")
    return 0


PROMPT = """These lines come from machine transcripts of Richard Nixon's White House conversations (1971-1973). The text is
normalized: lowercase, no punctuation, and every "'s" was expanded to the word "is". So a possessive such as "Agnew's
phone" appears as "agnew is phone", and "for God's sake" as "god is sake". In each line, one pair is marked in
brackets, like [agnew is]. Decide whether the marked "is" stands for a possessive "'s" (the word owns what follows:
"[agnew is] phone", "[brezhnev is] speech", "[god is] sake", "[haldeman is] office"), or is the verb "is" (or the
contraction of it): "[ziegler is] en route", "[mitchell is] going to", "[the president is] here".

Return JSON only: {"possessive": [the ids of the lines whose marked "is" is a possessive]}.

"""


def ask_chunk(chunk: list[tuple[int, str]]) -> tuple[list[int], dict]:
    import requests
    key = (ROOT / ".secrets" / "gemini_key.txt").read_text(encoding="utf-8").strip()
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT + "\n".join(f"{i}: {t}" for i, t in chunk)}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
    import time
    for attempt in range(10):
        if attempt:
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))  # rate limits (HTTP 429): back off and retry
        try:
            r = requests.post(f"{API}/v1beta/models/{MODEL}:generateContent", json=body, timeout=600,
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            r.raise_for_status()
            d = r.json()
            text = d["candidates"][0]["content"]["parts"][0]["text"]
            ids = [int(i) for i in json.loads(text)["possessive"]]
            return [i for i in ids if any(i == c for c, _ in chunk)], d.get("usageMetadata", {})
        except Exception as e:  # noqa: BLE001
            err = e
    raise SystemExit(f"chunk starting {chunk[0][0]} failed: {err}")


def ask() -> int:
    cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    todo = [(i, c["text"]) for i, c in enumerate(cands) if c["whisper"] is None]
    chunks = [todo[k:k + 300] for k in range(0, len(todo), 300)]
    yes, tin, tout = [], 0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for ids, use in ex.map(ask_chunk, chunks):
            yes.extend(ids)
            tin += use.get("promptTokenCount", 0)
            tout += use.get("candidatesTokenCount", 0) + use.get("thoughtsTokenCount", 0)
    (OUT / "answers.json").write_text(json.dumps({"possessive": sorted(yes), "asked": len(todo), "requests": len(chunks),
                                                  "tokens_in": tin, "tokens_out": tout}), encoding="utf-8")
    print(f"asked {len(todo)} in {len(chunks)} requests; possessive {len(yes)}; tokens in {tin:,} out {tout:,}")
    return 0


# bare nouns that do follow a real "is" ("ford is president", "he is home"): never joined by the grammar rule
COPULA_NOUNS = set("""president chairman secretary chief head boss director ambassador governor senator speaker leader
member part home time nominee candidate counsel attorney general minister premier king queen
today tomorrow tonight yesterday morning afternoon evening week month year monday tuesday wednesday thursday friday
saturday sunday""".split())


def rule() -> int:
    """No model available (spending cap): a grammar rule decides the candidates Whisper left open. A real "is" is not
    followed by a bare common noun in English, so "[brezhnev is] statement" is a possessive; "[connally is] likely"
    is not. Joined only when the next word is tagged NN/NNS, is not a name (casing lexicon) and is not a role noun."""
    import nltk
    cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    lex = json.loads((DATA_DIR / "site_build" / "casing.json").read_text(encoding="utf-8"))["cased"]
    yes = []
    for i, c in enumerate(cands):
        if c["whisper"] is not None:
            continue
        left, rest = c["text"].split("[", 1)
        mid, right = rest.split("]", 1)
        toks = left.split() + mid.split() + right.split()
        k = len(left.split()) + 2  # the word after "is"
        if k >= len(toks):
            continue
        nxt = toks[k]
        if nxt in COPULA_NOUNS or not nxt.isalpha() or lex.get(nxt, nxt)[:1].isupper():
            continue
        if nltk.pos_tag(toks)[k][1] in ("NN", "NNS"):
            yes.append(i)
    (OUT / "answers.json").write_text(json.dumps({"possessive": yes, "method": "rule",
                                                  "asked": sum(1 for c in cands if c["whisper"] is None)}),
                                      encoding="utf-8")
    print(f"rule: {len(yes)} of {sum(1 for c in cands if c['whisper'] is None)} undecided are possessives")
    return 0


def write() -> int:
    cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    ans = json.loads((OUT / "answers.json").read_text(encoding="utf-8"))
    yes, how = set(ans["possessive"]), ans.get("method", "model")
    fixes = [{"conv": c["conv"], "at": c["at"], "word": c["word"], "by": "whisper" if c["whisper"] == "s" else how}
             for i, c in enumerate(cands) if c["whisper"] == "s" or (c["whisper"] is None and i in yes)]
    FIXES.write_text(json.dumps({"about": "Possessives restored in release v1.2 (src/possessives.py): the \"is\" at `at` "
                                 "(seconds) is joined to the word before it as \"word's\". by: whisper = Whisper's raw "
                                 "output has \"word's\" there; model = Gemini judged it a possessive.",
                                 "fixes": fixes}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(fixes)} possessives -> {FIXES} (whisper {sum(f['by'] == 'whisper' for f in fixes)}, "
          f"model {sum(f['by'] == 'model' for f in fixes)})")
    return 0


if __name__ == "__main__":
    sys.exit({"find": find, "ask": ask, "rule": rule, "write": write}[sys.argv[1]]())
