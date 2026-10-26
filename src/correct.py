"""LLM correction pass, in two steps around an edit-list writer (the language model, called through its API).

  prep   merged transcript (merge.py) -> out/<conv>/correct_input.txt: ~60 s blocks, lowercase, with
         [parakeet words | whisper: alternative] at every disagreement; plus correct_context.txt
         holding NARA's date, place, participants and the conversation's subject-log entry.
  apply  out/<conv>/edits.json (a JSON list of {block, old, new, confidence, reason}) ->
         out/<conv>/corrected.json: the merged words with the edits applied (each edited word keeps
         its time and records the original), and corrected.txt: a readable transcript with
         speakers taken from speakers.json. The raw files are never changed.

    python src/correct.py prep  <root> --jobs jobs.jsonl
    python src/correct.py apply <root> --jobs jobs.jsonl
    python src/correct.py prep  <root> --jobs jobs.jsonl --parakeet parakeet_v5.json --marks 0.01 --workers 16
    python src/correct.py apply <root> --jobs jobs.jsonl --edits-name edits_argon.json --propagate   (after argon.py)
With --marks, runs of Parakeet words below that confidence are wrapped in {braces} for the corrector (7i).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from merge import merge
from score import norm_words

LOG_DIR = Path(os.environ.get("NIXON_DATA", Path.home() / "nixon-data")) / "manifest" / "subject_logs" / "extracted"
LOG_PREFIX = {"OVAL": "oval", "EOB": "eob", "WHT": "wht", "CAB": "cab", "CDHW": "cdhw", "CDST": "cdst", "CDSD": "cdsd"}
BLOCK_S = 60
DIARY_DIR = Path(os.environ.get("NIXON_DATA", Path.home() / "nixon-data")) / "haldeman" / "text"


def haldeman(date: str) -> str:
    """H. R. Haldeman's diary entries for the tape's date and the day after (he often dictated the next
    morning), from NARA's daily PDFs as text; '' if none. Coverage ends April 30, 1973."""
    import datetime
    try:
        day = datetime.datetime.strptime(date.strip(), "%B %d, %Y").date()
    except ValueError:
        return ""
    out = []
    for k, label in ((0, "same day"), (1, "next day")):
        stamp = (day + datetime.timedelta(days=k)).strftime("%Y%m%d")
        # a file's name holds the date(s) it covers, e.g. ...-19710101-pa.txt or ...-19721223-19721226_-pa.txt
        for f in sorted(DIARY_DIR.glob(f"*{stamp}*.txt")):
            body = "\n".join(l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
                             if l.strip() and "Haldeman Diaries Collection" not in l
                             and "Online Public Access Catalog" not in l and not l.strip().startswith("Page |"))
            out.append(f"--- Haldeman diary, {label} ({stamp}) ---\n{body}")
    return "\n\n".join(out)


@lru_cache(maxsize=4096)
def _log_files(prefix: str, tape: str) -> tuple[str, ...]:
    """The tape's subject log: <loc>-<tape>.rtf, else the PDF copy or NARA's 37-wht-audiotape-<tape>-log.pdf."""
    rtf = glob.glob(str(LOG_DIR / "**" / f"{prefix}-{tape}.rtf"), recursive=True)
    pdf = (glob.glob(str(LOG_DIR / "**" / f"{prefix}-{tape}.pdf"), recursive=True)
           + glob.glob(str(LOG_DIR / "**" / f"37-wht-audiotape-{tape}-log.pdf"), recursive=True))
    return tuple(rtf + pdf)


@lru_cache(maxsize=64)  # a tape's log holds every conversation on it: parse it once, not once per conversation
def _log_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        import subprocess
        return subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
    from striprtf.striprtf import rtf_to_text
    return rtf_to_text(open(path, encoding="latin-1").read())


LOG_FURNITURE = re.compile(r"^\s*(Page \|.*|Revised and re-released.*|White House Tapes of the Nixon Administration.*|"
                           r"Richard Nixon Presidential Library.*|Nixon Presidential Materials.*|NIXON PRESIDENTIAL.*)$", re.M)


def nara_log(job: dict) -> str:
    """The conversation's entry in NARA's subject log (RTF, else PDF), or '' if none is found. Logs number
    conversations with or without leading zeros ("Conversation No. 51-1", "No. 886-8"); a PDF log opens with a
    table of contents listing every number, so the longest match (the entry itself) is taken, without page furniture."""
    tape, num = job["conv"].split("-")
    best = ""
    for f in _log_files(LOG_PREFIX.get(job["location"], "*"), tape):
        t = _log_text(f)
        for m in re.finditer(rf"Conversation No\.\s*0*{int(tape)}-0*{int(num)}\b", t):
            nxt = re.search(r"Conversation No\.", t[m.end():])
            seg = t[m.start():m.end() + (nxt.start() if nxt else 4000)]
            if len(seg) > len(best):
                best = seg
        if "Date:" in best:
            break
    return re.sub(r"\n{3,}", "\n\n", LOG_FURNITURE.sub("", best)).strip()


def blocks(merged: list[dict], unsure_below: float | None = None) -> list[tuple[float, str]]:
    """~60 s blocks of the merged words, with [parakeet | whisper: alt] at disagreements. With unsure_below,
    runs of Parakeet words whose confidence is below it are wrapped in {braces} (METHODS 7i); inside a
    bracket, the Parakeet side is wrapped when any of its words is unsure."""
    unsure = [unsure_below is not None and x.get("c") is not None and x["c"] < unsure_below for x in merged]
    out, cur, t0, k = [], [], None, 0
    while k < len(merged):
        x = merged[k]
        t0 = x["s"] if t0 is None else t0
        if "alt" in x:
            j = k
            while j < len(merged) and merged[j].get("alt") == x["alt"] and merged[j]["src"] == x["src"]:
                j += 1
            prim = " ".join(y["w"] for y in merged[k:j])
            cur.append("[" + ("{" + prim + "}" if any(unsure[k:j]) else prim) + " | whisper: " + x["alt"] + "]")
        else:
            j = k
            while (j < len(merged) and "alt" not in merged[j] and unsure[j] == unsure[k]
                   and (j == k or merged[j - 1]["s"] - t0 < BLOCK_S)):
                j += 1
            run = " ".join(y["w"] for y in merged[k:j])
            cur.append("{" + run + "}" if unsure[k] else run)
        k = j
        if merged[k - 1]["s"] - t0 >= BLOCK_S or k == len(merged):
            out.append((t0, " ".join(cur)))
            cur, t0 = [], None
    return out


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def prep_one(root: Path, job: dict, parakeet: str = "parakeet.json", marks: float | None = None) -> str:
    d = root / "out" / job["conv"]
    pk = json.loads((d / parakeet).read_text(encoding="utf-8"))
    wh = json.loads((d / "whisper.json").read_text(encoding="utf-8"))
    merged = merge(pk, wh)
    (d / "merged.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    text = "\n".join(f"[{mmss(t)}] {b}" for t, b in blocks(merged, marks)) + "\n" if merged else ""
    (d / "correct_input.txt").write_text(text, encoding="utf-8", newline="\n")
    (d / "correct_context.txt").write_text(context_text(job), encoding="utf-8", newline="\n")
    n_alt = sum("alt" in x for x in merged)
    n_unsure = text.count("{")
    return f"{job['conv']}: {len(merged)} words, {n_alt} in disagreements, {n_unsure} unsure runs, {text.count(chr(10))} blocks"


def context_text(job: dict) -> str:
    return (f"Conversation {job['conv']}, {job['date']}, location code {job['location']}.\n"
            f"NARA participants: {'; '.join(job['participants'])}\n\nNARA subject log entry:\n{nara_log(job) or '(none found)'}\n"
            f"\nH. R. Haldeman's dictated diary (his own account of the day; he was not at every meeting):\n"
            f"{haldeman(job['date']) or '(no entry)'}\n")


def _context_star(a):
    root, job = a
    (root / "out" / job["conv"] / "correct_context.txt").write_text(context_text(job), encoding="utf-8", newline="\n")
    return job["conv"]


def cmd_context(root: Path, jobs: list[dict], workers: int = 1) -> None:
    """Rewrite only correct_context.txt (after a change to the NARA or diary lookups); transcripts untouched."""
    from multiprocessing import Pool
    with Pool(max(workers, 1)) as pool:
        n = sum(1 for _ in pool.imap_unordered(_context_star, [(root, j) for j in jobs], chunksize=16))
    print(f"rewrote the context of {n} tapes")


def _prep_star(a):
    try:
        return prep_one(*a)
    except Exception as e:  # one bad tape must not stop 19,000 others
        return f"{a[1]['conv']}: FAILED {type(e).__name__}: {e}"


def cmd_prep(root: Path, jobs: list[dict], parakeet: str = "parakeet.json", marks: float | None = None,
             workers: int = 1) -> None:
    args = [(root, job, parakeet, marks) for job in jobs]
    if workers <= 1:
        results = map(_prep_star, args)
    else:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap_unordered(_prep_star, args, chunksize=8)
    failed = 0
    for i, line in enumerate(results, 1):
        failed += "FAILED" in line
        if "FAILED" in line or workers <= 1 or i % 1000 == 0:
            print(f"{line}" if workers <= 1 or "FAILED" in line else f"{i}/{len(args)} prepared", flush=True)
    print(f"prepared {len(args) - failed} of {len(args)} tapes, {failed} failed")


def apply_edits(merged: list[dict], edits: list[dict], starts: list[float],
                keep_unclear: bool = False) -> tuple[list[dict], int, int]:
    """Apply an edit list to the merged words. An [unclear] edit removes its words; with keep_unclear (for
    display, not scoring) one "[unclear]" placeholder is left in their place."""
    words = [dict(x) for x in merged]
    applied = missing = 0
    for e in edits:
        m, s = e["block"].strip("[]").split(":")
        a = int(m) * 60 + int(s) - 0.5
        later = [t for t in starts if t > a + 0.5]
        b = (later[0] - 0.5) if later else 1e12
        old = e["old"].lower().split()
        new = [t for w in e["new"].split() for t in norm_words(w)] or []
        idx = [i for i, x in enumerate(words) if a <= x["s"] < b]
        hit = next((i for k, i in enumerate(idx) if [x["w"] for x in words[i:i + len(old)]] == old), None)
        if hit is None or not old:
            missing += 1
            continue
        span = words[hit:hit + len(old)]
        t0, t1 = span[0]["s"], span[-1]["e"]
        step = (t1 - t0) / max(len(new), 1)
        repl = [{"w": t, "s": round(t0 + k * step, 2), "e": round(t0 + (k + 1) * step, 2), "src": "llm",
                 "was": e["old"], "confidence": e.get("confidence"), "reason": e.get("reason")} for k, t in enumerate(new)]
        if not repl and keep_unclear and "unclear" in str(e.get("new", "")):
            repl = [{"w": "[unclear]", "s": t0, "e": t1, "src": "llm", "was": e["old"], "reason": e.get("reason")}]
        words = words[:hit] + repl + words[hit + len(old):]
        applied += 1
    return words, applied, missing


LM_DIR = Path(os.environ.get("NIXON_DATA", Path.home() / "nixon-data")) / "lm"
COMMON_MIN = 3  # a word seen at least this often in the era text (era_lm.py's clean sources) counts as ordinary


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """Lowercase words seen at least COMMON_MIN times in WSPF, NARA, the diary and FRUS (22.7M words); cached."""
    f = LM_DIR / "common_words.txt"
    if not f.exists():
        from collections import Counter
        cnt = Counter()
        for name in ("wspf", "nara", "diary", "frus"):
            for line in open(LM_DIR / "clean" / f"{name}.txt", encoding="utf-8"):
                cnt.update(re.findall(r"[a-z']+", line.lower()))
        f.write_text("\n".join(sorted(w for w, n in cnt.items() if n >= COMMON_MIN)) + "\n", encoding="utf-8")
    return frozenset(f.read_text(encoding="utf-8").split())


def propagate(words: list[dict], edits: list[dict]) -> tuple[list[dict], int]:
    """Repeat a name fix elsewhere in the tape (7k): an edit whose old words (1-3) include one the era text hardly
    knows ("scalli") is also applied to identical, unedited occurrences. An old phrase the corrector mapped to two
    different fixes is left alone. Copies are marked "propagated"."""
    common = common_words()
    rules: dict[tuple, set] = {}
    for e in edits:
        old = tuple(str(e.get("old", "")).lower().split())
        new = tuple(t for w in str(e.get("new", "")).split() for t in norm_words(w))
        if (not 1 <= len(old) <= 3 or not new or "unclear" in str(e.get("new", "")) or old == new
                or e.get("confidence") != "high"):
            continue
        if any(re.fullmatch(r"[a-z][a-z']{3,}", w) and w not in common for w in old):  # a word, not "1st" or "."
            rules.setdefault(old, set()).add(new)
    by_first: dict[str, list] = {}
    for old, news in rules.items():
        if len(news) == 1:
            by_first.setdefault(old[0], []).append((old, next(iter(news))))
    out, i, n = [], 0, 0
    while i < len(words):
        hit = next(((o, nw) for o, nw in by_first.get(words[i]["w"], [])
                    if tuple(x["w"] for x in words[i:i + len(o)]) == o and all(x.get("src") != "llm" for x in words[i:i + len(o)])), None)
        if hit is None:
            out.append(words[i]); i += 1
            continue
        old, new = hit
        t0, t1 = words[i]["s"], words[i + len(old) - 1]["e"]
        step = (t1 - t0) / len(new)
        out += [{"w": t, "s": round(t0 + k * step, 2), "e": round(t0 + (k + 1) * step, 2), "src": "llm",
                 "was": " ".join(old), "propagated": True} for k, t in enumerate(new)]
        i += len(old); n += 1
    return out, n


def readable(words: list[dict], named: list[dict]) -> str:
    def who(t):
        for s in named:
            if s["start"] - 0.3 <= t < s["end"] + 0.3:
                return "?" if s["speaker"].startswith("Unknown") else s["speaker"].upper()
        return "?"
    lines, cur = [], None
    for x in words:
        sp = who(x["s"])
        if cur and cur[0] == sp and x["s"] - cur[2] < 4:
            cur[3].append(x["w"]); cur[2] = x["e"]
        else:
            cur and lines.append(cur)
            cur = [sp, x["s"], x["e"], [x["w"]]]
    cur and lines.append(cur)
    return "\n".join(f"[{mmss(s)}] {sp}: {' '.join(w)}" for sp, s, _, w in lines) + "\n"


def cmd_apply(root: Path, jobs: list[dict], edits_name: str = "edits.json", do_propagate: bool = False) -> None:
    for job in jobs:
        d = root / "out" / job["conv"]
        if not (d / edits_name).exists():
            continue
        merged = json.loads((d / "merged.json").read_text(encoding="utf-8"))
        edits = json.loads((d / edits_name).read_text(encoding="utf-8"))
        for e in edits:  # the model sometimes copies the confidence braces into "old"
            e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
        starts = [int(m[1]) * 60 + int(m[2]) for m in re.finditer(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
        words, applied, missing = apply_edits(merged, edits, starts)
        n_prop = 0
        if do_propagate:
            words, n_prop = propagate(words, edits)
        (d / "corrected.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        sp = next((d / n for n in ("speakers.json", "speakers_v4.json") if (d / n).exists()), None)
        named = json.loads(sp.read_text(encoding="utf-8")) if sp else []
        (d / "corrected.txt").write_text(readable(words, named), encoding="utf-8", newline="\n")
        print(f"{job['conv']}: {applied} edits applied, {missing} not located" + (f", {n_prop} name fixes repeated" if n_prop else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "context", "apply"])
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--parakeet", default="parakeet.json", help="prep: Parakeet file name in out/<conv>/ (v5 re-run: parakeet_v5.json)")
    ap.add_argument("--marks", type=float, help="prep: wrap Parakeet words with confidence below this in {braces} (corpus: 0.01)")
    ap.add_argument("--workers", type=int, default=1, help="prep: parallel processes")
    ap.add_argument("--edits-name", default="edits.json", help="apply: edit-list file in out/<conv>/ (Argon pass: edits_argon.json)")
    ap.add_argument("--propagate", action="store_true", help="apply: repeat rare-word (name) fixes elsewhere in the tape")
    args = ap.parse_args()
    jobs = [json.loads(l) for l in args.jobs.read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = [j for j in jobs if (args.root / "out" / j["conv"] / "whisper.json").exists()]
    if args.cmd == "prep":
        jobs = [j for j in jobs if (args.root / "out" / j["conv"] / args.parakeet).exists()]
        cmd_prep(args.root, jobs, args.parakeet, args.marks, args.workers)
    elif args.cmd == "context":
        cmd_context(args.root, jobs, args.workers)
    else:
        cmd_apply(args.root, jobs, args.edits_name, args.propagate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
