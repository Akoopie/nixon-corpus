"""Argon correction pass at corpus scale (Gemini 4 Argon; METHODS 7c-2, 7i, 7k), from `correct.py prep` outputs.

  requests  out/<conv>/{correct_input.txt, correct_context.txt} -> one request per tape; a long tape is split at
            block boundaries into parts of at most PART_WORDS words, each sent with the full context
            -> <batch dir>/requests.jsonl and <batch dir>/estimate.json (tokens and cost at the batch price)
            --by-day: one request per day instead (or per run of consecutive conversations up to PART_WORDS
            words), conversations in time order under headers, each with its NARA record, the diary sent once;
            each edit names its conversation (groups.json maps request keys to conversations)
  submit    uploads requests.jsonl (Files API) and creates the Batch API job -> <batch dir>/batch.json
  status    the job's state
  fetch     downloads the results: per tape out/<conv>/edits_argon.json (all parts' edits, in order) and
            argon_usage.json; a response that is not a JSON edit list is kept as argon_raw_<part>.txt
  direct    the same requests through the standard API (threads), for small runs and tests
  report    pipeline checks over the fetched tapes: parse failures, unlocated edits, [unclear] stretches,
            share of words edited, tokens and cost per word

The prompt is the setting chosen in 7c-2 and 7i: the open prompt, Whisper alternatives at disagreements,
{braces} around words Parakeet was unsure of, [unclear] for hopeless stretches, NARA record + Haldeman diary.
The key is read from .secrets/gemini_key.txt and never printed.

    python src/argon.py requests <root> --convs convs.txt --out <batch dir>
    python src/argon.py submit <batch dir>  |  status <batch dir>  |  fetch <batch dir> --root <root>
    python src/argon.py direct <batch dir> --root <root> [--workers 8]
    python src/argon.py report <root> --convs convs.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from paths import SECRETS_DIR

MODEL = "gemini-4-argon"
API = "https://generativelanguage.googleapis.com"
PART_WORDS = 25000                     # transcript words per request (markup not counted); longer tapes are split
PRICE_IN, PRICE_OUT = 1.00, 5.00       # USD per 1M tokens, Batch tier, October 2026
CHARS_PER_TOKEN = 3.84                 # calibrated on the 7i run (5 tapes: prompt characters per input token)
OUT_PER_IN = 0.35                      # output (incl. thinking) per input token in the 7i run
OUT_PER_TRANSCRIPT = 0.584             # output tokens per transcript token, calibrated on the 604-tape pilot (7k)

INTRO = (
    "You are correcting a machine transcript of a Nixon White House tape (1971-73, a poor room recording). The speech "
    "recognizer is often wrong, above all on names, places, bills, organizations and any unfamiliar word, which it tends "
    "to turn into one or more ordinary words that sound alike: \"para leans\" in a talk about birds is \"parakeets\", "
    "\"early man\" is \"ehrlichman\", \"clean dienst\" is \"kleindienst\". The transcript is lowercase and unpunctuated, in "
    "~60 s blocks starting with [mm:ss]. You also have NARA's record of the conversation (date, place, participants, "
    "subject log) and H. R. Haldeman's dictated diary for that day and the next; he was Nixon's chief of staff and "
    "wrote about the people, meetings and issues of each day, though he was not at every meeting.\n")
WHISPER = ("Where a second speech model (Whisper) heard something different, the transcript shows "
           "[primary words | whisper: other words]; the primary model is usually better, but either may be right.\n")
CONF = ("Words the primary model itself was unsure of are wrapped in {braces}. Most of its errors are inside "
        "braces, but many braced words are right: use the braces to see where to look, and never change a word "
        "only because it is braced.\n")
RULES = """Task: correct every word or phrase you are confident was mis-transcribed, using the conversation's own flow, the NARA record and the diary. Use your knowledge of the period too (who held which job, what was in the news).
- Keep fillers (uh, um), repetitions, false starts and the speakers' grammar; only fix words that were heard wrong.
- Do not add sentences or ideas that are not in the transcript, and do not paraphrase.
- If a stretch is garbled and you cannot tell what was said, replace it with [unclear] ("new": "[unclear]") rather than leave words a reader could take as real. Do this only for stretches that make no sense in context; keep words that plausibly fit, even if braced.

Output ONLY a JSON array. Each element is {"block": "mm:ss" (the block's start time), "old": "the exact words as they appear in that block (for a bracket, the primary words; without braces)", "new": "replacement words, lowercase, no punctuation", "confidence": "high" | "medium", "reason": "short reason"}. Keep "old" spans short (1-12 words) and copy them exactly."""
# Corpus run from 2026-10-21 on (7l): the reason is one word, which cuts the output by about a quarter.
REASON_CODES = ('"reason": one word for the main evidence: "whisper" (the other model\'s version), "nara" (the NARA '
                'record), "diary" (Haldeman\'s diary), "day" (another conversation of the day), "context" (this '
                'conversation\'s own flow), "knowledge" (knowledge of the period), or "sound" (an obvious mishearing)}')
SHORT_REASONS = True


def rules(day: bool) -> str:
    r = RULES_DAY if day else RULES
    return r.replace('"reason": "short reason"}', REASON_CODES) if SHORT_REASONS else r


# --- by-day requests (7k): a day's conversations in time order, the diary sent once ------------------------------
INTRO_DAY = (
    "You are correcting machine transcripts of the Nixon White House tapes recorded on one day (1971-73; room "
    "microphones and telephone taps, often poor). The speech recognizer is often wrong, above all on names, places, "
    "bills, organizations and any unfamiliar word, which it tends to turn into one or more ordinary words that sound "
    "alike: \"para leans\" in a talk about birds is \"parakeets\", \"early man\" is \"ehrlichman\", \"clean dienst\" is "
    "\"kleindienst\". The day's conversations are given in time order, each under a header === id | place | time ===; "
    "each transcript is lowercase and unpunctuated, in ~60 s blocks starting with [mm:ss] (time within that conversation). "
    "Conversations of one day share people and topics: a name heard clearly in one can show what a garbled stretch in "
    "another is. You also have NARA's record of each conversation (participants, subject log) and H. R. Haldeman's "
    "dictated diary for that day and the next; he was Nixon's chief of staff and wrote about the people, meetings and "
    "issues of each day, though he was not at every meeting.\n")
RULES_DAY = (RULES.replace("using the conversation's own flow, the NARA record and the diary",
                           "using each conversation's own flow, the day's other conversations, the NARA records and the diary")
             .replace('Each element is {"block": "mm:ss" (the block\'s start time)',
                      'Each element is {"conv": "the conversation id from its header", "block": "mm:ss" (the block\'s '
                      'start time within that conversation)'))
DIARY_HEAD = "\nH. R. Haldeman's dictated diary"


def day_prompt(convs: list[tuple[str, str, str]], diary: str) -> str:
    """convs: (header, transcript, NARA record) in time order; diary: the day's diary section, sent once."""
    body = "\n".join(f"=== {h} ===\n{t}" for h, t, _ in convs)
    nara = "\n".join(f"--- NARA record, {h.split(' | ')[0]} ---\n{r.strip()}\n" for h, _, r in convs)
    return (INTRO_DAY + WHISPER + CONF + "\n<transcripts>\n" + body + "\n</transcripts>\n\n<context>\n" + nara +
            diary + "\n</context>\n\n" + rules(day=True))


def prompt(transcript: str, context: str, part: tuple[int, int] | None = None) -> str:
    note = (f"This is part {part[0]} of {part[1]} of a long conversation; correct only this part.\n"
            if part and part[1] > 1 else "")
    return (INTRO + WHISPER + CONF + note + "\n<transcript>\n" + transcript + "\n</transcript>\n\n<context>\n" + context +
            "\n</context>\n\n" + rules(day=False))


def n_words(text: str) -> int:
    """Transcript words, both models' (markup and block times not counted)."""
    return len(re.sub(r"^\[\d+:\d+\]|\||whisper:|[\[\]{}]", " ", text, flags=re.M).split())


def parts(transcript: str) -> list[str]:
    """Split at block (line) boundaries into pieces of at most PART_WORDS words."""
    out, cur, n = [], [], 0
    for line in transcript.splitlines():
        w = n_words(line)
        if cur and n + w > PART_WORDS:
            out.append("\n".join(cur) + "\n")
            cur, n = [], 0
        cur.append(line)
        n += w
    return out + (["\n".join(cur) + "\n"] if cur else [])


def request(text: str) -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}


def key() -> str:
    return (SECRETS_DIR / "gemini_key.txt").read_text(encoding="utf-8").strip()


def cmd_requests(root: Path, convs: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_req = chars = words = tchars = 0
    with open(out / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for conv in convs:
            d = root / "out" / conv
            transcript = (d / "correct_input.txt").read_text(encoding="utf-8")
            if not transcript.strip():
                continue
            ctx = (d / "correct_context.txt").read_text(encoding="utf-8")
            ps = parts(transcript)
            for i, p in enumerate(ps, 1):
                text = prompt(p, ctx, (i, len(ps)))
                f.write(json.dumps({"key": f"{conv}#{i}of{len(ps)}", "request": request(text)}, ensure_ascii=False) + "\n")
                n_req += 1
                chars += len(text)
                tchars += len(p)
            words += len(transcript.split())
    write_estimate(out, len(convs), n_req, chars, words, tchars)


def write_estimate(out: Path, tapes: int, n_req: int, chars: int, words: int, transcript_chars: int | None = None) -> None:
    tin = chars / CHARS_PER_TOKEN
    # output follows the transcript, not the context: 0.40 output tokens per transcript token on the long 7i
    # tapes, 1.46x that on the corpus pilot (short tapes draw more edits per word): 0.584
    tout = (transcript_chars / CHARS_PER_TOKEN * OUT_PER_TRANSCRIPT) if transcript_chars is not None else tin * OUT_PER_IN
    est = {"tapes": tapes, "requests": n_req, "transcript_words": words, "input_tokens": round(tin),
           "output_tokens": round(tout), "usd_batch": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 2)}
    (out / "estimate.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
    print(json.dumps(est))


def cmd_requests_by_day(root: Path, convs: list[str], out: Path, solo_above: int | None = None) -> None:
    """One request per day (or per run of consecutive conversations up to PART_WORDS words); a conversation longer
    than that gets requests of its own, one per part. With solo_above, a conversation of at least that many words
    gets a request of its own too (hybrid: the long room conversations alone, as tested in 7i; short calls grouped).
    groups.json maps each request key to its conversations."""
    import sqlite3
    from paths import CORPUS_DB
    db = sqlite3.connect(CORPUS_DB)
    meta = {r[0].removeprefix("37-wht-conversation-"): r[1:] for r in
            db.execute("SELECT conversation_id, date, start_time, end_time, location FROM conversations")}
    days: dict[str, list] = {}
    for conv in convs:
        d = root / "out" / conv
        t = (d / "correct_input.txt").read_text(encoding="utf-8")
        if not t.strip():
            continue
        ctx = (d / "correct_context.txt").read_text(encoding="utf-8")
        i = ctx.find(DIARY_HEAD)
        iso, start, end, loc = meta.get(conv, ("", "", "", ""))
        if not iso:  # 8 conversations are missing from the manifest: take the date from the NARA context line
            import datetime
            m = re.match(r"Conversation \S+, (.+?), location code (\S+)\.", ctx)
            try:
                iso = datetime.datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%dT00:00") if m else ""
                loc = loc or (m.group(2) if m else "")
            except ValueError:
                iso = ""
            if not iso:
                print(f"{conv}: no date; left out")
                continue
        head = f"{conv} | {loc or 'unknown place'} | {start or '?'}-{end or '?'}"
        days.setdefault(iso[:10], []).append((iso, conv, head, t, ctx[:i], ctx[i:]))
    out.mkdir(parents=True, exist_ok=True)
    groups, n_req, chars, tchars, words = {}, 0, 0, 0, 0
    with open(out / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for day, items in sorted(days.items()):
            items.sort()
            chunks, cur, n = [], [], 0
            for iso, conv, head, t, nara, diary in items:
                w = n_words(t)
                ps = parts(t) if w > PART_WORDS else [t]
                if len(ps) > 1 or (solo_above and w >= solo_above):  # a long conversation: its own request(s)
                    if cur:
                        chunks.append(cur); cur, n = [], 0
                    chunks += [[(conv, f"{head} | part {k} of {len(ps)}, correct only this part" if len(ps) > 1 else head, p, nara)]
                               for k, p in enumerate(ps, 1)]
                    continue
                if cur and n + w > PART_WORDS:
                    chunks.append(cur); cur, n = [], 0
                cur.append((conv, head, t, nara)); n += w
            if cur:
                chunks.append(cur)
            diary = items[0][5]
            for k, ch in enumerate(chunks, 1):
                key = f"{day}#{k}of{len(chunks)}"
                text = day_prompt([(h, t, r) for _, h, t, r in ch], diary)
                f.write(json.dumps({"key": key, "request": request(text)}, ensure_ascii=False) + "\n")
                groups[key] = {"convs": [c for c, *_ in ch], "words": [n_words(t) for _, _, t, _ in ch]}
                n_req += 1; chars += len(text); tchars += sum(len(t) for _, _, t, _ in ch)
            words += sum(len(it[3].split()) for it in items)
    (out / "groups.json").write_text(json.dumps(groups, indent=0), encoding="utf-8")
    write_estimate(out, sum(len(v) for v in days.values()), n_req, chars, words, tchars)


def parse_edits(resp: dict) -> tuple[list | None, str, dict]:
    """(edits or None, raw text, usage) from a GenerateContentResponse."""
    cands = resp.get("candidates") or []
    text = "".join(p.get("text", "") for c in cands[:1] for p in (c.get("content") or {}).get("parts", []) if not p.get("thought"))
    usage = resp.get("usageMetadata") or {}
    # The model sometimes thinks aloud before the list, with brackets in the prose ("should I return `[]`?"),
    # so take the LAST complete JSON list in the text rather than everything from the first "[".
    dec, found = json.JSONDecoder(), None
    for i in [m.start() for m in re.finditer(r"\[", text)]:
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, list) and all(isinstance(e, dict) for e in obj):
            found = obj
    if found is None:  # invalid JSON, e.g. unescaped quotes inside a "reason": read each edit's fields one by one
        found = lenient_edits(text) or None
    if found is None:
        return None, text, usage
    return [e for e in found if "old" in e and "new" in e and "block" in e], text, usage


OBJ_START = re.compile(r'\{\s*"(?:conv|block)"\s*:')


def lenient_edits(text: str) -> list[dict]:
    """Edits from a list that is not valid JSON: split at each object's start and read the short fields (which hold
    no quotes) by name, in any order; the reason is whatever lies between its key and the object's end."""
    starts = [m.start() for m in OBJ_START.finditer(text)]
    out = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        seg = text[a:b]
        e = {k: m.group(1) for k in ("conv", "block", "old", "new", "confidence")
             if (m := re.search(rf'"{k}"\s*:\s*"([^"]*)"', seg))}
        if m := re.search(r'"reason"\s*:\s*"(.*)"\s*\}', seg, re.S):
            e["reason"] = m.group(1)
        if {"block", "old", "new"} <= e.keys():
            out.append({**e, "parsed": "lenient"})
    return out


def save_results(root: Path, results: dict[str, dict], groups: dict | None = None, model: str = MODEL) -> None:
    """results: request key -> GenerateContentResponse or {"error": ...}; writes per-tape files.
    With groups (by-day requests), each edit goes to the conversation it names; a request's tokens are shared
    among its conversations by transcript words. Edits naming no conversation of their request are counted.
    model: recorded in argon_usage.json (deepseek.py passes the DeepSeek model it used)."""
    if groups is not None:
        save_day_results(root, results, groups, model)
        return
    by_conv: dict[str, dict] = {}
    for k, r in results.items():
        conv, part = k.split("#")
        by_conv.setdefault(conv, {})[part] = r
    for conv, ps in sorted(by_conv.items()):
        d = root / "out" / conv
        edits, usage, failed = [], {"input": 0, "output": 0, "thoughts": 0}, []
        for part in sorted(ps, key=lambda p: int(p.split("of")[0])):
            r = ps[part]
            if "error" in r:
                failed.append({"part": part, "error": r["error"]})
                continue
            e, text, u = parse_edits(r)
            usage["input"] += u.get("promptTokenCount", 0)
            usage["output"] += u.get("candidatesTokenCount", 0)
            usage["thoughts"] += u.get("thoughtsTokenCount", 0) or 0
            if e is None:
                (d / f"argon_raw_{part}.txt").write_text(text, encoding="utf-8")
                failed.append({"part": part, "error": "response is not a JSON edit list",
                               "finish": ((r.get("candidates") or [{}])[0]).get("finishReason")})
                continue
            edits += e
        (d / "edits_argon.json").write_text(json.dumps(edits, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "argon_usage.json").write_text(json.dumps({"model": model, "parts": len(ps), "usage": usage, "failed": failed}), encoding="utf-8")


def save_day_results(root: Path, results: dict[str, dict], groups: dict, model: str = MODEL) -> None:
    per: dict[str, dict] = {}
    unassigned = []
    for k, g in groups.items():
        convs, words = g["convs"], g["words"]
        for c in convs:
            per.setdefault(c, {"edits": [], "usage": {"input": 0, "output": 0, "thoughts": 0}, "failed": [], "parts": 0})
            per[c]["parts"] += 1
        r = results.get(k, {"error": "no response for this request"})
        if "error" in r:
            for c in convs:
                per[c]["failed"].append({"part": k, "error": r["error"]})
            continue
        e, text, u = parse_edits(r)
        total = max(sum(words), 1)
        for c, w in zip(convs, words):
            for name, field in (("input", "promptTokenCount"), ("output", "candidatesTokenCount"), ("thoughts", "thoughtsTokenCount")):
                per[c]["usage"][name] += round((u.get(field, 0) or 0) * w / total)
        if e is None:
            (root / f"argon_raw_{k.replace('#', '_')}.txt").write_text(text, encoding="utf-8")
            for c in convs:
                per[c]["failed"].append({"part": k, "error": "response is not a JSON edit list"})
            continue
        for x in e:
            named = str(x.get("conv", "")).strip().removeprefix("37-wht-conversation-")
            c = convs[0] if len(convs) == 1 else named if named in convs else None
            if c is None:
                unassigned.append({"request": k, **x})
            else:
                per[c]["edits"].append(x)
    for c, v in per.items():
        d = root / "out" / c
        (d / "edits_argon.json").write_text(json.dumps(v["edits"], ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "argon_usage.json").write_text(json.dumps({"model": model, "parts": v["parts"], "usage": v["usage"],
                                                        "failed": v["failed"], "by_day": True}), encoding="utf-8")
    (root / "argon_unassigned.json").write_text(json.dumps(unassigned, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(per)} conversations; {len(unassigned)} edits named no conversation of their request")


def cmd_submit(bdir: Path) -> None:
    import requests as rq
    data = (bdir / "requests.jsonl").read_bytes()
    h = {"x-goog-api-key": key()}
    start = rq.post(f"{API}/upload/v1beta/files", timeout=120, json={"file": {"display_name": f"nixon-argon-{bdir.name}"}},
                    headers={**h, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                             "X-Goog-Upload-Header-Content-Length": str(len(data)),
                             "X-Goog-Upload-Header-Content-Type": "application/jsonl"})
    start.raise_for_status()
    url = start.headers["X-Goog-Upload-URL"]
    up = rq.post(url, data=data, timeout=1800, headers={"Content-Length": str(len(data)), "X-Goog-Upload-Offset": "0",
                                                         "X-Goog-Upload-Command": "upload, finalize"})
    up.raise_for_status()
    fname = up.json()["file"]["name"]
    r = rq.post(f"{API}/v1beta/models/{MODEL}:batchGenerateContent", headers=h, timeout=120,
                json={"batch": {"display_name": f"nixon-argon-{bdir.name}", "input_config": {"file_name": fname}}})
    if r.status_code != 200:
        print("create failed:", r.status_code, r.text[:800]); r.raise_for_status()
    job = r.json()
    (bdir / "batch.json").write_text(json.dumps({"input_file": fname, "job": job}, indent=1), encoding="utf-8")
    print(f"uploaded {len(data) / 1e6:.1f} MB as {fname}; batch {job.get('name')}")


def get_job(bdir: Path) -> dict:
    import requests as rq
    name = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))["job"]["name"]
    r = rq.get(f"{API}/v1beta/{name}", headers={"x-goog-api-key": key()}, timeout=120)
    r.raise_for_status()
    return r.json()


def job_state(j: dict) -> str:
    return (j.get("metadata") or {}).get("state") or j.get("state") or ("done" if j.get("done") else "?")


def cmd_status(bdir: Path) -> None:
    j = get_job(bdir)
    meta = j.get("metadata") or {}
    print(f"{job_state(j)}; stats {meta.get('batchStats') or meta.get('batch_stats')}; done {j.get('done', False)}")


def download_results(bdir: Path) -> bool:
    """Download the batch's results file to <batch dir>/results.jsonl; False if there is none yet."""
    import requests as rq
    j = get_job(bdir)
    resp = j.get("response") or (j.get("metadata") or {}).get("output") or {}
    fname = resp.get("responsesFile") or (resp.get("output") or {}).get("responsesFile")
    if not fname:
        print("no results file yet:", job_state(j), json.dumps(j)[:600])
        return False
    r = rq.get(f"{API}/download/v1beta/{fname}:download", params={"alt": "media"}, headers={"x-goog-api-key": key()}, timeout=1800)
    r.raise_for_status()
    (bdir / "results.jsonl").write_bytes(r.content)
    return True


def failed_share(bdir: Path) -> tuple[int, int]:
    """(failed, total) responses in results.jsonl: a request that came back without candidates (quota, block, error)."""
    rows = [json.loads(l) for l in (bdir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return sum(1 for x in rows if not (x.get("response") or {}).get("candidates")), len(rows)


def cmd_fetch(bdir: Path, root: Path, local: bool = False) -> None:
    """Download the results file (or, with local, re-read the copy saved by an earlier fetch) and save per tape."""
    if not local and not download_results(bdir):
        return
    results = {}
    for line in (bdir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            x = json.loads(line)
            results[x["key"]] = x.get("response") or {"error": x.get("error") or x.get("status") or "no response"}
    save_results(root, results, load_groups(bdir))
    print(f"fetched {len(results)} responses for {len({k.split('#')[0] for k in results})} tapes")


def load_groups(bdir: Path) -> dict | None:
    f = bdir / "groups.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def cmd_direct(bdir: Path, root: Path, workers: int) -> None:
    import requests as rq
    from concurrent.futures import ThreadPoolExecutor
    reqs = [json.loads(l) for l in (bdir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    k = key()

    def call(x):
        for attempt in range(6):
            try:
                r = rq.post(f"{API}/v1beta/models/{MODEL}:generateContent", json=x["request"], timeout=1200,
                            headers={"x-goog-api-key": k, "Content-Type": "application/json"})
                if r.status_code == 200:
                    return x["key"], r.json()
                if r.status_code not in (429, 500, 503):
                    return x["key"], {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
            except rq.RequestException as e:
                err = str(e)
            time.sleep(15 * (attempt + 1))
        return x["key"], {"error": "gave up after retries"}

    with ThreadPoolExecutor(workers) as ex:
        results = dict(ex.map(call, reqs))
    save_results(root, results, load_groups(bdir))
    print(f"{len(results)} responses, {sum('error' in r for r in results.values())} errors")


def cmd_report(root: Path, convs: list[str]) -> None:
    from correct import apply_edits
    tot = dict(tapes=0, words=0, edits=0, applied=0, unlocated=0, unclear=0, unclear_words=0, changed=0,
               tin=0, tout=0, failed_parts=0)
    worst = []
    for conv in convs:
        d = root / "out" / conv
        if not (d / "argon_usage.json").exists():
            continue
        u = json.loads((d / "argon_usage.json").read_text(encoding="utf-8"))
        merged = json.loads((d / "merged.json").read_text(encoding="utf-8"))
        edits = json.loads((d / "edits_argon.json").read_text(encoding="utf-8"))
        for e in edits:
            e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
        starts = [int(a) * 60 + int(b) for a, b in re.findall(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
        words, applied, missing = apply_edits(merged, edits, starts)
        changed = sum(x.get("src") == "llm" for x in words)
        unc = [e for e in edits if "unclear" in str(e.get("new", ""))]
        tot["tapes"] += 1; tot["words"] += len(merged); tot["edits"] += len(edits); tot["applied"] += applied
        tot["unlocated"] += missing; tot["unclear"] += len(unc); tot["unclear_words"] += sum(len(e["old"].split()) for e in unc)
        tot["changed"] += changed; tot["failed_parts"] += len(u["failed"])
        tot["tin"] += u["usage"]["input"]; tot["tout"] += u["usage"]["output"] + u["usage"]["thoughts"]
        share = len([e for e in edits]) and sum(len(e["old"].split()) for e in edits) / max(len(merged), 1)
        worst.append((share, conv, len(merged), len(edits), missing, len(u["failed"])))
    t = tot
    if not t["tapes"]:
        print("no fetched tapes"); return
    cost = t["tin"] / 1e6 * PRICE_IN + t["tout"] / 1e6 * PRICE_OUT
    print(f"{t['tapes']} tapes, {t['words']:,} words; {t['edits']:,} edits, {t['applied']:,} applied, {t['unlocated']:,} not located "
          f"({t['unlocated'] / max(t['edits'], 1):.1%}); [unclear] {t['unclear']} stretches, {t['unclear_words']} words; "
          f"failed parts {t['failed_parts']}")
    print(f"tokens {t['tin']:,} in / {t['tout']:,} out (incl. thinking); ${cost:.2f} at the batch price; "
          f"${cost / t['words'] * 1e6:.2f} per million transcript words")
    print("most-edited tapes (share of words inside edited spans):")
    for share, conv, n, ne, miss, fp in sorted(worst, reverse=True)[:8]:
        print(f"  {conv}: {share:.1%} of {n} words in {ne} edits, {miss} unlocated, {fp} failed parts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["requests", "submit", "status", "fetch", "direct", "report"])
    ap.add_argument("path", type=Path, help="requests/report: the prep root; others: the batch dir")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--convs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--local", action="store_true", help="fetch: re-read the saved results.jsonl instead of downloading")
    ap.add_argument("--by-day", action="store_true", help="requests: group each day's conversations, diary sent once")
    ap.add_argument("--solo-above", type=int, help="requests --by-day: conversations of at least this many words get their own request")
    args = ap.parse_args()
    convs = [l.strip() for l in args.convs.read_text(encoding="utf-8").splitlines() if l.strip()] if args.convs else []
    if args.cmd == "requests":
        if args.by_day:
            cmd_requests_by_day(args.path, convs, args.out, args.solo_above)
        else:
            cmd_requests(args.path, convs, args.out)
    elif args.cmd == "submit":
        cmd_submit(args.path)
    elif args.cmd == "status":
        cmd_status(args.path)
    elif args.cmd == "fetch":
        cmd_fetch(args.path, args.root, args.local)
    elif args.cmd == "direct":
        cmd_direct(args.path, args.root, args.workers)
    else:
        cmd_report(args.path, convs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
