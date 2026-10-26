"""Held-out test of the released corpus on FRUS tape transcripts (METHODS 7o).

The State Department's Foreign Relations of the United States (FRUS) series prints transcripts that its
editors made from the tapes "specifically for this volume". None of them took part in any decision: they
were not training data, not used to choose a model, a learning rate, a prompt or a threshold, and not given
to the corrector. (FRUS text was one of the sources of the era 6-gram language model, METHODS 7j, which was
not adopted, so it is in no released layer.)

  refs   every FRUS document whose source note says the editors transcribed a tape and that cites exactly
         one conversation -> <out>/refs/<conv>_frus.json, in the validation-reference format (turns with
         speaker, text and segment; a new segment at every editorial omission, so each printed excerpt is
         aligned on its own). Documents printing the same conversation are merged; turns another document
         already printed are dropped. -> <out>/documents.tsv
  score  excludes conversations that took part in training, validation, listening or are parallel
         recordings of those (-> exclusions.tsv), scores four layers of each remaining conversation
         (whisper, parakeet_v5, merged, corrected) with score.py's alignment, pools the counts over
         conversations with bootstrap 95% intervals, breaks them down by location, and tests the speaker
         names (speakers_v4) on conversations outside the voiceprint classifier's training set.
         -> per_conv.tsv, results.json

    python src/frus_test.py refs
    python src/frus_test.py score [--workers 12] [--boot 10000]
    python src/frus_test.py show <conv> [--layer corrected]       # eyeball an alignment
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import jiwer

from paths import DATA_DIR, VALIDATION_DIR
from score import SHORT_TURN, locate, norm_words, ref_name_set
from score_lmsweep import SMALL

OUT = DATA_DIR / "frus_test"
FRUS = DATA_DIR / "lm" / "sources" / "frus"
CORPUS = DATA_DIR / "corpus_v5" / "out"
NIXON_DB = DATA_DIR / "corpus" / "nixon.db"
TEI = "{http://www.tei-c.org/ns/1.0}"
CONV_RE = re.compile(r"Conversation (?:No\.\s*)?(\d{1,3})\s*[–-]\s*(\d{1,3})\b")
# The source note must say the editors transcribed what is printed (not a tape quoted in a note of a memo):
# "The editors transcribed the portions of the tape recording printed here specifically for this volume."
TRANSCRIBED = re.compile(
    r"editors?\s+transcribed\s+(the\s+|this\s+|these\s+)?(portions?\s+(of\s+)?)?(the\s+)*"
    r"(tape\s+recording|conversation|discussion|portions?|nixon.s\s+remarks)?[^.;]{0,40}?(printed|published|specifically)"
    r"|(conversation|discussion|portions?)\s+transcribed\s+(here|was|are)", re.I)
# "Nixon: ...", "Kissinger: ...", "Unidentified voice: ...". At most four words before the colon.
SPEAKER = re.compile(r"^\s*\[?([A-Z][A-Za-z.’'\-]*(?:\s+[A-Za-z.’'\-]+){0,3})\]?\s*:\s+(.*)$", re.S)
OMISSION = re.compile(r"\[[^\]]*(omitted|not declassified|less than \d|lines? of source text|pages? of source text)[^\]]*\]", re.I)
LAYERS = {"whisper": "whisper.json", "parakeet_v5": "parakeet_v5.json", "merged": "merged.json",
          "corrected": "corrected.json"}
PLACES = {"OVAL": "Oval Office", "EOB": "Executive Office Building", "WHT": "White House telephone",
          "CDHW": "Camp David", "CDST": "Camp David", "CDSD": "Camp David", "CAB": "Cabinet Room"}
DEV = ["429-018", "538-015", "856-004", "862-006", "903-006"]  # fine-tuning dev conversations (METHODS 4)
CLASSES = {"Nixon", "Kissinger", "Haldeman", "Ehrlichman", "Colson", "Dean", "Ziegler", "Mitchell", "Rogers",
           "Shultz", "Haig", "Woods", "Connally", "Butterfield"}  # voices of speaker classifier v4 (METHODS 5)
VAL_ROOM = {"886-008": "886-008_wspf", "488-015": "488-015_nara", "779-002": "779-002_wspf",
            "421-018": "421-018_wspf", "051-001": "051-001_wspf"}  # the five room validation tapes
# Files that record what the listener heard, labeled or corrected (METHODS 7d, 7f, 7g) and the listener's
# voiceprint spans (METHODS 5): every conversation named in them is excluded.
LISTENER_FILES = ["listening/session/labels.jsonl", "listening/session/clips.json",
                  "listening/session/listen_train.jsonl", "listening/session/listen_test.jsonl",
                  "listening/session/listen_train_v1.jsonl", "listening/session/listen_test_v1.jsonl",
                  "listening/convs.txt", "listening/convs2.txt", "listening/jobs.jsonl", "listening/jobs2.jsonl",
                  "listening/jobs_all48.jsonl", "listening/jobs_pilot.jsonl",
                  "correct1/session/labels.jsonl", "correct1/session/session_full30.json",
                  "correct1/triage/labels.jsonl", "correct1/triage/session.json",
                  "correct1/v4data/train.jsonl", "correct1/v4data/test.jsonl",
                  "correct1/v5data/speech.jsonl", "correct1/v5data/crosstalk.jsonl",
                  "correct1/v5data/train_silence.jsonl", "kissinger/listener/segments.jsonl",
                  "round5k/heldout_all.txt", "round5k/listen_test.jsonl"]
ID_RE = re.compile(r"(?<![\d-])(\d{3}-\d{3})(?![\d-])")


def conv_id(tape: str, num: str) -> str:
    return f"{int(tape):03d}-{int(num):03d}"


def text_of(el) -> str:
    """Text of an element without its notes (footnotes, source notes), keeping the text after them."""
    parts = [el.text or ""]
    for ch in el:
        if ch.tag != TEI + "note":
            parts.append(text_of(ch))
        parts.append(ch.tail or "")
    return "".join(parts)


def paragraphs(el, in_note: bool = False):
    """<p> elements of a document that are not inside a note or the heading."""
    for ch in el:
        if ch.tag in (TEI + "note", TEI + "head"):
            continue
        if ch.tag == TEI + "p":
            yield ch
        else:
            yield from paragraphs(ch)


HEADER_LABELS = {"Conversation", "Date", "Time", "Location", "Participants", "Source", "Note", "Subject", "Place"}
ALIASES = {"President": "Nixon", "Kissigner": "Kissinger", "Macarthur": "MacArthur"}  # FRUS typos and forms


def surname(label: str) -> str:
    label = label.strip(" .[]")
    if re.match(r"(unidentified|unknown|several|both|all|voices?)\b", label, re.I):
        return "Unidentified"
    name = re.sub(r"[’']s$", "", label.split()[-1])
    return ALIASES.get(name, name)


def parse_document(el) -> tuple[list[dict], dict]:
    """Turns of one transcript document; segment numbers advance at every editorial omission."""
    turns, seg, spk, info = [], 0, None, Counter()
    for p in paragraphs(el):
        text = re.sub(r"\s+", " ", text_of(p)).strip()
        if not text:
            continue
        pieces = OMISSION.split(text)  # [before, kind, after, kind, ...]
        for i, piece in enumerate(pieces[::2]):
            if i:
                seg += 1
                info["omissions"] += 1
            piece = piece.strip()
            if not piece:
                continue
            if re.fullmatch(r"\[[^\]]*\]|…|\.\.\.|\. \. \.", piece):  # standalone editorial line or ellipsis
                if "…" in piece or "..." in piece or ". . ." in piece:
                    seg += 1
                    info["omissions"] += 1
                continue
            m = SPEAKER.match(piece) if i == 0 else None
            if m and m.group(1).strip() in HEADER_LABELS:  # a header block ("Date: ...", "Participants: ...")
                info["header_lines"] += 1
                continue
            if m and not re.search(r"\d", m.group(1)):
                spk = surname(m.group(1))
                info["speaker_paragraphs"] += 1
                piece = m.group(2)
            elif spk is None:  # editorial text before the first speaker
                info["preamble"] += 1
                continue
            turns.append({"speaker": spk, "text": piece, "segment": seg})
        info["paragraphs"] += 1
    return turns, info


def cmd_refs() -> None:
    db = sqlite3.connect(NIXON_DB)
    meta = {r[0].split("-", 3)[-1]: r[1:] for r in db.execute(
        "SELECT conversation_id, date, start_time, end_time, location, participants FROM conversations")}
    docs, by_conv = [], defaultdict(list)
    for f in sorted(FRUS.glob("frus1969-76*.xml")):
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == TEI + "div" and el.get("type") == "document":
                src = " ".join("".join(n.itertext()) for n in el.iter(TEI + "note") if n.get("type") == "source")
                src = re.sub(r"\s+", " ", src)
                if TRANSCRIBED.search(src) and CONV_RE.search(src):
                    convs = sorted({conv_id(*m.groups()) for m in CONV_RE.finditer(src)})
                    turns, info = parse_document(el)
                    head = re.sub(r"\s+", " ", text_of(el.find(TEI + "head")) if el.find(TEI + "head") is not None else "")
                    row = {"doc": f"{f.stem}#{el.get('{http://www.w3.org/XML/1998/namespace}id')}",
                           "convs": convs, "head": head.strip()[:160], "turns": len(turns),
                           "words": sum(len(t["text"].split()) for t in turns), **info}
                    if len(convs) > 1:
                        row["status"] = "skipped: cites several conversations"
                    elif info["speaker_paragraphs"] < 5:
                        row["status"] = "skipped: not a dialogue transcript (<5 speaker lines)"
                    else:
                        row["status"] = "used"
                        by_conv[convs[0]].append((row["doc"], turns))
                    docs.append(row)
                el.clear()
    refs = OUT / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    for old in refs.glob("*.json"):
        old.unlink()
    for conv, items in sorted(by_conv.items()):
        items.sort(key=lambda x: -sum(len(t["text"].split()) for t in x[1]))  # longest document first
        seen, merged, seg_base = set(), [], 0
        for doc, turns in items:
            # Flag turns another document already printed (half or more of their 6-word sequences seen).
            flags = []
            for t in turns:
                w = norm_words(t["text"])
                grams = {tuple(w[i:i + 6]) for i in range(len(w) - 5)}
                flags.append(None if not grams else len(grams & seen) / len(grams) >= 0.5)
            for i, fl in enumerate(flags):  # short turns follow the nearest long turn before them
                if fl is None:
                    prev = next((flags[j] for j in range(i - 1, -1, -1) if flags[j] is not None), None)
                    flags[i] = prev if prev is not None else next((x for x in flags[i:] if x is not None), False)
            seg, last = seg_base, None
            for t, dup in zip(turns, flags):
                if dup:
                    last = None
                    continue
                if last is None or t["segment"] != last:
                    seg += 1
                last = t["segment"]
                merged.append({**t, "segment": seg, "doc": doc})
            seg_base = seg + 1
            for t in turns:
                w = norm_words(t["text"])
                seen |= {tuple(w[i:i + 6]) for i in range(len(w) - 5)}
        m = meta.get(conv)
        out = {"name": f"{conv}_frus", "conversation_id": f"37-wht-conversation-{conv}",
               "source": "; ".join(d for d, _ in items), "note": PLACES.get(m[3], m[3]) if m else "",
               "coverage": "excerpts", "date": m[0] if m else "", "start": m[1] if m else "",
               "end": m[2] if m else "", "location": m[3] if m else "", "participants": m[4] if m else "",
               "words": sum(len(t["text"].split()) for t in merged), "turns": merged}
        (refs / f"{conv}_frus.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "documents.tsv", "w", encoding="utf-8") as fh:
        cols = ["doc", "convs", "status", "turns", "words", "speaker_paragraphs", "omissions", "preamble", "head"]
        fh.write("\t".join(cols) + "\n")
        for r in docs:
            fh.write("\t".join(str(",".join(r[c]) if c == "convs" else r.get(c, "")) for c in cols) + "\n")
    st = Counter(r["status"] for r in docs)
    print(f"{len(docs)} FRUS documents transcribe a cited tape: {dict(st)}; {len(by_conv)} conversations -> {refs}")
    spk = Counter(t["speaker"] for f in refs.glob("*.json")
                  for t in json.loads(f.read_text(encoding="utf-8"))["turns"])
    print("speakers:", spk.most_common(40))


# ---------------------------------------------------------------- exclusions

def minutes(clock: str) -> int | None:
    try:
        t = datetime.strptime(clock.strip().upper(), "%I:%M %p")
        return t.hour * 60 + t.minute
    except (ValueError, AttributeError):
        return None


def exclusions(candidates: list[str]) -> dict[str, str]:
    """Reason to exclude each candidate conversation ('' = keep)."""
    val = {re.match(r"(\d{3}-\d{3})", p.name).group(1) for p in (VALIDATION_DIR / "refs").glob("*.json")}
    train = {p.stem for p in (DATA_DIR / "training" / "transcripts").glob("*.json")}
    listener = set()
    for rel in LISTENER_FILES:
        p = DATA_DIR / rel
        if p.exists():
            listener |= set(ID_RE.findall(p.read_text(encoding="utf-8", errors="replace")))
    listener |= {re.match(r"(\d{3}-\d{3})", p.name).group(1) for p in (DATA_DIR / "correct1" / "drafts").glob("*.json")}
    used = val | train | set(DEV) | listener
    db = sqlite3.connect(NIXON_DB)
    when = {}
    for cid, date, st, en in db.execute("SELECT conversation_id, date, start_time, end_time FROM conversations"):
        a, b = minutes(st), minutes(en)
        if date and a is not None:
            when[cid.split("-", 3)[-1]] = (date[:10], a, b if b is not None and b >= a else a)
    used_when = [(c, *when[c]) for c in used if c in when]
    import csv
    tier = {r["conv_id"]: r["tier"] for r in csv.DictReader(open(DATA_DIR / "manifest" / "audio_manifest.csv", encoding="utf-8"))}
    reasons = {}
    for c in candidates:
        if c in val:
            r = "validation conversation"
        elif c in DEV:
            r = "fine-tuning dev conversation"
        elif c in train:
            r = "WSPF training conversation"
        elif c in listener:
            r = "listener heard, labeled or corrected it"
        else:
            r = ""
            if c in when:
                d, a, b = when[c]
                for u, ud, ua, ub in used_when:
                    if ud == d and u != c and a <= ub and ua <= b:
                        r = f"parallel recording of {u} (same date, overlapping time)"
                        break
        if not r:
            files = {k: CORPUS / c / v for k, v in LAYERS.items()}
            missing = [k for k, f in files.items() if not f.exists()]
            if missing and len(missing) < len(files):
                r = f"missing layer(s): {','.join(missing)}"
            elif missing and tier.get(c, "").startswith("D_"):
                r = "not in the corpus: released only as a whole-tape file"
            elif missing:
                r = "not in the corpus: not in the job list"
        reasons[c] = r
    return reasons


# ---------------------------------------------------------------- scoring

@lru_cache(maxsize=None)
def norm_token(w: str) -> tuple[str, ...]:
    return tuple(norm_words(w))


def layer_words(path: Path) -> tuple[list[str], list[int], list[dict]]:
    """Normalised hypothesis words, the index of the source word each came from, and the source words."""
    d = json.loads(path.read_text(encoding="utf-8"))
    words = d["words"] if isinstance(d, dict) else d
    hyp, src = [], []
    for i, w in enumerate(words):
        for t in norm_token(w["w"]):
            hyp.append(t)
            src.append(i)
    return hyp, src, words


def speaker_at(segs: list[dict], starts: list[float], t: float) -> str:
    from bisect import bisect_right
    i = bisect_right(starts, t) - 1
    if i >= 0 and segs[i]["start"] - 0.05 <= t <= segs[i]["end"] + 0.25:
        return segs[i]["speaker"]
    return ""


def score_conv(args) -> dict:
    """Raw counts for every layer of one conversation, plus the speaker comparison on the corrected layer."""
    conv, ref_path = args
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    units = [(t["segment"], t["text"], t["speaker"]) for t in ref["turns"]]
    names = ref_name_set([t for _, t, _ in units])
    out = {"conv": conv, "location": ref.get("location", ""), "date": ref.get("date", ""), "layers": {}}
    spk_segs = []
    sp = CORPUS / conv / "speakers_v4.json"
    if sp.exists():
        spk_segs = sorted(json.loads(sp.read_text(encoding="utf-8")), key=lambda s: s["start"])
    spk_starts = [s["start"] for s in spk_segs]
    for layer, fname in LAYERS.items():
        hyp, src, words = layer_words(CORPUS / conv / fname)
        c = Counter()
        spk_rows = Counter()
        inside, hit_src, name_src = set(), set(), set()  # source words in scored stretches / matching the reference / a name
        for seg in sorted({s for s, _, _ in units}):
            full, owner = [], []
            for s, t, who in units:
                if s == seg:
                    w = norm_words(t)
                    full += w
                    owner += [(who, len(w) <= SHORT_TURN)] * len(w)
            if not full:
                continue
            ra, rb, a, b = locate(full, hyp)
            c["ref_total"] += len(full)
            rw, hw = full[ra:rb], hyp[a:b]
            if not rw:
                continue
            al = jiwer.process_words(" ".join(rw), " ".join(hw) or "<empty>")
            c["N"] += len(rw)
            c["S"] += al.substitutions
            c["D"] += al.deletions
            c["I"] += al.insertions
            c["shown"] += len(hw)
            inside.update(src[a:b])
            hit = {}
            chunks = al.alignments[0]
            eq = [k for k, ch in enumerate(chunks) if ch.type == "equal"]
            for k, ch in enumerate(chunks):  # insertions outside the first..last match: excerpt-edge slack
                if ch.type == "insert" and (not eq or k < eq[0] or k > eq[-1]):
                    c["I_edge"] += ch.hyp_end_idx - ch.hyp_start_idx
            for ch in chunks:
                hs = hw[ch.hyp_start_idx:ch.hyp_end_idx]
                if ch.type == "equal":
                    for k in range(ch.ref_end_idx - ch.ref_start_idx):
                        hit[ch.ref_start_idx + k] = a + ch.hyp_start_idx + k
                if ch.type in ("substitute", "insert"):
                    bad = [w for w in hs if w not in SMALL]
                    c["shown_bad"] += len(bad)
                    if ch.type == "substitute":
                        c["content"] += len(bad)
            for i, w in enumerate(rw):
                who, short = owner[ra + i]
                h = i in hit
                if who == "Nixon":
                    c["nixon_N"] += 1
                    c["nixon_hit"] += h
                if short:
                    c["short_N"] += 1
                    c["short_hit"] += h
                if w in names:
                    c["names"] += 1
                    c["names_hit"] += h
                if h:
                    hit_src.add(src[hit[i]])
                    if w in names:
                        name_src.add(src[hit[i]])
                if layer == "corrected" and h and spk_segs:
                    sw = words[src[hit[i]]]
                    label = speaker_at(spk_segs, spk_starts, (sw["s"] + sw["e"]) / 2)
                    spk_rows[(who, label)] += 1
        out["layers"][layer] = dict(c)
        if layer == "corrected":
            out["speakers"] = [[a, b, n] for (a, b), n in spk_rows.items()]
            # The corrector's words inside the printed stretches vs elsewhere in the same conversation, and how
            # often its words inside match the reference (by reason): a memorising corrector would edit more,
            # and more accurately, where the printed transcript exists.
            llm = Counter()
            for k, w in enumerate(words):
                if w.get("src") != "llm":
                    continue
                code = reason_code(w.get("reason", ""))
                if k in inside:
                    llm[f"in|{code}"] += 1
                    llm[f"in_hit|{code}"] += k in hit_src
                    llm[f"in_name_hit|{code}"] += k in name_src
                else:
                    llm[f"out|{code}"] += 1
            llm["words_in"] = len(inside)
            llm["words_out"] = len(words) - len(inside)
            out["llm"] = dict(llm)
    return out


def reason_code(r: str) -> str:
    r = (r or "").strip().lower()
    for code in ("whisper", "nara", "diary", "day", "context", "knowledge", "sound"):
        if r == code:
            return code
    if not r:
        return "none"
    if r.startswith(("nara", "log")):
        return "nara (free text)"
    if "diary" in r or "haldeman" in r:
        return "diary (free text)"
    if "whisper" in r or "other model" in r or "alternative" in r:
        return "whisper (free text)"
    return "other free text"


def pooled(rows: list[dict], layer: str, idx=None) -> dict:
    tot = Counter()
    for k in (idx if idx is not None else range(len(rows))):
        tot.update(rows[k]["layers"][layer])
    n = max(tot["N"], 1)
    return {"N": tot["N"], "recall": 1 - (tot["S"] + tot["D"]) / n, "wer": (tot["S"] + tot["D"] + tot["I"]) / n,
            "sub": tot["S"] / n, "del": tot["D"] / n, "ins": tot["I"] / n,
            "ins_edge": tot["I_edge"] / n, "wer_interior": (tot["S"] + tot["D"] + tot["I"] - tot["I_edge"]) / n,
            "wrong_content_shown": tot["shown_bad"] / max(tot["shown"], 1),
            "names": tot["names_hit"] / max(tot["names"], 1), "names_n": tot["names"],
            "nixon": tot["nixon_hit"] / max(tot["nixon_N"], 1), "nixon_n": tot["nixon_N"],
            "short": tot["short_hit"] / max(tot["short_N"], 1), "short_n": tot["short_N"],
            "ref_used": tot["N"] / max(tot["ref_total"], 1)}


def bootstrap(rows: list[dict], layers: list[str], boot: int, seed: int = 1974) -> dict:
    """95% intervals by resampling conversations; differences are paired (same resample)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keys = ["N", "S", "D", "I", "names", "names_hit", "nixon_N", "nixon_hit"]
    arr = {l: np.array([[r["layers"][l].get(k, 0) for k in keys] for r in rows], float) for l in layers}
    stats = {l: {"recall": [], "wer": [], "names": [], "nixon": []} for l in layers}
    diffs = defaultdict(list)
    for _ in range(boot):
        idx = rng.integers(0, len(rows), len(rows))
        rec = {}
        for l in layers:
            s = arr[l][idx].sum(0)
            n = max(s[0], 1)
            rec[l] = 1 - (s[1] + s[2]) / n
            stats[l]["recall"].append(rec[l])
            stats[l]["wer"].append((s[1] + s[2] + s[3]) / n)
            stats[l]["names"].append(s[5] / max(s[4], 1))
            stats[l]["nixon"].append(s[7] / max(s[6], 1))
        for a, b in [("parakeet_v5", "whisper"), ("merged", "whisper"), ("merged", "parakeet_v5"),
                     ("corrected", "merged"), ("corrected", "whisper")]:
            if a in rec and b in rec:
                diffs[f"{a} - {b}"].append(rec[a] - rec[b])
                diffs[f"names: {a} - {b}"].append(stats[a]["names"][-1] - stats[b]["names"][-1])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"resamples": boot, "seed": seed,
            "layers": {l: {k: ci(v) for k, v in s.items()} for l, s in stats.items()},
            "differences": {k: ci(v) for k, v in diffs.items()}}


def llm_summary(rows: list[dict]) -> dict:
    """The corrector's words inside the scored stretches vs outside them, and their accuracy inside, by reason."""
    tot = sum((Counter(r.get("llm", {})) for r in rows), Counter())
    codes = sorted({k.split("|", 1)[1] for k in tot if "|" in k})
    words_in, words_out = tot["words_in"], tot["words_out"]
    n_in = sum(tot[f"in|{c}"] for c in codes)
    n_out = sum(tot[f"out|{c}"] for c in codes)
    return {"words_in_scored_stretches": words_in, "words_outside": words_out,
            "llm_words_in": n_in, "llm_words_out": n_out,
            "llm_share_in": n_in / max(words_in, 1), "llm_share_out": n_out / max(words_out, 1),
            "llm_in_match_reference": sum(tot[f"in_hit|{c}"] for c in codes) / max(n_in, 1),
            "by_reason": {c: {"in": tot[f"in|{c}"], "out": tot[f"out|{c}"],
                              "in_match_reference": tot[f"in_hit|{c}"] / max(tot[f"in|{c}"], 1),
                              "in_name_matches": tot[f"in_name_hit|{c}"]} for c in codes}}


def cmd_score(workers: int, boot: int) -> None:
    refs = {p.name.split("_")[0]: p for p in sorted((OUT / "refs").glob("*_frus.json"))}
    reasons = exclusions(list(refs))
    trial = {json.loads(l)["conv"] for l in (DATA_DIR / "jobs_trial.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
    keep = [c for c in refs if not reasons[c]]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(score_conv, [(c, str(refs[c])) for c in keep], chunksize=2))
    # Reference quality: too little text, or no layer finds the excerpt in the audio (a wrong citation, or our
    # audio file covers another part of the tape). The test is the same for every layer.
    final = []
    for r in rows:
        ref_words = max(r["layers"][l].get("ref_total", 0) for l in LAYERS)
        best_used = max(r["layers"][l].get("N", 0) / max(r["layers"][l].get("ref_total", 1), 1) for l in LAYERS)
        best_rec = max(1 - (r["layers"][l].get("S", 0) + r["layers"][l].get("D", 0)) / max(r["layers"][l].get("N", 1), 1)
                       for l in LAYERS)
        why = ""
        if ref_words < 50:
            why = "reference under 50 words"
        elif best_used < 0.5:
            why = "under half of the reference aligns in any layer"
        elif best_rec < 0.25:
            why = "no layer recovers a quarter of the reference (excerpt not found in our audio)"
        r["whisper_mode"] = "batched (trial fifth)" if r["conv"] in trial else "sequential"
        if why:
            reasons[r["conv"]] = why
        else:
            final.append(r)
    with open(OUT / "exclusions.tsv", "w", encoding="utf-8") as fh:
        fh.write("conv\treason\tref_words\n")
        for c in refs:
            if reasons[c]:
                w = json.loads(refs[c].read_text(encoding="utf-8"))["words"]
                fh.write(f"{c}\t{reasons[c]}\t{w}\n")
    layers = list(LAYERS)
    res = {"n_candidates": len(refs), "n_scored": len(final),
           "excluded": dict(Counter(re.sub(r" of \d{3}-\d{3}.*", "", v) for v in reasons.values() if v)),
           "ref_words_scored": sum(r["layers"]["corrected"]["N"] for r in final),
           "pooled": {l: pooled(final, l) for l in layers},
           "bootstrap": bootstrap(final, layers, boot) if boot else {},
           "by_location": {}, "by_whisper_mode": {}}
    groups = defaultdict(list)
    for i, r in enumerate(final):
        groups[PLACES.get(r["location"], r["location"] or "unknown")].append(i)
    for g, idx in sorted(groups.items(), key=lambda x: -len(x[1])):
        res["by_location"][g] = {"n": len(idx), **{l: pooled(final, l, idx) for l in layers}}
    modes = defaultdict(list)
    for i, r in enumerate(final):
        modes[r["whisper_mode"]].append(i)
    for g, idx in modes.items():
        res["by_whisper_mode"][g] = {"n": len(idx), "whisper": pooled(final, "whisper", idx),
                                    "corrected": pooled(final, "corrected", idx)}
    # Speaker names, on conversations the voiceprint classifier never trained on.
    vp = {json.loads(l)["conv"] for l in (DATA_DIR / "voiceprints_v4" / "segments.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
    spk = Counter()
    per = defaultdict(Counter)
    n_conv = 0
    for r in final:
        if r["conv"] in vp or not r.get("speakers"):
            continue
        n_conv += 1
        for ref_spk, label, n in r["speakers"]:
            kind = ("named" if label and not label.startswith("Unknown") and label != "other"
                    else "possibly" if label.startswith("Unknown (possibly") else "unnamed")
            spk[kind] += n
            per[ref_spk][kind] += n
            if ref_spk == "Unidentified":
                continue
            cls = "in_class" if ref_spk in CLASSES else "out_of_class"
            spk[f"{cls}_words"] += n
            if kind == "named":
                spk["named_judged"] += n
                spk["named_right"] += n * (label == ref_spk)
                spk[f"{cls}_named"] += n
                spk[f"{cls}_named_right"] += n * (label == ref_spk)
                per[ref_spk]["named_right"] += n * (label == ref_spk)
            if kind == "possibly":
                guess = label[len("Unknown (possibly "):].rstrip(")")
                spk["possibly_judged"] += n
                spk["possibly_right"] += n * (guess == ref_spk)
    total = spk["named"] + spk["possibly"] + spk["unnamed"]
    res["speakers"] = {
        "conversations": n_conv, "aligned_words": total,
        "named_share": spk["named"] / max(total, 1), "possibly_share": spk["possibly"] / max(total, 1),
        "named_precision": spk["named_right"] / max(spk["named_judged"], 1), "named_judged": spk["named_judged"],
        "possibly_precision": spk["possibly_right"] / max(spk["possibly_judged"], 1),
        "possibly_judged": spk["possibly_judged"],
        **{f"{cls}_{k}": v for cls in ("in_class", "out_of_class") for k, v in {
            "words": spk[f"{cls}_words"], "named": spk[f"{cls}_named"],
            "named_share": spk[f"{cls}_named"] / max(spk[f"{cls}_words"], 1),
            "named_precision": spk[f"{cls}_named_right"] / max(spk[f"{cls}_named"], 1)}.items()},
        "per_speaker": {k: {"words": sum(v[x] for x in ("named", "possibly", "unnamed")),
                            "named": v["named"], "named_right": v["named_right"]}
                        for k, v in sorted(per.items(), key=lambda x: -sum(x[1].values()))
                        if sum(v[x] for x in ("named", "possibly", "unnamed")) >= 300}}
    res["llm"] = llm_summary(final)
    # The same measures on the validation set (the corpus's released layers), for comparison.
    val_rows = [score_conv((c, str(VALIDATION_DIR / "refs" / f"{r}.json"))) for c, r in VAL_ROOM.items()]
    res["validation_room"] = {"n": len(val_rows), **{l: pooled(val_rows, l) for l in layers},
                              "llm": llm_summary(val_rows)}
    res["notes"] = {"whisper_mode": "batched if the conversation is in the trial job list (jobs_trial.jsonl); "
                                    "trial conversations that failed then were re-run sequentially, so a few may be mislabeled",
                    "stock_parakeet": "no stock-Parakeet transcripts exist for these conversations; not scored"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(OUT / "per_conv.tsv", "w", encoding="utf-8") as fh:
        fh.write("conv\tdate\tlocation\twhisper_mode\tref_words\t" + "\t".join(f"{l}_recall" for l in layers) + "\tnames\n")
        for r in final:
            p = {l: pooled([r], l) for l in layers}
            fh.write(f"{r['conv']}\t{r['date'][:10]}\t{r['location']}\t{r['whisper_mode']}\t{p['corrected']['N']}\t"
                     + "\t".join(f"{p[l]['recall']:.3f}" for l in layers) + f"\t{p['corrected']['names_n']}\n")
    print(json.dumps({k: res[k] for k in ("n_candidates", "n_scored", "excluded", "ref_words_scored")}, indent=1))
    print(f"{'layer':12} {'recall':>7} {'95% CI':>15} {'WER':>6} {'S':>5} {'D':>5} {'I':>5} {'wrong':>6} {'names':>6} {'nixon':>6} {'short':>6}")
    for l in layers:
        p, b = res["pooled"][l], res["bootstrap"].get("layers", {}).get(l, {}).get("recall", [0, 0])
        print(f"{l:12} {p['recall']:7.1%} [{b[0]:5.1%},{b[1]:5.1%}] {p['wer']:6.1%} {p['sub']:5.1%} {p['del']:5.1%} "
              f"{p['ins']:5.1%} {p['wrong_content_shown']:6.1%} {p['names']:6.1%} {p['nixon']:6.1%} {p['short']:6.1%}")
    for g, v in res["by_location"].items():
        print(f"  {g:28} n={v['n']:3} words={v['corrected']['N']:6} " +
              " ".join(f"{l}={v[l]['recall']:.1%}" for l in layers))
    print("speakers:", json.dumps({k: v for k, v in res["speakers"].items() if k != "per_speaker"}))
    print("llm:", json.dumps(res["llm"], indent=1))
    v = res["validation_room"]
    print("validation room:", {l: round(v[l]["recall"], 4) for l in layers}, "names",
          {l: round(v[l]["names"], 4) for l in layers})
    print("validation llm:", json.dumps({k: x for k, x in v["llm"].items() if k != "by_reason"}))


def cmd_show(conv: str, layer: str) -> None:
    ref = json.loads((OUT / "refs" / f"{conv}_frus.json").read_text(encoding="utf-8"))
    hyp, _, _ = layer_words(CORPUS / conv / LAYERS[layer])
    for seg in sorted({t["segment"] for t in ref["turns"]}):
        full = [w for t in ref["turns"] if t["segment"] == seg for w in norm_words(t["text"])]
        ra, rb, a, b = locate(full, hyp)
        print(f"--- segment {seg}: ref {len(full)} words, used {ra}:{rb}, hyp {a}:{b}")
        print("REF:", " ".join(full[ra:ra + 40]))
        print("HYP:", " ".join(hyp[a:a + 40]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refs")
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--boot", type=int, default=10000)
    sh = sub.add_parser("show")
    sh.add_argument("conv")
    sh.add_argument("--layer", default="corrected", choices=list(LAYERS))
    args = ap.parse_args()
    if args.cmd == "refs":
        cmd_refs()
    elif args.cmd == "score":
        cmd_score(args.workers, args.boot)
    else:
        cmd_show(args.conv, args.layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
