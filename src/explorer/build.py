"""Build the corpus explorer's database (no language model involved): days, conversations, people, topics,
full-text search.

Sources:
  NARA's manifest (corpus/nixon.db: date, time, place, participants, NARA's one-line description)
  the corpus job list (corpus_v5/jobs.jsonl) and each tape's NARA subject-log entry (correct.nara_log): the
      archivists' outline of the conversation, the source of its topics
  speakers_v4.json (who speaks when, from the voiceprint classifier, METHODS 5)
  the best transcript layer: corrected.json (after Argon) or merged.json (v5 + Whisper)
  H. R. Haldeman's diary (the day's narrative) and the archive server's download ledger (where each tape's audio is)

People: one per NARA participant entry ("Kissinger, Henry A."). A conversation links a person three ways:
  participant  NARA lists them
  speaking     the classifier names them (seconds of speech; a named class is matched to the participant
               with that surname, else to the most frequent bearer of it)
  mentioned    their surname occurs in the transcript (surnames that are mostly capitalized in the era text,
               4+ letters; attributed to the most frequent bearer of the surname)
Topics: the subject-log outline lines; a line without a leading dash is a top-level topic, dashed lines are
its sub-topics (indent level kept for display).

    python src/explorer/build.py            -> ~/nixon-data/explorer/explorer.db
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import CORPUS_DB, DATA_DIR  # noqa: E402

CORPUS = DATA_DIR / "corpus_v5"
OUT = DATA_DIR / "explorer"
LM_CLEAN = DATA_DIR / "lm" / "clean"
GROUPS = ("[unknown person(s)]", "white house operator", "cabinet officers", "unknown person(s)")
EVENT = re.compile(r"^(Date|Time|Location):|^Conversation No|^\[|^The President (met|talked|spoke)|"
                   r"\b(entered|left|joined|met with|talked with|spoke with)\b.*\b(\d{1,2}:\d{2}|unknown time)", re.I)
# page furniture in the logs, not topics: rows of asterisks, withdrawn-item markers, running heads
FURNITURE = re.compile(r"^[\W_]+$|^(BEGIN|END) WITHDRAWN ITEM|^Tape Subject Log|^\(?rev\.|^Withdrawn item|"
                       r"^(Personal returnable|Privacy|National security|Statutory)\b.*\[|^Duration:|^Archivist|^Segment Cat", re.I)


def person_key(entry: str) -> tuple[str, str, str, str]:
    """NARA participant entry -> (id, display name, surname, kind)."""
    e = entry.strip().rstrip(";")
    nick = re.search(r'\("([^"]+)"\)', e) or re.search(r"\(“([^”]+)”\)", e)
    base = re.sub(r"\s*\(.*?\)", "", e).strip()
    if "," not in base or base.lower() in GROUPS:
        return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "unknown", base, "", "group"
    last, rest = [x.strip() for x in base.split(",", 1)]
    suffix = re.match(r"^(.*?),\s*(Jr\.?|Sr\.?|III?|IV)$", rest)
    first, suf = (suffix.group(1), suffix.group(2)) if suffix else (rest, "")
    display = f"{first} {last}" + (f", {suf}" if suf else "") + (f" (\"{nick.group(1)}\")" if nick else "")
    return re.sub(r"[^a-z0-9]+", "-", f"{last} {first}".lower()).strip("-"), display, last, "person"


EVENT2 = re.compile(r"\b(talked|met|spoke) with\b.*\.$|^The (President|First Lady)\b.*\.$", re.I)
ROUTINE = re.compile(r"schedule$|^greetings?$|^refreshments?$|^farewells?$|^photographs?$|^gifts?$|^introductions?$|"
                     r"^small talk$|^weather$|^telephone call|^call(s)? (to|from)\b|^recording$|^seating$|^arrangements?$|"
                     r"^(the )?president'?s? (gift|schedule)", re.I)


def tidy_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("�", "'").replace("’", "'").replace("‘", "'")
                  .replace("“", '"').replace("”", '"')).strip()


def topic_key(t: str) -> str:
    return re.sub(r"^the ", "", re.sub(r"[.,;:]+$", "", t.lower())).strip()


def topics(log: str) -> list[tuple[int, int, str]]:
    """(level, dashed, text) for each outline line of a subject-log entry."""
    out = []
    for line in log.splitlines():
        if not line.strip() or EVENT.search(line.strip()) or EVENT2.search(line.strip()):
            continue
        lead = len(line) - len(line.lstrip(" \t"))
        indent = line[:lead].count("\t") + (line[:lead].count(" ") // 6)
        text = tidy_text(line)
        dashed = text.startswith("-")
        text = text.lstrip("-•").strip()
        if 2 <= len(text) <= 140 and not text.startswith("Page |") and not FURNITURE.search(text):
            out.append((indent, int(dashed), text))
    return out


def read_tape(args):
    """Per-tape work for the pool: transcript tokens, layer, speaker seconds by class, topic lines."""
    conv, job = args
    from correct import nara_log
    d = CORPUS / "out" / conv
    layer = "corrected" if (d / "corrected.json").exists() else "merged" if (d / "merged.json").exists() else ""
    toks = [x["w"] for x in json.loads((d / f"{layer}.json").read_text(encoding="utf-8"))] if layer else []
    speak = Counter()
    if (d / "speakers_v4.json").exists():
        for s in json.loads((d / "speakers_v4.json").read_text(encoding="utf-8")):
            speak[s["speaker"]] += max(0.0, s["end"] - s["start"])
    secs = 0.0
    if (d / "parakeet_v5.json").exists():
        secs = json.loads((d / "parakeet_v5.json").read_text(encoding="utf-8")).get("audio_seconds") or 0.0
    log = nara_log(job)
    return conv, layer, toks, dict(speak), secs, log


def name_like_surnames(candidates: set[str]) -> set[str]:
    """Surnames (lowercase) that are mostly capitalized in the spoken and FRUS era text: names, not words."""
    cap, low = Counter(), Counter()
    want = {c for c in candidates if len(c) >= 4}
    for name in ("wspf", "nara", "frus"):
        for w in re.findall(r"[A-Za-z']+", (LM_CLEAN / f"{name}.txt").read_text(encoding="utf-8")):
            k = w.lower()
            if k in want:
                (cap if w[0].isupper() else low)[k] += 1
    return {k for k in want if cap[k] > low[k]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in open(CORPUS / "jobs.jsonl", encoding="utf-8") if l.strip()}
    man = sqlite3.connect(CORPUS_DB)
    meta = {r[0].removeprefix("37-wht-conversation-"): r[1:] for r in man.execute(
        "SELECT conversation_id, date, start_time, end_time, location, participants, summary FROM conversations")}
    ledger = {}
    ledger_file = next((OUT / n for n in ("audio_downloads.jsonl", "audio_downloads.jsonl") if (OUT / n).exists()),
                       OUT / "audio_downloads.jsonl")  # the archive server's download log (either name)
    for l in open(ledger_file, encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            ledger[x["url"]] = x["path"]

    convs = sorted(jobs)
    with Pool(16) as pool:
        tapes = {r[0]: r[1:] for r in pool.imap_unordered(read_tape, [(c, jobs[c]) for c in convs], chunksize=32)}
    print(f"read {len(tapes)} tapes", flush=True)

    # people
    people: dict[str, dict] = {}
    conv_part: dict[str, list[str]] = {}
    for c in convs:
        ids = []
        for p in jobs[c]["participants"]:
            pid, disp, last, kind = person_key(p)
            people.setdefault(pid, {"name": disp, "surname": last, "kind": kind, "n": 0})
            people[pid]["n"] += 1
            ids.append(pid)
        conv_part[c] = list(dict.fromkeys(ids))
    bearer: dict[str, str] = {}  # surname (lowercase) -> its most frequent participant
    for pid, p in sorted(people.items(), key=lambda kv: -kv[1]["n"]):
        if p["kind"] == "person":
            bearer.setdefault(p["surname"].lower(), pid)
    mention_names = name_like_surnames(set(bearer))
    print(f"{len(people)} people and groups; {len(mention_names)} surnames counted as mentions", flush=True)

    db_path = OUT / "explorer.db"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript("""
    CREATE TABLE conversations (id TEXT PRIMARY KEY, date TEXT, iso TEXT, start TEXT, end TEXT, place TEXT, loc TEXT,
        minutes REAL, words INTEGER, layer TEXT, participants TEXT, summary TEXT, log TEXT, audio TEXT, parts INTEGER,
        alt_source INTEGER);
    CREATE TABLE days (date TEXT PRIMARY KEY, n INTEGER, minutes REAL, words INTEGER, diary TEXT);
    CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT, surname TEXT, kind TEXT, n_part INTEGER, n_speak INTEGER,
        speak_min REAL, n_mention_convs INTEGER, mentions INTEGER);
    CREATE TABLE conv_people (conv TEXT, person TEXT, participant INTEGER, speak_s REAL, mentions INTEGER,
        PRIMARY KEY (conv, person));
    CREATE TABLE topics (conv TEXT, pos INTEGER, level INTEGER, dashed INTEGER, text TEXT, root TEXT, root_key TEXT,
        routine INTEGER);
    CREATE VIRTUAL TABLE text_fts USING fts5(conv UNINDEXED, text);
    CREATE INDEX cp_person ON conv_people(person);
    CREATE INDEX t_conv ON topics(conv);
    CREATE INDEX t_root ON topics(root_key);
    CREATE INDEX c_date ON conversations(date);
    """)
    from correct import haldeman
    day_rows: dict[str, list] = defaultdict(lambda: [0, 0.0, 0, ""])
    for c in convs:
        layer, toks, speak, secs, log = tapes[c]
        j = jobs[c]
        iso, start, end, place, parts_s, summary = meta.get(c, ("", "", "", "", "", ""))
        if not iso:
            import datetime
            try:
                iso = datetime.datetime.strptime(j["date"], "%B %d, %Y").strftime("%Y-%m-%dT00:00")
            except ValueError:
                iso = ""
        date = iso[:10]
        audio = [ledger[u] for u in j["urls"] if u in ledger]
        alt = 0
        if len(audio) < len(j["urls"]):
            for fb in j.get("fallbacks", []):
                got = [ledger[u] for u in fb if u in ledger]
                if len(got) == len(fb):
                    audio, alt = got, 1
                    break
        db.execute("INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (c, date, iso, start, end, place or j["location"], j["location"], round(secs / 60, 1), len(toks), layer,
                    "; ".join(j["participants"]), summary, log, json.dumps(audio), len(j["urls"]), alt))
        # people links
        links: dict[str, list] = {pid: [1, 0.0, 0] for pid in conv_part[c]}
        surnames_here = {people[p]["surname"].lower(): p for p in conv_part[c] if people[p]["kind"] == "person"}
        for cls, s in speak.items():
            if cls.startswith("Unknown") or cls == "other":
                continue
            pid = surnames_here.get(cls.lower()) or bearer.get(cls.lower())
            if pid:
                links.setdefault(pid, [0, 0.0, 0])[1] += s
        for name, n in Counter(t for t in toks if t in mention_names).items():
            pid = surnames_here.get(name) or bearer[name]
            links.setdefault(pid, [0, 0.0, 0])[2] += n
        db.executemany("INSERT INTO conv_people VALUES (?,?,?,?,?)",
                       [(c, pid, v[0], round(v[1], 1), v[2]) for pid, v in links.items()])
        root = ""
        rows = []
        for k, (lvl, dashed, text) in enumerate(topics(log)):
            if not dashed:
                root = text
            r = root or text
            rows.append((c, k, lvl, dashed, text, r, topic_key(r), int(bool(ROUTINE.search(topic_key(r))))))
        db.executemany("INSERT INTO topics VALUES (?,?,?,?,?,?,?,?)", rows)
        if toks:
            db.execute("INSERT INTO text_fts VALUES (?,?)", (c, " ".join(toks)))
        if date:
            r = day_rows[date]
            r[0] += 1; r[1] += secs / 60; r[2] += len(toks)
            r[3] = r[3] or j["date"]
    for date, (n, mins, words, jd) in sorted(day_rows.items()):
        db.execute("INSERT INTO days VALUES (?,?,?,?,?)", (date, n, round(mins, 1), words, haldeman(jd)))
    for pid, p in people.items():
        s = db.execute("SELECT SUM(participant), SUM(speak_s > 30), SUM(speak_s), SUM(mentions > 0), SUM(mentions) "
                       "FROM conv_people WHERE person = ?", (pid,)).fetchone()
        db.execute("INSERT INTO people VALUES (?,?,?,?,?,?,?,?,?)",
                   (pid, p["name"], p["surname"], p["kind"], s[0] or 0, s[1] or 0, round((s[2] or 0) / 60, 1), s[3] or 0, s[4] or 0))
    db.commit()
    n = db.execute("SELECT COUNT(*), SUM(minutes), SUM(words), SUM(layer = 'corrected') FROM conversations").fetchone()
    print(f"explorer.db: {n[0]} conversations, {n[1] / 60:.0f} audio hours, {n[2]:,} words, {n[3]} Argon-corrected; "
          f"{db.execute('SELECT COUNT(*) FROM days').fetchone()[0]} days, {len(people)} people and groups, "
          f"{db.execute('SELECT COUNT(*) FROM topics').fetchone()[0]:,} topic lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
