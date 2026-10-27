"""The corpus explorer's local web server (standard library only): JSON over explorer.db, transcripts built on
request from the tape's files, and audio fetched from the archive server on first play and cached.

  /                       the single-page app (static/)
  /api/calendar           every day with tape: conversations, minutes, words
  /api/day/<date>         the day's conversations in time order, with people and top topics; Haldeman's diary
  /api/conv/<id>          one conversation: NARA record, people, topic outline, and the transcript as display words
  /api/person/<id>        a person: counts, conversations by date and role, topics with dates, frequent company
  /api/people?q=          people by number of conversations (optionally filtered by name)
  /api/search?q=          full-text search over the transcripts
  /audio/<id>             the conversation's audio (MP3, byte ranges for seeking)

Display words: v5's cased, punctuated words where the transcript layer kept them; Argon's replacements marked
with what they replaced and why (an [unclear] stretch shown as such); repeated name fixes marked; Whisper-only
words marked; each word tagged with the speaker whose segment contains it (speakers_v4.json).

    python src/explorer/server.py [--port 8787]      then open http://localhost:8813
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correct import apply_edits, propagate  # noqa: E402
from listener_fixes import apply_possessives, apply_words, drop_tone_words, override, speaker_spans  # noqa: E402
from paths import DATA_DIR  # noqa: E402

DB = DATA_DIR / "explorer" / "explorer.db"
CORPUS = DATA_DIR / "corpus_v5" / "out"
CACHE = DATA_DIR / "explorer" / "audio_cache"
STATIC = Path(__file__).resolve().parent / "static"
LOCK = threading.Lock()


def q(sql: str, args=()) -> list[dict]:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    finally:
        db.close()


def people_of(conv_ids: list[str]) -> dict[str, list[dict]]:
    if not conv_ids:
        return {}
    marks = ",".join("?" * len(conv_ids))
    out: dict[str, list[dict]] = {}
    for r in q(f"""SELECT cp.conv, p.id, p.name, p.kind, cp.participant, cp.speak_s, cp.mentions FROM conv_people cp
                   JOIN people p ON p.id = cp.person WHERE cp.conv IN ({marks})
                   ORDER BY cp.participant DESC, cp.speak_s DESC, cp.mentions DESC""", conv_ids):
        out.setdefault(r.pop("conv"), []).append(r)
    return out


def api_calendar(_):
    return q("SELECT date, n, minutes, words FROM days ORDER BY date")


def api_day(date):
    day = q("SELECT * FROM days WHERE date = ?", (date,))
    convs = q("""SELECT id, start, end, place, loc, minutes, words, layer, summary FROM conversations
                 WHERE date = ? ORDER BY iso, id""", (date,))
    ppl = people_of([c["id"] for c in convs])
    tops = {}
    for r in q(f"""SELECT conv, text FROM topics WHERE dashed = 0 AND routine = 0 AND conv IN ({",".join("?" * len(convs))})
                   ORDER BY conv, pos""", [c["id"] for c in convs]) if convs else []:
        tops.setdefault(r["conv"], []).append(r["text"])
    for c in convs:
        c["people"] = [p for p in ppl.get(c["id"], []) if p["participant"]]
        c["topics"] = list(dict.fromkeys(tops.get(c["id"], [])))[:8]
    days = [r["date"] for r in q("SELECT date FROM days ORDER BY date")]
    i = days.index(date) if date in days else -1
    return {"date": date, "day": day[0] if day else None, "conversations": convs,
            "prev": days[i - 1] if i > 0 else None, "next": days[i + 1] if 0 <= i < len(days) - 1 else None}


def display_words(conv: str) -> tuple[list[dict], dict]:
    d = CORPUS / conv
    if not (d / "merged.json").exists():
        return [], {}
    merged = json.loads((d / "merged.json").read_text(encoding="utf-8"))
    info = {"layer": "merged", "edits": 0, "unclear": 0, "propagated": 0}
    words = merged
    if (d / "edits_argon.json").exists():
        edits = json.loads((d / "edits_argon.json").read_text(encoding="utf-8"))
        for e in edits:
            e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
        starts = [int(a) * 60 + int(b) for a, b in
                  re.findall(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
        words, applied, _ = apply_edits(merged, edits, starts, keep_unclear=True)
        words, n_prop = propagate(words, edits)
        info.update(layer="argon", edits=applied, propagated=n_prop)
    pk = json.loads((d / "parakeet_v5.json").read_text(encoding="utf-8")) if (d / "parakeet_v5.json").exists() else {}
    cased = {(x["s"], x["e"]): x["w"] for x in pk.get("words", [])}
    # as in the release from v1.1 (listener_fixes.py): no words inside withdrawal tones; corrections heard by ear
    words, _ = drop_tone_words(words, pk.get("tones") or [])
    words, _ = apply_words(conv, words)
    words, _ = apply_possessives(conv, words)
    spans = speaker_spans(conv)
    segs = sorted(json.loads((d / "speakers_v4.json").read_text(encoding="utf-8")), key=lambda s: s["start"]) \
        if (d / "speakers_v4.json").exists() else []
    seg_starts = [s["start"] for s in segs]

    def speaker(t: float) -> str:
        if spans and (o := override(spans, t)):
            return o
        i = bisect.bisect_right(seg_starts, t) - 1
        for k in (i, i + 1):
            if 0 <= k < len(segs) and segs[k]["start"] - 0.5 <= t <= segs[k]["end"] + 0.5:
                return segs[k]["speaker"]
        return ""

    out, i = [], 0
    while i < len(words):
        x = words[i]
        if x.get("src") == "llm":
            j = i + 1
            while j < len(words) and words[j].get("src") == "llm" and words[j].get("was") == x.get("was") \
                    and words[j].get("propagated") == x.get("propagated"):
                j += 1
            text = " ".join(w["w"] for w in words[i:j])
            item = {"t": text, "s": x["s"], "e": words[j - 1]["e"], "argon": {"was": x.get("was"), "why": x.get("reason")}}
            if x.get("propagated"):
                item["argon"]["repeated"] = True
            if text == "[unclear]":
                info["unclear"] += 1
            out.append(item)
            i = j
            continue
        j = i + 1
        while j < len(words) and words[j].get("src") != "llm" and (words[j]["s"], words[j]["e"]) == (x["s"], x["e"]):
            j += 1
        item = {"t": cased.get((x["s"], x["e"])) if x.get("src") != "whisper" else None, "s": x["s"], "e": x["e"]}
        item["t"] = item["t"] or " ".join(w["w"] for w in words[i:j])
        if x.get("src") == "whisper":
            item["whisper"] = True
        if x.get("c") is not None and x["c"] < 0.01:
            item["unsure"] = True
        out.append(item)
        i = j
    for w in out:
        w["sp"] = speaker((w["s"] + w["e"]) / 2)
    return out, info


FRESH: dict | None = None
DISCOVER = DATA_DIR / "discover"


def fresh_spans() -> dict:
    """conv -> (time spans in no earlier transcript, prior transcripts) from compare_easynixon.py gaps and the
    discover.py registry: a conversation transcribed elsewhere (WSPF, NARA, FRUS) has no fresh spans."""
    global FRESH
    if FRESH is None:
        FRESH = {}
        reg = {}
        if (DISCOVER / "registry.json").exists():
            reg = json.loads((DISCOVER / "registry.json").read_text(encoding="utf-8"))["transcribed"]
        if (DISCOVER / "easynixon_gaps.jsonl").exists():
            for line in open(DISCOVER / "easynixon_gaps.jsonl", encoding="utf-8"):
                r = json.loads(line)
                spans = [] if r["conv"] in reg else [[g["s"], g["e"]] for g in r["gaps"]]
                FRESH[r["conv"]] = (spans, reg.get(r["conv"], []), r["has_theirs"])
        for conv, why in reg.items():
            FRESH.setdefault(conv, ([], why, True))
    return FRESH


def api_conv(cid):
    c = q("SELECT * FROM conversations WHERE id = ?", (cid,))
    if not c:
        return None
    c = c[0]
    c["audio"] = json.loads(c["audio"] or "[]")
    c["people"] = people_of([cid]).get(cid, [])
    c["topics"] = q("SELECT level, dashed, text FROM topics WHERE conv = ? ORDER BY pos", (cid,))
    c["words"], c["transcript"] = display_words(cid)
    spans, prior, easynixon = fresh_spans().get(cid, ([], [], True))
    c["fresh"], c["prior"], c["easynixon"] = spans, prior, easynixon
    for w in c["words"]:
        if any(a <= w["s"] <= b for a, b in spans):
            w["fresh"] = True
    return c


def api_person(pid):
    p = q("SELECT * FROM people WHERE id = ?", (pid,))
    if not p:
        return None
    convs = q("""SELECT c.id, c.date, c.start, c.place, c.loc, c.minutes, c.layer, cp.participant, cp.speak_s, cp.mentions
                 FROM conv_people cp JOIN conversations c ON c.id = cp.conv WHERE cp.person = ? ORDER BY c.iso, c.id""", (pid,))
    topics = q("""SELECT MIN(t.root) AS topic, t.root_key AS key, COUNT(DISTINCT t.conv) AS n, MIN(c.date) AS first,
                         MAX(c.date) AS last, GROUP_CONCAT(DISTINCT t.conv) AS convs
                  FROM topics t JOIN conv_people cp ON cp.conv = t.conv AND cp.person = ? AND cp.participant = 1
                  JOIN conversations c ON c.id = t.conv
                  WHERE t.dashed = 0 AND t.routine = 0 GROUP BY t.root_key ORDER BY n DESC LIMIT 60""", (pid,))
    for t in topics:
        t["convs"] = (t["convs"] or "").split(",")
    company = q("""SELECT p.id, p.name, COUNT(*) AS n FROM conv_people a JOIN conv_people b ON a.conv = b.conv
                   AND b.person != a.person JOIN people p ON p.id = b.person
                   WHERE a.person = ? AND a.participant = 1 AND b.participant = 1 AND p.kind = 'person'
                   GROUP BY b.person ORDER BY n DESC LIMIT 16""", (pid,))
    return {"person": p[0], "conversations": convs, "topics": topics, "company": company}


def api_people(qs):
    name = (qs.get("q", [""])[0] or "").strip()
    if name:
        return q("""SELECT id, name, kind, n_part, n_speak, speak_min, n_mention_convs FROM people
                    WHERE name LIKE ? ORDER BY n_part DESC LIMIT 40""", (f"%{name}%",))
    return q("""SELECT id, name, kind, n_part, n_speak, speak_min, n_mention_convs FROM people WHERE kind = 'person'
                ORDER BY n_part DESC LIMIT 60""")


def api_search(qs):
    text = (qs.get("q", [""])[0] or "").strip().lower()
    terms = re.findall(r"[a-z0-9']+", text)
    if not terms:
        return []
    match = " ".join(f'"{t}"' for t in terms) if len(terms) == 1 or not text.startswith('"') else f'"{" ".join(terms)}"'
    return q("""SELECT f.conv AS id, c.date, c.start, c.place, c.minutes, c.layer,
                       snippet(text_fts, 1, '<mark>', '</mark>', ' … ', 14) AS snippet
                FROM text_fts f JOIN conversations c ON c.id = f.conv WHERE text_fts MATCH ?
                ORDER BY rank LIMIT 120""", (match,))


SIZES: dict[str, int] = {}


def audio_paths(cid: str) -> list[str]:
    c = q("SELECT audio FROM conversations WHERE id = ?", (cid,))
    return json.loads(c[0]["audio"] or "[]") if c else []


def remote_size(path: str) -> int | None:
    if path not in SIZES:
        r = subprocess.run(["ssh", os.environ.get("NIXON_AUDIO_HOST", "audio-host"), f"stat -c %s '{path}'"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        SIZES[path] = int(r.stdout.strip())
    return SIZES[path]


def join_mp3(parts: list[Path], out: Path) -> bool:
    """Join MP3 parts with ffmpeg's concat demuxer (no re-encoding). Byte-wise concatenation keeps the first part's
    Xing/Info header, which describes that part alone, and browsers then seek to the wrong place (a seek to 9:45 in
    916-019 played 2:06). ffmpeg writes one header for the whole file."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-map", "0:a", "-c", "copy", "-map_metadata", "-1", "-f", "mp3", str(out)],
                       capture_output=True, timeout=600)
    lst.unlink(missing_ok=True)
    return r.returncode == 0 and out.exists()


def audio_file(cid: str) -> Path | None:
    """A multi-part tape's MP3 in the local cache, copied from the archive server on first request (parts joined in order)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{cid}.mp3"
    if f.exists():
        return f
    c = q("SELECT audio FROM conversations WHERE id = ?", (cid,))
    paths = json.loads(c[0]["audio"] or "[]") if c else []
    if not paths:
        return None
    with LOCK:
        if f.exists():
            return f
        parts = []
        for i, p in enumerate(paths):
            part = CACHE / f"{cid}.part{i}.mp3"
            r = subprocess.run(["ssh", os.environ.get("NIXON_AUDIO_HOST", "audio-host"), f"cat '{p}'"], capture_output=True, timeout=600)
            if r.returncode != 0:
                for x in parts:
                    x.unlink(missing_ok=True)
                return None
            part.write_bytes(r.stdout)
            parts.append(part)
        tmp = CACHE / f"{cid}.joining.mp3"
        ok = join_mp3(parts, tmp)
        for x in parts:
            x.unlink(missing_ok=True)
        if not ok:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(f)
    return f


ROUTES = [(re.compile(r"^/api/calendar$"), api_calendar), (re.compile(r"^/api/day/([\d-]+)$"), api_day),
          (re.compile(r"^/api/conv/([\w-]+)$"), api_conv), (re.compile(r"^/api/person/([\w-]+)$"), api_person)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        path, qs = unquote(u.path), parse_qs(u.query)
        try:
            if path == "/api/people":
                return self.send_json(api_people(qs))
            if path == "/api/search":
                return self.send_json(api_search(qs))
            for pat, fn in ROUTES:
                m = pat.match(path)
                if m:
                    res = fn(m.group(1) if pat.groups else None)
                    return self.send_json(res if res is not None else {"error": "not found"}, 200 if res is not None else 404)
            if path.startswith("/audio/"):
                return self.send_audio(path.split("/")[2])
            return self.send_static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # report, keep serving
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def send_static(self, path):
        f = (STATIC / ("index.html" if path in ("/", "") else path.lstrip("/"))).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            f = STATIC / "index.html"
        types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", types.get(f.suffix, "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_audio(self, cid):
        """Single-file tapes stream straight from the archive server (one ssh read per play or seek, the bytes asked for);
        multi-part tapes come from the local cache."""
        paths = audio_paths(cid)
        cached = CACHE / f"{cid}.mp3"
        remote = paths[0] if len(paths) == 1 and not cached.exists() else None
        f = None if remote else audio_file(cid)
        size = remote_size(remote) if remote else (f.stat().st_size if f else None)
        if not size:
            return self.send_json({"error": "no audio for this conversation"}, 404)
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        a, b = 0, size - 1
        if rng:
            a = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2)))
            b = min(int(rng.group(2)), size - 1) if rng.group(1) and rng.group(2) else size - 1
        if a >= size or b < a:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        if remote:
            proc = subprocess.Popen(["ssh", os.environ.get("NIXON_AUDIO_HOST", "audio-host"), f"tail -c +{a + 1} '{remote}' | head -c {b - a + 1}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                while chunk := proc.stdout.read(1 << 16):
                    self.wfile.write(chunk)
            finally:  # the browser drops a stream when the listener seeks: stop the read on the archive server too
                proc.kill()
            return
        with open(f, "rb") as fh:
            fh.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8813)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"Nixon tapes explorer: http://localhost:{args.port}  (data {DB})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
