"""Build the public release package of the Nixon transcript corpus (nothing is uploaded; the corpus is only read).

    python src/export_release.py                         # full build -> ~/nixon-data/release/nixon-corpus-v1.1/
    python src/export_release.py --limit 200             # quick trial of the first 200 conversations (own folder)
    python src/export_release.py --finalize              # rewrite README / DATA_CARD copy / SHA256SUMS only
    python src/export_release.py --sample 871-004 ...    # copy named conversations into data/sample/ (needs a built release)

Per conversation (every one of the job list's 19,426, including those with no words), parallel over processes:
  conversations/<id>.json.gz   the full record: NARA metadata, audio URLs, withdrawn-tone spans, words with speaker,
                               confidence and source, the LLM edit list (applied or not), speaker segments, turns
  transcripts/<id>.txt         the readable cased transcript, one line per speaker turn with a timestamp
  raw/<id>.json.gz             Parakeet v5 words (with confidence), Whisper words (with probability), speaker segments
Then: metadata.csv, the archives (conversations and raw by year as .tar of .json.gz, transcripts as one .zip),
SHA256SUMS, MANIFEST.json (sizes, counts, checks) and README.md.

The word layer is the explorer's display layer (explorer/server.py display_words): Parakeet v5's cased, punctuated words;
Whisper words where Parakeet was silent; the LLM corrector's replacements (correct.apply_edits keep_unclear=True, then
correct.propagate) marked with what they replaced; each word tagged with the voiceprint classifier's speaker for the
segment that contains it. From v1.1 (listener_fixes.py): words inside withdrawal tones are dropped, and corrections
confirmed by listening (docs/listener_corrections.json) replace words or name speakers. Methods: docs/METHODS.md 5, 6,
7b, 7h, 7k, 7l, 7s, 8b.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tarfile
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct import apply_edits, propagate  # noqa: E402
from listener_fixes import apply_possessives, apply_words, drop_tone_words, override, speaker_spans  # noqa: E402
from paths import DATA_DIR  # noqa: E402
from score import norm_words  # noqa: E402

VERSION = "1.2"
NAME = f"nixon-corpus-v{VERSION}"
CORPUS = DATA_DIR / "corpus_v5"
DB = DATA_DIR / "explorer" / "explorer.db"
RELEASE = DATA_DIR / "release"
REPO = Path(__file__).resolve().parent.parent
LOCATIONS = {"CAB": "Cabinet Room", "CDHW": "Camp David Hard Wire", "CDSD": "Camp David Study Desk",
             "CDST": "Camp David Study Table", "EOB": "Old Executive Office Building", "OVAL": "Oval Office",
             "WHT": "White House Telephone"}
WORD_FIELDS = ["text", "start", "end", "speaker", "confidence", "source", "edit"]
SOURCES = {"both": "both", "parakeet": "parakeet", "whisper": "whisper-fill"}
TURN_GAP = 4.0  # s of silence that starts a new line within one speaker's turn (as correct.readable)
GEN = {"mtime": 0}


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def gz_write(path: Path, obj) -> int:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as g:
        g.write(data)
    return len(data)


def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def apply_edits_tracked(merged, edits, starts):
    """correct.apply_edits (keep_unclear=True) with the same matching rule, plus whether each edit was located.
    Checked against the original in every conversation (the 'apply_check' column of metadata.csv)."""
    words = [dict(x) for x in merged]
    status = []
    for e in edits:
        try:
            m, s = e["block"].strip("[]").split(":")
            a = int(m) * 60 + int(s) - 0.5
        except (KeyError, ValueError):
            status.append(False)
            continue
        later = [t for t in starts if t > a + 0.5]
        b = (later[0] - 0.5) if later else 1e12
        old = str(e.get("old", "")).lower().split()
        new_s = str(e.get("new", ""))
        new = [t for w in new_s.split() for t in norm_words(w)] or []
        idx = [i for i, x in enumerate(words) if a <= x["s"] < b]
        hit = next((i for i in idx if [x["w"] for x in words[i:i + len(old)]] == old), None)
        if hit is None or not old:
            status.append(False)
            continue
        span = words[hit:hit + len(old)]
        t0, t1 = span[0]["s"], span[-1]["e"]
        step = (t1 - t0) / max(len(new), 1)
        repl = [{"w": t, "s": round(t0 + k * step, 2), "e": round(t0 + (k + 1) * step, 2), "src": "llm",
                 "was": e["old"], "confidence": e.get("confidence"), "reason": e.get("reason")} for k, t in enumerate(new)]
        if not repl and "unclear" in new_s:
            repl = [{"w": "[unclear]", "s": t0, "e": t1, "src": "llm", "was": e["old"], "reason": e.get("reason")}]
        words = words[:hit] + repl + words[hit + len(old):]
        status.append(True)
    return words, status


def display_rows(words, cased, speaker):
    """The explorer's display units (server.display_words), as release rows
    [text, start, end, speaker, confidence, source, edit]; `edit` is null or {was, reason, edit_confidence}."""
    out, i = [], 0
    while i < len(words):
        x = words[i]
        if x.get("src") in ("llm", "listener"):
            j = i + 1
            while j < len(words) and words[j].get("src") == x["src"] and words[j].get("was") == x.get("was") \
                    and words[j].get("propagated") == x.get("propagated"):
                j += 1
            text = " ".join(w["w"] for w in words[i:j])
            ed = {"was": x.get("was"), "reason": x.get("reason")}
            if x.get("confidence"):
                ed["edit_confidence"] = x.get("confidence")
            src = "listener-edit" if x["src"] == "listener" else "llm-propagated" if x.get("propagated") else "llm-edit"
            out.append([text, x["s"], words[j - 1]["e"], None, None, src, ed])
            i = j
            continue
        j = i + 1
        while j < len(words) and words[j].get("src") != "llm" and (words[j]["s"], words[j]["e"]) == (x["s"], x["e"]):
            j += 1
        t = cased.get((x["s"], x["e"])) if x.get("src") != "whisper" else None
        t = t or " ".join(w["w"] for w in words[i:j])
        c = x.get("c")
        out.append([t, x["s"], x["e"], None, None if c is None else round(c, 3), SOURCES.get(x.get("src"), x.get("src")), None])
        i = j
    for r in out:
        r[3] = speaker((r[1] + r[2]) / 2)
    return out


def process(args):
    cid, job, row, out_dir = args
    d = CORPUS / "out" / cid
    try:
        return _process(cid, job, row, out_dir, d)
    except Exception as e:  # report, keep going: the build is checked for failures afterwards
        import traceback
        return {"id": cid, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}


def _process(cid, job, row, out_dir, d):
    pk = jload(d / "parakeet_v5.json") if (d / "parakeet_v5.json").exists() else {"words": [], "tones": [], "audio_seconds": 0}
    wh = jload(d / "whisper.json") if (d / "whisper.json").exists() else None
    merged = jload(d / "merged.json") if (d / "merged.json").exists() else None
    segs = sorted(jload(d / "speakers_v4.json"), key=lambda s: s["start"]) if (d / "speakers_v4.json").exists() else []
    usage = jload(d / "argon_usage.json") if (d / "argon_usage.json").exists() else None
    edits_raw = jload(d / "edits_argon.json") if (d / "edits_argon.json").exists() else None
    cased = {(x["s"], x["e"]): x["w"] for x in pk["words"]}
    seg_starts = [s["start"] for s in segs]
    spans = speaker_spans(cid)

    def speaker(t):
        if spans and (o := override(spans, t)):
            return o
        i = bisect.bisect_right(seg_starts, t) - 1
        for k in (i, i + 1):
            if 0 <= k < len(segs) and segs[k]["start"] - 0.5 <= t <= segs[k]["end"] + 0.5:
                nm = segs[k]["speaker"]
                return "unknown" if nm.startswith("Unknown") or nm == "other" else nm
        return "unknown"

    # --- the text layer
    if merged is None:  # the five tapes whose first-run download failed: Parakeet v5 only, no merge, no correction
        layer = "parakeet-v5-only" if pk["words"] else "none"
        words = [{"w": x["w"], "s": x["s"], "e": x["e"], "src": "parakeet", **({"c": x["c"]} if "c" in x else {})}
                 for x in pk["words"]]
        cased = {}
    else:
        layer, words = "merged", merged
    n_applied = n_prop = 0
    edits_out, apply_check = [], "n/a"
    if merged is not None and edits_raw is not None:
        edits = [dict(e) for e in edits_raw]
        for e in edits:
            e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
        starts = [int(a) * 60 + int(b) for a, b in
                  re.findall(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
        words, status = apply_edits_tracked(merged, edits, starts)
        ref_words, ref_applied, _ = apply_edits(merged, edits, starts, keep_unclear=True)
        apply_check = "ok" if ([w["w"] for w in ref_words] == [w["w"] for w in words] and ref_applied == sum(status)) else "MISMATCH"
        n_applied = sum(status)
        words, n_prop = propagate(words, edits)
        layer = "merged+llm"
        for e, ok in zip(edits_raw, status):
            o = {k: e.get(k) for k in ("block", "old", "new", "confidence", "reason")}
            o["applied"] = ok
            if e.get("parsed"):
                o["parsed"] = e["parsed"]
            edits_out.append(o)
    # --- v1.1 (listener_fixes.py): words inside withdrawal tones dropped, corrections heard by the author applied
    def n_real(ws):  # tokens as the pipeline counts them (corrected.json): not the corrector's [unclear] placeholders
        return len(ws) - sum(1 for w in ws if w.get("src") == "llm" and w["w"] == "[unclear]")

    pre_tokens, had_words = n_real(words), bool(words)  # the correction status below describes the text before v1.1
    words, _ = drop_tone_words(words, pk.get("tones") or [])
    n_tone = pre_tokens - n_real(words)
    words, fixes = apply_words(cid, words)
    words, n_poss = apply_possessives(cid, words)  # v1.2: "agnew is phone" -> "agnew's phone"
    rows = display_rows(words, cased, speaker)
    n_unclear = sum(1 for r in rows if r[0] == "[unclear]" and r[5] == "llm-edit")
    n_tokens = n_real(words)
    explorer_words = row["words"]
    token_check = "n/a"
    if (d / "corrected.json").exists():
        token_check = "ok" if pre_tokens == len(jload(d / "corrected.json")) else "MISMATCH"
    word_count = n_tokens

    # --- argon status
    failed = (usage or {}).get("failed") or []
    if merged is None:
        argon = "not-run" if had_words else "no-words"
    elif not had_words:
        argon = "no-words"
    elif failed:
        argon = "failed"
    elif edits_raw:
        argon = "corrected"
    else:
        argon = "no-edits"

    # --- speaker segments and turns
    seg_out = []
    for s in segs:
        nm = s["speaker"]
        poss = nm[len("Unknown (possibly "):-1] if nm.startswith("Unknown (possibly ") else ""
        seg_out.append([s["start"], s["end"], "unknown" if nm.startswith("Unknown") or nm == "other" else nm, poss,
                        s.get("confidence")])
    turns, cur = [], None
    for r in rows:
        if cur and cur["speaker"] == r[3] and r[1] - cur["end"] < TURN_GAP:
            cur["parts"].append(r[0]); cur["end"] = r[2]
        else:
            cur and turns.append(cur)
            cur = {"speaker": r[3], "start": r[1], "end": r[2], "parts": [r[0]]}
    cur and turns.append(cur)
    turns_out = [{"speaker": t["speaker"], "start": t["start"], "end": t["end"], "text": " ".join(t["parts"])} for t in turns]
    sp_words = Counter(r[3] for r in rows)
    named = [(n, k) for n, k in sp_words.most_common() if n != "unknown" and k >= 20]
    confs = [r[4] for r in rows if r[4] is not None]
    tones = pk.get("tones") or []

    iso = row["iso"] or ""
    rec = {
        "format": f"nixon-corpus/{VERSION}",
        "id": cid,
        "tape": cid.split("-")[0],
        "date": row["date"],
        "start_time": row["start"], "end_time": row["end"], "start_iso": iso,
        "nara_location_code": job["location"], "nara_location": LOCATIONS.get(job["location"], job["location"]),
        "nara_participants": job["participants"],
        "nara_summary": row["summary"],
        "nara_subject_log": row["log"],
        "audio": {"urls": job["urls"], "fallback_urls": job.get("fallbacks", []), "parts": len(job["urls"]),
                  "duration_s": pk.get("audio_seconds"),
                  "note": "urls are the NARA catalog or Nixon Library files this transcript was made from (first choice); "
                          "fallback_urls are alternate copies of the same conversation"},
        "withdrawn_tone_spans": tones,
        "text_layer": layer,
        "correction": {"status": argon, "edits_total": len(edits_out), "edits_applied": n_applied,
                       "edits_not_located": len(edits_out) - n_applied, "propagated_units": n_prop,
                       "unclear_stretches": n_unclear, "failed_parts": [f.get("part") for f in failed]},
        "listener_corrections": {"words": [{k: f[k] for k in ("at", "old", "new", "reason")} for f in fixes],
                                 "speaker_spans": [{k: s[k] for k in ("start", "end", "speaker", "reason")} for s in spans]},
        "stats": {"word_count": word_count, "words_dropped_in_tones": n_tone, "possessives_restored": n_poss,
                  "display_units": len(rows),
                  "words_with_confidence": len(confs),
                  "named_speaker_words": sum(k for n, k in sp_words.items() if n != "unknown")},
        "speaker_segments_fields": ["start", "end", "speaker", "possibly", "confidence"],
        "speaker_segments": seg_out,
        "word_fields": WORD_FIELDS,
        "words": rows,
        "edits": edits_out,
        "turns": turns_out,
    }
    gz_write(out_dir / "conversations" / f"{cid}.json.gz", rec)

    # --- plain text
    head = [f"# Nixon Corpus v{VERSION} | conversation {cid} | {row['date']} | {row['start'] or '?'} - {row['end'] or '?'} | "
            f"{rec['nara_location']} ({job['location']})",
            f"# NARA participants: {'; '.join(job['participants'])}",
            f"# Machine transcript, not an official record. Text layer: {layer}; LLM correction: {argon}. "
            f"Speaker names come from a voiceprint classifier (UNKNOWN = not named). Listen to the audio before quoting."]
    body = [f"[{hms(t['start'])}] {t['speaker'].upper()}: {t['text']}" for t in turns_out] or ["(no speech recognized)"]
    (out_dir / "transcripts" / f"{cid}.txt").write_text("\n".join(head + [""] + body) + "\n", encoding="utf-8", newline="\n")

    # --- raw layers
    raw = {"id": cid, "parakeet_v5": {"model": "Parakeet TDT, fine-tuned (v5)", "audio_seconds": pk.get("audio_seconds"),
                                       "fields": ["text", "start", "end", "confidence"],
                                       "words": [[x["w"], x["s"], x["e"], x.get("c")] for x in pk["words"]]},
           "whisper": None if wh is None else {"model": "whisper-large-v3", "fields": ["text", "start", "end", "probability"],
                                               "words": [[x["w"], x["s"], x["e"], x.get("p")] for x in wh["words"]]},
           "speaker_segments_fields": ["start", "end", "text", "confidence", "speaker"],
           "speaker_segments": [[s["start"], s["end"], s.get("text"), s.get("confidence"), s["speaker"]] for s in segs]}
    gz_write(out_dir / "raw" / f"{cid}.json.gz", raw)

    return {
        "id": cid, "tape": rec["tape"], "date": row["date"], "start_time": row["start"], "end_time": row["end"], "start_iso": iso,
        "nara_location_code": job["location"], "nara_location": rec["nara_location"],
        "duration_s": pk.get("audio_seconds"), "word_count": word_count,
        "text_layer": layer, "correction_status": argon,
        "edits_total": len(edits_out), "edits_applied": n_applied, "propagated_units": n_prop, "unclear_stretches": n_unclear,
        "words_with_confidence": len(confs),
        "named_speaker_share": round(rec["stats"]["named_speaker_words"] / len(rows), 3) if rows else "",
        "named_speakers": ";".join(n for n, _ in named),
        "withdrawn_spans": len(tones), "withdrawn_seconds": round(sum(b - a for a, b in tones), 1),
        "words_dropped_in_tones": n_tone, "listener_fixes": len(fixes) + len(spans), "possessives_restored": n_poss,
        "audio_parts": len(job["urls"]), "audio_alt_source": row["alt_source"],
        "audio_urls": "|".join(job["urls"]), "audio_fallback_urls": "|".join("+".join(f) for f in job.get("fallbacks", [])),
        "nara_participants": "; ".join(job["participants"]), "nara_summary": row["summary"],
        "explorer_word_count": explorer_words, "apply_check": apply_check, "token_check": token_check,
    }


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def make_tar(path: Path, files: list[Path]):
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as t:
        for f in files:
            ti = tarfile.TarInfo(f.name)
            ti.size = f.stat().st_size
            ti.mtime = 0
            ti.mode = 0o644
            with open(f, "rb") as fh:
                t.addfile(ti, fh)


def make_zip(path: Path, files: list[Path]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            zi = zipfile.ZipInfo(f.name, date_time=(2026, 10, 25, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, f.read_bytes())


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def build(limit: int | None, workers: int) -> int:
    t0 = time.time()
    jobs = {}
    for l in open(CORPUS / "jobs.jsonl", encoding="utf-8"):
        if l.strip():
            j = json.loads(l)
            jobs[j["conv"]] = j
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in db.execute("SELECT id, date, iso, start, end, summary, log, words, layer, alt_source FROM conversations")}
    convs = sorted(jobs)
    assert set(convs) == set(rows), "job list and explorer.db disagree on the conversation set"
    root = RELEASE / (NAME if not limit else f"{NAME}-trial{limit}")
    work = RELEASE / ("_build" if not limit else f"_build-trial{limit}")
    if limit:
        convs = convs[::max(1, len(convs) // limit)][:limit]
    for p in (root, work):
        if p.exists():
            shutil.rmtree(p)
    for sub in ("conversations", "transcripts", "raw"):
        (work / sub).mkdir(parents=True)
    root.mkdir(parents=True)
    print(f"building {len(convs)} conversations with {workers} processes -> {root}", flush=True)
    meta, errors = [], []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, [(c, jobs[c], rows[c], work) for c in convs], chunksize=8), 1):
            (errors if "error" in r else meta).append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(convs)}  {time.time() - t0:.0f}s", flush=True)
    for e in errors[:10]:
        print("ERROR", e["id"], e["error"], e.get("trace", ""), flush=True)
    if errors:
        print(f"{len(errors)} conversations failed; stopping", flush=True)
        return 1
    meta.sort(key=lambda r: r["id"])

    cols = list(meta[0])
    with open(root / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        w.writerows(meta)
    with open(root / "uncorrected_conversations.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "date", "word_count", "correction_status", "text_layer"])
        for r in meta:
            if r["word_count"] and r["correction_status"] in ("failed", "not-run"):
                w.writerow([r["id"], r["date"], r["word_count"], r["correction_status"], r["text_layer"]])

    # --- archives
    by_year: dict[str, list[str]] = {}
    for r in meta:
        by_year.setdefault((r["date"] or "0000")[:4], []).append(r["id"])
    archives = []
    for year, ids in sorted(by_year.items()):
        for kind in ("conversations", "raw"):
            p = root / f"{NAME}-{kind}-{year}.tar"
            make_tar(p, [work / kind / f"{c}.json.gz" for c in ids])
            archives.append((p, len(ids)))
    p = root / f"{NAME}-transcripts-txt.zip"
    make_zip(p, [work / "transcripts" / f"{r['id']}.txt" for r in meta])
    archives.append((p, len(meta)))

    # --- manifest and checksums
    sums = {}
    for p in sorted(root.iterdir()):
        if p.is_file():
            sums[p.name] = sha256(p)
    chk = Counter()
    for r in meta:
        chk["apply_check_" + r["apply_check"]] += 1
        chk["token_check_" + r["token_check"]] += 1
    tot_words = sum(r["word_count"] for r in meta)
    manifest = {
        "name": NAME, "version": VERSION, "built_pt": time.strftime("%Y-%m-%d %H:%M"), "conversations": len(meta),
        "with_words": sum(1 for r in meta if r["word_count"]), "words": tot_words,
        "words_dropped_in_tones": sum(r["words_dropped_in_tones"] for r in meta),
        "convs_with_words_dropped_in_tones": sum(1 for r in meta if r["words_dropped_in_tones"]),
        "convs_emptied_by_tones": sum(1 for r in meta if r["words_dropped_in_tones"] and not r["word_count"]),
        "listener_fixes": sum(r["listener_fixes"] for r in meta),
        "possessives_restored": sum(r["possessives_restored"] for r in meta),
        "convs_with_possessives_restored": sum(1 for r in meta if r["possessives_restored"]),
        "audio_hours": round(sum(r["duration_s"] or 0 for r in meta) / 3600, 1),
        "text_layer": dict(Counter(r["text_layer"] for r in meta)),
        "correction_status": dict(Counter(r["correction_status"] for r in meta)),
        "checks": dict(chk),
        "files": [{"name": p.name, "bytes": p.stat().st_size, "size": human(p.stat().st_size), "files": n,
                   "sha256": sums[p.name]} for p, n in archives],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    write_readme(root, manifest, meta)
    card = REPO / "docs" / "DATA_CARD.md"
    if card.exists():
        shutil.copy(card, root / "DATA_CARD.md")
    sums = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (root / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("conversations", "with_words", "words", "words_dropped_in_tones",
                                                "convs_with_words_dropped_in_tones", "convs_emptied_by_tones",
                                                "listener_fixes", "possessives_restored", "audio_hours",
                                                "text_layer", "correction_status", "checks")}, indent=1))
    for f in manifest["files"]:
        print(f"  {f['name']:52s} {f['size']:>10s}  {f['files']} files")
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


def write_readme(root: Path, m: dict, meta: list[dict]):
    files = "\n".join(f"| `{f['name']}` | {f['size']} | {f['files']:,} | `{f['sha256'][:16]}...` |" for f in m["files"])
    st = m["correction_status"]
    text = f"""# The Nixon Corpus, release v{VERSION}

Machine transcripts, with speaker names, of {m['conversations']:,} White House tape conversations released as individual
recordings ({m['audio_hours']:,} audio hours; {m['with_words']:,} conversations have words, {m['words']:,} words).
Full documentation, accuracy figures and limitations are in `DATA_CARD.md`. Read its limitations section before quoting
anything: **these are machine transcripts, a finding aid, not the official record. Listen to the audio before you quote.**

## Contents

| File | Size | Conversations | SHA-256 (start) |
|---|---|---|---|
{files}
| `metadata.csv` | {human((root / 'metadata.csv').stat().st_size)} | {m['conversations']:,} | one row per conversation |
| `uncorrected_conversations.csv` | small | list | conversations whose LLM correction did not run or failed (see below) |
| `MANIFEST.json`, `SHA256SUMS`, `DATA_CARD.md` | | | |

Full checksums are in `SHA256SUMS` (`sha256sum -c SHA256SUMS`).

* `*-conversations-<year>.tar`: one `<id>.json.gz` per conversation, the full record (words, speakers, edits, turns).
* `*-transcripts-txt.zip`: one `<id>.txt` per conversation, readable, one line per speaker turn with a timestamp.
* `*-raw-<year>.tar`: the raw layers, Parakeet v5 words with confidence and Whisper large-v3 words with probability, plus
  the speaker segments. Optional; the conversation files already contain what most uses need.
* `metadata.csv`: one row per conversation (date, times, NARA location, participants, duration, word count, correction
  status, audio URLs, quality flags).

## Reading a conversation file

```python
import gzip, json, tarfile
tar = tarfile.open("{NAME}-conversations-1973.tar")
rec = json.load(gzip.open(tar.extractfile("871-004.json.gz")))
rec["word_fields"]    # {WORD_FIELDS}
for text, start, end, speaker, conf, source, edit in rec["words"][:5]:
    print(start, speaker, text)
```

## Correction status (important)

The corpus was corrected by a language model (Google Gemini 4 Argon) in a hybrid by-day run (METHODS 7k, 7l). In this release
{st.get('corrected', 0):,} conversations carry at least one edit, {st.get('no-edits', 0):,} came back from the corrector with
no edits (mostly very short calls), and **{st.get('failed', 0):,} conversations ({sum(r['word_count'] for r in meta if r['correction_status'] == 'failed'):,} words) are
uncorrected because their correction requests failed** (quota error, 1972-10-13 to 1973-03-07; their `argon_usage.json`
records the failure). {st.get('not-run', 0)} further conversations with Parakeet words were never merged or corrected.
For these the text is the Parakeet v5 + Whisper merge; the field `correction.status` in each file and the column
`correction_status` in `metadata.csv` say which case applies. `uncorrected_conversations.csv` lists them.

## Release v1.2

This is the first public release. Release v1.1 was prepared but never published; v1.2 adds one change to it:

* **Possessives restored.** Corrected words are stored in normalized form, where a possessive "'s" written by the correction model or a Whisper fill had become the word "is" ("agnew is phone"). {m.get('possessives_restored', 0):,} such words in {m.get('convs_with_possessives_restored', 0):,} conversations are joined back ("agnew's phone"): where Whisper's raw output has the possessive form, or where the word after "is" is a common noun (a real "is" is not followed by a bare noun; names, titles and time words excluded) (`src/possessives.py`, `docs/possessive_fixes.json`; `stats.possessives_restored` in each record).

## Release v1.1

It differs from the internal build v1.0 in two ways (METHODS 7s):

* **Words over withdrawn passages removed.** NARA's withdrawal tones were muted to silence before transcription, so any
  word placed inside one was invented. {m.get('words_dropped_in_tones', 0):,} such words in
  {m.get('convs_with_words_dropped_in_tones', 0):,} conversations were dropped (`stats.words_dropped_in_tones`); in
  {m.get('convs_emptied_by_tones', 0)} of them these were the only words, so their `word_count` is now 0 (their
  `correction.status` still describes the correction step).
* **Corrections confirmed by listening.** Word and speaker corrections heard by the author replace the machine output
  (source `listener-edit` for words; `listener_corrections` in each record lists them; {m.get('listener_fixes', 0)} in
  this release).

## License

The transcripts, annotations and edit lists are released under CC BY 4.0: reuse requires attribution, given by citing the
paper and this release (see `DATA_CARD.md`). The recordings and the government transcripts and finding aids they derive
from are public domain.

## Citation and contact

Arvin Akoopie (ORCID 0000-0001-7833-4172). Corrections are welcome through the repository's issue tracker. Full citation in `DATA_CARD.md`.
"""
    (root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def finalize() -> int:
    """Rewrite README.md, DATA_CARD.md and SHA256SUMS of a built release (after editing the README text or the card)."""
    root = RELEASE / NAME
    m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    meta = [{**r, "word_count": int(r["word_count"])} for r in csv.DictReader(open(root / "metadata.csv", encoding="utf-8"))]
    write_readme(root, m, meta)
    shutil.copy(REPO / "docs" / "DATA_CARD.md", root / "DATA_CARD.md")
    sums = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (root / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8", newline="\n")
    print(f"finalized {root}")
    return 0


def make_sample(ids: list[str]) -> int:
    root = RELEASE / NAME
    work = RELEASE / "_build"
    out = REPO / "data" / "sample"
    for sub in ("conversations", "transcripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meta = list(csv.DictReader(open(root / "metadata.csv", encoding="utf-8")))
    keep = [r for r in meta if r["id"] in ids]
    assert len(keep) == len(ids), f"missing: {set(ids) - {r['id'] for r in keep}}"
    total = 0
    for r in keep:
        for sub, ext in (("conversations", ".json.gz"), ("transcripts", ".txt")):
            shutil.copy(work / sub / f"{r['id']}{ext}", out / sub / f"{r['id']}{ext}")
            total += (out / sub / f"{r['id']}{ext}").stat().st_size
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(keep[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    print(f"sample: {len(keep)} conversations, {human(total)} -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="trial run on about this many conversations, spread over the corpus")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--finalize", action="store_true", help="rewrite README, DATA_CARD copy and SHA256SUMS of the built release")
    ap.add_argument("--sample", nargs="+", metavar="ID", help="copy these conversations from the built release into data/sample/")
    args = ap.parse_args()
    if args.finalize:
        return finalize()
    if args.sample:
        return make_sample(args.sample)
    return build(args.limit, args.workers)


if __name__ == "__main__":
    sys.exit(main())
