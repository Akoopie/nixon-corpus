"""Find candidate passages for "never before published" claims, by code (METHODS 7m).

  registry  conversations that already have a published transcript, from sources on disk:
              WSPF transcripts (NARA's list of the special prosecutor's transcripts)
              NARA's own transcript files in the subject-log folders
              the House Judiciary Committee / trial / validation references
              FRUS documents whose editors transcribed the tape (the source note says so)
            FRUS footnotes that only cite a conversation are recorded as "cited", not as transcribed.
            Not yet covered (checked by hand later): the Miller Center's PRDE, Brinkley & Nichter's
            The Nixon Tapes, Kutler's Abuse of Power, nixontapes.org, books and newspapers.
  rank      60-second windows of Argon-corrected conversations outside the registry, scored by
              charged terms (money, surveillance, the FBI and IRS, leaks, threats, slurs, cover-ups...)
              x names in the window that the conversation's NARA subject log never mentions
              x how clearly the audio was heard (share of words the recognizer was sure of)
            one best window per conversation -> <out>/candidates.jsonl, top ones printed

    python src/discover.py registry
    python src/discover.py rank [--top 40]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from paths import DATA_DIR, MANIFEST_DIR, VALIDATION_DIR

OUT = DATA_DIR / "discover"
CORPUS = DATA_DIR / "corpus_v5" / "out"
EXPLORER_DB = DATA_DIR / "explorer" / "explorer.db"
FRUS = DATA_DIR / "lm" / "sources" / "frus"
TEI = "{http://www.tei-c.org/ns/1.0}"
CONV_RE = re.compile(r"Conversation (?:No\.\s*)?(\d{1,3})\s*[–-]\s*(\d{1,3})\b")
CHARGED = re.compile(
    r"\b(irs|tax returns?|audit(ed|ing)?|fbi|cia|hoover|wiretap\w*|tap(ped|ping)? (his|her|their|the) (phone|line)s?|bug(ged|ging)?|"
    r"surveillance|leak(s|ed|er|ers|ing)?|plumbers|break ?in|burglar\w*|cover ?up|hush|blackmail|bribe\w*|kickback\w*|"
    r"cash|slush|contribut(ion|ions|or|ors)|secret(ly)?|illegal\w*|indict\w*|perjur\w*|obstruct\w*|clemency|pardon\w*|"
    r"destroy (him|them|it)|ruin (him|them)|get (him|them)|screw (him|them)|kill(ed|ing)? (him|them)|enemies|smear|dirt on|"
    r"jews?|jewish|blacks?|negro\w*|nigger\w*|kike\w*|wop\w*|homosexual\w*|queers?|"
    r"bomb(ed|ing|s)? (hanoi|haiphong|the dikes|them)|nuclear|assassinat\w*|coup|overthrow|"
    r"payoff|pay ?off|shredd?\w*|burn (the|them|it)|files? on)\b")


def conv_id(tape: str, num: str) -> str:
    return f"{int(tape):03d}-{int(num):03d}"


def cmd_registry() -> dict:
    reg: dict[str, set] = {}
    add = lambda c, why: c and reg.setdefault(c, set()).add(why)
    for l in open(MANIFEST_DIR / "sources" / "wspf_list.jsonl", encoding="utf-8"):
        if l.strip():
            add(json.loads(l)["conv"], "WSPF transcript")
    logs = MANIFEST_DIR / "subject_logs" / "extracted"
    for f in logs.rglob("transcript_*"):
        m = re.search(r"(\d{3})-(\d{3})$", f.stem)
        if m:
            add(conv_id(*m.groups()), "NARA transcript")
    for f in (VALIDATION_DIR / "refs").glob("*.json"):
        m = re.match(r"(\d{3})-(\d{3})_", f.name)
        if m:
            add(conv_id(*m.groups()), f"reference ({f.stem.split('_', 1)[1]})")
    cited = {}
    for f in sorted(FRUS.glob("frus1969-76*.xml")):
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == TEI + "div" and el.get("type") == "document":
                src = " ".join("".join(n.itertext()) for n in el.iter(TEI + "note") if n.get("type") == "source")
                transcribed = bool(re.search(r"transcribed|prepared (this|the) transcript", src, re.I))
                for m in CONV_RE.finditer(src):
                    if transcribed:
                        add(conv_id(*m.groups()), f"FRUS transcript ({f.stem})")
                body = " ".join("".join(n.itertext()) for n in el.iter(TEI + "note") if n.get("type") != "source")
                for m in CONV_RE.finditer(body + " " + (src if not transcribed else "")):
                    cited.setdefault(conv_id(*m.groups()), set()).add(f.stem)
                el.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"transcribed": {c: sorted(v) for c, v in sorted(reg.items())},
           "frus_cited": {c: sorted(v) for c, v in sorted(cited.items()) if c not in reg}}
    (OUT / "registry.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    kinds = {}
    for v in reg.values():
        for w in v:
            k = w.split(" (")[0]
            kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(reg)} conversations already transcribed somewhere on disk: {kinds}; "
          f"{len(out['frus_cited'])} more only cited in FRUS footnotes")
    return out


def cmd_rank(top: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "explorer"))
    reg = json.loads((OUT / "registry.json").read_text(encoding="utf-8"))
    skip = set(reg["transcribed"])
    db = sqlite3.connect(EXPLORER_DB)
    meta = {r[0]: r[1:] for r in db.execute("SELECT id, date, start, place, participants, log FROM conversations WHERE layer = 'corrected'")}
    names = {r[0].lower() for r in db.execute("SELECT DISTINCT surname FROM people WHERE kind = 'person' AND LENGTH(surname) >= 4")}
    from build import name_like_surnames
    names = name_like_surnames(names)
    rows = []
    for conv, (date, start, place, parts, log) in meta.items():
        if conv in skip or not (CORPUS / conv / "corrected.json").exists():
            continue
        words = json.loads((CORPUS / conv / "corrected.json").read_text(encoding="utf-8"))
        if len(words) < 150:
            continue
        logl = (log or "").lower() + " " + (parts or "").lower()
        best = None
        i = 0
        while i < len(words):
            t0, j = words[i]["s"], i
            while j < len(words) and words[j]["s"] < t0 + 60:
                j += 1
            win = words[i:j]
            text = " ".join(w["w"] for w in win)
            charged = len(CHARGED.findall(text))
            if charged >= 2:
                new = {w["w"] for w in win if w["w"] in names and w["w"] not in logl}
                conf = [w["c"] for w in win if w.get("c") is not None]
                clear = sum(c >= 0.3 for c in conf) / len(conf) if len(conf) >= 20 else 0.5
                score = charged * (1 + 0.5 * len(new)) * clear
                if not best or score > best[0]:
                    best = (score, t0, charged, sorted(new), round(clear, 2), text)
            i = max(j, i + 1)
        if best:
            rows.append({"conv": conv, "date": date, "start": start, "place": place, "score": round(best[0], 2), "t": best[1],
                         "charged": best[2], "unlogged_names": best[3], "clear": best[4], "frus_cited": conv in reg["frus_cited"],
                         "participants": parts, "text": best[5]})
    rows.sort(key=lambda r: -r["score"])
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} conversations with a candidate window (of {len(meta)} Argon-corrected; {len(skip)} skipped as already transcribed)")
    for r in rows[:top]:
        print(f"{r['score']:6.1f} {r['conv']} {r['date']} {r['start']:>8} {r['place'][:12]:12} at {int(r['t'] // 60)}:{int(r['t'] % 60):02d} "
              f"charged {r['charged']} clear {r['clear']} new names {r['unlogged_names'][:5]}{' FRUS-cited' if r['frus_cited'] else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["registry", "rank"])
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    cmd_registry() if args.cmd == "registry" else cmd_rank(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
