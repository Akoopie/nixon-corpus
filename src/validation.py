"""Build the validation set: public-domain human transcripts paired with audio.

These are the yardstick for the model bake-off and the LLM correction pilot.
All references are US-government work (Watergate Special Prosecution Force,
House Judiciary Committee 1974, trial exhibits, NARA archivists).

    python src/validation.py fetch    # download transcripts + exhibit audio
    python src/validation.py parse    # write validation/refs/<name>.json for each reference

Each ref JSON holds the speaker turns of the human transcript plus the audio
it belongs to. Most references cover only part of a recording, so scoring
locates the matching stretch of ASR output by alignment, not by timestamps.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from acquire import USER_AGENT, fetch, load_ledger
from paths import AUDIO_DIR, VALIDATION_DIR

SOURCES = VALIDATION_DIR / "sources"
REFS = VALIDATION_DIR / "refs"
TAPES = "https://www.nixonlibrary.gov/sites/default/files/forresearchers/find/tapes/"
WSPF, TRIAL = TAPES + "watergate/wspf/", TAPES + "watergate/trial/"
HJC_URL = "https://archive.org/download/transcriptsofeig00nixo/transcriptsofeig00nixo_djvu.txt"
PDF, TXT = ("application/pdf",), ("text/plain",)
MP3 = ("audio/", "application/octet-stream")


def conv_audio(cid: str) -> list[str]:
    return sorted(str(p) for p in (AUDIO_DIR / "conversations" / cid).glob("*.mp3"))


def src(name: str) -> str:
    return str(SOURCES / name)


# name -> reference definition. "audio" is what the reference was transcribed
# from; "coverage" says how much of that audio the reference covers.
REFERENCES = {
    "886-008_wspf":  dict(conv="886-008", parser="pdf", path=src("wspf_886-008.pdf"), url=WSPF + "886-008.pdf",
                          audio=lambda: conv_audio("886-008"), coverage="10:12-11:55 portion (Dean present)",
                          note="HJC staff transcript of 1974-07-08, reviewed by NARA"),
    "886-008_hjc":   dict(conv="886-008", parser="hjc", path=src("hjc_8rpc_djvu.txt"), url=HJC_URL, pages=(79, 130),
                          audio=lambda: conv_audio("886-008"), coverage="10:12-11:55 portion",
                          note="House Judiciary Committee 1974 print (OCR of BPL scan)"),
    "886-008_trial": dict(conv="886-008", parser="pdf", path=src("trial_exhibit_12.pdf"), url=TRIAL + "exhibit_12.pdf",
                          audio=lambda: [src(f"trial_exhibit_12_0{i}.mp3") for i in (1, 2, 3)],
                          coverage="exhibit excerpt", note="U.S. v. Mitchell exhibit 12"),
    "421-018_wspf":  dict(conv="421-018", parser="pdf", path=src("wspf_421-018.pdf"), url=WSPF + "421-018.pdf",
                          audio=lambda: conv_audio("421-018"), coverage="5:20-6:01 pm portion (last ~41 min)",
                          note="HJC staff transcript, EOB"),
    "038-034_wspf":  dict(conv="038-034", parser="pdf", path=src("wspf_038-034.pdf"), url=WSPF + "038-034.pdf",
                          audio=lambda: conv_audio("038-034"), coverage="full", note="telephone"),
    "038-034_trial": dict(conv="038-034", parser="pdf", path=src("trial_exhibit_21.pdf"), url=TRIAL + "exhibit_21.pdf",
                          audio=lambda: [src("trial_exhibit_21.mp3")], coverage="exhibit excerpt", note="telephone"),
    "051-001_wspf":  dict(conv="051-001", parser="pdf", path=src("wspf_051-001.pdf"), url=WSPF + "051-001.pdf",
                          audio=lambda: conv_audio("051-001"), coverage="nearly full",
                          note="Cabinet Room, ~28 speakers"),
    "488-015_nara":  dict(conv="488-015", parser="pdf", path=src("nara_trnoval488-15.pdf"),
                          url=TAPES + "finding_aids/tapesubjectlogs/trnoval488-15.pdf",
                          audio=lambda: conv_audio("488-015"), coverage="7 partial segments",
                          note="NARA archivist verbatim transcript (keeps uh/false starts)"),
    "741-002_trial": dict(conv="741-002", parser="pdf", path=src("trial_exhibit_01.pdf"), url=TRIAL + "exhibit_01.pdf",
                          audio=lambda: [src("trial_exhibit_01.mp3")], coverage="full (exhibit audio = transcript)",
                          note="June 23, 1972 'smoking gun' excerpt"),
    "779-002_wspf":  dict(conv="779-002", parser="pdf", path=src("wspf_779-002.pdf"), url=WSPF + "779-002.pdf",
                          audio=lambda: conv_audio("779-002"), coverage="per transcript", note="Oval"),
    "037-205_trial": dict(conv="037-205", parser="pdf", path=src("trial_exhibit_14.pdf"), url=TRIAL + "exhibit_14.pdf",
                          audio=lambda: [src("trial_exhibit_14.mp3")], coverage="partial (pp. 5-8 of the call)",
                          note="telephone, Colson"),
}
EXTRA_AUDIO = {f"trial_exhibit_{n}.mp3": TRIAL + f"exhibit_{n}.mp3" for n in ("01", "14", "21")}
EXTRA_AUDIO.update({f"trial_exhibit_12_0{i}.mp3": TRIAL + f"exhibit_12_0{i}.mp3" for i in (1, 2, 3)})


def cmd_fetch() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ledger, failed = load_ledger(), 0
    for name, ref in REFERENCES.items():
        accept = TXT if ref["path"].endswith(".txt") else PDF
        if not fetch(session, ref["url"], Path(ref["path"]), ledger, accept=accept):
            failed += 1
    for fname, url in EXTRA_AUDIO.items():
        if not fetch(session, url, SOURCES / fname, ledger, accept=MP3):
            failed += 1
    print(f"Validation sources in {SOURCES} ({failed} failed)")
    return 1 if failed else 0


# --- parsing -----------------------------------------------------------------

# "PRESIDENT:", "Kissinger:", "UNIDENTIFIED VOICE:" at the start of a block, or the
# abbreviated "P." / "D." / "H" on a line of its own that some WSPF transcripts use.
LABEL = re.compile(r"^\s*([A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){0,3})\s*:\s*(.*)"
                   r"|^\s*([A-Z]{1,3})\.?[ \t]*\n(.*)", re.S)


def _label(m: re.Match) -> tuple[str, str]:
    return (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
HEADER = re.compile(
    r"^(TRANSCRIPT OF|NIXON PRESIDENTIAL MATERIALS|Tape Subject Log|\(rev\.|Conv\. No\.|Conversation No\."
    r"|Transcript #\d|Part of a conversation|\*{3,}|[-\d\s]+$|[A-Z][a-z]+ \d{1,2}, 19\d\d FROM"
    r"|(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+19\d\d)",
    re.I)
NOT_SPEAKERS = {"note", "page", "date", "time", "location", "transcript", "conv", "segment"}
# Words that start lines of speech often enough to look like labels ("Yeah.  That's right").
COMMON_WORDS = {"yeah", "yes", "no", "right", "sure", "okay", "well", "oh", "uh", "uhhuh", "and",
                "now", "good", "so", "but", "i", "all", "fine", "true", "exactly", "huh", "hmm"}


def label_vocabulary(labels: list[str]) -> dict[str, str]:
    """Map raw labels to canonical speaker names. Labels used >= 3 times are real;
    rarer ones are OCR/typing variants ("Presddent", "Pres Ident", "Dear") snapped
    to the closest real label, or kept as-is when nothing is close."""
    from collections import Counter
    from difflib import SequenceMatcher

    def key(s: str) -> str:
        return re.sub(r"[^a-z]", "", s.lower())

    counts = Counter(key(l) for l in labels)
    real = [k for k, n in counts.items() if n >= 3 and k not in COMMON_WORDS]
    vocab = {}
    for raw in set(labels):
        k = key(raw)
        if k in real:
            vocab[raw] = k
            continue
        best = max(real, key=lambda r: SequenceMatcher(None, k, r).ratio(), default=None)
        if best and SequenceMatcher(None, k, best).ratio() >= 0.75:
            vocab[raw] = best
        elif k not in COMMON_WORDS:
            vocab[raw] = k
    return {raw: speaker_name(k) for raw, k in vocab.items()}


def speaker_name(label: str) -> str:
    label = label.strip().rstrip(".")
    if label.lower() in ("president", "thepresident", "the president", "p"):
        return "Nixon"
    return label if len(label) <= 3 else label.title()  # keep abbreviations ("D", "H") as written


def _tidy(turns: list[dict]) -> list[dict]:
    for t in turns:
        t["text"] = re.sub(r"\s+", " ", t["text"]).strip()
    return [t for t in turns if t["text"]]


def parse_pdf(path: str) -> list[dict]:
    """Speaker turns from a WSPF / trial-exhibit / NARA transcript PDF.
    NARA's multi-excerpt transcripts ("Transcript #3: Part of a conversation ...")
    get a `segment` number on each turn, since each excerpt is aligned separately."""
    import pymupdf

    # A block can start with header lines and continue into speech, so headers
    # are stripped line by line; a "Transcript #N" line opens excerpt N.
    blocks = []  # (segment opened by this block or None, remaining text)
    with pymupdf.open(path) as doc:
        for page in doc:
            for b in page.get_text("blocks", sort=True):
                opened, keep = None, []
                for line in b[4].splitlines():
                    s = line.strip()
                    if (m := re.match(r"Transcript #(\d+)", s)):
                        opened = int(m.group(1))
                    elif s and not HEADER.match(s):
                        keep.append(s)
                blocks.append((opened, "\n".join(keep)))
    raw_labels = [_label(m)[0] for _, b in blocks if (m := LABEL.match(b)) and len(_label(m)[0]) <= 40]
    vocab = label_vocabulary(raw_labels)

    turns: list[dict] = []
    segment, in_header = 0, False
    for opened, text in blocks:
        if opened is not None:
            segment, in_header = opened, True
        if not text:
            continue
        m = LABEL.match(text)
        label, body = _label(m) if m else (None, None)
        if m and label in vocab and label.lower() not in NOT_SPEAKERS:
            turns.append({"speaker": vocab[label], "text": body, "segment": segment})
            in_header = False
        elif turns and not in_header:  # header prose before an excerpt's first turn is skipped
            turns[-1]["text"] += " " + text
    return _tidy(turns)


HJC_LABEL = re.compile(r"^([A-Z][A-Za-z]+(?: [A-Z][a-z]+)?)\.(?:\s+(.*)|$)")


def parse_hjc(path: str, pages: tuple[int, int]) -> list[dict]:
    """Turns from the OCR text of the 1974 House Judiciary Committee print,
    restricted to a page range. Page numbers appear as lines holding only digits.

    The OCR sometimes reads the speaker-label column apart from the speech
    ("Dean." alone on a line), so speaker attribution here is unreliable; the
    word sequence is what this reference is good for."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    first, last = pages
    start = end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.isdigit():
            n = int(s)
            if n == first and start is None:
                start = i + 1
            elif n == last + 1 and start is not None:
                end = i
                break
    if start is None:
        raise ValueError(f"page {first} marker not found in {path}")
    body = [l.strip() for l in lines[start:end] if l.strip() and not l.strip().isdigit()]
    vocab = label_vocabulary([m.group(1) for s in body if (m := HJC_LABEL.match(s))])
    turns: list[dict] = []
    for s in body:
        m = HJC_LABEL.match(s)
        if m and m.group(1) in vocab and m.group(1).lower() not in NOT_SPEAKERS:
            turns.append({"speaker": vocab[m.group(1)], "text": m.group(2) or ""})
        elif turns:
            turns[-1]["text"] += " " + s
    return _tidy(turns)


def cmd_parse() -> int:
    REFS.mkdir(parents=True, exist_ok=True)
    print(f"{'reference':16} {'turns':>6} {'words':>7}  speakers")
    for name, ref in REFERENCES.items():
        if not Path(ref["path"]).exists():
            print(f"{name:16} missing source (run fetch)")
            continue
        turns = parse_hjc(ref["path"], ref["pages"]) if ref["parser"] == "hjc" else parse_pdf(ref["path"])
        audio = ref["audio"]()
        out = {"name": name, "conversation_id": "37-wht-conversation-" + ref["conv"],
               "source": ref["path"], "source_url": ref["url"], "note": ref["note"],
               "coverage": ref["coverage"], "audio": audio, "turns": turns}
        (REFS / f"{name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        words = sum(len(t["text"].split()) for t in turns)
        spk = {}
        for t in turns:
            spk[t["speaker"]] = spk.get(t["speaker"], 0) + 1
        top = ", ".join(f"{k} {v}" for k, v in sorted(spk.items(), key=lambda x: -x[1])[:5])
        flag = "" if all(Path(a).exists() for a in audio) and audio else "  [AUDIO MISSING]"
        print(f"{name:16} {len(turns):6} {words:7}  {top}{flag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "parse"])
    args = ap.parse_args()
    return cmd_fetch() if args.cmd == "fetch" else cmd_parse()


if __name__ == "__main__":
    sys.exit(main())
