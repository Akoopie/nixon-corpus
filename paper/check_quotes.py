"""Check every corpus quotation in the paper against the final text layer (METHODS 7s).

The final layer is what release v1.1 ships: the merged transcript with the Argon edits applied (correct.apply_edits,
keep_unclear, then propagate, as in export_release.py), words inside withdrawal tones dropped, and the corrections the
author confirmed by ear (docs/listener_corrections.json) applied. A quotation passes when its words occur in that layer
contiguously, in order (segments separated by an ellipsis may skip words between them). Quotations that the paper
presents as the recognizers' text before correction are checked against the merged layer instead.

    python paper/check_quotes.py          # every quotation in QUOTES; also lists \\machine{} quotes in the .tex not covered
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from correct import propagate  # noqa: E402
from export_release import apply_edits_tracked, CORPUS  # noqa: E402
from listener_fixes import apply_words, drop_tone_words  # noqa: E402
from score import norm_words  # noqa: E402

SECTIONS = ROOT / "paper" / "arxiv" / "sections"
FINAL, MERGED = "final", "merged"

# (conversation, layer, quotation, where it appears). Ellipses are written "[...]".
QUOTES = [
    # Figure 1 (make_fig1.py): Haldeman gives his name on the telephone
    ("018-078", MERGED, "this is bob alderman", "Fig. 1 Parakeet/merged"),
    ("018-078", FINAL, "hello mister president this is bob haldeman 0 general hughes", "Fig. 1 released"),
    # Section 4.3: real mishearings that Whisper heard right
    ("003-118", MERGED, "check a date with the whole different tomorrow", "4.3 'the whole different'"),
    ("003-118", FINAL, "check a date with haldeman tomorrow", "4.3 corrected"),
    ("014-154", MERGED, "generally please thank you", "4.3 'generally'"),
    ("014-154", FINAL, "john ehrlichman please thank you", "4.3 corrected"),
    # Section 5.4: a same-day correction from a parallel recording
    ("476-021", MERGED, "want doctor burns marjorie burns i am pleased to call", "5.4 Burns, before"),
    ("001-061", MERGED, "find doctor burns arthur burns i placed a call", "5.4 Burns, the telephone recording itself"),
    ("476-021", FINAL, "find doctor burns arthur burns i placed a call", "5.4 Burns, after"),
    ("001-061", FINAL, "find doctor burns arthur burns i placed a call", "5.4 Burns, after (parallel)"),
    # Section 7.1, 871-004
    ("871-004", FINAL, "i want to put gray in there i told him that hoover had told me and i am going to make him lie "
     "because i think gray is not handling himself well what he said yesterday with regard to the fact that it was "
     "that you know this jolly well bullshit and all that sort of thing that you were that it was highly improper for "
     "you to be present i am god damn it we are not going to let him get away with that", "7.1 Dean passage"),
    ("871-004", FINAL, "that in the 1968 campaign that the fbi at the request of the white house never say whom at the "
     "request of the white house it is an electronic the electronic surveillance of vice president agnew is phone when "
     "he was in phoenix is this true or not", "7.1 Agnew passage"),
    # Section 7.2, 916-019
    ("916-019", FINAL, "the problem with the memcon does not involve you but it does potentially involve john according "
     "to according to al who and al should not talk to you about it for the reason that it makes it appear it could "
     "make it appear that john was trying to get walters to talk to gray [...] now my point is john is story should be "
     "perfect should be and i hope you talked to john john is story is perfect john should say his only concern was "
     "the whole cia involvement and so forth and wanted to be sure that the this investigation of watergate did not "
     "get out as he put it did not get out of hand [...] getting out of hand had nothing to do with with apprehending "
     "the watergate criminals and trying them getting out of hand had to do with exactly what is now happening yeah "
     "which is unloading all this pentagon papers investigation type stuff the leak the plumbers operation type stuff "
     "the exactly wiretapping of the reporters and and staff members type stuff", "7.2 main passage"),
    ("916-019", FINAL, "ehrlichman and me", "7.2"), ("916-019", FINAL, "dean was not there", "7.2"),
    ("916-019", FINAL, "ehrlichman did do all the talking", "7.2"), ("916-019", FINAL, "bob", "7.2"),
    ("916-019", FINAL, "be sure that you and john talked exactly that way", "7.2"),
    ("916-019", FINAL, "john ehrlichman", "7.2"), ("916-019", FINAL, "according to al", "7.2"),
    ("916-019", FINAL, "and we have still got another one as we know that that i think we have got to figure it out henry",
     "7.2 henry"),
    ("916-019", MERGED, "i i do not need to faint picture", "7.2 'faint' before the author's correction"),
    ("916-019", MERGED, "set up a phone story", "7.2 'phone' before the correction step"),
    ("916-019", FINAL, "i i do not need to paint a picture i just want to be sure that john does not indicate that the "
     "president directed him to get walters to get gray to turn this off and set up a phony story", "7.2 Nixon"),
    ("916-019", FINAL, "john would never indicate that in the 1st place [...] a it was not true b that is under the that "
     "area that we are not we do not testify in any way which is both good and bad what the president tells us to do is "
     "not not public public consumption", "7.2 Haldeman"),
    # Section 7.3, 021-068
    ("021-068", FINAL, "instead of doing the business of the government this congress is this is a smear a day congress "
     "rather than a law day of congress", "7.3"),
    ("021-068", FINAL, "it will go down in history as a smear a day rather than the law day", "7.3"),
    # Appendix table
    ("789-006", FINAL, "super apathy among the blacks", "App."), ("789-006", FINAL, "just exactly where you want it", "App."),
    ("789-006", FINAL, "not too keen for mcgovern", "App."),
    ("233-010", FINAL, "make sweden the black post i want a black for sweden", "App."),
    ("690-011", FINAL, "get at least 150 out of it", "App."),
    ("819-002", FINAL, "liddy is case will collapse", "App. (paper prints the normalized form)"),
    ("224-015", FINAL, "a loyalist", "App."),
    # Figure 3, "Corrected layer" lines (src/whisper_artifacts.py EXAMPLES)
    ("421-018", FINAL, "but john you you want to do that well i do not think it is good now let us let us try another let "
     "us try another comment for that supposing mitchell were to step out on that same day", "Fig. 3a"),
    ("886-008", FINAL, "wonder i am just trying to think as to why that we just finished the moscow trip i mean we were "
     ". that is right democrats were not doing that magruder", "former Fig. 4b (no longer drawn)"),
]
EMPTY = [("099-010", 0.0, 3342.0, "Fig. 3b: no words over the withdrawn passage")]


def final_words(conv: str) -> list[dict]:
    d = CORPUS / "out" / conv
    merged = json.loads((d / "merged.json").read_text(encoding="utf-8"))
    words = merged
    if (d / "edits_argon.json").exists():
        edits = json.loads((d / "edits_argon.json").read_text(encoding="utf-8"))
        for e in edits:
            e["old"] = str(e.get("old", "")).replace("{", "").replace("}", "")
        starts = [int(a) * 60 + int(b) for a, b in
                  re.findall(r"^\[(\d+):(\d+)\]", (d / "correct_input.txt").read_text(encoding="utf-8"), re.M)]
        words, _ = apply_edits_tracked(merged, edits, starts)
        words, _ = propagate(words, edits)
    tones = json.loads((d / "parakeet_v5.json").read_text(encoding="utf-8")).get("tones") or []
    words, _ = drop_tone_words(words, tones)
    words, _ = apply_words(conv, words)
    return words


def merged_words(conv: str) -> list[dict]:
    return json.loads((CORPUS / "out" / conv / "merged.json").read_text(encoding="utf-8"))


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def find(seg: list[str], toks: list[str], start: int) -> int:
    n = len(seg)
    for i in range(start, len(toks) - n + 1):
        if toks[i:i + n] == seg:
            return i
    return -1


def check(conv: str, layer: str, quote: str) -> tuple[bool, str]:
    words = final_words(conv) if layer == FINAL else merged_words(conv)
    toks = [w["w"] for w in words]
    segs = [s.split() for s in quote.split("[...]")]
    pos, spans = 0, []
    for seg in segs:
        i = find(seg, toks, pos)
        if i < 0:
            # report the longest matching prefix, to show where the quotation departs from the layer
            best = (0, -1)
            for k in range(len(seg), 0, -1):
                j = find(seg[:k], toks, pos)
                if j >= 0:
                    best = (k, j)
                    break
            k, j = best
            ctx = " ".join(toks[j + k:j + k + 8]) if j >= 0 else ""
            return False, (f"departs after {k} of {len(seg)} words"
                           + (f" at {mmss(words[j]['s'])}: quote has {' '.join(seg[k:k + 6])!r}, layer has {ctx!r}"
                              if j >= 0 else ": first words not found"))
        spans.append(f"{mmss(words[i]['s'])}-{mmss(words[i + len(seg) - 1]['e'])}")
        pos = i + len(seg)
    return True, ", ".join(spans)


def norm(text: str) -> str:
    """Whisper-normalized words (case, punctuation and contractions removed), padded for whole-word matching."""
    return " " + " ".join(t for w in text.split() for t in norm_words(w)) + " "


def uncovered() -> list[str]:
    """\\machine{} quotations in the .tex whose words (normalized: the paper quotes the released transcript with its
    casing and punctuation) are not part of a quotation in QUOTES, which are checked against the corpus above."""
    known = [norm(p) for _, _, q, _ in QUOTES for p in q.split("[...]")]
    out = []
    for f in sorted(SECTIONS.glob("*.tex")):
        for m in re.finditer(r"\\machine\{([^}]*)\}", f.read_text(encoding="utf-8")):
            q = re.sub(r"\s+", " ", m.group(1).replace("[\\ldots]", "[...]")).strip()
            if not all(any(norm(part) in k for k in known) for part in q.split("[...]")):
                out.append(f"{f.name}: {q[:90]}")
    return out


def main() -> int:
    bad = 0
    for conv, layer, q, where in QUOTES:
        ok, info = check(conv, layer, q)
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {conv} [{layer}] {where}: {info}")
    for conv, a, b, where in EMPTY:
        n = sum(1 for w in final_words(conv) if a <= w["s"] < b)
        bad += n > 0
        print(f"{'OK  ' if n == 0 else 'FAIL'} {conv} [final] {where}: {n} words in {mmss(a)}-{mmss(b)}")
    rest = uncovered()
    print(f"\\machine quotations in the .tex not covered above: {len(rest)}")
    for r in rest:
        print("   ", r)
    print(f"{bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
