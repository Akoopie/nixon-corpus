"""Nixon-era n-gram language model for Parakeet's decoder (shallow fusion in NeMo's TDT decoding).

The model is a KenLM n-gram over Parakeet's own subword pieces, which NeMo 3.0 loads as a GPU
n-gram (NGramGPULanguageModel) and adds to the decoder's scores with weight ngram_lm_alpha.

  text    sources -> <lm>/text/<source>.txt, one sentence per line, cased and punctuated like the
          ASR training targets (finetune_parakeet.tidy), a blank line between documents; speaker
          labels ("Mr. Clements:") are stripped:
            wspf      WSPF transcripts of the training conversations (spoken; same people)
            nara      NARA's own transcripts (transcript_<loc>-<tape>-<conv>.rtf in the subject-log
                      folders; RTF preferred over the PDF copy), except of held-out conversations
            diary     H. R. Haldeman's dictated diary
            logs      NARA subject logs (topic outlines) and the names / acronyms / places indexes
            frus      FRUS 1969-76: document text, without the editors' notes
  filter  held-out text = the validation references and the dev conversations' transcripts. A
          document with >= 3 lines and >= 10% of its lines sharing an 8-word sequence with it is
          dropped whole (probably a copy of a held-out conversation), otherwise just those lines.
          In the written sources (diary, logs, frus), lines repeated more than 50 times are page
          furniture and dropped; spoken sources keep their "Yeah." -> <lm>/clean/<source>.txt
  tokens  clean lines -> Parakeet's subword ids, one line per sentence, each id written as
          chr(id + 100) (NeMo's DEFAULT_TOKEN_OFFSET) -> <lm>/tokens/<source>.txt
  build   KenLM lmplz (run in WSL) -> <lm>/<name>.arpa
  boost   per-tape phrase lists for NeMo's boosting tree -> <out>/<conv>.txt: NARA's participants
          (surname, first + surname, nickname) and the names and places in the tape's subject-log
          entry that NARA's names / places indexes know, unless the word is mostly an ordinary
          word in the era text ("White", "Price")

    python src/era_lm.py text
    python src/era_lm.py filter
    python src/era_lm.py tokens --nemo <parakeet .nemo>      (nemotest env: needs sentencepiece)
    python src/era_lm.py build --name era6 --order 6 [--sources wspf diary logs frus]
    python src/era_lm.py boost --jobs jobs.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from paths import DATA_DIR, MANIFEST_DIR, VALIDATION_DIR

LM_DIR = DATA_DIR / "lm"
WSPF_DIR = DATA_DIR / "training" / "transcripts"
DIARY_DIR = DATA_DIR / "haldeman" / "text"
LOG_ROOT = MANIFEST_DIR / "subject_logs" / "extracted" / "Nixon White House Tapes" / "Complete Conversations"
FRUS_DIR = LM_DIR / "sources" / "frus"
# fine-tuning dev conversations (METHODS §4): the weight of the LM is tuned on them
DEV_CONVS = {"429-018", "538-015", "856-004", "862-006", "903-006"}
HELD_CONVS = {"886-008", "741-002", "779-002", "488-015", "421-018", "051-001", "038-034", "037-205"} | DEV_CONVS
OVERLAP_N = 8
DOC_HITS, DOC_SHARE = 3, 0.10  # held-out lines that make a whole document count as held-out text
BOILERPLATE = 50   # a line repeated more often than this in one written source is page furniture
WRITTEN = {"diary", "logs", "frus"}
TOKEN_OFFSET = 100
TEI = "{http://www.tei-c.org/ns/1.0}"
SOURCES = ("wspf", "nara", "diary", "logs", "frus")


def tidy(text: str) -> str:
    """As finetune_parakeet.tidy: transcript typography -> plain text the tokenizer knows."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace('"', "")
    text = text.replace("�", "'")
    text = re.sub(r"\s*(--|—|–)\s*", ", ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r",\s*([,.?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Gen|Sen|Rep|Gov|St|Jr|Sr|Lt|Col|Adm|Amb|No|U\.S|D\.C|Vol|p)\.$")
# "Mr. Clements:", "The President:", "Kissinger:", "HALDEMAN:" at the start of a line
LABEL = re.compile(r"^((Mr|Mrs|Ms|Dr|Gen|Amb|Adm|Sen|Secretary|Ambassador|General|President|The President|"
                   r"Chairman|Prime Minister|Foreign Minister|Minister)\.? )?[A-Z][A-Za-z'\-]+( [A-Z][A-Za-z'\-]+)?\s*:\s+")


def sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, not after common abbreviations (Mr., U.S.)."""
    out, cur = [], ""
    for piece in SENT_END.split(tidy(LABEL.sub("", paragraph.strip()))):
        cur = f"{cur} {piece}".strip()
        if not ABBREV.search(cur):
            out.append(cur)
            cur = ""
    return out + ([cur] if cur else [])


def text_wspf() -> list[list[str]]:
    docs = []
    for f in sorted(WSPF_DIR.glob("*.json")):
        if f.stem in DEV_CONVS:
            continue
        docs.append([s for t in json.loads(f.read_text(encoding="utf-8"))["turns"] for s in sentences(t["text"])])
    return docs


def text_diary() -> list[list[str]]:
    docs = []
    for f in sorted(DIARY_DIR.glob("*.txt")):
        body = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
                if "Haldeman Diaries Collection" not in l and "Online Public Access Catalog" not in l
                and not l.strip().startswith("Page |")]
        lines, para = [], []
        for l in body + [""]:  # blank lines end paragraphs; pdftotext -layout wrapped the rest
            if l:
                para.append(l[:-1] if l.endswith("-") and len(l) > 1 and l[-2].isalpha() else l + " ")
            elif para:
                lines += sentences("".join(para))
                para = []
        docs.append(lines)
    return docs


def log_files(transcripts: bool) -> list[str]:
    """Subject-log folder files, one per document (the RTF when a PDF copy exists too): NARA's
    transcripts (transcript_oval-472-004) or the logs (oval-472, 37-wht-audiotape-477-log)."""
    by_stem = {}
    for f in sorted(glob.glob(str(LOG_ROOT / "Logs and Transcripts" / "**" / "*.*"), recursive=True)):
        stem, ext = Path(f).stem, Path(f).suffix.lower()
        if ext in (".rtf", ".pdf") and stem.startswith("transcript_") == transcripts:
            if stem not in by_stem or ext == ".rtf":
                by_stem[stem] = f
    return [f for s, f in sorted(by_stem.items())
            if not (m := re.search(r"(\d{3})-(\d{3})$", s)) or f"{m.group(1)}-{m.group(2)}" not in HELD_CONVS]


def read_log(f: str) -> str:
    from striprtf.striprtf import rtf_to_text
    if f.lower().endswith(".rtf"):
        return rtf_to_text(open(f, encoding="cp1252", errors="replace").read())
    return subprocess.run(["pdftotext", f, "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def text_nara() -> list[list[str]]:
    docs = []
    for f in log_files(transcripts=True):
        paras = re.split(r"\n\s*\n", read_log(f))
        docs.append([s for p in paras for s in sentences(" ".join(p.split()))
                     if not re.match(r"^(Page \d|Conversation No\.|NARA|Nixon Presidential|\*The quotation)", s)])
    return docs


def text_logs() -> list[list[str]]:
    from striprtf.striprtf import rtf_to_text
    docs = []
    for f in log_files(transcripts=False):
        lines = []
        for l in read_log(f).splitlines():
            l = tidy(LABEL.sub("", l.strip().lstrip("-• \t")))
            if (len(l.split()) >= 2 and not re.match(r"^(Page \d|Conversation No\.|NARA|Nixon Presidential)", l)
                    and re.search(r"[A-Za-z]{3,}", re.sub(r"\b[ap]m\b", "", l))):  # not a bare clock time
                lines.append(l)
        docs.append(lines)
    for name in ("Names", "Acronyms", "Geograph"):
        f = LOG_ROOT / "Indices" / "Acronyms, Geographic, Names" / f"{name}.rtf"
        lines = []
        for l in rtf_to_text(open(f, encoding="cp1252", errors="replace").read()).splitlines():
            l = re.sub(r":\s*[\d,\s\-]+$", "", l.strip())  # drop the page/tape numbers
            if name == "Names" and (m := re.match(r"^([^,(]+),\s*([^,(]+?)(,\s*(Jr|Sr|II|III)\.?)?(\s*\(.*\))?$", l)):
                l = f"{m.group(2).strip()} {m.group(1).strip()}" + (f", {m.group(4)}." if m.group(4) else "")
            if l and len(l) > 2:
                lines.append(tidy(l))
        docs.append(lines)
    return docs


def tei_text(el, skip=("note",)) -> str:
    """An element's text without the content of editors' <note>s (their tails are kept)."""
    parts = [el.text or ""]
    for ch in el:
        if ch.tag.replace(TEI, "") not in skip:
            parts.append(tei_text(ch, skip))
        parts.append(ch.tail or "")
    return "".join(parts)


def text_frus() -> list[list[str]]:
    docs = []
    for f in sorted(FRUS_DIR.glob("frus1969-76*.xml")):
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == TEI + "div" and el.get("type") == "document":
                docs.append([s for p in el.iter(TEI + "p") for s in sentences(tei_text(p))])
                el.clear()
    return docs


def norm8(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def read_docs(path: Path) -> list[list[str]]:
    docs, cur = [], []
    for l in path.read_text(encoding="utf-8").splitlines() + [""]:
        if l:
            cur.append(l)
        elif cur:
            docs.append(cur)
            cur = []
    return docs


def cmd_text() -> None:
    (LM_DIR / "text").mkdir(parents=True, exist_ok=True)
    for name, fn in (("wspf", text_wspf), ("nara", text_nara), ("diary", text_diary), ("logs", text_logs), ("frus", text_frus)):
        docs = [[l for l in d if l.strip()] for d in fn()]
        docs = [d for d in docs if d]
        (LM_DIR / "text" / f"{name}.txt").write_text("\n\n".join("\n".join(d) for d in docs) + "\n", encoding="utf-8", newline="\n")
        print(f"{name}: {len(docs):,} documents, {sum(map(len, docs)):,} lines, "
              f"{sum(len(l.split()) for d in docs for l in d):,} words", flush=True)


def cmd_filter() -> None:
    held = set()
    refs = list((VALIDATION_DIR / "refs").glob("*.json")) + [WSPF_DIR / f"{c}.json" for c in sorted(DEV_CONVS)]
    for f in refs:
        d = json.loads(f.read_text(encoding="utf-8"))
        units = d["turns"] if "turns" in d else d.get("segments", [])
        w = norm8(" ".join(u["text"] for u in units))
        held.update(tuple(w[i:i + OVERLAP_N]) for i in range(len(w) - OVERLAP_N + 1))
    print(f"{len(refs)} held-out texts, {len(held):,} distinct {OVERLAP_N}-word sequences")
    (LM_DIR / "clean").mkdir(parents=True, exist_ok=True)
    from collections import Counter
    for name in SOURCES:
        docs = read_docs(LM_DIR / "text" / f"{name}.txt")
        often = {l for l, n in Counter(l for d in docs for l in d).items() if n > BOILERPLATE} if name in WRITTEN else set()
        kept, dropped, n_docs, n_lines, n_boiler = [], [], 0, 0, 0
        for d in docs:
            hit = [any(tuple(w[i:i + OVERLAP_N]) in held for i in range(len(w) - OVERLAP_N + 1)) for w in map(norm8, d)]
            if sum(hit) >= DOC_HITS and sum(hit) >= DOC_SHARE * len(d):
                dropped += d + [""]
                n_docs += 1
                continue
            for l, h in zip(d, hit):
                if h:
                    dropped.append(l)
                    n_lines += 1
                elif l in often:
                    n_boiler += 1
                else:
                    kept.append(l)
        (LM_DIR / "clean" / f"{name}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
        (LM_DIR / "clean" / f"{name}.dropped.txt").write_text("\n".join(dropped) + "\n", encoding="utf-8", newline="\n")
        print(f"{name}: kept {len(kept):,} lines ({sum(len(l.split()) for l in kept):,} words); dropped {n_docs} whole documents "
              f"and {n_lines} more lines sharing an {OVERLAP_N}-word sequence with held-out text, {n_boiler:,} boilerplate lines "
              f"({len(often)} distinct)")


def cmd_tokens(nemo: Path) -> None:
    import sentencepiece as spm
    with tarfile.open(nemo) as t:
        m = next(x for x in t.getmembers() if x.name.endswith("_tokenizer.model"))
        (LM_DIR / "tokenizer.model").write_bytes(t.extractfile(m).read())
    sp = spm.SentencePieceProcessor(model_file=str(LM_DIR / "tokenizer.model"))
    print(f"tokenizer from {nemo.name}: {sp.get_piece_size()} pieces")
    (LM_DIR / "tokens").mkdir(parents=True, exist_ok=True)
    dev = [s for c in sorted(DEV_CONVS) for t in json.loads((WSPF_DIR / f"{c}.json").read_text(encoding="utf-8"))["turns"]
           for s in sentences(t["text"])]
    (LM_DIR / "clean" / "dev.txt").write_text("\n".join(dev) + "\n", encoding="utf-8", newline="\n")
    for name in SOURCES + ("dev",):
        lines = (LM_DIR / "clean" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        n = 0
        with open(LM_DIR / "tokens" / f"{name}.txt", "w", encoding="utf-8", newline="\n") as f:
            for ids in sp.encode(lines):
                if ids:
                    f.write(" ".join(chr(i + TOKEN_OFFSET) for i in ids) + "\n")
                    n += len(ids)
        print(f"{name}: {n:,} tokens")


def cmd_build(name: str, order: int, sources: list[str], prune: str) -> None:
    """One lmplz model per source, then KenLM's log-linear interpolation with weights tuned on the dev
    conversations' transcripts (tokens/dev.txt), all in WSL (~/kenlm/build/bin, libraries in ~/kenlm-env/lib)."""
    def wsl(p: Path) -> str:
        return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]
    kenlm, work = "LD_LIBRARY_PATH=~/kenlm-env/lib ~/kenlm/build/bin", f"~/lm_work/{name}"
    steps = [f"rm -rf {work} && mkdir -p {work}"]
    for s in sources:
        steps.append(f"{kenlm}/lmplz -o {order} --discount_fallback --prune {prune} -S 30% -T /tmp/lmplz_ "
                     f"--intermediate {work}/{s} < '{wsl(LM_DIR / 'tokens' / f'{s}.txt')}' 2> {work}/{s}.log")
    out = LM_DIR / f"{name}.arpa"
    models = " ".join(f"{work}/{s}" for s in sources)
    tune = wsl(LM_DIR / "tokens" / "dev.txt")
    steps.append(f"{kenlm}/interpolate -m {models} -t '{tune}' --just_tune > {work}/weights.txt 2> {work}/tune.log")
    steps.append(f"{kenlm}/interpolate -m {models} -w $(cat {work}/weights.txt) -S 30% -T /tmp/interp_ > '{wsl(out)}' 2> {work}/interp.log")
    for st in steps:
        print(st, flush=True)
        subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", "set -o pipefail; " + st], check=True)
    weights = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", f"cat {work}/weights.txt"],
                             capture_output=True, text=True).stdout.split()
    (LM_DIR / f"{name}.weights.txt").write_text(" ".join(f"{s}={w}" for s, w in zip(sources, weights)) + "\n", encoding="utf-8")
    print(f"{out}: {out.stat().st_size / 1e6:.0f} MB; weights " + ", ".join(f"{s} {w}" for s, w in zip(sources, weights)))


MAX_PHRASES = 80
CAPSEQ = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:\s+(?:[A-Z]\.\s+)?[A-Z][a-zA-Z'\-]+){0,2}")


def cmd_boost(jobs_path: Path, out: Path) -> None:
    from collections import Counter
    from striprtf.striprtf import rtf_to_text
    from correct import nara_log
    idx = LOG_ROOT / "Indices" / "Acronyms, Geographic, Names"
    full, surnames, places = set(), set(), set()
    for l in rtf_to_text(open(idx / "Names.rtf", encoding="cp1252", errors="replace").read()).splitlines():
        if m := re.match(r"^([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?),\s*([A-Z][A-Za-z'\-]+)", l.strip()):
            surnames.add(m.group(1)); full.add(f"{m.group(2)} {m.group(1)}")
    for l in rtf_to_text(open(idx / "Geograph.rtf", encoding="cp1252", errors="replace").read()).splitlines():
        if p := re.sub(r":\s*[\d,\s\-]+$", "", l.strip()):
            places.add(p)
    case = Counter()  # how often each word is capitalized vs lowercase in the spoken and FRUS text
    for name in ("wspf", "nara", "frus"):
        for w in re.findall(r"[A-Za-z'\-]+", (LM_DIR / "clean" / f"{name}.txt").read_text(encoding="utf-8")):
            case[w] += 1
    ordinary = lambda w: case[w.lower()] > case[w]  # "White", "Price", "Long": mostly ordinary words
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for job in jobs:
        phrases = []
        for p in job["participants"]:
            if m := re.match(r"^([^,\[(]+),\s*([A-Za-z.\s]+?)(?:,\s*(?:Jr|Sr|III?)\.?)?\s*(?:\((?:\"(\w+)\")?.*\))?$", p.strip()):
                last, first = m.group(1).strip(), m.group(2).split()[0].strip(".")
                phrases += [last] + ([f"{first} {last}"] if len(first) > 1 else [])  # not "H Haldeman"
                phrases += [m.group(3), f"{m.group(3)} {last}"] if m.group(3) else []
        for seq in CAPSEQ.findall(nara_log(job)):
            seq = " ".join(seq.split())
            if seq in full or seq in places or (seq in surnames and not ordinary(seq) and len(seq) > 3):
                phrases.append(seq)
        seen, keep = set(), []
        for p in phrases:
            if p and p not in seen and not all(ordinary(w) for w in p.split()):
                seen.add(p); keep.append(p)
        (out / f"{job['conv']}.txt").write_text("\n".join(keep[:MAX_PHRASES]) + "\n", encoding="utf-8", newline="\n")
        print(f"{job['conv']}: {len(keep[:MAX_PHRASES])} phrases: {', '.join(keep[:12])}{' ...' if len(keep) > 12 else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["text", "filter", "tokens", "build", "boost"])
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--nemo", type=Path)
    ap.add_argument("--name", default="era6")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--prune", default="0 0 1", help="lmplz --prune: count thresholds per order (singletons of order >= 3 dropped)")
    args = ap.parse_args()
    if args.cmd == "text":
        cmd_text()
    elif args.cmd == "filter":
        cmd_filter()
    elif args.cmd == "tokens":
        cmd_tokens(args.nemo)
    elif args.cmd == "boost":
        cmd_boost(args.jobs, args.out)
    else:
        cmd_build(args.name, args.order, args.sources, args.prune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
