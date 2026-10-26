"""Cut (audio clip, human transcript) training pairs out of long conversations.

Per conversation:
  1. ASR anchors: WhisperX large-v3 transcribes the audio and force-aligns its own
     words, giving a time for every word it heard.
  2. The human transcript is aligned to those words; where they agree, the human
     word inherits the time. Each sentence of the human text gets an estimated
     time span from its anchored words (interpolated when it has none).
  3. Sentences are grouped into chunks of up to ~20 s, and the HUMAN text of each
     chunk is force-aligned to its padded time window (wav2vec2 CTC). That times
     every human word, including the ones Whisper missed, which are exactly the
     hard passages the fine-tune should learn from.
  4. A chunk becomes a training clip only if the alignment is confident, enough of
     its words were also heard by Whisper (proof the text belongs to this stretch
     of audio), its speaking rate is plausible and it holds no [unintelligible] or
     other annotations. Kept fillers (uh, um) teach a verbatim style.

Runs on a GPU pod (whisperx env):
    python align_clips.py --turns DIR --audio-root DIR --out DIR [--convs 038-037 ...]
Writes <out>/clips/<conv>/<n>.wav, <out>/manifest.jsonl (NeMo format) and <out>/stats.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

from score import norm_words

SR = 16000
MAX_CHUNK_S, MIN_CLIP_S, MAX_CLIP_S = 20.0, 1.5, 25.0
PAD_S = 1.0                      # search margin around a chunk's estimated span
MIN_ASR_MATCH = 0.30             # share of a chunk's words Whisper also heard
MIN_ALIGN_SCORE = 0.35           # mean wav2vec2 word score
MIN_TIMED = 0.90                 # share of words the aligner could place
WPS_RANGE = (1.0, 6.0)           # plausible speaking rate, words per second
ANNOTATION = re.compile(r"[\[\]()]|unintelligible|inaudible|withdrawn", re.I)


def load_audio(paths: list[Path]):
    import numpy as np
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", f"concat=n={len(paths)}:v=0:a=1", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0


def sentences(turns: list[dict]) -> list[dict]:
    """Split turns into sentences: [{"text", "norm"}], at turn ends, . ? ! and --."""
    out = []
    for t in turns:
        for s in re.split(r"(?<=[.?!])\s+|\s+--\s*|--$", t["text"]):
            s = s.strip()
            if s:
                out.append({"text": s, "norm": norm_words(s)})
    return [s for s in out if s["norm"]]


def asr_words(model, align_model, meta, audio) -> list[dict]:
    import whisperx
    result = model.transcribe(audio, batch_size=16)
    aligned = whisperx.align(result["segments"], align_model, meta, audio, "cuda", return_char_alignments=False)
    words = []
    for w in aligned.get("word_segments", []):
        if "start" in w and "end" in w:
            for n in norm_words(w["word"]):
                words.append({"norm": n, "start": w["start"], "end": w["end"]})
    return words


def estimate_spans(sents: list[dict], hyp: list[dict]) -> None:
    """Give every sentence an estimated (start, end) from anchored words, interpolating gaps."""
    ref_norm = [n for s in sents for n in s["norm"]]
    owner = [i for i, s in enumerate(sents) for _ in s["norm"]]
    times = {}
    sm = SequenceMatcher(None, ref_norm, [h["norm"] for h in hyp], autojunk=False)
    for b in sm.get_matching_blocks():
        if b.size >= 2:
            for k in range(b.size):
                times[b.a + k] = (hyp[b.b + k]["start"], hyp[b.b + k]["end"])
    for s in sents:
        s["matched"] = 0
    for i, (t0, t1) in times.items():
        s = sents[owner[i]]
        s["matched"] += 1
        s["t0"] = min(s.get("t0", t0), t0)
        s["t1"] = max(s.get("t1", t1), t1)
    # Interpolate unanchored sentences between their anchored neighbours.
    anchored = [i for i, s in enumerate(sents) if "t0" in s]
    for i, s in enumerate(sents):
        if "t0" in s:
            continue
        prev = max((j for j in anchored if j < i), default=None)
        nxt = min((j for j in anchored if j > i), default=None)
        if prev is None or nxt is None:
            continue
        s["t0"], s["t1"] = sents[prev]["t1"], sents[nxt]["t0"]
        s["interpolated"] = True


def chunks(sents: list[dict]) -> list[list[dict]]:
    """Consecutive timed sentences grouped into chunks of up to MAX_CHUNK_S. An
    untimed or annotated sentence ("(Inaudible)") breaks the chain, so only that
    sentence is lost rather than its whole chunk."""
    out, cur = [], []
    for s in sents:
        if "t0" not in s or s["t1"] <= s["t0"] or ANNOTATION.search(s["text"]):
            if cur:
                out.append(cur)
            cur = []
            continue
        if cur and s["t1"] - cur[0]["t0"] > MAX_CHUNK_S:
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def main() -> int:
    global MIN_ASR_MATCH, MIN_ALIGN_SCORE, MIN_TIMED  # --min-* options override them
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", type=Path, required=True, help="dir of <conv>.json transcripts")
    ap.add_argument("--audio-root", type=Path, required=True, help="dir holding conversations/<conv>/*.mp3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--convs", nargs="*", help="limit to these conversation ids (smoke test)")
    ap.add_argument("--asr-cache", type=Path, help="save/reuse Whisper word timings here (voiceprints.py reads them)")
    ap.add_argument("--min-asr-match", type=float, help=f"override MIN_ASR_MATCH ({MIN_ASR_MATCH}); round 2 used a looser re-cut")
    ap.add_argument("--min-align-score", type=float, help=f"override MIN_ALIGN_SCORE ({MIN_ALIGN_SCORE})")
    ap.add_argument("--min-timed", type=float, help=f"override MIN_TIMED ({MIN_TIMED})")
    args = ap.parse_args()

    import soundfile as sf
    import whisperx

    model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
    align_model, meta = whisperx.load_align_model(language_code="en", device="cuda")
    MIN_ASR_MATCH = args.min_asr_match if args.min_asr_match is not None else MIN_ASR_MATCH
    MIN_ALIGN_SCORE = args.min_align_score if args.min_align_score is not None else MIN_ALIGN_SCORE
    MIN_TIMED = args.min_timed if args.min_timed is not None else MIN_TIMED
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = open(args.out / "manifest.jsonl", "a", encoding="utf-8", newline="\n")
    stats = {}
    for tpath in sorted(args.turns.glob("*.json")):
        conv = tpath.stem
        if args.convs and conv not in args.convs:
            continue
        mp3s = sorted((args.audio_root / "conversations" / conv).glob("*.mp3"))
        if not mp3s or (args.out / "clips" / conv / "DONE").exists():
            continue
        audio = load_audio(mp3s)
        sents = sentences(json.loads(tpath.read_text(encoding="utf-8"))["turns"])
        cache = args.asr_cache / f"{conv}.json" if args.asr_cache else None
        if cache and cache.exists():
            hyp = json.loads(cache.read_text(encoding="utf-8"))
        else:
            hyp = asr_words(model, align_model, meta, audio)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(hyp), encoding="utf-8")
        estimate_spans(sents, hyp)
        kept = dropped = 0
        reasons: dict[str, int] = {}
        clip_dir = args.out / "clips" / conv
        clip_dir.mkdir(parents=True, exist_ok=True)
        for n, ch in enumerate(chunks(sents)):
            text = " ".join(s["text"] for s in ch)
            nwords = sum(len(s["norm"]) for s in ch)
            why = None
            if ANNOTATION.search(text):
                why = "annotation"
            elif sum(s["matched"] for s in ch) / nwords < MIN_ASR_MATCH:
                why = "asr_mismatch"
            if why is None:
                w0 = max(ch[0]["t0"] - PAD_S, 0.0)
                w1 = min(ch[-1]["t1"] + PAD_S, len(audio) / SR)
                res = whisperx.align([{"start": w0, "end": w1, "text": text}], align_model, meta, audio, "cuda",
                                     return_char_alignments=False)
                words = [w for seg in res["segments"] for w in seg.get("words", [])]
                timed = [w for w in words if "start" in w]
                if not words or len(timed) / len(words) < MIN_TIMED:
                    why = "untimed"
                else:
                    c0, c1 = timed[0]["start"] - 0.15, timed[-1]["end"] + 0.15
                    dur = c1 - c0
                    score = sum(w.get("score", 0) for w in timed) / len(timed)
                    if not MIN_CLIP_S <= dur <= MAX_CLIP_S:
                        why = "duration"
                    elif not WPS_RANGE[0] <= nwords / dur <= WPS_RANGE[1]:
                        why = "rate"
                    elif score < MIN_ALIGN_SCORE:
                        why = "low_score"
            if why:
                dropped += 1
                reasons[why] = reasons.get(why, 0) + 1
                continue
            clip = clip_dir / f"{n:05d}.wav"
            sf.write(str(clip), audio[int(c0 * SR):int(c1 * SR)], SR)
            manifest.write(json.dumps({"audio_filepath": str(clip), "duration": round(dur, 3), "text": text,
                                       "conv": conv, "align_score": round(score, 3)}, ensure_ascii=False) + "\n")
            manifest.flush()
            kept += 1
        (clip_dir / "DONE").write_text("")
        clip_s = sum(json.loads(l)["duration"] for l in open(args.out / "manifest.jsonl", encoding="utf-8")
                     if json.loads(l)["conv"] == conv)
        stats[conv] = {"audio_min": round(len(audio) / SR / 60, 1), "clips": kept, "clip_min": round(clip_s / 60, 1),
                       "dropped": dropped, "reasons": reasons}
        print(f"{conv}: {stats[conv]}", flush=True)
    prev = json.loads((args.out / "stats.json").read_text()) if (args.out / "stats.json").exists() else {}
    prev.update(stats)
    (args.out / "stats.json").write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
