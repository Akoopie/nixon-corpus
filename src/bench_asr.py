"""Model bake-off harness: run ASR configurations over the validation audio, then score.

    python src/bench_asr.py run whisperx-batched fw-sequential --refs 741-002_trial 038-034_wspf
    python src/bench_asr.py score       # table of every config x reference run so far

Each run writes bench/<config>/<audio_key>.json (same segment format as
transcribe.py, plus timing). References that share audio (886-008 has three)
are all scored against the one run. Runs on CUDA when available, else CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from paths import AUDIO_DIR, BENCH_DIR
from score import REFS, score_one, load_units, norm_words


def device() -> tuple[str, str]:
    import torch
    return ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8")


def audio_key(ref: dict) -> str:
    first = Path(ref["audio"][0])
    if first.parent.parent == AUDIO_DIR / "conversations":
        return first.parent.name
    # trial_exhibit_12_01.mp3 + _02 + _03 -> trial_exhibit_12; a single file keeps its stem
    return re.sub(r"_0\d$", "", first.stem) if len(ref["audio"]) > 1 else first.stem


def wav_for(ref: dict, key: str) -> Path:
    """One 16 kHz mono WAV per audio key (multi-part recordings are concatenated)."""
    out = BENCH_DIR / "_audio" / f"{key}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        inputs = [a for p in ref["audio"] for a in ("-i", p)]
        n = len(ref["audio"])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                        "-filter_complex", f"concat=n={n}:v=0:a=1", "-ar", "16000", "-ac", "1", str(out)],
                       check=True)
    return out


def read_wav(wav: Path):
    """16 kHz mono 16-bit WAV -> float32 array, without PyAV (newer PyAV releases
    dropped an argument faster-whisper's own decoder still passes)."""
    import wave
    import numpy as np
    with wave.open(str(wav), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# --- ASR configurations ------------------------------------------------------
# Each takes a 16 kHz WAV path and returns [{"start", "end", "text"}, ...].

def run_whisperx_batched(wav: Path) -> list[dict]:
    """The June pipeline: WhisperX batched large-v3 (no fallback, thresholds unused)."""
    import whisperx
    dev, ctype = device()
    model = whisperx.load_model("large-v3", dev, compute_type=ctype, language="en")
    result = model.transcribe(whisperx.load_audio(str(wav)), batch_size=8)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]


def run_fw_sequential(wav: Path) -> list[dict]:
    """faster-whisper large-v3, sequential, with temperature fallback and the
    compression / log-prob / no-speech checks that batched mode skips."""
    from faster_whisper import WhisperModel
    dev, ctype = device()
    model = WhisperModel("large-v3", device=dev, compute_type=ctype)
    segments, _ = model.transcribe(
        read_wav(wav), language="en", beam_size=5, condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.6, vad_filter=False)
    return [{"start": s.start, "end": s.end, "text": s.text.strip(), "avg_logprob": s.avg_logprob}
            for s in segments]


def run_whisperx_chunk20(wav: Path) -> list[dict]:
    """WhisperX batched with the fix suggested for dropped windows: 20 s chunks
    and a more sensitive VAD (onset 0.35 / offset 0.25 instead of 0.5 / 0.363)."""
    import whisperx
    dev, ctype = device()
    model = whisperx.load_model("large-v3", dev, compute_type=ctype, language="en",
                                vad_options={"chunk_size": 20, "vad_onset": 0.35, "vad_offset": 0.25})
    result = model.transcribe(whisperx.load_audio(str(wav)), batch_size=16, chunk_size=20)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]


PARAKEET_CHUNK_S = 600


def run_parakeet_v3(wav: Path) -> list[dict]:
    """NVIDIA Parakeet TDT 0.6B v3 (a transducer: it drops words rather than looping).
    Local attention lets it take a whole tape in one pass; if that runs out of
    GPU memory, the audio is cut into 10-minute pieces instead."""
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.change_attention_model("rel_pos_local_attn", [256, 256])
    model.change_subsampling_conv_chunking_factor(1)

    def segments(path: str, offset: float = 0.0) -> list[dict]:
        out = model.transcribe([path], timestamps=True)[0]
        return [{"start": s["start"] + offset, "end": s["end"] + offset, "text": s["segment"].strip()}
                for s in out.timestamp["segment"]]

    try:
        return segments(str(wav))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    import soundfile as sf
    audio, sr = sf.read(str(wav))
    segs, step = [], PARAKEET_CHUNK_S * sr
    for i in range(0, len(audio), step):
        piece = wav.with_name(f"{wav.stem}.chunk{i // step}.wav")
        sf.write(str(piece), audio[i:i + step], sr)
        segs += segments(str(piece), offset=i / sr)
        piece.unlink()
    return segs


def _parakeet_windows(wav: Path, source: str, window_s: int = 300) -> list[dict]:
    """Parakeet over fixed 5-minute windows (how the batch would run it; whole-file
    decoding of long tapes degraded badly on 421-018). `source` is a hub name or .nemo path."""
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    model = (nemo_asr.models.ASRModel.restore_from(source) if source.endswith(".nemo")
             else nemo_asr.models.ASRModel.from_pretrained(source))
    audio, sr = sf.read(str(wav))
    segs, step = [], window_s * sr
    for i in range(0, len(audio), step):
        piece = wav.with_name(f"{wav.stem}.win{i // step}.wav")
        sf.write(str(piece), audio[i:i + step], sr)
        out = model.transcribe([str(piece)], timestamps=True)[0]
        segs += [{"start": s["start"] + i / sr, "end": s["end"] + i / sr, "text": s["segment"].strip()}
                 for s in out.timestamp["segment"]]
        piece.unlink()
    return segs


def run_parakeet_v3_windows(wav: Path) -> list[dict]:
    return _parakeet_windows(wav, "nvidia/parakeet-tdt-0.6b-v3")


def run_parakeet_nixon(wav: Path) -> list[dict]:
    """The Nixon-tape fine-tune (path in env PARAKEET_FT), same 5-minute windows."""
    import os
    return _parakeet_windows(wav, os.environ["PARAKEET_FT"])


QWEN_CHUNK_S = 120


def run_qwen3_asr(wav: Path) -> list[dict]:
    """Qwen3-ASR-1.7B (Jan 2026, LLM-based, strong on noisy short-form audio).
    Decoded in 2-minute pieces, one segment per piece: scoring needs the words,
    and its forced aligner (for timestamps) is a separate model."""
    import torch
    import soundfile as sf
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=8, max_new_tokens=2048)
    audio, sr = sf.read(str(wav))
    step, segs = QWEN_CHUNK_S * sr, []
    for i in range(0, len(audio), step):
        piece = audio[i:i + step]
        r = model.transcribe(audio=(piece, sr), language="English")[0]
        segs.append({"start": i / sr, "end": (i + len(piece)) / sr, "text": getattr(r, "text", str(r)).strip()})
    return segs


CONFIGS = {
    "whisperx-batched": run_whisperx_batched,
    "whisperx-chunk20": run_whisperx_chunk20,
    "fw-sequential": run_fw_sequential,
    "parakeet-v3": run_parakeet_v3,
    "qwen3-asr": run_qwen3_asr,
    "parakeet-v3-win": run_parakeet_v3_windows,
    "parakeet-nixon": run_parakeet_nixon,
}


def cmd_run(configs: list[str], ref_names: list[str], name: str | None = None) -> int:
    refs = {n: json.loads((REFS / f"{n}.json").read_text(encoding="utf-8")) for n in ref_names}
    keys = {}
    for n, r in refs.items():
        keys.setdefault(audio_key(r), r)
    for cfg in configs:
        for key, ref in keys.items():
            label = name or cfg  # --name keeps a new model's results apart from earlier runs of the same config
            out = BENCH_DIR / label / f"{key}.json"
            if out.exists():
                print(f"{label} / {key}: already run, skipping")
                continue
            t0 = time.time()
            try:
                wav = wav_for(ref, key)
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                            "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout)
                print(f"{cfg} / {key}: {dur / 60:.1f} min of audio ...", flush=True)
                t0 = time.time()
                segs = CONFIGS[cfg](wav)
            except Exception as e:  # one failing model/file must not sink the whole run
                import traceback
                traceback.print_exc()
                print(f"   FAILED {cfg} / {key}: {e}", flush=True)
                continue
            elapsed = time.time() - t0
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"config": label, "audio_key": key, "device": device()[0],
                                       "audio_seconds": dur, "elapsed_seconds": elapsed,
                                       "segments": segs}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"   done in {elapsed / 60:.1f} min ({dur / elapsed:.1f}x realtime), {len(segs)} segments")
    return 0


# The project's purpose is room-mic audio (Oval Office, EOB, Cabinet Room); phone
# calls already transcribe well and are tracked only as a regression check.
ROOM_REFS = ["886-008_wspf", "741-002_trial", "779-002_wspf", "488-015_nara", "421-018_wspf", "051-001_wspf"]
PHONE_REFS = ["038-034_wspf", "037-205_trial"]


def summarize(rows: dict[str, dict[str, dict]]) -> None:
    """Word-weighted ROOM score (the headline) and phone score per config."""
    print(f"\n{'config':18} {'ROOM words':>10} {'ROOM Nixon':>10} {'ROOM short':>10} {'phone words':>11}  room refs")
    for cfg, by_ref in sorted(rows.items()):
        room = [by_ref[r] for r in ROOM_REFS if r in by_ref]
        phone = [by_ref[r] for r in PHONE_REFS if r in by_ref]
        if not room:
            continue
        w = lambda rs, k: sum(r[k] * r["N"] for r in rs) / max(sum(r["N"] for r in rs), 1)
        print(f"{cfg:18} {w(room, 'complete'):10.1%} {w(room, 'nixon_recall'):10.1%} "
              f"{w(room, 'short_recall'):10.1%} {w(phone, 'complete') if phone else float('nan'):11.1%}  "
              f"{len(room)}/{len(ROOM_REFS)}")


def cmd_score(bench_dir: Path = BENCH_DIR) -> int:
    refs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(REFS.glob("*.json"))}
    rows: dict[str, dict[str, dict]] = {}
    print(f"{'config':18} {'reference':16} {'used':>5} {'WER':>6} {'del':>6} {'ins':>6} {'complete':>9} "
          f"{'names':>6} {'nixon':>6} {'short':>6} {'speed':>8}")
    for run in sorted(bench_dir.glob("*/*.json")):
        if run.parent.name.startswith("_"):
            continue
        meta = json.loads(run.read_text(encoding="utf-8"))
        if "config" not in meta:
            continue  # June-era files
        hyp = [w for _, t, _ in load_units(run) for w in norm_words(t)]
        for name, ref in refs.items():
            if audio_key(ref) != meta["audio_key"]:
                continue
            r = score_one(REFS / f"{name}.json", hyp)
            rows.setdefault(run.parent.name, {})[name] = r  # by folder: renamed or --name runs stay apart
            speed = meta["audio_seconds"] / meta["elapsed_seconds"]
            print(f"{run.parent.name:18} {name:16} {r['ref_used']:5.0%} {r['wer']:6.1%} {r['del']:6.1%} {r['ins']:6.1%} "
                  f"{r['complete']:9.1%} {r['name_recall']:6.0%} {r['nixon_recall']:6.0%} "
                  f"{r['short_recall']:6.0%} {speed:6.1f}x {meta['device']}")
    summarize(rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("configs", nargs="*", help=f"configs to run: {', '.join(CONFIGS)}")
    ap.add_argument("--refs", nargs="*", default=[], help="reference names whose audio to transcribe")
    ap.add_argument("--dir", type=Path, default=BENCH_DIR, help="score runs under this directory")
    ap.add_argument("--name", help="run: save under this name instead of the config name (one config only)")
    args = ap.parse_args()
    if args.cmd == "score":
        return cmd_score(args.dir)
    unknown = set(args.configs) - set(CONFIGS)
    if unknown or not args.configs or not args.refs:
        ap.error(f"need configs from {list(CONFIGS)} and --refs")
    return cmd_run(args.configs, args.refs, args.name)


if __name__ == "__main__":
    sys.exit(main())
