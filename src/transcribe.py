"""Transcribe + diarize a single Nixon audio file with WhisperX.

Pipeline (per the research-recommended stack):
  1. faster-whisper backend (int8 on CPU) for the transcript
  2. wav2vec2 forced alignment -> word-level timestamps
  3. pyannote diarization -> speaker labels, merged onto segments

Output: data/transcripts/<conversation_id>.json  (list of segments with
speaker / start / end / text), ready for src/index.py.

Usage:
    python src/transcribe.py data/audio/oval-692-003.mp3 --conv 692-003
    python src/transcribe.py <audio> --conv <id> --model small --no-diarize

Notes:
  * --model: tiny/base/small/medium/large-v3. On a 22-core CPU, "small" is a
    good speed/accuracy MVP default; "large-v3" is most accurate but slow.
  * Diarization needs a Hugging Face token (env HF_TOKEN) AND you must accept
    the model terms for pyannote/speaker-diarization-3.1 on huggingface.co.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from paths import SECRETS_DIR, TRANSCRIPT_DIR


def transcribe(audio: Path, conv_id: str, model_name: str, diarize: bool,
               compute_type: str, language: str | None) -> dict:
    import whisperx

    device = "cpu"
    print(f"[1/3] Loading WhisperX model '{model_name}' ({compute_type}, {device})")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

    print(f"[1/3] Transcribing {audio.name}")
    audio_data = whisperx.load_audio(str(audio))
    result = model.transcribe(audio_data, batch_size=8)
    lang = result.get("language", language or "en")

    print(f"[2/3] Aligning words (lang={lang})")
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio_data, device,
                            return_char_alignments=False)

    if diarize:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            # Fallback: read from a local gitignored file so the secret never
            # has to be pasted into a shell command or chat.
            token_file = SECRETS_DIR / "hf_token.txt"
            if token_file.exists():
                hf_token = token_file.read_text(encoding="utf-8").strip()
        if not hf_token:
            print("[3/3] WARNING: HF_TOKEN not set -> skipping diarization (speakers will be UNKNOWN).")
        else:
            print("[3/3] Diarizing speakers")
            try:
                # DiarizationPipeline moved to whisperx.diarize in recent versions;
                # fall back to the old top-level location for older installs.
                try:
                    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
                except ImportError:
                    from whisperx import DiarizationPipeline, assign_word_speakers
                # Newer whisperx uses token=; older used use_auth_token=.
                try:
                    diar = DiarizationPipeline(token=hf_token, device=device)
                except TypeError:
                    diar = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diar_segments = diar(audio_data)
                result = assign_word_speakers(diar_segments, result)
            except Exception as e:
                print(f"[3/3] Diarization failed ({e}); continuing without speaker labels.")
    else:
        print("[3/3] Diarization disabled (--no-diarize)")

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "conversation_id": conv_id,
            "speaker": seg.get("speaker", "UNKNOWN"),
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "text": (seg.get("text") or "").strip(),
        })
    return {
        "conversation_id": conv_id,
        "audio": str(audio),
        "model": model_name,
        "language": lang,
        "segments": segments,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="path to an audio file (mp3/wav/m4a/...)")
    ap.add_argument("--conv", required=True, help="NARA conversation_id this audio belongs to")
    ap.add_argument("--model", default="small", help="whisper model size (default: small)")
    ap.add_argument("--compute-type", default="int8", help="ctranslate2 compute type (default: int8)")
    ap.add_argument("--language", default="en", help="force language (default: en; '' to auto-detect)")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker diarization")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}", file=sys.stderr)
        return 1

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out = transcribe(
        audio=args.audio,
        conv_id=args.conv,
        model_name=args.model,
        diarize=not args.no_diarize,
        compute_type=args.compute_type,
        language=args.language or None,
    )

    out_path = TRANSCRIPT_DIR / f"{args.conv}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out['segments'])} segments -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
