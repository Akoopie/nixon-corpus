"""Batch worker for the full transcription run.

Each tape gets everything later steps need, so nothing has to be re-run on a GPU:
  parakeet  stage A (NeMo env):  fine-tuned Parakeet, 5-minute windows, word timestamps
  whisper   stage B (whisperx env): Whisper large-v3 (sequential, with fallbacks), word
            timestamps; pyannote speaker separation; a voiceprint for every speaker turn
            >= 2 s and for every speaker overall
Both stages mute withdrawn passages (NARA's 1 kHz tone) before transcribing, and record where they were.

Many pods can run the same command against one job list on a shared network volume:
a tape is claimed with an atomic mkdir, finished tapes are skipped, and a claim not
refreshed for STALE_S (a pod that died) is taken over.

On a fleet pod both stages share one GPU and one download: whisper (the slow stage) walks the
job list, and parakeet --follow transcribes each tape whisper has just downloaded, then sweeps
the job list for leftovers once whisper is done. Each stage touches a done-flag on exit, which
the other reads (--partner-done) to know nobody on the pod still needs a tape's audio.

    python src/worker.py plan --out jobs.jsonl [--tiers A B] [--only 038-034 ...]   # local
    python src/worker.py parakeet --jobs jobs.jsonl --root /workspace/corpus --model FT.nemo [--follow] [--partner-done FLAG]
    python src/worker.py whisper  --jobs jobs.jsonl --root /workspace/corpus [--partner-done FLAG]
    python src/worker.py status   --jobs jobs.jsonl --root /workspace/corpus
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

SR = 16000
HEARTBEAT_S = 5 * 60  # a claim is refreshed this often while its tape is processed (some tapes take an hour)
STALE_S = 15 * 60     # three missed heartbeats: the pod holding the claim is gone
WINDOW_S = 300
CACHE = Path(os.environ.get("WORKER_CACHE", "/root/cache"))
TONE_HZ, TONE_BAND, TONE_SHARE, TONE_MIN_S = 1000.0, 40.0, 0.6, 1.0


# --- planning (local) -----------------------------------------------------------

def participants_by_conv() -> dict[str, list[str]]:
    """NARA's participant list per conversation (from the manifest loaded into the corpus DB)."""
    from db import connect
    conn = connect()
    out = {}
    for row in conn.execute("SELECT conversation_id, participants FROM conversations"):
        names = [p.strip() for p in (row["participants"] or "").split(";") if p.strip()]
        out[row["conversation_id"].removeprefix("37-wht-conversation-")] = names
    conn.close()
    return out


def cmd_plan(out: Path, tiers: list[str], only: list[str] | None) -> int:
    from acquire import conversation_options, read_manifest
    people = participants_by_conv()
    jobs = []
    for r in read_manifest():
        if r["tier"][0] not in tiers or (only and r["conv_id"] not in only):
            continue
        opts = conversation_options(r)
        if opts:
            jobs.append({"conv": r["conv_id"], "location": r["location"], "date": r["date"],
                         "participants": people.get(r["conv_id"], []),
                         "urls": opts[0], "fallbacks": opts[1:], "bytes": int(r["primary_bytes"] or 0)})
    jobs.sort(key=lambda j: -j["bytes"])  # longest first, so the fleet doesn't end on one huge tape
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8", newline="\n")
    print(f"{len(jobs)} jobs -> {out}")
    return 0


# --- shared helpers -------------------------------------------------------------

def load_jobs(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_stamp(path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(str(time.time()))
    tmp.replace(path)  # atomic: a reader never sees a half-written stamp


def claim(root: Path, stage: str, conv: str) -> bool:
    d = root / "claims" / stage / conv
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        # Another pod may be between its mkdir and its first stamp: a missing or unreadable stamp
        # counts as a fresh claim, dated by the directory itself. (Reading it as stale let two pods
        # take one tape; reading an empty stamp as a number crashed 8 of 10 trial pods.)
        try:
            claimed = float((d / "claimed_at").read_text())
        except (OSError, ValueError):
            claimed = d.stat().st_mtime
        age = time.time() - claimed
        if age < STALE_S:
            return False
        print(f"  taking over stale claim on {conv} ({age / 60:.0f} min old)", flush=True)
    write_stamp(d / "claimed_at")
    (d / "host").write_text(socket.gethostname())
    return True


def fetch_wav(job: dict) -> Path:
    """Download the tape's audio (all parts, falling back to other sources) as one 16 kHz mono WAV.

    Both stages run side by side and want the same file. The first takes a lock and converts to a
    temporary name, renamed into place only when complete; the other waits for the finished file.
    (Without this, stage A once read a file stage B was still writing and transcribed 31 of 82 min.)"""
    CACHE.mkdir(parents=True, exist_ok=True)
    wav = CACHE / f"{job['conv']}.wav"
    lock = CACHE / f"{job['conv']}.lock"
    while not wav.exists():
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            break  # we hold the lock: download below
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 1800:  # holder died mid-download
                lock.unlink(missing_ok=True)
            time.sleep(2)
    else:
        return wav
    try:
        return _download(job, wav)
    finally:
        lock.unlink(missing_ok=True)


def _download(job: dict, wav: Path) -> Path:
    tmp = wav.with_name(wav.stem + ".tmp.wav")
    for urls in [job["urls"]] + job.get("fallbacks", []):
        parts = []
        try:
            for i, u in enumerate(urls):
                p = CACHE / f"{job['conv']}.part{i}"
                subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "4", "-A", "nixon-corpus-archiver/0.1",
                                "-o", str(p), u], check=True)
                parts.append(p)
            ins = [a for p in parts for a in ("-i", str(p))]
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ins, "-filter_complex",
                            f"concat=n={len(parts)}:v=0:a=1", "-ar", str(SR), "-ac", "1", str(tmp)], check=True)
            tmp.replace(wav)  # atomic: the final name only ever holds a complete file
            return wav
        except subprocess.CalledProcessError as e:
            print(f"  source failed for {job['conv']} ({e}); trying next", flush=True)
        finally:
            for p in parts:
                p.unlink(missing_ok=True)
    raise RuntimeError(f"no audio source worked for {job['conv']}")


def read_audio(wav: Path):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR
    return audio if audio.ndim == 1 else audio.mean(1).astype(np.float32)


def find_tones(audio) -> list[list[float]]:
    """Intervals [start, end] (s) dominated by a ~1 kHz tone: NARA's marker for withdrawn audio."""
    import numpy as np
    frame = SR // 2
    n = len(audio) // frame
    if n == 0:
        return []
    frames = audio[:n * frame].reshape(n, frame) * np.hanning(frame)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1 / SR)
    band = (freqs > TONE_HZ - TONE_BAND) & (freqs < TONE_HZ + TONE_BAND)
    share = spec[:, band].sum(1) / (spec.sum(1) + 1e-12)
    loud = np.sqrt((frames ** 2).mean(1)) > 0.01
    hit = (share > TONE_SHARE) & loud
    spans, start = [], None
    for i, h in enumerate(list(hit) + [False]):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if (i - start) * 0.5 >= TONE_MIN_S:
                spans.append([start * 0.5, i * 0.5])
            start = None
    return spans


def muted(audio, tones):
    out = audio.copy()
    for a, b in tones:
        out[int(a * SR):int(b * SR)] = 0.0
    return out


def heartbeat(stamp: Path, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_S):
        try:
            write_stamp(stamp)
        except OSError:  # e.g. a momentary network-volume error: the next beat retries
            pass


def downloaded_by_partner(jobs: list[dict], partner_done: Path):
    """Jobs whose audio the other stage on this pod has downloaded, in arrival order, until it exits."""
    by_conv, seen = {j["conv"]: j for j in jobs}, set()
    while True:
        # a download in progress (<conv>.tmp.wav) has stem "<conv>.tmp", which is never a job
        new = sorted((p for p in CACHE.glob("*.wav") if p.stem in by_conv and p.stem not in seen),
                     key=lambda p: p.stat().st_mtime)
        for p in new:
            seen.add(p.stem)
            yield by_conv[p.stem]
        if not new:
            if partner_done.exists():
                return
            time.sleep(5)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(f".{socket.gethostname()}.{os.getpid()}.tmp")  # unique: two writers can't trip over one tmp file
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)  # atomic: a half-written file never looks finished


def run_stage(stage: str, jobs_path: Path, root: Path, process, output_name: str, other_output: str,
              partner_done: Path | None = None, follow: bool = False) -> int:
    jobs = load_jobs(jobs_path)
    # The second pass over the list picks up tapes whose claims went stale while this pod worked
    # (their pod died mid-tape); finished and live-claimed tapes are skipped.
    queue = itertools.chain(downloaded_by_partner(jobs, partner_done) if follow else [], jobs, jobs)
    other_stage = other_output.removesuffix(".json")
    done = failed = 0
    t0 = time.time()

    def release(conv: str) -> None:
        """Delete the tape's local audio once the other stage on this pod can't still need it."""
        if ((root / "out" / conv / other_output).exists() or (root / "failed" / f"{other_stage}-{conv}.txt").exists()
                or (partner_done is not None and partner_done.exists())):
            (CACHE / f"{conv}.wav").unlink(missing_ok=True)

    for job in queue:
        out_dir = root / "out" / job["conv"]
        if (out_dir / output_name).exists() or not claim(root, stage, job["conv"]):
            continue
        t = time.time()
        beat = threading.Event()
        threading.Thread(target=heartbeat, args=(root / "claims" / stage / job["conv"] / "claimed_at", beat),
                         daemon=True).start()
        try:
            wav = fetch_wav(job)
            audio = read_audio(wav)
            tones = find_tones(audio)
            result = process(muted(audio, tones), job)
            result.update({"conversation_id": "37-wht-conversation-" + job["conv"], "stage": stage,
                           "audio_seconds": round(len(audio) / SR, 2), "tones": tones,
                           "source_urls": job["urls"], "host": socket.gethostname(),
                           "elapsed_seconds": round(time.time() - t, 1)})
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / output_name, result)
            done += 1
            rate = len(audio) / SR / max(time.time() - t, 1e-6)
            print(f"[{stage}] {job['conv']}: {len(audio) / SR / 60:5.1f} min audio in {time.time() - t:5.0f} s "
                  f"({rate:5.0f}x realtime), {len(tones)} withdrawn spans", flush=True)
        except Exception as e:
            failed += 1
            print(f"[{stage}] {job['conv']}: FAILED {type(e).__name__}: {e}", flush=True)
            (root / "failed").mkdir(parents=True, exist_ok=True)
            (root / "failed" / f"{stage}-{job['conv']}.txt").write_text(f"{type(e).__name__}: {e}\n")
        finally:
            beat.set()
        release(job["conv"])
    print(f"[{stage}] finished: {done} done, {failed} failed, {(time.time() - t0) / 3600:.2f} h", flush=True)
    return 0


# --- stage A: Parakeet ------------------------------------------------------------

def cmd_parakeet(jobs: Path, root: Path, model_path: str, partner_done: Path | None, follow: bool,
                 output_name: str = "parakeet.json", confidence: bool = False, ngram_lm: str | None = None,
                 ngram_alpha: float = 0.0, boost_dir: Path | None = None, boost_alpha: float = 0.5) -> int:
    import copy
    import tempfile
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict
    model = (nemo_asr.models.ASRModel.restore_from(model_path) if model_path.endswith(".nemo")
             else nemo_asr.models.ASRModel.from_pretrained(model_path))
    model.eval()
    # plain: the decoding a window falls back to when word confidence fails; same fusion, no confidence
    plain = copy.deepcopy(model.cfg.decoding)
    if ngram_lm:  # shallow fusion with the era n-gram LM (era_lm.py); NeMo supports it in greedy_batch
        with open_dict(plain):
            plain.strategy = "greedy_batch"
            plain.greedy.ngram_lm_model = ngram_lm
            plain.greedy.ngram_lm_alpha = ngram_alpha
    cfg = copy.deepcopy(plain)
    if confidence:  # per-word confidence (entropy-based, the word's least confident token), for later passes
        with open_dict(cfg):
            cfg.confidence_cfg = OmegaConf.create({
                "preserve_word_confidence": True, "preserve_token_confidence": True, "exclude_blank": True,
                "aggregation": "min", "method_cfg": {"name": "entropy", "entropy_type": "tsallis", "alpha": 0.5,
                                                     "entropy_norm": "exp"}})
    if confidence or ngram_lm:
        model.change_decoding_strategy(cfg)
    tmp = Path(tempfile.mkdtemp())

    def boost(job) -> int:
        """Per-tape phrase boosting (era_lm.py boost): this tape's names, or none. Returns the phrase count."""
        from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
        f = boost_dir / f"{job['conv']}.txt"
        n = len([l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]) if f.exists() else 0
        for c in (cfg, plain):
            with open_dict(c):
                c.strategy = "greedy_batch"
                c.greedy.boosting_tree = OmegaConf.structured(BoostingTreeModelConfig(key_phrases_file=str(f))) if n else None
                c.greedy.boosting_tree_alpha = boost_alpha if n else 0.0
        model.change_decoding_strategy(cfg)
        return n

    def process(audio, job):
        n_boost = boost(job) if boost_dir else 0
        words, segments = [], []
        step = WINDOW_S * SR
        for i in range(0, len(audio), step):
            piece, off = tmp / "win.wav", i / SR
            sf.write(str(piece), audio[i:i + step], SR)
            try:
                hyp = model.transcribe([str(piece)], timestamps=True, verbose=False)[0]
            except RuntimeError:  # NeMo's word-confidence aggregation fails on some audio: redo this window without it
                if not confidence:
                    raise
                model.change_decoding_strategy(plain)
                hyp = model.transcribe([str(piece)], timestamps=True, verbose=False)[0]
                model.change_decoding_strategy(cfg)
            ws = hyp.timestamp.get("word", [])
            conf = [float(c) for c in (getattr(hyp, "word_confidence", None) or [])]
            words += [{"w": w["word"], "s": round(w["start"] + off, 2), "e": round(w["end"] + off, 2),
                       **({"c": round(conf[k], 3)} if len(conf) == len(ws) else {})} for k, w in enumerate(ws)]
            segments += [{"text": s["segment"].strip(), "start": round(s["start"] + off, 2),
                          "end": round(s["end"] + off, 2)} for s in hyp.timestamp.get("segment", [])]
        result = {"model": model_path, "segments": segments, "words": words}
        if ngram_lm or boost_dir:
            result["decoding"] = {"ngram_lm": ngram_lm, "ngram_alpha": ngram_alpha if ngram_lm else 0.0,
                                  "boost_phrases": n_boost, "boost_alpha": boost_alpha if n_boost else 0.0}
        return result

    # a re-run with a new model writes e.g. parakeet_v2.json and claims under its own stage name
    return run_stage(output_name.removesuffix(".json"), jobs, root, process, output_name, "whisper.json", partner_done, follow)


# --- stage B: Whisper + speakers ---------------------------------------------------

def cmd_whisper(jobs: Path, root: Path, partner_done: Path | None) -> int:
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Inference, Model, Pipeline

    token = os.environ.get("HF_TOKEN")
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for local smoke tests
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token).to(torch.device(dev))
    embed = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=token),
                      window="whole", device=torch.device(dev))

    def voiceprint(audio, a, b):
        x = torch.from_numpy(audio[int(a * SR):int(b * SR)]).unsqueeze(0)
        return np.asarray(embed({"waveform": x, "sample_rate": SR}), dtype=np.float32)

    mode = os.environ.get("WORKER_WHISPER_MODE", "sequential")
    name = os.environ.get("WORKER_WHISPER", "large-v3")
    compute = "float16" if dev == "cuda" else "int8"
    if mode == "batched":  # ~4x faster; merge ceiling 75.4% vs 76.9% on the room validation set
        import whisperx
        wx = whisperx.load_model(name, dev, compute_type=compute, language="en")
        wx_align = whisperx.load_align_model(language_code="en", device=dev)  # cached under TORCH_HOME
    else:
        asr = WhisperModel(name, device=dev, compute_type=compute)

    def transcribe(audio):
        """Whisper segments as (text, start, end, avg_logprob, compression_ratio, words)."""
        if mode == "batched":
            import whisperx
            res = wx.transcribe(audio, batch_size=16)
            res = whisperx.align(res["segments"], *wx_align, audio, dev, return_char_alignments=False)
            return [(s["text"], s["start"], s["end"], None, None,
                     [(w["word"], w["start"], w["end"], w.get("score")) for w in s.get("words", []) if "start" in w])
                    for s in res["segments"]]
        segs, _ = asr.transcribe(audio, language="en", beam_size=5, condition_on_previous_text=False,
                                 temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], compression_ratio_threshold=2.4,
                                 log_prob_threshold=-1.0, no_speech_threshold=0.6, vad_filter=False,
                                 word_timestamps=True)
        return [(s.text, s.start, s.end, s.avg_logprob, s.compression_ratio,
                 [(w.word, w.start, w.end, w.probability) for w in (s.words or [])]) for s in segs]

    def process(audio, job):
        segments, words, seg_prints = [], [], []
        for text, start, end, logprob, ratio, seg_words in transcribe(audio):
            s = argparse.Namespace(start=start, end=end)
            seg = {"text": text.strip(), "start": round(start, 2), "end": round(end, 2)}
            if logprob is not None:
                seg.update(avg_logprob=round(logprob, 3), compression_ratio=round(ratio, 2))
            # A voiceprint wherever Whisper heard speech: pyannote's own speech detection missed
            # ~98% of an EOB recording in testing, so naming speakers must not depend on it.
            if s.end - s.start >= 1.5:
                seg["vp"] = len(seg_prints)
                seg_prints.append(voiceprint(audio, s.start, min(s.end, s.start + 15.0)))
            segments.append(seg)
            words += [{"w": w.strip(), "s": round(ws, 2), "e": round(we, 2), "p": round(p, 3) if p is not None else None}
                      for w, ws, we, p in seg_words]
        # NARA's participant list bounds the speaker count; pyannote alone often undercounts
        # (it heard one voice in a two-person phone call in testing).
        n = len([p for p in job.get("participants", []) if "operator" not in p.lower()])
        bounds = {"min_speakers": 2, "max_speakers": min(n + 2, 12)} if n >= 2 else {}
        result = diar({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, **bounds)
        annotation = getattr(result, "speaker_diarization", result)
        turns = [{"speaker": spk, "start": round(seg.start, 2), "end": round(seg.end, 2)}
                 for seg, _, spk in annotation.itertracks(yield_label=True)]
        prints, by_speaker = [], {}
        for t in turns:
            if t["end"] - t["start"] >= 2.0:
                v = voiceprint(audio, t["start"], t["end"])
                t["voiceprint"] = len(prints)
                prints.append(v)
                by_speaker.setdefault(t["speaker"], []).append(v)
        centroids = {k: np.mean(v, 0).round(5).tolist() for k, v in by_speaker.items()}
        return {"model": f"whisper-{name} {mode}", "segments": segments, "words": words,
                "diarization": {"model": "pyannote/speaker-diarization-community-1", "turns": turns},
                "voiceprints": {"model": "pyannote/wespeaker-voxceleb-resnet34-LM",
                                "segments": [p.round(5).tolist() for p in seg_prints],
                                "turns": [p.round(5).tolist() for p in prints], "speakers": centroids}}

    return run_stage("whisper", jobs, root, process, "whisper.json", "parakeet.json", partner_done)


def cmd_status(jobs: Path, root: Path) -> int:
    jl = load_jobs(jobs)
    now = time.time()
    for stage, name in (("parakeet", "parakeet.json"), ("whisper", "whisper.json")):
        stamps = [p.stat().st_mtime for j in jl if (p := root / "out" / j["conv"] / name).exists()]
        recent = sum(now - s < 3600 for s in stamps)
        eta = f", ~{(len(jl) - len(stamps)) / recent:.1f} h to go at that rate" if recent else ""
        print(f"{stage:9} {len(stamps)}/{len(jl)} tapes done, {recent} in the last hour{eta}")
    failed = sorted(p.name for p in (root / "failed").glob("*.txt")) if (root / "failed").exists() else []
    print(f"failed: {len(failed)} {failed[:10]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "parakeet", "whisper", "status"])
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--root", type=Path, default=Path("/workspace/corpus"))
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--tiers", nargs="*", default=["A", "B"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--partner-done", type=Path, help="flag file the other stage on this pod touches when it exits")
    ap.add_argument("--output-name", default="parakeet.json", help="parakeet: output file (a re-run keeps the old one)")
    ap.add_argument("--confidence", action="store_true", help="parakeet: also save per-word confidence")
    ap.add_argument("--ngram-lm", help="parakeet: n-gram LM (.nemo or ARPA, era_lm.py) for shallow fusion")
    ap.add_argument("--ngram-alpha", type=float, default=0.0, help="parakeet: weight of the n-gram LM")
    ap.add_argument("--boost-dir", type=Path, help="parakeet: per-tape phrase lists <conv>.txt for the boosting tree")
    ap.add_argument("--boost-alpha", type=float, default=0.5, help="parakeet: weight of the boosting tree")
    ap.add_argument("--follow", action="store_true",
                    help="parakeet: first take tapes the other stage downloaded, until --partner-done appears")
    args = ap.parse_args()
    if args.follow and not args.partner_done:
        ap.error("--follow needs --partner-done")
    if args.cmd == "plan":
        return cmd_plan(args.out, args.tiers, args.only)
    if args.cmd == "parakeet":
        return cmd_parakeet(args.jobs, args.root, args.model, args.partner_done, args.follow,
                            args.output_name, args.confidence, args.ngram_lm, args.ngram_alpha,
                            args.boost_dir, args.boost_alpha)
    if args.cmd == "whisper":
        return cmd_whisper(args.jobs, args.root, args.partner_done)
    return cmd_status(args.jobs, args.root)


if __name__ == "__main__":
    sys.exit(main())
