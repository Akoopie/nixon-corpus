"""Voiceprints for the main players (Nixon, Haldeman, Ehrlichman, Dean, ...).

The training transcripts label who speaks every turn; aligning them to the audio
yields thousands of single-speaker clips with known speakers. A pretrained
speaker-embedding model turns each clip into a voiceprint vector, and a small
classifier learns each person. It only DETECTS who is speaking; nothing here
separates or rewrites audio.

    python src/voiceprints.py segments --turns DIR --audio-root DIR --out DIR   # GPU: turn time spans
    python src/voiceprints.py embed --out DIR                                   # GPU: one embedding per turn
    python src/voiceprints.py train --out DIR                                   # CPU: classifier + report

Evaluation holds out whole conversations, so a speaker is always recognized in a
conversation the classifier never saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from score import norm_words

MIN_TURN_S, PIECE_S, MIN_ANCHORED = 2.0, 10.0, 3  # long turns are cut into ~10 s pieces, not dropped
MIN_SEGMENTS = 30          # speakers with fewer segments are folded into "other"
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def cpu_words(model, audio) -> list[dict]:
    """Word timings from faster-whisper's own word timestamps (no wav2vec2 pass: fast on CPU).
    Accuracy barely matters here; the words only anchor transcript turns in time."""
    segments, _ = model.transcribe(audio, language="en", word_timestamps=True, beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False)
    return [{"norm": n, "start": w.start, "end": w.end}
            for s in segments for w in (s.words or []) for n in norm_words(w.word)]


def cmd_segments(turns_dir: Path, audio_root: Path, out: Path, convs: list[str] | None, device: str,
                 asr_cache: Path | None = None) -> int:
    """Time span of each transcript turn, from words both the ASR and the transcript have."""
    import time
    from align_clips import SR, load_audio

    cache_dir = asr_cache or out / "asr_cache"
    models: dict = {}  # loaded lazily: with a full cache (from align_clips) no ASR model is needed
    if device == "cuda":
        def words_of(audio):
            import whisperx
            from align_clips import asr_words
            if not models:
                models["m"] = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="en")
                models["a"] = whisperx.load_align_model(language_code="en", device="cuda")
            return asr_words(models["m"], *models["a"], audio)
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=16)
        words_of = lambda audio: cpu_words(model, audio)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "segments.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for tpath in sorted(turns_dir.glob("*.json")):
            conv = tpath.stem
            mp3s = sorted((audio_root / "conversations" / conv).glob("*.mp3"))
            if not mp3s or (convs and conv not in convs):
                continue
            turns = json.loads(tpath.read_text(encoding="utf-8"))["turns"]
            t_start = time.time()
            audio = load_audio(mp3s)
            cache = cache_dir / f"{conv}.json"  # ASR is the slow part; rule changes reuse it
            if cache.exists():
                hyp = json.loads(cache.read_text(encoding="utf-8"))
            else:
                hyp = words_of(audio)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(hyp), encoding="utf-8")
            ref, owner = [], []
            for i, t in enumerate(turns):
                w = norm_words(t["text"])
                ref += w
                owner += [i] * len(w)
            spans: dict[int, list] = {}
            sm = SequenceMatcher(None, ref, [h["norm"] for h in hyp], autojunk=False)
            for b in sm.get_matching_blocks():
                if b.size >= 2:
                    for k in range(b.size):
                        spans.setdefault(owner[b.a + k], []).append(hyp[b.b + k])
            n = 0
            for i, words in spans.items():
                if len(words) < MIN_ANCHORED:
                    continue
                # Cut the anchored span into ~PIECE_S pieces at word boundaries (a long
                # monologue becomes several clips instead of being thrown away).
                piece = [words[0]]
                for w in words[1:] + [None]:
                    if w is not None and w["end"] - piece[0]["start"] <= PIECE_S:
                        piece.append(w)
                        continue
                    t0, t1 = piece[0]["start"], piece[-1]["end"]
                    if t1 - t0 >= MIN_TURN_S:
                        f.write(json.dumps({"conv": conv, "speaker": turns[i]["speaker"], "start": round(t0, 2),
                                            "end": round(t1, 2), "audio": [str(p) for p in mp3s]}) + "\n")
                        n += 1
                    piece = [w] if w is not None else []
            mins = len(audio) / SR / 60
            print(f"{conv}: {mins:5.1f} min of audio -> {n:4} labeled speaker segments "
                  f"({time.time() - t_start:4.0f} s)", flush=True)
    return 0


def cmd_embed(out: Path, device: str) -> int:
    import os
    import numpy as np
    from align_clips import SR, load_audio
    from paths import SECRETS_DIR
    from pyannote.audio import Inference, Model
    import torch

    token = os.environ.get("HF_TOKEN") or (SECRETS_DIR / "hf_token.txt").read_text(encoding="utf-8").strip()
    model = Model.from_pretrained(EMBED_MODEL, token=token)
    infer = Inference(model, window="whole", device=torch.device(device))
    segs = [json.loads(l) for l in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    embs, cache = [], {}
    for s in segs:
        key = tuple(s["audio"])
        if key not in cache:
            cache.clear()
            cache[key] = load_audio([Path(p) for p in s["audio"]])
        wav = cache[key][int(s["start"] * SR):int(s["end"] * SR)]
        embs.append(infer({"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": SR}))
    np.save(out / "embeddings.npy", np.vstack(embs).astype("float32"))
    print(f"{len(embs)} embeddings -> {out / 'embeddings.npy'}")
    return 0


def conversation_normalize(X, segs):
    """Subtract each conversation's mean voiceprint: removes what the room/microphone
    adds to every voice in it, leaving what differs between the voices."""
    import numpy as np
    X = X.copy()
    for conv in {s["conv"] for s in segs}:
        idx = np.array([s["conv"] == conv for s in segs])
        X[idx] -= X[idx].mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def cmd_train(out: Path, val_convs: list[str], epochs: int, lr: float, conv_norm: bool = False) -> int:
    """Train a small classifier head on top of the frozen voiceprint model, printing
    every epoch. Validation conversations are never trained on, so their score shows
    how well it recognizes voices in conversations it has never heard."""
    import numpy as np
    import torch
    from torch import nn

    torch.manual_seed(0)
    segs = [json.loads(l) for l in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    X = np.load(out / "embeddings.npy")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if conv_norm:
        X = conversation_normalize(X, segs)
    train_mask = np.array([s["conv"] not in val_convs for s in segs])
    counts = Counter(s["speaker"] for s, m in zip(segs, train_mask) if m)
    names = sorted(k for k, n in counts.items() if n >= MIN_SEGMENTS and len(k) > 3)  # skip "D"/"H" abbreviations
    classes = names + ["other"]
    y = np.array([classes.index(s["speaker"]) if s["speaker"] in names else len(names) for s in segs])
    Xt, yt = torch.tensor(X[train_mask]), torch.tensor(y[train_mask])
    Xv, yv = torch.tensor(X[~train_mask]), torch.tensor(y[~train_mask])
    print(f"train: {len(yt)} segments from {len({s['conv'] for s, m in zip(segs, train_mask) if m})} conversations"
          f" | validation: {len(yv)} segments from {len(val_convs)} held-out conversations")
    print("speakers:", ", ".join(f"{c} ({int((yt == i).sum())})" for i, c in enumerate(classes)))

    head = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(classes)))
    freq = torch.bincount(yt, minlength=len(classes)).float()
    loss_fn = nn.CrossEntropyLoss(weight=(freq.sum() / (freq.clamp(min=1) * len(classes))))  # Nixon talks most
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, patience = float("inf"), None, 0
    print(f"\n{'epoch':>5} {'train loss':>10} {'train acc':>9} {'val loss':>9} {'val acc':>8}")
    for epoch in range(1, epochs + 1):
        head.train()
        perm, tot = torch.randperm(len(yt)), 0.0
        for i in range(0, len(yt), 32):  # mini-batches of 32 segments
            b = perm[i:i + 32]
            loss = loss_fn(head(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        head.eval()
        with torch.no_grad():
            tr_acc = (head(Xt).argmax(1) == yt).float().mean().item()
            vlog = head(Xv)
            vloss, vacc = loss_fn(vlog, yv).item(), (vlog.argmax(1) == yv).float().mean().item()
        mark = ""
        if vloss < best:
            best, best_state, patience, mark = vloss, {k: v.clone() for k, v in head.state_dict().items()}, 0, "  <- best so far"
        else:
            patience += 1
        print(f"{epoch:5} {tot / len(yt):10.3f} {tr_acc:9.1%} {vloss:9.3f} {vacc:8.1%}{mark}", flush=True)
        if patience >= 10:
            print(f"stopping early: validation loss hasn't improved for 10 epochs (overfitting starts)")
            break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pv = head(Xv).argmax(1).numpy()
    yvn = yv.numpy()
    print("\nper speaker, on the held-out conversations (each 2-30 s segment judged alone):")
    for i, c in enumerate(classes):
        m = yvn == i
        if m.sum():
            print(f"  {c:12} {int(m.sum()):4} segs  recall {(pv[m] == i).mean():6.1%}  "
                  f"precision {(yvn[pv == i] == i).mean() if (pv == i).any() else 0:6.1%}")
    # A whole speaker "pile" (all of someone's speech in a conversation) is judged far more reliably.
    piles = {}
    for s, xv, t in zip([s for s, m in zip(segs, train_mask) if not m], Xv, yvn):
        piles.setdefault((s["conv"], int(t)), []).append(xv)
    with torch.no_grad():
        hits = [int(head(torch.stack(v).mean(0, keepdim=True)).argmax(1).item() == t) for (_, t), v in piles.items()]
    print(f"whole speaker piles (all of one person's speech in a conversation): {sum(hits)}/{len(hits)} named correctly")
    torch.save({"state": best_state, "classes": classes, "embed_model": EMBED_MODEL}, out / "speaker_head.pt")
    print("saved", out / "speaker_head.pt")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["segments", "embed", "train"])
    ap.add_argument("--turns", type=Path)
    ap.add_argument("--audio-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--convs", nargs="*", help="segments: only these conversations")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--val", nargs="*", default=[], help="train: held-out validation conversations")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--conv-norm", action="store_true", help="train: subtract each conversation's mean voiceprint")
    ap.add_argument("--asr-cache", type=Path, help="segments: Whisper word timings saved by align_clips.py")
    args = ap.parse_args()
    if args.cmd == "segments":
        return cmd_segments(args.turns, args.audio_root, args.out, args.convs, args.device, args.asr_cache)
    if args.cmd == "embed":
        return cmd_embed(args.out, args.device)
    return cmd_train(args.out, args.val, args.epochs, args.lr, args.conv_norm)


if __name__ == "__main__":
    sys.exit(main())
