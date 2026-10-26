"""Experiments on the voiceprint classifier: a cross-validated settings sweep and a grokking test.

    python src/voiceprint_lab.py sweep --out DIR    # leave-one-conversation-out over every setting
    python src/voiceprint_lab.py grok --out DIR     # very long training with strong weight decay

Both read segments.jsonl + embeddings.npy written by voiceprints.py. The sweep
never scores a setting on clips it trained on: each conversation takes a turn as
the held-out one, and room voiceprints are computed from the training side only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from finetune_data import TURNS_DIR

ROOMS = {"Oval": "oval", "Executive": "eob", "Telephone": "phone", "Cabinet": "cabinet"}


JUNK_LABELS = {"continued", "cont", "contd"}  # transcript artifacts, not speakers


def load(out: Path):
    segs = [json.loads(l) for l in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    X = np.load(out / "embeddings.npy").astype("float32")
    keep = np.array([len(s["speaker"]) > 2 and s["speaker"].lower() not in JUNK_LABELS for s in segs])
    segs, X = [s for s, k in zip(segs, keep) if k], X[keep]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    loc = {}
    for s in segs:  # clips from tapes without a human transcript carry their room themselves
        if s.get("room"):
            loc[s["conv"]] = s["room"]
    for conv in {s["conv"] for s in segs} - set(loc):
        place = json.loads((TURNS_DIR / f"{conv}.json").read_text(encoding="utf-8"))["location"]
        loc[conv] = next((v for k, v in ROOMS.items() if k in place), "other")
    # Name only people heard in >= 2 conversations, so a held-out conversation always
    # has training examples of them; everyone else is "other".
    convs_of = defaultdict(set)
    for s in segs:
        convs_of[s["speaker"]].add(s["conv"])
    counts = Counter(s["speaker"] for s in segs)
    names = sorted(k for k in counts if len(convs_of[k]) >= 2 and counts[k] >= 15 and len(k) > 3)
    classes = names + ["other"]
    y = np.array([classes.index(s["speaker"]) if s["speaker"] in names else len(names) for s in segs])
    return segs, X, y, classes, loc


def normalize(X, segs, train_idx, mode, loc):
    """none | conversation (subtract each conversation's mean) | room (subtract the mean
    voiceprint of that room, computed from TRAINING conversations only)."""
    if mode == "none":
        return X
    X = X.copy()
    convs = np.array([s["conv"] for s in segs])
    if mode == "conversation":
        for c in set(convs):
            X[convs == c] -= X[convs == c].mean(0)
    else:
        rooms = np.array([loc[c] for c in convs])
        tr = np.zeros(len(X), bool)
        tr[train_idx] = True
        overall = X[tr].mean(0)
        for r in set(rooms):
            m = rooms == r
            X[m] -= X[m & tr].mean(0) if (m & tr).any() else overall
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def train_head(Xtr, ytr, n_classes, hidden, dropout, wd, lr, epochs, seed=0, log=None, Xv=None, yv=None, every=1):
    torch.manual_seed(seed)
    layers = [nn.Linear(Xtr.shape[1], n_classes)] if hidden == 0 else [
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)]
    head = nn.Sequential(*layers)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    freq = torch.bincount(yt, minlength=n_classes).float()
    loss_fn = nn.CrossEntropyLoss(weight=freq.sum() / (freq.clamp(min=1) * n_classes))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 32):
            b = perm[i:i + 32]
            loss = loss_fn(head(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log is not None and (epoch % every == 0 or epoch == 1):
            head.eval()
            with torch.no_grad():
                lt = head(Xt)
                lv = head(torch.tensor(Xv))
                log.append([epoch, loss_fn(lt, yt).item(), (lt.argmax(1) == yt).float().mean().item() * 100,
                            loss_fn(lv, torch.tensor(yv)).item(), (lv.argmax(1).numpy() == yv).mean() * 100])
    head.eval()
    return head


def evaluate(X, y, segs, classes, loc, cfg, folds: int = 0):
    """Hold out conversations (one at a time, or `folds` groups); returns balanced accuracy over
    named speakers, pile accuracy, and per-speaker recall. Never scores a conversation it trained on."""
    convs = np.array([s["conv"] for s in segs])
    pred = np.empty_like(y)
    pile_hits = pile_total = 0
    if folds:
        from sklearn.model_selection import GroupKFold
        splits = [(tr, te) for tr, te in GroupKFold(n_splits=folds).split(X, y, convs)]
    else:
        splits = [(np.where(convs != c)[0], np.where(convs == c)[0]) for c in sorted(set(convs))]
    for tr, te in splits:
        Xn = normalize(X, segs, tr, cfg["norm"], loc)
        head = train_head(Xn[tr], y[tr], len(classes), cfg["hidden"], cfg["dropout"], cfg["wd"], cfg["lr"], cfg["epochs"])
        with torch.no_grad():
            logits = head(torch.tensor(Xn[te]))
        pred[te] = logits.argmax(1).numpy()
        for held in set(convs[te]):  # one pile = all of one person's speech in one held-out conversation
            in_conv = convs[te] == held
            for c in set(y[te][in_conv]):
                m = in_conv & (y[te] == c)
                if c < len(classes) - 1 and m.sum() >= 3:
                    pile_total += 1
                    pile_hits += int(logits[torch.tensor(m)].mean(0).argmax().item() == c)
    recalls = [(pred[y == c] == c).mean() for c in range(len(classes) - 1) if (y == c).any()]
    return float(np.mean(recalls)), pile_hits, pile_total, {classes[c]: float((pred[y == c] == c).mean())
                                                             for c in range(len(classes) - 1)}


def cmd_sweep(out: Path, folds: int = 0, quick: bool = False) -> int:
    segs, X, y, classes, loc = load(out)
    print(f"{len(segs)} segments, {len({s['conv'] for s in segs})} conversations; named speakers: "
          + ", ".join(f"{c} ({int((y == i).sum())})" for i, c in enumerate(classes[:-1])))
    grid = {"norm": ["none", "conversation", "room"], "hidden": [0, 64, 256], "dropout": [0.3],
            "wd": [1e-2, 1e-1], "lr": [1e-3], "epochs": [10, 40]}
    if quick:  # big data: fewer, more distinct settings
        grid = {"norm": ["none", "room"], "hidden": [0, 256], "dropout": [0.3], "wd": [1e-1], "lr": [1e-3],
                "epochs": [15]}
    results = []
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        bal, hits, total, per = evaluate(X, y, segs, classes, loc, cfg, folds)
        results.append({**cfg, "balanced_acc": round(bal, 4), "piles": f"{hits}/{total}", "pile_acc": hits / max(total, 1),
                        "per_speaker": {k: round(v, 3) for k, v in per.items()}})
        print(f"norm={cfg['norm']:12} hidden={cfg['hidden']:3} wd={cfg['wd']:<5} epochs={cfg['epochs']:3}  "
              f"balanced acc {bal:6.1%}  piles {hits}/{total}", flush=True)
    results.sort(key=lambda r: (r["pile_acc"], r["balanced_acc"]), reverse=True)
    (out / "sweep.json").write_text(json.dumps({"classes": classes, "results": results}, indent=1))
    best = results[0]
    print(f"\nbest: norm={best['norm']} hidden={best['hidden']} wd={best['wd']} epochs={best['epochs']} "
          f"-> balanced acc {best['balanced_acc']:.1%}, piles {best['piles']}")
    print("  per speaker:", ", ".join(f"{k} {v:.0%}" for k, v in best["per_speaker"].items()))
    return 0


def cmd_grok(out: Path, val_convs: list[str], steps_epochs: int, wd: float) -> int:
    segs, X, y, classes, loc = load(out)
    convs = np.array([s["conv"] for s in segs])
    tr, te = np.where(~np.isin(convs, val_convs))[0], np.where(np.isin(convs, val_convs))[0]
    Xn = normalize(X, segs, tr, "none", loc)
    log: list = []
    print(f"grokking test: {steps_epochs} epochs, weight decay {wd}, hidden 256, no dropout")
    train_head(Xn[tr], y[tr], len(classes), 256, 0.0, wd, 1e-3, steps_epochs, log=log, Xv=Xn[te], yv=y[te], every=25)
    for row in log[::8] + [log[-1]]:
        print(f"  epoch {row[0]:5}  train loss {row[1]:.3f}  train acc {row[2]:5.1f}%  val loss {row[3]:.3f}  val acc {row[4]:5.1f}%")
    (out / "grok.json").write_text(json.dumps({"wd": wd, "epochs": steps_epochs, "log": log}))
    return 0


def cmd_final(out: Path, norm: str, hidden: int, wd: float, epochs: int) -> int:
    """Train the classifier on ALL clips with the sweep's winning setting and save everything
    inference needs: weights, class names, and each room's mean voiceprint."""
    segs, X, y, classes, loc = load(out)
    rooms = np.array([loc[s["conv"]] for s in segs])
    room_means = {str(r): X[rooms == r].mean(0).tolist() for r in set(rooms)}
    Xn = normalize(X, segs, np.arange(len(X)), norm, loc)
    head = train_head(Xn, y, len(classes), hidden, 0.3, wd, 1e-3, epochs)
    path = out / "speaker_classifier.pt"
    torch.save({"state": head.state_dict(), "classes": classes, "norm": norm, "hidden": hidden,
                "room_means": room_means, "embed_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "trained_on": f"{len(segs)} clips, {len(set(s['conv'] for s in segs))} conversations"}, path)
    print(f"saved {path}: classes {classes}, rooms {sorted(room_means)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sweep", "grok", "final"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", nargs="*", default=["538-015", "854-017", "427-010"])
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=0, help="sweep: hold out conversations in this many groups (0 = one at a time)")
    ap.add_argument("--quick", action="store_true", help="sweep: small grid for big data")
    args = ap.parse_args()
    if args.cmd == "sweep":
        return cmd_sweep(args.out, args.folds, args.quick)
    if args.cmd == "final":
        return cmd_final(args.out, "room", 0, 0.1, 15)  # the sweep's winner on the 54-conversation data
    return cmd_grok(args.out, args.val, args.epochs, args.wd)


if __name__ == "__main__":
    sys.exit(main())
