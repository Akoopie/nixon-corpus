"""Fine-tune NVIDIA Parakeet TDT 0.6B v3 on Nixon-tape clips (NeMo).

Input: the clip manifest from align_clips.py. Steps:
  1. filter    drop clips the base model can barely match (< MIN_MATCH of the text:
               misaligned or leaked headers), tidy text for the tokenizer, split
               train/dev BY CONVERSATION so no conversation is on both sides
  2. train     full fine-tune at a low learning rate; keep the checkpoint with the
               best dev WER; export a .nemo file

Runs on a GPU pod (NeMo env, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1):
    python finetune_parakeet.py filter --manifest data/manifest.jsonl --out ft
    python finetune_parakeet.py train --out ft [--epochs 12 --lr 3e-5 --batch 16]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

BASE = "nvidia/parakeet-tdt-0.6b-v3"
MIN_MATCH = 0.25      # clips below this base-model match are treated as misaligned
FUSED_BATCH = 1       # utterances per slice when computing the TDT joint/loss (the default 4 = our whole batch)
DEV_SHARE = 0.10
HEADER = re.compile(r"\b[A-Z][A-Z ,.'&]{5,}\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|"
                    r"OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+19\d\d")


def tidy(text: str) -> str:
    """Transcript typography -> plain text the tokenizer knows (keeps case, punctuation, fillers)."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace('"', "")
    text = re.sub(r"\s*(--|—|–)\s*", ", ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r",\s*([,.?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def cmd_filter(manifest: Path, out: Path, min_match: float = MIN_MATCH) -> int:
    import jiwer
    import nemo.collections.asr as nemo_asr
    from score import norm_words

    clips = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    clips = list({c["audio_filepath"]: c for c in clips}.values())  # a resumed alignment can repeat lines
    clips = [c for c in clips if not HEADER.search(c["text"])]
    model = nemo_asr.models.ASRModel.from_pretrained(BASE)
    hyps = model.transcribe([c["audio_filepath"] for c in clips], batch_size=32)
    kept, matches = [], []
    for c, h in zip(clips, hyps):
        ref = norm_words(c["text"])
        r = jiwer.process_words(" ".join(ref), " ".join(norm_words(h.text)) or "x")
        c["base_match"] = round(r.hits / max(len(ref), 1), 3)
        matches.append(c["base_match"])
        if c["base_match"] >= min_match:
            c["text"] = tidy(c["text"])
            kept.append(c)
    convs = sorted({c["conv"] for c in kept})
    random.Random(1974).shuffle(convs)
    dev = set(convs[:max(1, round(len(convs) * DEV_SHARE))])
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", [c for c in kept if c["conv"] not in dev]), ("dev", [c for c in kept if c["conv"] in dev])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for c in part:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} clips, {sum(c['duration'] for c in part) / 3600:.1f} h")
    print(f"dropped {len(clips) - len(kept)} misaligned clips; base model matched "
          f"{sum(matches) / max(len(matches), 1):.0%} of clip words on average; dev conversations: {sorted(dev)}")
    return 0


def freeze_batchnorm(model) -> int:
    """Keep every BatchNorm layer on its pretrained statistics: with 4 clips per batch, updating
    them makes the running statistics noisy. Lightning calls model.train() each epoch, so each
    layer's own train() is replaced with a no-op that leaves it in eval mode."""
    import types
    import torch.nn as nn
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.train = types.MethodType(lambda self, mode=True: self, m)
            n += 1
    return n


def cmd_train(out: Path, epochs: int, lr: float, batch: int, max_steps: int = -1, freeze_bn: bool = False,
              init: str | None = None, warmup: int = 300) -> int:
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from omegaconf import OmegaConf, open_dict

    # continue from an earlier fine-tune (.nemo) or start from NVIDIA's model
    model = nemo_asr.models.ASRModel.restore_from(init) if init else nemo_asr.models.ASRModel.from_pretrained(BASE)
    # Compute the TDT joint + loss in sub-batches: the batch x frames x tokens x vocab tensor is what
    # runs out of GPU memory. Fail loudly if it does not take: unfused training OOMs on a 24 GB card.
    model.joint.set_fuse_loss_wer(True, loss=model.loss, metric=model.wer)
    model.joint.set_fused_batch_size(FUSED_BATCH)
    assert model.joint.fuse_loss_wer and model.joint.fused_batch_size == FUSED_BATCH, "fused joint did not take"
    import torch
    gpu = torch.cuda.is_available()  # the same script smoke-tests on a CPU before any GPU is rented
    data = dict(sample_rate=16000, batch_size=batch, num_workers=8 if gpu and sys.platform != "win32" else 0,
                pin_memory=gpu, max_duration=22.0, min_duration=1.0, use_lhotse=False, use_start_end_token=False)
    model.setup_training_data(OmegaConf.create({**data, "manifest_filepath": str(out / "train.jsonl"), "shuffle": True}))
    model.setup_validation_data(OmegaConf.create({**data, "manifest_filepath": str(out / "dev.jsonl"), "shuffle": False}))
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create({
            "name": "adamw", "lr": lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup, "min_lr": lr / 20}})
    trainer = pl.Trainer(
        devices=1, accelerator="gpu" if gpu else "cpu", precision="bf16-mixed" if gpu else "32-true",
        max_epochs=epochs, max_steps=max_steps,
        accumulate_grad_batches=max(1, 32 // batch),  # effective batch ~32 whatever fits in memory
        gradient_clip_val=1.0, log_every_n_steps=20, val_check_interval=0.5, enable_checkpointing=True,
        logger=pl.loggers.CSVLogger(str(out / "logs")), default_root_dir=str(out),
        callbacks=[ModelCheckpoint(dirpath=str(out / "ckpt"), monitor="val_wer", mode="min", save_top_k=1,
                                   filename="best-{epoch}-{val_wer:.4f}"), LearningRateMonitor()])
    model.set_trainer(trainer)
    model.setup_optimization(model.cfg.optim)
    if freeze_bn:
        print(f"froze {freeze_batchnorm(model)} BatchNorm layers on their pretrained statistics")
    trainer.validate(model)  # dev WER of the base model, for comparison
    trainer.fit(model)
    best = trainer.checkpoint_callback.best_model_path
    print("best checkpoint:", best, "val_wer:", trainer.checkpoint_callback.best_model_score)
    if best:  # load the best epoch's weights into this model (rebuilding from the .ckpt loses the tokenizer path)
        model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["state_dict"])
    model.save_to(str(out / "parakeet-tdt-0.6b-v3-nixon.nemo"))
    print("saved", out / "parakeet-tdt-0.6b-v3-nixon.nemo")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["filter", "train"])
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--min-match", type=float, default=MIN_MATCH, help="filter: base-model match needed to keep a clip")
    ap.add_argument("--init", help="start from this .nemo instead of the base model (continued training)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm layers on pretrained statistics")
    args = ap.parse_args()
    if args.cmd == "filter":
        return cmd_filter(args.manifest, args.out, args.min_match)
    return cmd_train(args.out, args.epochs, args.lr, args.batch, args.max_steps, args.freeze_bn, args.init, args.warmup)


if __name__ == "__main__":
    sys.exit(main())
