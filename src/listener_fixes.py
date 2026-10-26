"""Fixes applied to the word layer from release v1.1 on (docs/corrections_v1.1.md, METHODS 7s).

1. Words whose midpoint lies inside a NARA withdrawal tone are dropped. The tones were muted to silence before either
   recognizer ran, so any word there is invented (METHODS 7p: 675 words in 227 conversations of release v1.0).
2. Word and speaker corrections confirmed by the author's listening (docs/listener_corrections.json).

Used by export_release.py and explorer/server.py, so the release and the browsing tool show the same text.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "docs" / "listener_corrections.json"


def in_tones(x: dict, tones: list) -> bool:
    m = (x["s"] + x["e"]) / 2
    return any(a <= m <= b for a, b in tones)


def drop_tone_words(words: list[dict], tones: list) -> tuple[list[dict], int]:
    """The same midpoint rule as paper/whisper_artifacts.py (cmd_extra), which counted the 675 words."""
    if not tones:
        return words, 0
    kept = [x for x in words if x.get("s") is None or not in_tones(x, tones)]
    return kept, len(words) - len(kept)


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {"words": [], "speakers": []}


def apply_words(conv: str, words: list[dict]) -> tuple[list[dict], list[dict]]:
    """Replace each listened token with the heard words, spread evenly over its span, or delete it when `new` is empty;
    returns (words, fixes applied). New tokens carry src 'listener' and what they replaced."""
    done = []
    for f in _load()["words"]:
        if f["conv"] != conv:
            continue
        hit = next((i for i, x in enumerate(words) if x["w"] == f["old"] and abs(x["s"] - f["at"]) <= 1.0), None)
        if hit is None:
            raise ValueError(f"{conv}: listener fix {f['old']!r} at {f['at']} s not found")
        x = words[hit]
        new = f["new"].split()
        step = (x["e"] - x["s"]) / max(len(new), 1)
        repl = [{"w": t, "s": round(x["s"] + k * step, 2), "e": round(x["s"] + (k + 1) * step, 2), "src": "listener",
                 "was": f["old"], "reason": f["reason"]} for k, t in enumerate(new)]
        words = words[:hit] + repl + words[hit + 1:]
        done.append(f)
    return words, done


def speaker_spans(conv: str) -> list[dict]:
    return [s for s in _load()["speakers"] if s["conv"] == conv]


def override(spans: list[dict], t: float) -> str | None:
    return next((s["speaker"] for s in spans if s["start"] <= t <= s["end"]), None)
