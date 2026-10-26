"""Hours of speech attributed to each named voice in the released corpus -> paper/data/speaker_time.json.

    python paper/speaker_time.py

Sums the durations of the classifier's named segments (speakers_v4.json, the labels behind every corpus speaker;
METHODS 5) per voice, with the number of conversations in which each voice is named. Segments below the naming
threshold carry no name and are counted as unnamed.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import DATA_DIR  # noqa: E402

secs, convs = defaultdict(float), defaultdict(set)
total = 0.0
for f in (DATA_DIR / "corpus_v5/out").glob("*/speakers_v4.json"):
    for seg in json.loads(f.read_text(encoding="utf-8")):
        d = max(0.0, float(seg["end"]) - float(seg["start"]))
        total += d
        spk = seg.get("speaker") or "unknown"
        if not spk.lower().startswith(("unknown", "other")) and spk:
            secs[spk] += d
            convs[spk].add(f.parent.name)
out = {"total_segment_hours": round(total / 3600, 1),
       "named": {k: {"hours": round(v / 3600, 1), "conversations": len(convs[k])}
                 for k, v in sorted(secs.items(), key=lambda kv: -kv[1])}}
(ROOT / "paper/data/speaker_time.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
