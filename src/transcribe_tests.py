"""Transcribe small held-out test sets with several Parakeet models, for local scoring.

    python transcribe_tests.py --out results.json --model r3=a.nemo --model r4=b.nemo --set name=manifest.jsonl [...]

Writes {set: {model: {clip id: text}}}; a clip's id is its audio file's stem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path.nemo")
    ap.add_argument("--set", action="append", required=True, help="name=manifest.jsonl")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    sets = {}
    for s in a.set:
        name, path = s.split("=", 1)
        sets[name] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {name: {} for name in sets}
    for m in a.model:
        mname, path = m.split("=", 1)
        model = nemo_asr.models.ASRModel.restore_from(path)
        model.eval()
        for name, rows in sets.items():
            hyps = model.transcribe([r["audio_filepath"] for r in rows], batch_size=8, verbose=False)
            out[name][mname] = {Path(r["audio_filepath"]).stem: (h.text if hasattr(h, "text") else str(h))
                                for r, h in zip(rows, hyps)}
            print(mname, name, len(rows), flush=True)
        del model
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
