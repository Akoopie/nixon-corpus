#!/bin/bash
# After the v5 corpus run (corpus_v5.sh): gather every shard's parts from the global volume, check each tape of the
# job list against them, and pack one archive for the archive server. Runs on a CPU pod that mounts the global volume.
#   bash v5corpus_assemble.sh
# Output: /root/parakeet_v5_corpus.tar.xz (+ .sha256), also copied to results/corpus_v5/final/. The archive holds
#   parakeet_v5_corpus/out/<conv>/parakeet_v5.json, failed/, logs/, SUMMARY.txt and manifest.tsv
#   (one line per tape: conv, audio seconds, words, words with confidence).
set -euo pipefail
G=/workspace-global; R=$G/results/corpus_v5; F=parakeet_v5_corpus; W=/root/final
say() { echo "$(date +%T) $*"; }
rm -rf $W && mkdir -p $W/$F/logs
tar -xf $G/v5corpus/kit.tar -C $W v5corpus/jobs.jsonl
for d in shard_0 shard_1 shard_2 shard_3 shard_4 shard_5 shard_6 shard_7 retry; do  # retry last: its outputs win
  [ -d $R/$d ] || continue
  n=0
  for p in $R/$d/part_*.tar.xz; do
    cp --no-preserve=ownership $p /root/part.tar.xz
    [ "$(stat -c %s /root/part.tar.xz)" = "$(stat -c %s $p)" ] || { say "short read of $p"; exit 1; }
    tar --no-same-owner -xJf /root/part.tar.xz -C $W/$F; n=$((n + 1))
  done
  mkdir -p $W/$F/logs/$d && cp -r --no-preserve=ownership $R/$d/logs_* $W/$F/logs/$d/
  say "$d: $n parts"
done
rm -f /root/part.tar.xz
python3 - $W/v5corpus/jobs.jsonl $W/$F <<'EOF'
import json, sys
from pathlib import Path
jobs = [json.loads(l)["conv"] for l in open(sys.argv[1]) if l.strip()]
root = Path(sys.argv[2])
rows, failed, missing, bad = [], [], [], []
for c in jobs:
    p = root / "out" / c / "parakeet_v5.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            w = d["words"]
            rows.append((c, d["audio_seconds"], len(w), sum("c" in x for x in w)))
        except Exception as e:
            bad.append(f"{c}: {e}")
    elif (root / "failed" / f"parakeet_v5-{c}.txt").exists():
        failed.append(c)
    else:
        missing.append(c)
(root / "manifest.tsv").write_text("conv\taudio_seconds\twords\twords_with_confidence\n"
                                   + "".join(f"{c}\t{a}\t{n}\t{k}\n" for c, a, n, k in rows), encoding="utf-8")
h = sum(r[1] for r in rows) / 3600; nw = sum(r[2] for r in rows); nc = sum(r[3] for r in rows)
summary = (f"Parakeet v5 (round 5, lr 6e-5) over the corpus: {len(jobs)} tapes in the job list\n"
           f"  transcribed: {len(rows)} tapes, {h:.1f} audio hours, {nw:,} words, {nc / max(nw, 1):.1%} with confidence\n"
           f"  failed: {len(failed)} {failed[:20]}\n  missing: {len(missing)} {missing[:20]}\n  unreadable: {len(bad)} {bad[:5]}\n")
(root / "SUMMARY.txt").write_text(summary, encoding="utf-8")
print(summary, end="")
if missing or bad:
    sys.exit(1)
EOF
say "packing"
tar -C $W -cf - $F | xz -9 -T4 > /root/$F.tar.xz
xz -t /root/$F.tar.xz
(cd /root && sha256sum $F.tar.xz > $F.tar.xz.sha256)
mkdir -p $R/final && cp --no-preserve=ownership /root/$F.tar.xz /root/$F.tar.xz.sha256 $R/final/
for try in $(seq 20); do [ "$(stat -c %s $R/final/$F.tar.xz)" = "$(stat -c %s /root/$F.tar.xz)" ] && break; sleep 10; done
say "archive: $(stat -c %s /root/$F.tar.xz) bytes ($(du -sh $W/$F | cut -f1) unpacked), $(cat /root/$F.tar.xz.sha256)"
say "ASSEMBLY OK"
