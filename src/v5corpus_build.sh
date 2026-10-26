#!/bin/bash
# Build the kit for the v5 corpus run (corpus_v5.sh) on a CPU pod that mounts the global volume. Single files only:
#   v5corpus/kit.tar        bundle/src (code), bundle/env.sh, v5corpus/jobs.jsonl (19,426 tapes)
#   v5corpus/kit.env        sizes + sha256 of kit.tar and of the v5 model, and the model's path
#   v5corpus/corpus_v5.sh   the boot script
# Each file written to the global volume is read back and checked.
#   bash v5corpus_build.sh   (expects /root/src.tar, /root/jobs_v5.jsonl, /root/corpus_v5.sh)
set -euo pipefail
G=/workspace-global; V=$G/v5corpus
MODEL=$G/results/round5k/lr6e-5/parakeet-tdt-0.6b-v3-nixon.nemo  # v5 (round 5, lr 6e-5; METHODS 7g)
say() { echo "$(date +%T) $*"; }
wait_size() { for try in $(seq 20); do [ "$(stat -c %s "$1" 2>/dev/null)" = "$2" ] && return 0; sleep 15; done; return 1; }
mkdir -p $V /root/kit/bundle/src /root/kit/v5corpus

say "model"
MODEL_BYTES=$(stat -c %s $MODEL); MODEL_SHA=$(sha256sum $MODEL | cut -d' ' -f1)
tar -tf $MODEL | grep -q model_weights.ckpt && tar -tf $MODEL | grep -q model_config.yaml
say "   $MODEL: $MODEL_BYTES bytes, sha256 $MODEL_SHA, archive has config + weights"

say "kit"
tar --no-same-owner -xf /root/src.tar -C /root/kit/bundle/src
cp $G/bundle/env.sh /root/kit/bundle/env.sh
cp /root/jobs_v5.jsonl /root/kit/v5corpus/jobs.jsonl
[ -z "$(find /root/kit -type l)" ] && [ -z "$(find /root/kit -type f -size 0)" ]
python3 -c "
import json
j = [json.loads(l) for l in open('/root/kit/v5corpus/jobs.jsonl') if l.strip()]
assert len(j) == len({x['conv'] for x in j}) == 19426, len(j)
assert all(x['urls'] and all(u.startswith('http') for u in x['urls']) for x in j)
print('   jobs.jsonl:', len(j), 'tapes, all with source URLs')"
python3 -c "import ast; [ast.parse(open(f'/root/kit/bundle/src/{f}').read()) for f in ('worker.py', 'paths.py')]; print('   worker.py parses')"
tar -cf /root/kit.tar -C /root/kit bundle v5corpus
KIT_BYTES=$(stat -c %s /root/kit.tar); KIT_SHA=$(sha256sum /root/kit.tar | cut -d' ' -f1)
cp --no-preserve=ownership /root/kit.tar $V/kit.tar
wait_size $V/kit.tar $KIT_BYTES
[ "$(sha256sum $V/kit.tar | cut -d' ' -f1)" = $KIT_SHA ]
printf 'KIT_SHA=%s\nKIT_BYTES=%s\nMODEL=%s\nMODEL_SHA=%s\nMODEL_BYTES=%s\n' $KIT_SHA $KIT_BYTES $MODEL $MODEL_SHA $MODEL_BYTES > /root/kit.env
cp --no-preserve=ownership /root/kit.env $V/kit.env && cmp /root/kit.env $V/kit.env
sed -i 's/\r$//' /root/corpus_v5.sh && bash -n /root/corpus_v5.sh
cp --no-preserve=ownership /root/corpus_v5.sh $V/corpus_v5.sh && cmp /root/corpus_v5.sh $V/corpus_v5.sh
cat $V/kit.env
say "KIT BUILD OK"
