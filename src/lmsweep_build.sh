#!/bin/bash
# Build the kit for the era-LM sweep (lm_sweep.sh) on a CPU pod that mounts the global volume. Single files only:
#   lmsweep/kit.tar      bundle/src (code), bundle/env.sh, lmsweep_kit/{jobs.jsonl, boost/<conv>.txt, era6.nemo}
#   lmsweep/kit.env      sizes + sha256 of kit.tar, era6.nemo and the v5 model, and the model's path
#   lmsweep/lm_sweep.sh  the boot script
# Each file written to the global volume is read back and checked.
#   bash lmsweep_build.sh <era6.nemo sha256>   (expects /root/src.tar, /root/lmkit.tar with era6.nemo.xz, /root/lm_sweep.sh)
set -euo pipefail
G=/workspace-global; V=$G/lmsweep; LM_SHA=$1
MODEL=$G/results/round5k/lr6e-5/parakeet-tdt-0.6b-v3-nixon.nemo  # v5 (round 5, lr 6e-5; METHODS 7g)
say() { echo "$(date +%T) $*"; }
wait_size() { for try in $(seq 20); do [ "$(stat -c %s "$1" 2>/dev/null)" = "$2" ] && return 0; sleep 15; done; return 1; }
rm -rf /root/kit && mkdir -p $V /root/kit/bundle/src

say "model"
MODEL_BYTES=$(stat -c %s $MODEL); MODEL_SHA=$(sha256sum $MODEL | cut -d' ' -f1)
tar -tf $MODEL | grep -q model_weights.ckpt && tar -tf $MODEL | grep -q model_config.yaml
say "   $MODEL: $MODEL_BYTES bytes, sha256 $MODEL_SHA"

say "kit"
tar --no-same-owner -xf /root/src.tar -C /root/kit/bundle/src
tar --no-same-owner -xf /root/lmkit.tar -C /root/kit
xz -d /root/kit/lmsweep_kit/era6.nemo.xz
[ "$(sha256sum /root/kit/lmsweep_kit/era6.nemo | cut -d' ' -f1)" = "$LM_SHA" ]
say "   era6.nemo: $(stat -c %s /root/kit/lmsweep_kit/era6.nemo) bytes, sha256 matches"
cp $G/bundle/env.sh /root/kit/bundle/env.sh
find /root/kit -name '*.txt' -o -name '*.jsonl' -o -name '*.py' -o -name '*.sh' | xargs sed -i 's/\r$//'
[ -z "$(find /root/kit -type l)" ] && [ -z "$(find /root/kit -type f -size 0)" ]
python3 -c "
import json, os
j = [json.loads(l) for l in open('/root/kit/lmsweep_kit/jobs.jsonl') if l.strip()]
assert len(j) == len({x['conv'] for x in j}) == 11, len(j)
assert all(x['urls'] and all(u.startswith('http') for u in x['urls']) for x in j)
assert all(os.path.getsize(f\"/root/kit/lmsweep_kit/boost/{x['conv']}.txt\") > 0 for x in j)
print('   jobs.jsonl:', len(j), 'tapes, all with source URLs and a boost list')"
python3 -c "import ast; [ast.parse(open(f'/root/kit/bundle/src/{f}').read()) for f in ('worker.py', 'paths.py')]; print('   worker.py parses')"
grep -q -- "--ngram-lm" /root/kit/bundle/src/worker.py
tar -cf /root/kit.tar -C /root/kit bundle lmsweep_kit
KIT_BYTES=$(stat -c %s /root/kit.tar); KIT_SHA=$(sha256sum /root/kit.tar | cut -d' ' -f1)
cp --no-preserve=ownership /root/kit.tar $V/kit.tar
wait_size $V/kit.tar $KIT_BYTES
[ "$(sha256sum $V/kit.tar | cut -d' ' -f1)" = $KIT_SHA ]
printf 'KIT_SHA=%s\nKIT_BYTES=%s\nMODEL=%s\nMODEL_SHA=%s\nMODEL_BYTES=%s\nLM_SHA=%s\n' $KIT_SHA $KIT_BYTES $MODEL $MODEL_SHA $MODEL_BYTES $LM_SHA > /root/kit.env
cp --no-preserve=ownership /root/kit.env $V/kit.env && cmp /root/kit.env $V/kit.env
sed -i 's/\r$//' /root/lm_sweep.sh && bash -n /root/lm_sweep.sh
cp --no-preserve=ownership /root/lm_sweep.sh $V/lm_sweep.sh && cmp /root/lm_sweep.sh $V/lm_sweep.sh
cat $V/kit.env
say "KIT BUILD OK"
