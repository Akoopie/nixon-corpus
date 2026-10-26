#!/bin/bash
# Quick round-4 check: continue round 3 for 80 steps (lr 1e-5, warm-up 10) on round 3's data plus the listener's
# corrected chunks x10, then transcribe the held-out sets with round 3 and this model; validation tapes last.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
source /workspace/bundle/env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/round4q; LOG=$R/round4q.log
log() { echo "$(date +%T) $*" >> $LOG; }
terminate() { log "$1; terminating"; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; }
( sleep 5400; terminate "watchdog 90 min" ) &
log "boot $(nvidia-smi --query-gpu=name --format=csv,noheader)"
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || terminate "setup failed"
cd /workspace/bundle/src
mkdir -p $R/ft
cp /workspace/round3/ft/train.jsonl $R/ft/train.jsonl
for i in $(seq 10); do cat /workspace/correct1/v4data/train.jsonl >> $R/ft/train.jsonl; done
cp /workspace/round3/ft/dev.jsonl $R/ft/dev.jsonl
log "setup done; continuing round 3 on $(wc -l < $R/ft/train.jsonl) clips"
$NEMO finetune_parakeet.py train --out $R/ft --init /workspace/round3/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --lr 1e-5 --warmup 10 --batch 4 --epochs 99 --max-steps 80 > $R/train.log 2>&1 || terminate "training failed"
log "trained"
$NEMO transcribe_tests.py --out $R/tests.json \
    --model r3=/workspace/round3/ft/parakeet-tdt-0.6b-v3-nixon.nemo --model r4q=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --set listener=/workspace/listen_test/listen_test.jsonl --set corrections=/workspace/correct1/v4data/test.jsonl > $R/tests.log 2>&1
log "held-out transcripts exit $?"
REFS=$(ls $NIXON_DATA/validation/refs | sed 's/\.json$//' | tr '\n' ' ')
PARAKEET_FT=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo $NEMO bench_asr.py run parakeet-nixon --name parakeet-r4q --refs $REFS > $R/eval.log 2>&1
$WX bench_asr.py score > $R/score.txt 2>&1
log "validation done"
terminate "round 4 quick done"
