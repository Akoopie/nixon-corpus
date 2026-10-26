#!/bin/bash
# Parakeet round 4 = round 3's recipe (WSPF loose re-cut + listener clips x10, 8 epochs from the base model) plus the
# listener's corrected chunks x10. Then: validation tapes (bench --name parakeet-r4) and the held-out sets (listener
# clips, correction test windows) transcribed by round 3 and round 4. The pod terminates itself.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
source /workspace/bundle/env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/round4; LOG=$R/round4.log
log() { echo "$(date +%T) $*" >> $LOG; }
terminate() { log "$1; terminating"; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; }
( sleep $((3 * 3600)); terminate "watchdog 3 h" ) &
log "boot $(nvidia-smi --query-gpu=name --format=csv,noheader)"
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || terminate "setup failed"
cd /workspace/bundle/src

# Quick answer first: continue round 3 for 80 steps (lr 1e-5, warm-up 10) on the same data, then the held-out sets
Q=/workspace/round4q; mkdir -p $Q/ft
cp /workspace/round3/ft/train.jsonl $Q/ft/train.jsonl
for i in $(seq 10); do cat /workspace/correct1/v4data/train.jsonl >> $Q/ft/train.jsonl; done
cp /workspace/round3/ft/dev.jsonl $Q/ft/dev.jsonl
log "quick: continuing round 3 on $(wc -l < $Q/ft/train.jsonl) clips"
$NEMO finetune_parakeet.py train --out $Q/ft --init /workspace/round3/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --lr 1e-5 --warmup 10 --batch 4 --epochs 99 --max-steps 80 > $Q/train.log 2>&1 && {
  log "quick: trained"
  $NEMO transcribe_tests.py --out $Q/tests.json \
      --model r3=/workspace/round3/ft/parakeet-tdt-0.6b-v3-nixon.nemo --model r4q=$Q/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
      --set listener=/workspace/listen_test/listen_test.jsonl --set corrections=/workspace/correct1/v4data/test.jsonl > $Q/tests.log 2>&1
  log "quick: held-out transcripts exit $?"
} || log "quick: training failed"

mkdir -p $R/ft
cp /workspace/round3/ft/train.jsonl $R/ft/train.jsonl
for i in $(seq 10); do cat /workspace/correct1/v4data/train.jsonl >> $R/ft/train.jsonl; done
cp /workspace/round3/ft/dev.jsonl $R/ft/dev.jsonl
log "train manifest: $(wc -l < $R/ft/train.jsonl) clips; dev $(wc -l < $R/ft/dev.jsonl)"
OK=""
for B in 4 2; do
  $NEMO finetune_parakeet.py train --out $R/ft --epochs 8 --batch $B --lr 3e-5 > $R/train_b$B.log 2>&1 && { OK=$B; break; }
  log "train at batch $B failed"
done
[ -n "$OK" ] && [ -s $R/ft/parakeet-tdt-0.6b-v3-nixon.nemo ] || terminate "training failed"
log "trained at batch $OK"

REFS=$(ls $NIXON_DATA/validation/refs | sed 's/\.json$//' | tr '\n' ' ')
PARAKEET_FT=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo $NEMO bench_asr.py run parakeet-nixon --name parakeet-r4 --refs $REFS > $R/eval.log 2>&1
log "validation transcripts exit $?"
$WX bench_asr.py score > $R/score.txt 2>&1
$NEMO transcribe_tests.py --out $R/tests.json \
    --model r3=/workspace/round3/ft/parakeet-tdt-0.6b-v3-nixon.nemo --model r4=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --set listener=/workspace/listen_test/listen_test.jsonl --set corrections=/workspace/correct1/v4data/test.jsonl > $R/tests.log 2>&1
log "held-out transcripts exit $?"
terminate "round 4 done"
