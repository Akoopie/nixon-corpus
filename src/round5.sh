#!/bin/bash
# Parakeet round 5 = round 4's data (round 3's + the listener's corrections x10) plus the listener's triage
# silence (empty-text clips: "nothing said" and the noise seconds of "mixed" answers) x10, 8 epochs from the
# base model, one arm per learning rate (a learning-rate check). Then the validation tapes (bench --name
# parakeet-r5-<tag>) and the held-out correction chunks. Arms come from /workspace/round5/queue ("<lr> <tag>" per
# line); each pod claims arms with an atomic mkdir, so any number of pods can share the queue. The pod setup runs
# once; the pod terminates itself when the queue is empty.
#   bash round5.sh   (arguments are ignored)
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
source /workspace/bundle/env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/workspace/round5/pod_$RUNPOD_POD_ID.log
log() { echo "$(date +%T) $*" >> $LOG; }
terminate() { log "$1; terminating"; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; }
( sleep $((5 * 3600)); terminate "watchdog 5 h" ) &
log "boot $(nvidia-smi --query-gpu=name --format=csv,noheader)"
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || terminate "setup failed"
cd /workspace/bundle/src

while read -r LR TAG <&3; do  # the queue is on fd 3 so nothing inside the loop can read it from stdin
  [ -n "$TAG" ] || continue
  mkdir /workspace/round5/claim_$TAG 2>/dev/null || continue  # another pod has this arm
  R=/workspace/round5/$TAG; mkdir -p $R/ft
  log "arm $TAG: lr $LR"
  cp /workspace/round4/ft/train.jsonl $R/ft/train.jsonl
  for i in $(seq 10); do cat /workspace/correct1/v5data/train_silence.jsonl >> $R/ft/train.jsonl; done
  cp /workspace/round3/ft/dev.jsonl $R/ft/dev.jsonl
  log "train manifest: $(wc -l < $R/ft/train.jsonl) clips ($(grep -c '"text": ""' $R/ft/train.jsonl) empty-text); dev $(wc -l < $R/ft/dev.jsonl)"
  OK=""
  for B in 4 2; do
    $NEMO finetune_parakeet.py train --out $R/ft --epochs 8 --batch $B --lr $LR > $R/train_b$B.log 2>&1 && { OK=$B; break; }
    log "train at batch $B failed"
  done
  log "loader: $(grep -a -m2 -o 'Dataset loaded with [0-9]* files[^.]*' $R/train_b${OK:-4}.log | tr '\n' ';')"
  [ -n "$OK" ] && [ -s $R/ft/parakeet-tdt-0.6b-v3-nixon.nemo ] || { log "arm $TAG: training failed"; continue; }
  log "trained at batch $OK"

  REFS=$(ls $NIXON_DATA/validation/refs | sed 's/\.json$//' | tr '\n' ' ')
  PARAKEET_FT=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo $NEMO bench_asr.py run parakeet-nixon --name parakeet-r5-$TAG --refs $REFS > $R/eval.log 2>&1
  log "validation transcripts exit $?"
  $NEMO transcribe_tests.py --out $R/tests.json --model r5=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
      --set corrections=/workspace/correct1/v4data/test.jsonl --set listener=/workspace/listen_test/listen_test.jsonl > $R/tests.log 2>&1
  log "held-out transcripts exit $?"
  $WX bench_asr.py score > $R/score.txt 2>&1
done 3< /workspace/round5/queue
terminate "queue empty"
