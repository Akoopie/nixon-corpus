#!/bin/bash
# Round 5, one learning-rate arm, on a pod in any data center: inputs come from the global volume (mounted at
# /workspace-global) and are copied to the pod's own disk at /workspace, so every script keeps its usual paths
# and nothing is written to shared storage until the end. Results (logs, transcripts, scores, the model) are
# then copied to /workspace-global/results/round5/<tag>. The global volume is object storage: plain cp only,
# no renames or locks. The pod terminates itself.
#   bash round5g.sh <lr> <tag>
LR=$1; TAG=$2
G=/workspace-global; OUT=$G/results/round5/$TAG
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
mkdir -p /workspace/round5 && LOG=/workspace/round5/$TAG.log
log() { echo "$(date +%T) $*" >> $LOG; }
save() { mkdir -p $OUT && cp -r --no-preserve=ownership $LOG $OUT/ 2>/dev/null; }
terminate() { log "$1; terminating"; save; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; }
( sleep $((3 * 3600)); terminate "watchdog 3 h" ) &
log "boot $(nvidia-smi --query-gpu=name --format=csv,noheader) in $(printenv RUNPOD_DC_ID), lr $LR"
# Object storage costs a round trip per file: big files and small folders are copied, but the 5,500 WSPF
# training clips come as one tar (round2_wspf.tar) and are unpacked locally.
for p in bundle hf torch corpus correct1 listen_test nixon-data round3 round4; do
  cp -r --no-preserve=ownership $G/$p /workspace/ || terminate "copy $p from global volume failed"
done
# copy the tar to local disk and check its size first: a fresh object can read short for a while after it is
# written (the first attempt failed exactly here, with no error text kept)
WSPF_BYTES=2560839680
for try in 1 2 3 4 5 6; do
  cp --no-preserve=ownership $G/round2_wspf.tar /root/round2_wspf.tar 2>> $LOG
  [ "$(stat -c %s /root/round2_wspf.tar 2>/dev/null)" = "$WSPF_BYTES" ] && break
  log "round2_wspf.tar read $(stat -c %s /root/round2_wspf.tar 2>/dev/null) of $WSPF_BYTES bytes (try $try); retrying in 30 s"
  sleep 30
done
tar --no-same-owner -xf /root/round2_wspf.tar -C /workspace 2>> $LOG  # exit code ignored: ownership warnings are harmless
NWAV=$(find /workspace/round2/wspf/clips -name "*.wav" | wc -l)
[ "$NWAV" -ge 5546 ] && [ -s /workspace/round2/wspf/manifest.jsonl ] || terminate "unpacking round2_wspf.tar incomplete: $NWAV clips"
rm /root/round2_wspf.tar
log "WSPF clips unpacked: $(find /workspace/round2/wspf/clips -name '*.wav' | wc -l) files"
log "inputs copied: $(du -sh /workspace | cut -f1)"
source /workspace/bundle/env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the NeMo environment comes pre-built from the global volume (venv-nemo.tar, made on a pod with fast internet):
# no package downloads, so a host with a slow connection cannot stall setup. Whisper is not needed here.
VENV_BYTES=7998863360
for try in 1 2 3 4 5 6; do
  cp --no-preserve=ownership $G/venv-nemo.tar /root/venv-nemo.tar 2>> $LOG
  [ "$(stat -c %s /root/venv-nemo.tar 2>/dev/null)" = "$VENV_BYTES" ] && break
  log "venv-nemo.tar read $(stat -c %s /root/venv-nemo.tar 2>/dev/null) of $VENV_BYTES bytes (try $try); retrying in 30 s"; sleep 30
done
tar --no-same-owner -xf /root/venv-nemo.tar -C /root 2>> $LOG; rm -f /root/venv-nemo.tar
$NEMO -c "import nemo.collections.asr, jiwer, transformers" >> $LOG 2>&1 || terminate "NeMo environment does not import"
log "NeMo environment ready"
cd /workspace/bundle/src

R=/workspace/round5/$TAG; mkdir -p $R/ft
cp /workspace/round4/ft/train.jsonl $R/ft/train.jsonl
for i in $(seq 10); do cat /workspace/correct1/v5data/train_silence.jsonl >> $R/ft/train.jsonl; done
cp /workspace/round3/ft/dev.jsonl $R/ft/dev.jsonl
log "train manifest: $(wc -l < $R/ft/train.jsonl) clips ($(grep -c '"text": ""' $R/ft/train.jsonl) empty-text); dev $(wc -l < $R/ft/dev.jsonl)"
OK=""
for B in 16 4 2; do  # 16 fits an H200 (141 GB); effective batch stays 32 via accumulation
  $NEMO finetune_parakeet.py train --out $R/ft --epochs 8 --batch $B --lr $LR > $R/train_b$B.log 2>&1 && { OK=$B; break; }
  log "train at batch $B failed: $(grep -a -m1 -E 'Error|error:' $R/train_b$B.log | cut -c1-200)"
done
log "loader: $(grep -a -m2 -o 'Dataset loaded with [0-9]* files[^.]*' $R/train_b${OK:-4}.log | tr '\n' ';')"
[ -n "$OK" ] && [ -s $R/ft/parakeet-tdt-0.6b-v3-nixon.nemo ] || terminate "training failed"
log "trained at batch $OK"

REFS=$(ls $NIXON_DATA/validation/refs | sed 's/\.json$//' | tr '\n' ' ')
PARAKEET_FT=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo $NEMO bench_asr.py run parakeet-nixon --name parakeet-r5-$TAG --refs $REFS > $R/eval.log 2>&1
log "validation transcripts exit $?"
$NEMO transcribe_tests.py --out $R/tests.json --model r5=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --set corrections=/workspace/correct1/v4data/test.jsonl --set listener=/workspace/listen_test/listen_test.jsonl > $R/tests.log 2>&1
log "held-out transcripts exit $?"
$NEMO bench_asr.py score > $R/score.txt 2>&1
mkdir -p $OUT/bench
cp -r --no-preserve=ownership $R/tests.json $R/score.txt $R/eval.log $R/train_b*.log $R/ft/logs $R/ft/parakeet-tdt-0.6b-v3-nixon.nemo $OUT/ \
  && cp -r --no-preserve=ownership $NIXON_DATA/bench/parakeet-r5-$TAG $OUT/bench/ && log "results saved to $OUT" || log "saving results FAILED"
terminate "arm $TAG done"
