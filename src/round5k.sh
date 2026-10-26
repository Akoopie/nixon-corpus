#!/bin/bash
# Parakeet round 5 (v5), one learning-rate arm, on one GPU pod in any data center.
# Data = round 4's (round 3's WSPF clips + listener clips x10 + the listener's corrections x10) plus the listener's
# triage silence clips x10 (empty text), 8 epochs from NVIDIA's base model, AdamW, effective batch 32.
#
# Inputs are four single files on the global volume (no symlinks, no per-file copies), each checked on arrival:
#   venv-nemo.tar (pre-built NeMo environment), round2_wspf.tar (WSPF clips),
#   v5kit/kit.tar (code, manifests, other clips, validation refs + audio; sha256 in v5kit/kit.env),
#   v5kit/parakeet-tdt-0.6b-v3.nemo (base model; sha256 = its Hugging Face blob name)
# Everything runs on the pod's own disk. Order:
#   1. copy + check inputs   2. preflight (every manifest clip is a real 16 kHz WAV)
#   3. SMOKE: a few training steps on the longest clips at the chosen batch, save a model, transcribe with it,
#      run one validation tape, write the results back to the global volume
#   4. the real training   5. validation tapes + held-out sets   6. results to /workspace-global/results/round5k/<tag>
# The pod terminates itself at the end, on any failure, or after the watchdog's limit.
#   bash round5k.sh <lr> <tag>
LR=$1; TAG=$2
G=/workspace-global; K=$G/v5kit; OUT=$G/results/round5k/$TAG; W=/workspace
BASE=/root/models/parakeet-tdt-0.6b-v3.nemo
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
mkdir -p $W/round5 /root/models && LOG=$W/round5/$TAG.log
log() { echo "$(date +%T) $*" >> $LOG; }
save() { mkdir -p $OUT && cp --no-preserve=ownership $LOG $OUT/ 2>/dev/null; }
# never returns: if the API refuses five times, the script stops here (never runs on into the next step) and the
# pod waits, idle, to be terminated from outside
terminate() { log "$1; terminating"; save; [ -n "$DRYRUN" ] && exit 1; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; log "podTerminate did not take; waiting to be terminated"; save; sleep infinity; }
( sleep $((150 * 60)); terminate "watchdog 2.5 h" ) &
progress() {  # the newest training progress line, for watching from outside
  tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' | grep -a -E 'Epoch [0-9]+:' | tail -1 | cut -c1-160
}
( while sleep 120; do save; { tail -5 $LOG; progress $W/round5/$TAG/train.log; } > /root/progress.txt; \
  cp --no-preserve=ownership /root/progress.txt $OUT/progress.txt 2>/dev/null; done ) &
log "boot $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) in $(printenv RUNPOD_DC_ID), lr $LR"

# --- 1. inputs ---------------------------------------------------------------------------------------------
source $K/kit.env 2>> $LOG
[ -n "$KIT_SHA" ] && [ -n "$BASE_SHA" ] || terminate "v5kit/kit.env unreadable"
fetch() {  # src dst bytes [sha256]: copy one file to local disk; retry until size (and hash) match
  for try in 1 2 3 4 5 6; do
    cp --no-preserve=ownership "$1" "$2" 2>> $LOG
    if [ "$(stat -c %s "$2" 2>/dev/null)" = "$3" ]; then
      [ -z "$4" ] && return 0
      [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$4" ] && return 0
      log "$(basename $1): sha256 mismatch (try $try)"
    else
      log "$(basename $1): read $(stat -c %s "$2" 2>/dev/null) of $3 bytes (try $try)"
    fi
    sleep 30
  done
  return 1
}
T0=$(date +%s)
# DRYRUN=1 (CPU test pod): everything up to the smoke manifests, without the NeMo environment or a GPU
UNPACK="round2_wspf:$W kit:$W"
if [ -z "$DRYRUN" ]; then
  fetch $G/venv-nemo.tar /root/venv-nemo.tar 7998863360 & P1=$!
  UNPACK="venv-nemo:/root $UNPACK"
fi
# at most two reads at once: three parallel reads made the global volume's client fail with "Cannot allocate
# memory" on a 16 GB CPU pod (the retries recovered, but the margin is not worth it)
( fetch $G/round2_wspf.tar /root/round2_wspf.tar 2560839680 || exit 2
  fetch $K/kit.tar /root/kit.tar $KIT_BYTES $KIT_SHA || exit 3
  fetch $K/parakeet-tdt-0.6b-v3.nemo $BASE $BASE_BYTES $BASE_SHA || exit 4 ) & P2=$!
[ -n "$DRYRUN" ] || wait $P1 || terminate "copy venv-nemo.tar failed"
wait $P2; RC=$?; [ $RC -eq 0 ] || terminate "copy failed: $(case $RC in 2) echo round2_wspf.tar;; 3) echo kit.tar;; *) echo base model;; esac)"
log "inputs copied and checked in $(( $(date +%s) - T0 )) s"
for t in $UNPACK; do  # ownership warnings are harmless; the checks below decide
  tar --no-same-owner -xf /root/${t%%:*}.tar -C ${t#*:} 2> /root/tar_${t%%:*}.err
  grep -v -E 'Cannot change ownership|Exiting with failure status' /root/tar_${t%%:*}.err | head -3 >> $LOG
  rm /root/${t%%:*}.tar
done
source $W/bundle/env.sh  # NIXON_DATA=/workspace/nixon-data, NEMO=/root/venv-nemo/bin/python, cd bundle/src
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 2. preflight --------------------------------------------------------------------------------------------
python3 preflight.py --manifest $W/round5/train.jsonl --manifest $W/round5/dev.jsonl \
    --manifest $W/correct1/v4data/test.jsonl --manifest $W/listen_test/listen_test.jsonl \
    --model $BASE --sha256 $BASE_SHA --refs $NIXON_DATA/validation/refs --bench-audio $NIXON_DATA/bench/_audio >> $LOG 2>&1 \
  || terminate "preflight failed"
[ -n "$DRYRUN" ] || $NEMO -c "import nemo.collections.asr, jiwer, transformers, torch; assert torch.cuda.is_available()" >> $LOG 2>&1 \
  || terminate "NeMo environment does not import or sees no GPU"
log "preflight ok, NeMo ready ($(( $(date +%s) - T0 )) s since boot copy began)"

# --- 3. smoke: the whole chain on a tiny run ------------------------------------------------------------------
S=$W/round5/smoke_$TAG; mkdir -p $S/ft
python3 - $W/round5/train.jsonl $W/round5/dev.jsonl $S/ft <<'EOF'
# the 96 longest distinct training clips (the worst case for GPU memory) + 8 empty-text clips; 32 longest dev clips
import json, sys
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train, seen, pick = rows(sys.argv[1]), set(), []
for r in sorted((r for r in train if 1.0 <= r["duration"] <= 22.0 and r["text"]), key=lambda r: -r["duration"]):
    if r["audio_filepath"] not in seen:
        seen.add(r["audio_filepath"]); pick.append(r)
    if len(pick) == 96:
        break
pick += [r for r in train if r["text"] == ""][:8]
dev = sorted(rows(sys.argv[2]), key=lambda r: -r["duration"])[:32]
for name, part in (("train", pick), ("dev", dev)):
    with open(f"{sys.argv[3]}/{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
print(f"smoke: {len(pick)} train clips (longest {pick[0]['duration']} s), {len(dev)} dev clips")
EOF
[ -n "$DRYRUN" ] && { log "DRYRUN OK: $(wc -l < $S/ft/train.jsonl) smoke train / $(wc -l < $S/ft/dev.jsonl) dev clips"; save; exit 0; }
B=""
for b in 16 8 4; do
  $NEMO finetune_parakeet.py train --out $S/ft --init $BASE --epochs 99 --max-steps 4 --warmup 1 --batch $b --lr $LR \
      > $S/train_b$b.log 2>&1 && [ -s $S/ft/parakeet-tdt-0.6b-v3-nixon.nemo ] && { B=$b; break; }
  log "smoke at batch $b failed: $(grep -a -m1 -E 'OutOfMemory|Error|error:' $S/train_b$b.log | cut -c1-200)"
  rm -rf $S/ft/ckpt $S/ft/logs
done
[ -n "$B" ] || terminate "smoke training failed at every batch size"
log "smoke trained at batch $B: $(progress $S/train_b$B.log)"
$NEMO transcribe_tests.py --out $S/tests.json --model smoke=$S/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --set listener=$W/listen_test/listen_test.jsonl > $S/tests.log 2>&1 || terminate "smoke: transcribe_tests failed"
PARAKEET_FT=$S/ft/parakeet-tdt-0.6b-v3-nixon.nemo $NEMO bench_asr.py run parakeet-nixon --name smoke-$TAG \
    --refs 741-002_trial > $S/bench.log 2>&1
SB=$(ls $NIXON_DATA/bench/smoke-$TAG/*.json 2>/dev/null | head -1)
[ -s "$SB" ] || terminate "smoke: bench run wrote nothing ($(grep -a -m1 FAILED $S/bench.log))"
mkdir -p $OUT/smoke && cp --no-preserve=ownership $S/tests.json $SB $S/train_b$B.log $OUT/smoke/ \
  || terminate "smoke: writing results to the global volume failed"
for try in 1 2 3 4 5 6; do [ "$(stat -c %s $OUT/smoke/tests.json 2>/dev/null)" = "$(stat -c %s $S/tests.json)" ] && break; sleep 10; done
[ "$(stat -c %s $OUT/smoke/tests.json 2>/dev/null)" = "$(stat -c %s $S/tests.json)" ] || terminate "smoke: results do not read back from the global volume"
rm -rf $NIXON_DATA/bench/smoke-$TAG $S/ft/ckpt
log "SMOKE OK: train, save, transcribe, bench and write-back all work; batch $B"

# --- 4. training -----------------------------------------------------------------------------------------------
R=$W/round5/$TAG; mkdir -p $R/ft
cp $W/round5/train.jsonl $W/round5/dev.jsonl $R/ft/
log "training: $(wc -l < $R/ft/train.jsonl) clips ($(grep -c '"text": ""' $R/ft/train.jsonl) empty-text), dev $(wc -l < $R/ft/dev.jsonl), batch $B, lr $LR"
$NEMO finetune_parakeet.py train --out $R/ft --init $BASE --epochs 8 --batch $B --lr $LR > $R/train.log 2>&1
RC=$?
if [ $RC -ne 0 ] && grep -a -q OutOfMemory $R/train.log && [ $B -gt 4 ]; then
  log "training ran out of GPU memory at batch $B; once more at batch $((B / 2))"
  B=$((B / 2)); rm -rf $R/ft/ckpt $R/ft/logs; mv $R/train.log $R/train_oom.log
  $NEMO finetune_parakeet.py train --out $R/ft --init $BASE --epochs 8 --batch $B --lr $LR > $R/train.log 2>&1
  RC=$?
fi
log "loader: $(grep -a -m2 -o 'Dataset loaded with [0-9]* files[^.]*' $R/train.log | tr '\n' ';')"
[ $RC -eq 0 ] && [ -s $R/ft/parakeet-tdt-0.6b-v3-nixon.nemo ] \
  || { mkdir -p $OUT && cp --no-preserve=ownership $R/train.log $OUT/ 2>/dev/null; terminate "training failed (exit $RC): $(grep -a -m1 -E 'Error|error:' $R/train.log | cut -c1-200)"; }
log "trained: $(grep -a 'best checkpoint' $R/train.log | cut -c1-200)"

# --- 5. evaluation -----------------------------------------------------------------------------------------------
M=$R/ft/parakeet-tdt-0.6b-v3-nixon.nemo
REFS=$(ls $NIXON_DATA/validation/refs | sed 's/\.json$//' | tr '\n' ' ')
PARAKEET_FT=$M $NEMO bench_asr.py run parakeet-nixon --name parakeet-r5-$TAG --refs $REFS > $R/eval.log 2>&1
log "validation transcripts exit $?: $(ls $NIXON_DATA/bench/parakeet-r5-$TAG | wc -l) files, $(grep -a -c FAILED $R/eval.log) failed"
$NEMO transcribe_tests.py --out $R/tests.json --model r5=$M \
    --set corrections=$W/correct1/v4data/test.jsonl --set listener=$W/listen_test/listen_test.jsonl > $R/tests.log 2>&1
log "held-out transcripts exit $?"

# --- 6. results ----------------------------------------------------------------------------------------------------
mkdir -p $OUT/bench
cp -r --no-preserve=ownership $R/tests.json $R/eval.log $R/train.log $R/ft/logs $OUT/ \
  && cp -r --no-preserve=ownership $NIXON_DATA/bench/parakeet-r5-$TAG $OUT/bench/ \
  && log "results saved to $OUT" || log "saving results FAILED"
MB=$(stat -c %s $M)
cp --no-preserve=ownership $M $OUT/parakeet-tdt-0.6b-v3-nixon.nemo
for try in $(seq 20); do [ "$(stat -c %s $OUT/parakeet-tdt-0.6b-v3-nixon.nemo 2>/dev/null)" = "$MB" ] && break; sleep 15; done
[ "$(stat -c %s $OUT/parakeet-tdt-0.6b-v3-nixon.nemo 2>/dev/null)" = "$MB" ] && log "model saved ($MB bytes)" || log "saving the model FAILED"
$NEMO bench_asr.py score > $R/score.txt 2>&1; cp --no-preserve=ownership $R/score.txt $OUT/ 2>/dev/null
terminate "arm $TAG done"
