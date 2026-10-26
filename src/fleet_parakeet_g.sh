#!/bin/bash
# Parakeet corpus re-run on the global volume, pod <shard> of <shards>, in any data center.
#   bash fleet_parakeet_g.sh <shard> <shards> <model .nemo on the global volume> [output name]
# The pod takes every <shards>-th tape of corpus/jobs.jsonl (a fixed slice: the global volume is object storage
# and cannot hold the claim locks the old fleet used), copies only what it needs to its own disk, and runs
# worker.py parakeet with per-word confidence into a local corpus root. Every UPLOAD_S seconds the new outputs
# are packed into one tar.xz (transcripts compress ~5x) and written once to results/corpus/<output>/shard_<k>/.
# A restarted pod unpacks its earlier parts first, so finished tapes are skipped. The pod terminates itself.
K=$1; N=$2; MODEL=$3; OUTPUT=${4:-parakeet_v5.json}
G=/workspace-global; STAGE=${OUTPUT%.json}; OUT=$G/results/corpus/$STAGE/shard_$K; C=/workspace/corpus
UPLOAD_S=1200; MAX_HOURS=${MAX_HOURS:-6}
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
mkdir -p $C $OUT && LOG=$C/shard_$K.log
log() { echo "$(date +%T) $*" >> $LOG; }
upload() {  # pack outputs not yet uploaded into the next part; write-once, no renames on the global volume
  local new=$(cd $C && find out -name "$OUTPUT" -newer $C/.last_upload 2>/dev/null; cd $C && ls failed/$STAGE-*.txt 2>/dev/null)
  touch $C/.upload_start
  if [ -n "$new" ]; then
    local part=$OUT/part_$(date +%s).tar.xz
    (cd $C && echo "$new" | tar -cJf /tmp/part.tar.xz -T -) && cp /tmp/part.tar.xz $part && log "uploaded $(echo "$new" | wc -l) files to $(basename $part)"
  fi
  mv $C/.upload_start $C/.last_upload; cp $LOG $OUT/shard_$K.log 2>/dev/null
}
terminate() { log "$1; terminating"; upload; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; }
( sleep $((MAX_HOURS * 3600)); terminate "watchdog $MAX_HOURS h" ) &
log "boot shard $K of $N on $(nvidia-smi --query-gpu=name --format=csv,noheader) in $(printenv RUNPOD_DC_ID)"

mkdir -p /workspace/corpus
for p in bundle hf torch; do cp -r --no-preserve=ownership $G/$p /workspace/ || terminate "copy $p failed"; done
cp --no-preserve=ownership $G/$MODEL /workspace/model.nemo || terminate "copy model failed"
python3 -c "
import sys
lines = [l for l in open('$G/corpus/jobs.jsonl') if l.strip()]
open('$C/jobs.jsonl', 'w').write(''.join(l for i, l in enumerate(lines) if i % $N == $K))
"
for p in $OUT/part_*.tar.xz; do [ -f "$p" ] && tar -xJf "$p" -C $C; done  # resume: earlier parts of this shard
touch -d '1970-01-01' $C/.last_upload; touch $C/.last_upload  # files unpacked above count as uploaded
log "$(wc -l < $C/jobs.jsonl) tapes in this shard, $(find $C/out -name "$OUTPUT" 2>/dev/null | wc -l) already done"
source /workspace/bundle/env.sh
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WORKER_CACHE=/root/cache
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || terminate "setup failed"
cd /workspace/bundle/src
touch /workspace/no_partner  # tells worker.py nothing else on this pod needs a tape's audio afterwards

( while sleep $UPLOAD_S; do upload; done ) &
for try in 1 2 3 4 5; do
  $NEMO worker.py parakeet --jobs $C/jobs.jsonl --root $C --model /workspace/model.nemo --output-name $OUTPUT \
      --confidence --partner-done /workspace/no_partner >> $LOG 2>&1 && break
  log "parakeet attempt $try exited $?; resuming"
done
terminate "shard $K done: $(find $C/out -name "$OUTPUT" | wc -l) of $(wc -l < $C/jobs.jsonl) tapes, $(ls $C/failed 2>/dev/null | wc -l) failed"
