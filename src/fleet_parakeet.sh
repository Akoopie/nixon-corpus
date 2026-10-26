#!/bin/bash
# Boot script for a Parakeet-only re-run of the corpus with a new model (no Whisper): each pod downloads
# each tape it claims, transcribes it, writes out/<conv>/$OUTPUT (with per-word confidence) next to the
# earlier files, and terminates itself when the job list is exhausted. Same claims/heartbeat/watchdog
# rules as fleet_worker.sh.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 NIXON_DATA=/workspace/nixon-data PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WORKER_CACHE=/root/cache
C=${CORPUS_ROOT:-/workspace/corpus}
JOBS=${JOBS:-$C/jobs.jsonl}
MODEL=${MODEL:-/workspace/round2/ft/parakeet-tdt-0.6b-v3-nixon.nemo}
OUTPUT=${OUTPUT:-parakeet_v2.json}
MAX_HOURS=${MAX_HOURS:-4}
mkdir -p $C/logs_v2 $WORKER_CACHE
LOG=$C/logs_v2/${RUNPOD_POD_ID:-$(hostname)}.log

terminate() {
  echo "$(date +%T) $1; terminating" >> $LOG
  for i in 1 2 3 4 5; do
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
      https://api.runpod.io/graphql >> $LOG 2>&1
    sleep 60
  done
}

echo "$(date +%T) boot: $OUTPUT from $MODEL, $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $LOG
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || { terminate "environment setup failed"; exit 1; }
[ -s "$MODEL" ] || { terminate "model $MODEL missing"; exit 1; }
(
  t0=$(date +%s)
  while sleep 300; do
    idle=$(( $(date +%s) - $(stat -c %Y $LOG) ))
    [ $idle -gt 3600 ] && terminate "watchdog: log silent for $((idle / 60)) min"
    [ $(( $(date +%s) - t0 )) -gt $(( MAX_HOURS * 3600 )) ] && terminate "watchdog: past $MAX_HOURS h"
  done
) &
cd /workspace/bundle/src
for try in 1 2 3 4 5; do
  /root/venv-nemo/bin/python worker.py parakeet --jobs $JOBS --root $C --model $MODEL \
    --output-name $OUTPUT --confidence >> $LOG 2>&1 && break
  echo "$(date +%T) stage crashed (try $try of 5); restarting" >> $LOG
  sleep 30
done
terminate "job list exhausted"
