#!/bin/bash
# Boot script for every pod in the transcription fleet (all mount network volume ypkugjs7pc at /workspace).
#   environments -> stage B (Whisper + speakers) walks the shared job list; stage A (fine-tuned Parakeet)
#   follows it on the same GPU, transcribing each tape stage B downloads -> when the list is exhausted
#   the pod terminates itself.
# Models load from the volume's Hugging Face cache (HF_HUB_OFFLINE), so fleet pods carry no token.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 NIXON_DATA=/workspace/nixon-data PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WORKER_CACHE=/root/cache TORCH_HOME=/workspace/torch  # torch hub: whisperx alignment model
C=${CORPUS_ROOT:-/workspace/corpus}   # a test run points this elsewhere
JOBS=${JOBS:-$C/jobs.jsonl}
export WORKER_WHISPER_MODE=${WORKER_WHISPER_MODE:-sequential}
MAX_HOURS=${MAX_HOURS:-24}
mkdir -p $C/logs $WORKER_CACHE
LOG=$C/logs/${RUNPOD_POD_ID:-$(hostname)}.log

terminate() {  # retried: a pod that fails to terminate bills until someone notices
  echo "$(date +%T) $1; terminating" >> $LOG
  for i in 1 2 3 4 5; do
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
      https://api.runpod.io/graphql >> $LOG 2>&1
    sleep 60
  done
}

echo "$(date +%T) boot: jobs $JOBS, whisper mode $WORKER_WHISPER_MODE, $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $LOG
bash /workspace/bundle/setup_pinned.sh >> $LOG 2>&1 || { terminate "environment setup failed"; exit 1; }

# Watchdog: a log silent for 90 min (hung download, wedged GPU) or a pod past MAX_HOURS ends itself.
# Any tape it held stops getting heartbeats and is taken over by another pod after 15 min.
(
  t0=$(date +%s)
  while sleep 300; do
    idle=$(( $(date +%s) - $(stat -c %Y $LOG) ))
    [ $idle -gt 5400 ] && terminate "watchdog: log silent for $((idle / 60)) min"
    [ $(( $(date +%s) - t0 )) -gt $(( MAX_HOURS * 3600 )) ] && terminate "watchdog: past $MAX_HOURS h"
  done
) &

retry() {  # a stage that crashes is restarted, so one bug can't end a pod's work early (a trial lost 8 of 10 pods that way)
  for try in 1 2 3 4 5; do
    "$@" >> $LOG 2>&1 && return 0
    echo "$(date +%T) stage crashed (try $try of 5): $2; restarting" >> $LOG
    sleep 30
  done
}

cd /workspace/bundle/src
( retry /root/venv-wx/bin/python worker.py whisper --jobs $JOBS --root $C \
    --partner-done $WORKER_CACHE/parakeet.done
  touch $WORKER_CACHE/whisper.done ) &
B=$!
( retry /root/venv-nemo/bin/python worker.py parakeet --jobs $JOBS --root $C \
    --model /workspace/training/ft/parakeet-tdt-0.6b-v3-nixon.nemo \
    --follow --partner-done $WORKER_CACHE/whisper.done
  touch $WORKER_CACHE/parakeet.done ) &
A=$!
wait $A $B
terminate "job list exhausted"
