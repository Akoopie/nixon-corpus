#!/bin/bash
# Era-LM sweep (METHODS 7j): Parakeet v5 with shallow fusion of the era n-gram LM at five weights, each with and
# without per-tape name boosting, over the 5 dev and 6 validation tapes, on one GPU in any data center.
#   bash lm_sweep.sh
# Inputs are single files on the global volume, each checked on arrival: venv-nemo.tar (pre-built NeMo environment),
# lmsweep/kit.tar (code, job list, boost lists, era6.nemo) and the v5 model; sizes and sha256 in lmsweep/kit.env.
# Weight 0 runs without the LM (the corpus path), so a0.0_nb must reproduce the corpus's parakeet_v5.json.
# Two worker processes share the GPU, five settings each; every tape is downloaded once (the WAV cache is kept:
# no --partner-done). Outputs out/<conv>/parakeet_a<weight>_<b|nb>.json; every UPLOAD_S the new ones go into one
# write-once part in results/lmsweep/. Order: inputs -> NeMo check -> SMOKE (038-034, the shortest tape, at
# LM 0.3 + boost; output checked, uploaded and read back) -> the 10 settings -> final upload. The pod terminates
# itself when done, on any failure, or after MAX_MIN minutes. DRYRUN=1 (CPU pod): everything except the NeMo
# environment and the workers, with a stand-in output to exercise the check and the upload.
G=/workspace-global; V=$G/lmsweep; C=/workspace/lmsweep; K=/workspace/lmsweep_kit
OUT=$G/results/lmsweep${DRYRUN:+_dryrun}
UPLOAD_S=600; MAX_MIN=${MAX_MIN:-100}
SET_A="0.0:nb 0.2:nb 0.5:nb 0.1:b 0.3:b"   # weight:boost; the two lists split the 10 settings between the workers
SET_B="0.1:nb 0.3:nb 0.0:b 0.2:b 0.5:b"
export LC_ALL=C  # sort and comm must agree on order
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
mkdir -p $C $OUT /root/models && LOG=$C/sweep.log && touch $C/uploaded.txt
log() { echo "$(date +%T) $*" >> $LOG; }
wait_size() { for try in $(seq 12); do [ "$(stat -c %s "$1" 2>/dev/null)" = "$2" ] && return 0; sleep 10; done; return 1; }
upload() {  # outputs and failure records not yet uploaded -> one new part; tracked by name, not by mtime
  ( flock 9
    cd $C && { find out -name 'parakeet_a*.json'; find failed -name '*.txt' 2>/dev/null; } | sort > /root/all.txt
    sort -o $C/uploaded.txt $C/uploaded.txt
    comm -23 /root/all.txt $C/uploaded.txt > /root/new.txt
    if [ -s /root/new.txt ]; then
      P=part_$(date +%s).tar.xz
      if tar -cJf /root/$P -T /root/new.txt && cp --no-preserve=ownership /root/$P $OUT/$P \
          && wait_size $OUT/$P $(stat -c %s /root/$P); then
        cat /root/new.txt >> $C/uploaded.txt; log "uploaded $(wc -l < /root/new.txt) files as $P ($(stat -c %s /root/$P) bytes)"
      else
        log "upload of $P FAILED; its files go in the next part"
      fi
      rm -f /root/$P
    fi
    mkdir -p $OUT/logs_${RUNPOD_POD_ID:-dryrun}
    cp --no-preserve=ownership $LOG $C/worker_*.log $OUT/logs_${RUNPOD_POD_ID:-dryrun}/ 2>/dev/null
  ) 9> /root/upload.lock
}
# never returns: if the API refuses five times, the script stops here and the pod waits to be terminated from outside
terminate() { log "$1; terminating"; upload; [ -n "$DRYRUN" ] && exit 1; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; log "podTerminate did not take; waiting to be terminated"; sleep infinity; }
( sleep $((MAX_MIN * 60)); terminate "watchdog $MAX_MIN min" ) &
log "boot on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) in $(printenv RUNPOD_DC_ID), $(nproc) vCPU, $(free -g | awk '/Mem/ {print $2}') GB RAM"

# --- inputs --------------------------------------------------------------------------------------------------
source $V/kit.env 2>> $LOG
[ -n "$KIT_SHA" ] && [ -n "$MODEL_SHA" ] && [ -n "$MODEL" ] && [ -n "$LM_SHA" ] || terminate "lmsweep/kit.env unreadable"
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
[ -n "$DRYRUN" ] || { fetch $G/venv-nemo.tar /root/venv-nemo.tar 7998863360 & P1=$!; }
# at most two reads at once: more made the global volume's client fail with "Cannot allocate memory"
( fetch $V/kit.tar /root/kit.tar $KIT_BYTES $KIT_SHA || exit 2
  fetch $MODEL /root/models/parakeet_v5.nemo $MODEL_BYTES $MODEL_SHA || exit 3 ) & P2=$!
[ -n "$DRYRUN" ] || wait $P1 || terminate "copy venv-nemo.tar failed"
wait $P2; RC=$?; [ $RC -eq 0 ] || terminate "copy failed: $([ $RC -eq 2 ] && echo kit.tar || echo the v5 model)"
log "inputs copied and checked in $(( $(date +%s) - T0 )) s"
for t in $([ -n "$DRYRUN" ] || echo venv-nemo:/root) kit:/workspace; do
  tar --no-same-owner -xf /root/${t%%:*}.tar -C ${t#*:} 2> /root/tar_${t%%:*}.err
  grep -v -E 'Cannot change ownership|Exiting with failure status' /root/tar_${t%%:*}.err | head -3 >> $LOG
  rm /root/${t%%:*}.tar
done
LM=$K/era6.nemo
[ "$(sha256sum $LM | cut -d' ' -f1)" = "$LM_SHA" ] || terminate "era6.nemo in the kit does not match kit.env"
[ "$(ls $K/boost/*.txt | wc -l)" = "$(wc -l < $K/jobs.jsonl)" ] || terminate "boost lists do not match the job list"
source /workspace/bundle/env.sh  # NEMO=/root/venv-nemo/bin/python; cd /workspace/bundle/src
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WORKER_CACHE=/root/cache
[ -n "$DRYRUN" ] || $NEMO -c "import nemo.collections.asr, soundfile, torch; assert torch.cuda.is_available()" >> $LOG 2>&1 \
  || terminate "NeMo environment does not import or sees no GPU"
which ffmpeg curl flock > /dev/null || terminate "ffmpeg, curl or flock missing"
grep -q -- "--ngram-lm" worker.py || terminate "worker.py in the kit has no --ngram-lm"
M=/root/models/parakeet_v5.nemo
for p in $OUT/part_*.tar.xz; do  # resume: earlier parts count as done and uploaded
  [ -f "$p" ] && tar --no-same-owner -xJf "$p" -C $C && tar -tJf "$p" >> $C/uploaded.txt
done
log "$(wc -l < $K/jobs.jsonl) tapes x 10 settings; $(find $C/out -name 'parakeet_a*.json' 2>/dev/null | wc -l) outputs already done"

run_setting() {  # weight:boost -> out/<conv>/parakeet_a<weight>_<b|nb>.json for every tape in $2 (default: all)
  local a=${1%%:*} b=${1#*:} jobs=${2:-$K/jobs.jsonl}; local tag=a${a}_$b; local extra=""
  [ "$a" != "0.0" ] && extra="--ngram-lm $LM --ngram-alpha $a"  # weight 0 = no LM at all (the corpus path)
  [ "$b" = "b" ] && extra="$extra --boost-dir $K/boost --boost-alpha 0.5"
  for try in 1 2; do
    $NEMO worker.py parakeet --jobs $jobs --root $C --model $M --output-name parakeet_$tag.json --confidence $extra \
        >> $C/worker_$tag.log 2>&1 && return 0
    log "setting $tag exited $? (try $try); restarting"
    rm -rf $C/claims/parakeet_$tag; sleep 10
  done
  return 1
}

# --- smoke: one tape end to end, output checked, uploaded and read back -------------------------------------------
grep '"conv": "038-034"' $K/jobs.jsonl > $C/jobs_smoke.jsonl || terminate "smoke tape 038-034 not in the job list"
SO=$C/out/038-034/parakeet_a0.3_b.json
if [ -n "$DRYRUN" ]; then
  mkdir -p $(dirname $SO) && echo '{"words":[{"w":"x","s":0,"e":1,"c":0.9}],"segments":[],"audio_seconds":840,"decoding":{"ngram_alpha":0.3,"boost_phrases":5}}' > $SO
elif [ ! -f $SO ]; then
  run_setting 0.3:b $C/jobs_smoke.jsonl || terminate "smoke: worker failed"
fi
python3 - $SO <<'EOF' >> $LOG 2>&1 || terminate "smoke: output check failed"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
w, dec = d["words"], d.get("decoding") or {}
print(f"smoke output: {len(w)} words over {d['audio_seconds'] / 60:.1f} min, {sum('c' in x for x in w) / max(len(w), 1):.0%} "
      f"with confidence; decoding {dec}")
assert w and d["audio_seconds"] > 0 and dec.get("ngram_alpha") == 0.3 and dec.get("boost_phrases", 0) > 0
EOF
upload
grep -q "out/038-034/parakeet_a0.3_b.json" $C/uploaded.txt || terminate "smoke: upload to the global volume failed"
log "SMOKE OK: $(grep -a -o '[0-9.]* min audio in .*realtime' $C/worker_a0.3_b.log 2>/dev/null | head -1)"
[ -n "$DRYRUN" ] && { log "DRYRUN OK"; exit 0; }

# --- the sweep ----------------------------------------------------------------------------------------------------
( while sleep $UPLOAD_S; do upload; done ) &
run_list() { local rc=0; for s in $1; do run_setting $s || { log "setting $s FAILED"; rc=1; }; done; return $rc; }
run_list "$SET_A" & PA=$!
run_list "$SET_B" & PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
DONE=$(find $C/out -name 'parakeet_a*.json' | wc -l); FAILED=$(ls $C/failed 2>/dev/null | wc -l)
terminate "sweep finished (worker lists exit $RA/$RB): $DONE of $(( $(wc -l < $K/jobs.jsonl) * 10 )) outputs, $FAILED failure records"
