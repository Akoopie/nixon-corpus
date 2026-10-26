#!/bin/bash
# Parakeet v5 over the whole corpus (stage A only, no Whisper): pod <shard> of <shards>, in any data center.
#   bash corpus_v5.sh <shard> <shards>
# Inputs are single files on the global volume, each checked on arrival: venv-nemo.tar (pre-built NeMo environment),
# v5corpus/kit.tar (code + job list) and the v5 model; sizes and sha256 in v5corpus/kit.env. The pod takes every
# <shards>-th job (a fixed slice: the global volume can't hold claim locks) and splits it between two worker
# processes, so one downloads and decodes while the other transcribes. One walks its half longest-first and the
# other shortest-first, so two very long tapes (several GB of RAM each) don't meet. Everything runs on local disk;
# every UPLOAD_S the outputs not yet uploaded are packed into one write-once part in results/corpus_v5/shard_<k>/.
# A restarted pod unpacks its earlier parts, so finished tapes are skipped. Order: inputs -> NeMo check -> shard ->
# SMOKE (one ~6-min tape end to end, output checked, uploaded and read back) -> both workers -> final upload.
# The pod terminates itself when done, on any failure, or after MAX_MIN minutes. DRYRUN=1 (CPU pod): everything
# except the NeMo environment and the workers, with a stand-in output to exercise the upload.
K=$1; N=$2
G=/workspace-global; V=$G/v5corpus; C=/workspace/corpus; NAME=parakeet_v5.json
OUT=$G/results/corpus_v5${DRYRUN:+_dryrun}/${TAG:-shard_$K}  # a dry run never writes where a real pod would resume from
# TAG + JOBS_SRC (e.g. TAG=retry JOBS_SRC=$G/v5corpus/jobs_retry.jsonl, shard 0 of 1) re-run a list of failed tapes
UPLOAD_S=900; MAX_MIN=${MAX_MIN:-210}
export LC_ALL=C  # sort and comm must agree on order
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
mkdir -p $C $OUT /root/models && LOG=$C/shard_$K.log && touch $C/uploaded.txt
log() { echo "$(date +%T) $*" >> $LOG; }
wait_size() { for try in $(seq 12); do [ "$(stat -c %s "$1" 2>/dev/null)" = "$2" ] && return 0; sleep 10; done; return 1; }
upload() {  # outputs and failure records not yet uploaded -> one new part; tracked by name, not by mtime
  ( flock 9
    cd $C && { find out -name "$NAME"; find failed -name '*.txt' 2>/dev/null; } | sort > /root/all.txt
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
    mkdir -p $OUT/logs_${RUNPOD_POD_ID:-dryrun}  # one folder per pod: a restarted pod keeps its predecessor's logs
    cp --no-preserve=ownership $LOG $C/worker_*.log $OUT/logs_${RUNPOD_POD_ID:-dryrun}/ 2>/dev/null
  ) 9> /root/upload.lock
}
# never returns: if the API refuses five times, the script stops here and the pod waits to be terminated from outside
terminate() { log "$1; terminating"; upload; [ -n "$DRYRUN" ] && exit 1; for i in 1 2 3 4 5; do curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >> $LOG 2>&1; sleep 60; done; log "podTerminate did not take; waiting to be terminated"; sleep infinity; }
( sleep $((MAX_MIN * 60)); terminate "watchdog $MAX_MIN min" ) &
log "boot shard $K of $N on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) in $(printenv RUNPOD_DC_ID), $(nproc) vCPU, $(free -g | awk '/Mem/ {print $2}') GB RAM"

# --- inputs --------------------------------------------------------------------------------------------------
source $V/kit.env 2>> $LOG
[ -n "$KIT_SHA" ] && [ -n "$MODEL_SHA" ] && [ -n "$MODEL" ] || terminate "v5corpus/kit.env unreadable"
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
source /workspace/bundle/env.sh  # NEMO=/root/venv-nemo/bin/python; cd /workspace/bundle/src
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WORKER_CACHE=/root/cache
[ -n "$DRYRUN" ] || $NEMO -c "import nemo.collections.asr, soundfile, torch; assert torch.cuda.is_available()" >> $LOG 2>&1 \
  || terminate "NeMo environment does not import or sees no GPU"
which ffmpeg curl flock > /dev/null || terminate "ffmpeg, curl or flock missing"
M=/root/models/parakeet_v5.nemo
touch /workspace/no_partner  # worker.py: no other stage on this pod needs a tape's audio, so delete it once done

# --- shard ---------------------------------------------------------------------------------------------------
python3 - ${JOBS_SRC:-/workspace/v5corpus/jobs.jsonl} $K $N $C <<'EOF' >> $LOG
# this pod's jobs (every N-th, longest first), halves a (longest first) and b (shortest first), and a smoke tape
import json, sys
src, k, n, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lines = [l for l in open(src, encoding="utf-8") if l.strip()]
mine = [l for i, l in enumerate(lines) if i % n == k]
jobs = [json.loads(l) for l in mine]
smoke = min([j for j in jobs if j["bytes"] and not j["fallbacks"] and len(j["urls"]) == 1] or jobs,
            key=lambda j: abs(j["bytes"] - 15e6))  # about 6 minutes at 320 kbps: one full window and a partial one
for name, part in (("jobs", mine), ("jobs_a", mine[0::2]), ("jobs_b", mine[1::2][::-1]),
                   ("jobs_smoke", [json.dumps(smoke) + "\n"])):
    open(f"{c}/{name}.jsonl", "w", encoding="utf-8", newline="\n").writelines(part)
print(f"shard {k} of {n}: {len(mine)} of {len(lines)} jobs ({sum(j['bytes'] for j in jobs) / 1e9:.1f} GB listed); "
      f"smoke tape {smoke['conv']} ({smoke['bytes'] / 1e6:.1f} MB)")
EOF
[ -s $C/jobs_a.jsonl ] && [ -s $C/jobs_smoke.jsonl ] || terminate "sharding the job list failed"
for p in $OUT/part_*.tar.xz; do  # resume: this shard's earlier parts count as done and uploaded
  [ -f "$p" ] && tar --no-same-owner -xJf "$p" -C $C && tar -tJf "$p" >> $C/uploaded.txt
done
log "$(wc -l < $C/jobs.jsonl) tapes in this shard; $(find $C/out -name $NAME 2>/dev/null | wc -l) already done"

# --- smoke: one tape end to end, output checked, uploaded and read back -------------------------------------------
SC=$(python3 -c "import json; print(json.load(open('$C/jobs_smoke.jsonl'))['conv'])")
if [ -n "$DRYRUN" ]; then  # stand-in output, to exercise the check and the upload
  mkdir -p $C/out/$SC && echo '{"words":[{"w":"x","s":0,"e":1,"c":0.9}],"segments":[],"audio_seconds":360}' > $C/out/$SC/$NAME
elif [ ! -f $C/out/$SC/$NAME ]; then
  $NEMO worker.py parakeet --jobs $C/jobs_smoke.jsonl --root $C --model $M --output-name $NAME --confidence \
      --partner-done /workspace/no_partner > $C/worker_smoke.log 2>&1 || terminate "smoke: worker exited $?"
fi
python3 - $C/out/$SC/$NAME <<'EOF' >> $LOG 2>&1 || terminate "smoke: output check failed"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
w = d["words"]
share = sum("c" in x for x in w) / max(len(w), 1)
print(f"smoke output: {len(w)} words over {d['audio_seconds'] / 60:.1f} min, {share:.0%} with confidence")
# confidence is reported, not required: NeMo's word aggregation fails on ~7% of 5-min windows (word and token
# counts disagree), and worker.py then keeps the window's words without confidence (136-009 had 19%)
assert w and d["audio_seconds"] > 0
EOF
upload
grep -q "out/$SC/$NAME" $C/uploaded.txt || terminate "smoke: upload to the global volume failed"
log "SMOKE OK: $(grep -a -o '[0-9.]* min audio in .*realtime' $C/worker_smoke.log 2>/dev/null | head -1)"
[ -n "$DRYRUN" ] && { log "DRYRUN OK"; exit 0; }

# --- the shard --------------------------------------------------------------------------------------------------
( while sleep $UPLOAD_S; do upload; done ) &
run_half() {  # a crashed worker is restarted; finished tapes are skipped, and its stale claim is cleared first
  for try in 1 2 3; do
    $NEMO worker.py parakeet --jobs $C/jobs_$1.jsonl --root $C --model $M --output-name $NAME --confidence \
        --partner-done /workspace/no_partner >> $C/worker_$1.log 2>&1 && return 0
    log "worker $1 exited $? (try $try); restarting"
    rm -rf $C/claims/parakeet_v5; sleep 10
  done
  return 1
}
run_half a & PA=$!
run_half b & PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
DONE=$(find $C/out -name $NAME | wc -l); FAILED=$(ls $C/failed 2>/dev/null | wc -l)
terminate "shard $K finished (workers exit $RA/$RB): $DONE of $(wc -l < $C/jobs.jsonl) tapes done, $FAILED failed"
