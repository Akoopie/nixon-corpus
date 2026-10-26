#!/bin/bash
# Build the round-5 training kit on a CPU pod that mounts the Romania volume (/romania, read only) and the global
# volume (/workspace-global). The kit is three single files on the global volume, no symlinks anywhere:
#   v5kit/parakeet-tdt-0.6b-v3.nemo   NVIDIA's base model as a regular file, checked against its sha256
#   v5kit/kit.tar                     code, manifests, correction/listener/silence clips, validation refs + audio
#   (round2_wspf.tar and venv-nemo.tar stay where they are: both already unpack correctly on GPU pods)
# Every step is checked, then the kit is unpacked again from the global volume into an empty /workspace and the
# preflight runs on it, exactly as a GPU pod will do it.
#   bash v5kit_build.sh   (expects /root/src.tar with the repository's src/*.py)
set -euo pipefail
G=/workspace-global; K=$G/v5kit; W=/workspace
BASE_SHA=3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d
say() { echo "$(date +%T) $*"; }
mkdir -p $K $W

# a fresh object on the global volume can read short for a while: wait until it has the size we wrote
wait_size() { for try in $(seq 20); do [ "$(stat -c %s "$1" 2>/dev/null)" = "$2" ] && return 0; sleep 15; done; return 1; }

say "1. base model"
SRC=$(readlink -f /romania/hf/hub/models--nvidia--parakeet-tdt-0.6b-v3/blobs/$BASE_SHA)
cp --no-preserve=ownership "$SRC" /root/base.nemo
[ "$(sha256sum /root/base.nemo | cut -d' ' -f1)" = $BASE_SHA ] || { say "base model sha256 mismatch"; exit 1; }
BYTES=$(stat -c %s /root/base.nemo)
cp --no-preserve=ownership /root/base.nemo $K/parakeet-tdt-0.6b-v3.nemo
wait_size $K/parakeet-tdt-0.6b-v3.nemo $BYTES || { say "global copy of the model has the wrong size"; exit 1; }
[ "$(sha256sum $K/parakeet-tdt-0.6b-v3.nemo | cut -d' ' -f1)" = $BASE_SHA ] || { say "global copy sha256 mismatch"; exit 1; }
rm /root/base.nemo
say "   $K/parakeet-tdt-0.6b-v3.nemo: $BYTES bytes, sha256 ok (read back from the global volume)"

say "2. workspace layout"
mkdir -p $W/bundle/src $W/round5 $W/correct1 $W/nixon-data/validation $W/nixon-data/bench
tar --no-same-owner -xf /root/src.tar -C $W/bundle/src
cp --no-preserve=ownership $G/bundle/env.sh $W/bundle/env.sh
cp -rL --no-preserve=ownership $G/correct1/v4data $G/correct1/v5data $W/correct1/
cp -rL --no-preserve=ownership $G/listen_test $W/
cp -rL --no-preserve=ownership $G/nixon-data/validation/refs $W/nixon-data/validation/
cp -rL --no-preserve=ownership $G/nixon-data/bench/_audio $W/nixon-data/bench/
# round 5's data: round 4's training manifest (round 3's WSPF clips + listener clips x10 + the listener's
# corrections x10) plus the triage silence clips x10; round 3's dev set
cp $G/round4/ft/train.jsonl $W/round5/train.jsonl
for i in $(seq 10); do cat $W/correct1/v5data/train_silence.jsonl >> $W/round5/train.jsonl; done
cp $G/round3/ft/dev.jsonl $W/round5/dev.jsonl
python3 - <<'EOF'
# correction clips: duration from the file itself (correction_to_training.py wrote end - start, which is too long
# for a window's last chunk when the conversation ends first: 787-013_00360_14 is 7.79 s, not 15.0)
import json, wave
p = "/workspace/round5/train.jsonl"
rows, fixed = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()], set()
for r in rows:
    if r["audio_filepath"].startswith("/workspace/correct1/"):
        with wave.open(r["audio_filepath"], "rb") as w:
            secs = round(w.getnframes() / w.getframerate(), 2)
        if abs(secs - r["duration"]) > 0.25:
            fixed.add((r["audio_filepath"].split("/")[-1], r["duration"], secs))
        r["duration"] = secs
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print(f"   correction clip durations set from the files; changed: {sorted(fixed)}")
EOF
say "   train $(wc -l < $W/round5/train.jsonl) lines ($(grep -c '"text": ""' $W/round5/train.jsonl) empty-text), dev $(wc -l < $W/round5/dev.jsonl)"

say "3. WSPF clips"
WSPF_BYTES=2560839680
cp --no-preserve=ownership $G/round2_wspf.tar /root/round2_wspf.tar
[ "$(stat -c %s /root/round2_wspf.tar)" = $WSPF_BYTES ] || { say "round2_wspf.tar read short"; exit 1; }
tar --no-same-owner -xf /root/round2_wspf.tar -C $W || true  # ownership warnings only; the preflight checks every clip
rm /root/round2_wspf.tar

say "4. checks"
[ -z "$(find $W -type l | head -1)" ] || { say "symlinks in the workspace:"; find $W -type l | head; exit 1; }
[ -z "$(find $W -type f -size 0 ! -name DONE | head -1)" ] || { say "empty files:"; find $W -type f -size 0 ! -name DONE | head; exit 1; }  # DONE = the aligner's empty done-flags
python3 - <<'EOF'
# no clip may be both trained on and scored: held-out listener clips, held-out correction windows, dev set
import json
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train = {r["audio_filepath"] for r in rows("/workspace/round5/train.jsonl")}
for name in ("/workspace/round5/dev.jsonl", "/workspace/correct1/v4data/test.jsonl", "/workspace/listen_test/listen_test.jsonl"):
    both = train & {r["audio_filepath"] for r in rows(name)}
    print(f"   overlap train / {name}: {len(both)}")
    assert not both, sorted(both)[:5]
held = open("/workspace/correct1/v4data/heldout_windows.txt").read().split()
leak = [r["audio_filepath"] for r in rows("/workspace/round5/train.jsonl") if any(r["audio_filepath"].split("/")[-1].startswith(w) for w in held)]
print(f"   held-out windows {held}: {len(leak)} training clips from them")
assert not leak, leak[:5]
EOF
cd $W/bundle/src
python3 preflight.py --manifest $W/round5/train.jsonl --manifest $W/round5/dev.jsonl \
    --manifest $W/correct1/v4data/test.jsonl --manifest $W/listen_test/listen_test.jsonl \
    --model $K/parakeet-tdt-0.6b-v3.nemo --sha256 $BASE_SHA \
    --refs $W/nixon-data/validation/refs --bench-audio $W/nixon-data/bench/_audio

say "5. kit.tar"
tar -cf /root/kit.tar -C $W bundle round5 correct1 listen_test nixon-data
KIT_SHA=$(sha256sum /root/kit.tar | cut -d' ' -f1); KIT_BYTES=$(stat -c %s /root/kit.tar)
cp --no-preserve=ownership /root/kit.tar $K/kit.tar
wait_size $K/kit.tar $KIT_BYTES || { say "global kit.tar has the wrong size"; exit 1; }
[ "$(sha256sum $K/kit.tar | cut -d' ' -f1)" = $KIT_SHA ] || { say "global kit.tar sha256 mismatch"; exit 1; }
printf 'KIT_SHA=%s\nKIT_BYTES=%s\nBASE_SHA=%s\nBASE_BYTES=%s\n' $KIT_SHA $KIT_BYTES $BASE_SHA $BYTES > $K/kit.env
rm /root/kit.tar
say "   kit.tar: $KIT_BYTES bytes, sha256 $KIT_SHA"

say "6. round trip: unpack the global kit into an empty workspace and check again"
rm -rf $W/bundle $W/round5 $W/correct1 $W/listen_test $W/nixon-data
cp --no-preserve=ownership $K/kit.tar /root/kit.tar
[ "$(sha256sum /root/kit.tar | cut -d' ' -f1)" = $KIT_SHA ] || { say "kit.tar read back wrong"; exit 1; }
tar --no-same-owner -xf /root/kit.tar -C $W || true
rm /root/kit.tar
cd $W/bundle/src
python3 preflight.py --manifest $W/round5/train.jsonl --manifest $W/round5/dev.jsonl \
    --manifest $W/correct1/v4data/test.jsonl --manifest $W/listen_test/listen_test.jsonl \
    --refs $W/nixon-data/validation/refs --bench-audio $W/nixon-data/bench/_audio
cat $K/kit.env
say "KIT BUILD OK"
