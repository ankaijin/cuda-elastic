#!/usr/bin/env bash
# Checks that the SOTA binaries produce exactly the baseline's outputs.
#
# For each app, shape and input (every input mode; random and ties with
# several seeds) it runs the baseline in graph mode, the baseline in direct
# mode and the SOTA binary (its default, plus graph launch and its other
# --mode), each with --dump, and cmps every dump against the graph
# baseline's, byte for byte. Dumps hold the host output buffers: KWS writes
# batch x count float32 probabilities then batch int32 indices, VWW count
# int8 bytes. The SOTA dumps its input set 0, which is the baseline's input
# (the other --rotate sets are checked inside the SOTA binary itself).
#
#   ./compare_sota.sh              both apps
#   APPS=kws ./compare_sota.sh     one app
#   KWS_SHAPES=12:4194304 INPUTS=ties ./compare_sota.sh   a subset
#   SEEDS="1 2 3" REPLAYS=2 GPU_LOCK=/tmp/cuda-elastic-gpu.lock ./compare_sota.sh
# GPU_LOCK, if set, wraps every GPU run in flock on that file. Binaries come
# from <repo>/<gpu>/mlperf-classification/build (make all), dumps go to a
# temporary directory that is removed on exit.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD:-$HERE/../$("$HERE/../gpu.sh" name)/mlperf-classification/build}"
APPS="${APPS:-kws vww}"
SEEDS="${SEEDS:-1 2 3}"
INPUTS="${INPUTS:-smoke random ties}"  # input modes to run (VWW has no ties)
REPLAYS="${REPLAYS:-2}"
# KWS count:batch, VWW count: the check shapes plus the odd ones.
KWS_SHAPES="${KWS_SHAPES:-12:1 12:4096 12:4194304 12:1000003 7:4099}"
VWW_SHAPES="${VWW_SHAPES:-16 27648 268435456 1000003 9999991}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run() {
  if [[ -n "${GPU_LOCK:-}" ]]; then
    flock "$GPU_LOCK" "$@"
  else
    "$@"
  fi
}

wanted() { [[ " $INPUTS " == *" $1 "* ]]; }

failures=0
compared=0
# compare <app> <baseline> <sota> <sota's other mode> <args...>: every
# variant of one app on one input, against the graph baseline.
compare() {
  local app="$1" baseline="$2" sota="$3" altmode="$4"
  shift 4
  local args=("$@" --replays="$REPLAYS")
  local label="$app ${*}"
  if ! run "$BUILD/$baseline" "${args[@]}" --launch=graph \
      --dump="$TMP/graph" >/dev/null; then
    echo "FAIL $label: graph baseline failed"
    failures=$((failures + 1))
    return
  fi
  local variant before="$failures"
  for variant in "direct:$baseline --launch=direct" \
                 "sota:$sota" \
                 "sota-graph:$sota --launch=graph" \
                 "sota-$altmode:$sota --mode=$altmode"; do
    local name="${variant%%:*}" command=(${variant#*:})
    compared=$((compared + 1))
    if ! run "$BUILD/${command[0]}" "${command[@]:1}" "${args[@]}" \
        --dump="$TMP/$name" >/dev/null; then
      echo "FAIL $label: $name failed"
      failures=$((failures + 1))
    elif ! cmp -s "$TMP/graph" "$TMP/$name"; then
      echo "MISMATCH $label: $name differs from the graph baseline"
      failures=$((failures + 1))
    fi
  done
  if [[ "$failures" -eq "$before" ]]; then
    echo "ok $label ($(stat -c %s "$TMP/graph") B dumps, 4 variants equal)"
  fi
  rm -f "$TMP"/*
}

for app in $APPS; do
  case "$app" in
    kws)
      for shape in $KWS_SHAPES; do
        IFS=: read -r count batch <<< "$shape"
        if wanted smoke; then
          compare kws kws_postprocess kws_postprocess_sota pipeline \
            --count="$count" --batch="$batch" --inputs=smoke
        fi
        for inputs in random ties; do
          wanted "$inputs" || continue
          for seed in $SEEDS; do
            compare kws kws_postprocess kws_postprocess_sota pipeline \
              --count="$count" --batch="$batch" --inputs="$inputs" \
              --seed="$seed"
          done
        done
      done
      ;;
    vww)
      for count in $VWW_SHAPES; do
        if wanted smoke; then
          compare vww vww_preprocess vww_preprocess_sota zerocopy \
            --count="$count" --inputs=smoke
        fi
        wanted random || continue
        for seed in $SEEDS; do
          compare vww vww_preprocess vww_preprocess_sota zerocopy \
            --count="$count" --inputs=random --seed="$seed"
        done
      done
      ;;
    *)
      echo "unknown app $app" >&2
      exit 2
      ;;
  esac
done

echo "$compared comparisons, $failures mismatches"
[[ "$failures" -eq 0 ]]
