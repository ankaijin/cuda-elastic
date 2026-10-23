#!/usr/bin/env bash
set -euo pipefail

# Profiles one classification kernel at its large shape (random inputs).
#   APP=kws ./profile_cuda.sh   build/kws_postprocess, 12 x 4M rows (default)
#   APP=vww ./profile_cuda.sh   build/vww_preprocess, 256M bytes
#   APP=kws_sota, APP=vww_sota  the SOTA binaries at the same shapes; their
#                               --launch default is direct, so pass
#                               LAUNCH=direct to profile what they report
# Override the arguments with ARGS. LAUNCH=direct profiles the direct-launch
# mode instead of graph replays; set TIMESTAMP to give graph and direct runs
# the same file suffix.
APP="${APP:-kws}"
# VARIANT=sota is the same as APP=<app>_sota.
[[ "${VARIANT:-baseline}" == sota && "$APP" != *_sota ]] && APP="${APP}_sota"
LAUNCH="${LAUNCH:-graph}"
# Binaries and profiles live in <repo>/<gpu>/mlperf-classification/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-../$(../gpu.sh name)/mlperf-classification}"
case "$APP" in
  kws)
    BIN="$GPU_DIR/build/kws_postprocess"
    ARGS="${ARGS:---count=12 --batch=4194304 --inputs=random --replays=20}"
    ;;
  vww)
    BIN="$GPU_DIR/build/vww_preprocess"
    ARGS="${ARGS:---count=268435456 --inputs=random --replays=20}"
    ;;
  kws_sota)
    BIN="$GPU_DIR/build/kws_postprocess_sota"
    ARGS="${ARGS:---count=12 --batch=4194304 --inputs=random --replays=20}"
    ;;
  vww_sota)
    BIN="$GPU_DIR/build/vww_preprocess_sota"
    ARGS="${ARGS:---count=268435456 --inputs=random --replays=20}"
    ;;
  *)
    echo "APP must be kws, vww, kws_sota or vww_sota"
    exit 2
    ;;
esac
ARGS="$ARGS --launch=$LAUNCH"
OUT_DIR="$GPU_DIR/profiles"
STAMP="$(basename "$BIN")-$LAUNCH-${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT_DIR"

case "${1:-all}" in
  nsys)
    nsys profile \
      --trace=cuda,nvtx \
      --cuda-graph-trace=node \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      --output="$OUT_DIR/nsys-$STAMP" \
      $BIN $ARGS
    ;;

  ncu)
    # Each iteration runs one kernel (one graph node in graph mode). Skip
    # the correctness iteration's and capture the first kernels-only one.
    # The SOTA binaries time kernels-only first and check afterwards, so
    # their launches 0 and 1 are both kernels-only warm-up launches of the
    # same kernel; skip 1 / count 1 profiles a kernels-only launch for them
    # too.
    ncu \
      --set full \
      --launch-skip 1 \
      --launch-count 1 \
      --replay-mode kernel \
      --graph-profiling node \
      --target-processes application-only \
      --export "$OUT_DIR/ncu-$STAMP.ncu-rep" \
      --force-overwrite \
      $BIN $ARGS
    ;;

  all)
    "$0" nsys
    "$0" ncu
    ;;

  *)
    echo "Usage: [APP=kws|vww|kws_sota|vww_sota] $0 [nsys|ncu|all]"
    exit 2
    ;;
esac
