#!/usr/bin/env bash
set -euo pipefail

# Profiles one classification kernel at its large shape (random inputs).
#   APP=kws ./profile_cuda.sh   build/kws_postprocess, 12 x 4M rows (default)
#   APP=vww ./profile_cuda.sh   build/vww_preprocess, 256M bytes
# Override the arguments with ARGS.
APP="${APP:-kws}"
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
  *)
    echo "APP must be kws or vww"
    exit 2
    ;;
esac
OUT_DIR="$GPU_DIR/profiles"
STAMP="$(basename "$BIN")-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"

case "${1:-all}" in
  nsys)
    nsys profile \
      --trace=cuda \
      --cuda-graph-trace=node \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      --output="$OUT_DIR/nsys-$STAMP" \
      $BIN $ARGS
    ;;

  ncu)
    # Each graph holds one kernel node. Skip the correctness replay's node
    # and capture the first kernels-only replay.
    ncu \
      --set full \
      --launch-skip 1 \
      --launch-count 1 \
      --replay-mode kernel \
      --graph-profiling node \
      --target-processes application-only \
      --export "$OUT_DIR/ncu-$STAMP" \
      --force-overwrite \
      $BIN $ARGS
    ;;

  all)
    "$0" nsys
    "$0" ncu
    ;;

  *)
    echo "Usage: [APP=kws|vww] $0 [nsys|ncu|all]"
    exit 2
    ;;
esac
