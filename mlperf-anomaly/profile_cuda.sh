#!/usr/bin/env bash
set -euo pipefail

# Profiles build/anomaly_mse at the large shape (random inputs).
# Override the arguments with ARGS, e.g. ARGS="--count=640" ./profile_cuda.sh
# LAUNCH=direct profiles the direct-launch mode instead of graph replays; set
# TIMESTAMP to give graph and direct runs the same file suffix.
# Binaries and profiles live in <repo>/<gpu>/mlperf-anomaly/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-../$(../gpu.sh name)/mlperf-anomaly}"
BIN="$GPU_DIR/build/anomaly_mse"
LAUNCH="${LAUNCH:-graph}"
ARGS="${ARGS:---count=67108864 --inputs=random --replays=20} --launch=$LAUNCH"
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
    # Each iteration runs two kernels (graph nodes in graph mode):
    # mse_partial_kernel, then mse_finalize_kernel. Skip the correctness
    # iteration's two and capture one launch of each from the first
    # kernels-only iteration.
    ncu \
      --set full \
      --launch-skip 2 \
      --launch-count 2 \
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
    echo "Usage: $0 [nsys|ncu|all]"
    exit 2
    ;;
esac
