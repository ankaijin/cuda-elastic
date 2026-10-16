#!/usr/bin/env bash
set -euo pipefail

# Profiles build/anomaly_mse at the large shape (random inputs).
# Override the arguments with ARGS, e.g. ARGS="--count=640" ./profile_cuda.sh
# LAUNCH=direct profiles the direct-launch mode instead of graph replays; set
# TIMESTAMP to give graph and direct runs the same file suffix.
# VARIANT=sota profiles build/anomaly_mse_sota (make sota) instead.
# Binaries and profiles live in <repo>/<gpu>/mlperf-anomaly/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-../$(../gpu.sh name)/mlperf-anomaly}"
VARIANT="${VARIANT:-baseline}"
case "$VARIANT" in
  baseline) BIN="$GPU_DIR/build/anomaly_mse" ;;
  sota) BIN="$GPU_DIR/build/anomaly_mse_sota" ;;
  *) echo "VARIANT must be baseline or sota"; exit 2 ;;
esac
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
    # Baseline: each iteration runs two kernels (graph nodes in graph mode):
    # mse_partial_kernel, then mse_finalize_kernel. Skip the correctness
    # iteration's two and capture one launch of each from the first
    # kernels-only iteration.
    # SOTA: one mse_sota_kernel per kernels-only iteration. At 64M the
    # correctness pass runs one end-to-end iteration per rotated set (3 sets
    # x 4 chunk kernels), so skip 12 and capture the first kernels-only one.
    if [[ "$VARIANT" == sota ]]; then
      SKIP="${NCU_SKIP:-12}" COUNT="${NCU_COUNT:-1}"
    else
      SKIP="${NCU_SKIP:-2}" COUNT="${NCU_COUNT:-2}"
    fi
    ncu \
      --set full \
      --launch-skip "$SKIP" \
      --launch-count "$COUNT" \
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
