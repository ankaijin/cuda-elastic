#!/usr/bin/env bash
set -euo pipefail

# Profiles build/anomaly_mse at the large shape (random inputs).
# Override the arguments with ARGS, e.g. ARGS="--count=640" ./profile_cuda.sh
BIN="./build/anomaly_mse"
ARGS="${ARGS:---count=67108864 --inputs=random --replays=20}"
OUT_DIR="profiles"
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
    # Each graph holds two kernel nodes: mse_partial_kernel, then
    # mse_finalize_kernel. Skip the correctness replay's two nodes and capture
    # one launch of each from the first kernels-only replay.
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
