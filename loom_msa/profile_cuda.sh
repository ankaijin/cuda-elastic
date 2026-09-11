#!/usr/bin/env bash
set -euo pipefail

BIN="./multi_sensor_attention_graph"
OUT_DIR="profiles"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"

case "${1:-all}" in
  nsys)
    nsys profile \
      --trace=cuda \
      --cuda-graph-trace=graph \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      --output="$OUT_DIR/nsys-$STAMP" \
      "$BIN"
    ;;

  ncu)
    # The captured graph contains four kernel nodes, in this order:
    # audio projection, IMU projection, attention, and statistics reduction.
    # Skip the four warm-up nodes, then capture one launch of each node from
    # the first measured graph replay.
    ncu \
      --set basic \
      --launch-skip 4 \
      --launch-count 4 \
      --replay-mode kernel \
      --graph-profiling node \
      --target-processes application-only \
      --export "$OUT_DIR/ncu-all-kernels-$STAMP" \
      --force-overwrite \
      "$BIN"
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
