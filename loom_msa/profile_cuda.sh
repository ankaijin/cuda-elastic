#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Binaries and profiles live in <repo>/<gpu>/loom_msa/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/$("$SCRIPT_DIR/../gpu.sh" name)/loom_msa}"
# Defaults to the build the results spreadsheet uses (make builds it). Override
# to profile another variant, e.g. BIN=$GPU_DIR/build/msa_optimized_medium
BIN="${BIN:-$GPU_DIR/build/msa_optimized_large}"
OUT_DIR="$GPU_DIR/profiles"
STAMP="$(basename "$BIN")-$(date +%Y%m%d-%H%M%S)"

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not built; run make in $SCRIPT_DIR (or set BIN)" >&2
  exit 1
fi

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
