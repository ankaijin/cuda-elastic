#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Binaries and profiles live in <repo>/<gpu>/loom_msa/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/$("$SCRIPT_DIR/../gpu.sh" name)/loom_msa}"
# Defaults to the build the results spreadsheet uses (make builds it). Override
# to profile another variant, e.g. BIN=$GPU_DIR/build/msa_optimized_medium
BIN="${BIN:-$GPU_DIR/build/msa_optimized_large}"
# ARGS go to the binary; the default is the spreadsheet's random inputs.
# LAUNCH=direct profiles the direct-launch mode instead of graph replays (the
# msa_* harness binaries take --launch); set TIMESTAMP to give graph and
# direct runs the same file suffix.
LAUNCH="${LAUNCH:-graph}"
read -r -a BIN_ARGS <<< "${ARGS:---inputs=random} --launch=$LAUNCH"
OUT_DIR="$GPU_DIR/profiles"
STAMP="$(basename "$BIN")-$LAUNCH-${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not built; run make in $SCRIPT_DIR (or set BIN)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

case "${1:-all}" in
  nsys)
    nsys profile \
      --trace=cuda,nvtx \
      --cuda-graph-trace=graph \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      --output="$OUT_DIR/nsys-$STAMP" \
      "$BIN" "${BIN_ARGS[@]}"
    ;;

  ncu)
    # Each iteration runs four kernels (graph nodes in graph mode), in this
    # order: audio projection, IMU projection, attention, and statistics
    # reduction. Skip the first warm-up iteration's four, then capture one
    # launch of each from the second.
    ncu \
      --set basic \
      --launch-skip 4 \
      --launch-count 4 \
      --replay-mode kernel \
      --graph-profiling node \
      --target-processes application-only \
      --export "$OUT_DIR/ncu-all-kernels-$STAMP" \
      --force-overwrite \
      "$BIN" "${BIN_ARGS[@]}"
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
