#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Binaries and profiles live in <repo>/<gpu>/loom_msa/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/$("$SCRIPT_DIR/../gpu.sh" name)/loom_msa}"
# Defaults to the build the results spreadsheet uses (make builds it);
# VARIANT=sota profiles build/msa_sota_large (make sota) instead. Override
# to profile another variant, e.g. BIN=$GPU_DIR/build/msa_optimized_medium
VARIANT="${VARIANT:-baseline}"
case "$VARIANT" in
  baseline) BIN="${BIN:-$GPU_DIR/build/msa_optimized_large}" ;;
  sota) BIN="${BIN:-$GPU_DIR/build/msa_sota_large}"
        NCU_BIN="${NCU_BIN:-$GPU_DIR/build/msa_sota_large_1chunk}" ;;
  *) echo "VARIANT must be baseline or sota" >&2; exit 2 ;;
esac
# ARGS go to the binary; the default is the spreadsheet's random inputs.
# LAUNCH=direct profiles the direct-launch mode instead of graph replays (the
# msa_* harness binaries take --launch); set TIMESTAMP to give graph and
# direct runs the same file suffix.
LAUNCH="${LAUNCH:-graph}"
read -r -a BIN_ARGS <<< "${ARGS:---inputs=random} --launch=$LAUNCH"
OUT_DIR="$GPU_DIR/profiles"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
STAMP="$(basename "$BIN")-$LAUNCH-$TIMESTAMP"

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
    # Baseline: each iteration runs four kernels (graph nodes in graph
    # mode), in this order: audio projection, IMU projection, attention, and
    # statistics reduction. SOTA: msa_sota_large runs 13 concurrent
    # msa_sota_attention_kernel launches (one per imu chunk), which ncu would
    # serialize, so ncu profiles msa_sota_large_1chunk (NCU_BIN): the same
    # kernel as one launch per iteration (the last block of each query block
    # merges the splits and the statistics). Skip the first warm-up
    # iteration's kernels, then capture every launch of the second.
    if [[ "$VARIANT" == sota ]]; then PER_ITER=1; else PER_ITER=4; fi
    NCU_BIN="${NCU_BIN:-$BIN}"
    ncu \
      --set basic \
      --launch-skip "${NCU_SKIP:-$PER_ITER}" \
      --launch-count "${NCU_COUNT:-$PER_ITER}" \
      --replay-mode kernel \
      --graph-profiling node \
      --target-processes application-only \
      --export "$OUT_DIR/ncu-all-kernels-$(basename "$NCU_BIN")-$LAUNCH-$TIMESTAMP" \
      --force-overwrite \
      "$NCU_BIN" "${BIN_ARGS[@]}"
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
