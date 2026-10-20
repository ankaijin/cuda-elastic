#!/usr/bin/env bash
# Runs every whole-application timing that results/cuda_performance.xlsx
# uses, on the current GPU: each workload at its spreadsheet shape, launched
# directly, as a CUDA Graph, and as the agentic optimized (<name>_sota)
# version in its own launch mode. Every binary times one batch of BATCH
# back-to-back iterations and then single iterations, each launched alone on
# an idle device (see each harness); build_spreadsheet.py reads both.
#
# The three versions of a workload run interleaved, RUNS times each, so slow
# drift (clocks, temperature) hits them alike; the spreadsheet takes the
# median over runs. Each version's output goes to
#   <gpu>/<workload>/profiles/timing-<name>-<launch>-<stamp>.log
# with one "### run N" header per run. Build every workload first (the
# README's Build section); GPU commands run under flock $GPU_LOCK.
#
#   ./time_all.sh        the timings (RUNS runs of each version)
#   ./time_all.sh nsys   one run of each version under nsys, to
#                        <gpu>/<workload>/profiles/nsys-timing-<name>-<launch>-<stamp>
#                        (.nsys-rep, .sqlite and the run's .log); each timed
#                        iteration is an NVTX range, the single iterations'
#                        names end in .single
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/$("$ROOT/gpu.sh" name)"
STAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUNS="${RUNS:-5}"
BATCH="${BATCH:-50}"  # loom_msa's is compiled in (GRAPH_REPLAYS=50 for large)
GPU_LOCK="${GPU_LOCK:-/tmp/cuda-elastic-gpu.lock}"

# workload|name|launch|command (run with bash -c, so env assignments work).
# The agentic optimized versions' launch modes match SOTA_LAUNCH in
# results/build_spreadsheet.py.
CONFIGS=(
  "llama2c|llama2c|direct|$OUT/llama2c/llama2c_cuda_graph direct $BATCH 1024 16384 8192 1 100 nocopy coalesced"
  "llama2c|llama2c|graph|$OUT/llama2c/llama2c_cuda_graph graph $BATCH 1024 16384 8192 1 100 nocopy coalesced"
  "llama2c|llama2c_sota|graph|$OUT/llama2c/build/llama2c_sota graph $BATCH 1024 16384 8192 1 100 nocopy"
  "loom_msa|msa_optimized_large|direct|$OUT/loom_msa/build/msa_optimized_large --inputs=random --launch=direct"
  "loom_msa|msa_optimized_large|graph|$OUT/loom_msa/build/msa_optimized_large --inputs=random --launch=graph"
  "loom_msa|msa_sota_large|direct|$OUT/loom_msa/build/msa_sota_large --inputs=random --launch=direct"
  "pagerank|pagerank|direct|PAGERANK_MODE=bench PAGERANK_LAUNCH=direct GRAPH_REPLAYS=$BATCH $OUT/pagerank/gapbs_pagerank_cuda_graph"
  "pagerank|pagerank|graph|PAGERANK_MODE=bench PAGERANK_LAUNCH=graph GRAPH_REPLAYS=$BATCH $OUT/pagerank/gapbs_pagerank_cuda_graph"
  "pagerank|pagerank_sota|direct|PAGERANK_MODE=bench PAGERANK_LAUNCH=direct GRAPH_REPLAYS=$BATCH $OUT/pagerank/build/pagerank_sota"
  "mlperf-anomaly|anomaly_mse|direct|$OUT/mlperf-anomaly/build/anomaly_mse --count=67108864 --inputs=random --replays=$BATCH --launch=direct"
  "mlperf-anomaly|anomaly_mse|graph|$OUT/mlperf-anomaly/build/anomaly_mse --count=67108864 --inputs=random --replays=$BATCH --launch=graph"
  "mlperf-anomaly|anomaly_mse_sota|graph|$OUT/mlperf-anomaly/build/anomaly_mse_sota --count=67108864 --inputs=random --replays=$BATCH --launch=graph"
  "mlperf-classification|kws_postprocess|direct|$OUT/mlperf-classification/build/kws_postprocess --count=12 --batch=4194304 --inputs=random --replays=$BATCH --launch=direct"
  "mlperf-classification|kws_postprocess|graph|$OUT/mlperf-classification/build/kws_postprocess --count=12 --batch=4194304 --inputs=random --replays=$BATCH --launch=graph"
  "mlperf-classification|kws_postprocess_sota|direct|$OUT/mlperf-classification/build/kws_postprocess_sota --count=12 --batch=4194304 --inputs=random --replays=$BATCH --launch=direct"
  "mlperf-classification|vww_preprocess|direct|$OUT/mlperf-classification/build/vww_preprocess --count=268435456 --inputs=random --replays=$BATCH --launch=direct"
  "mlperf-classification|vww_preprocess|graph|$OUT/mlperf-classification/build/vww_preprocess --count=268435456 --inputs=random --replays=$BATCH --launch=graph"
  "mlperf-classification|vww_preprocess_sota|direct|$OUT/mlperf-classification/build/vww_preprocess_sota --count=268435456 --inputs=random --replays=$BATCH --launch=direct"
)

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r _ _ _ command <<< "$config"
  binary="$(tr ' ' '\n' <<< "$command" | grep -m1 "^$OUT/")"
  [[ -x "$binary" ]] || { echo "error: $binary missing; build it first" >&2; exit 1; }
done

if [[ "${1:-time}" == nsys ]]; then
  command -v nsys >/dev/null || { echo "error: nsys not in PATH" >&2; exit 127; }
  for config in "${CONFIGS[@]}"; do
    IFS='|' read -r workload name launch command <<< "$config"
    out="$OUT/$workload/profiles/nsys-timing-$name-$launch-$STAMP"
    env_part="${command%%"$OUT"/*}"  # leading VAR=value assignments, if any
    echo "### nsys $workload $name $launch"
    flock "$GPU_LOCK" bash -c "$env_part nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none --export=sqlite --force-overwrite=true \
      --output='$out' ${command#"$env_part"}" > "$out.log" 2>&1 ||
      { echo "error: nsys run failed, see $out.log" >&2; exit 1; }
  done
  echo "### done; traces: $OUT/*/profiles/nsys-timing-*-$STAMP.sqlite"
  exit
elif [[ "${1:-time}" != time ]]; then
  echo "Usage: $0 [time|nsys]" >&2
  exit 2
fi

for ((run = 1; run <= RUNS; ++run)); do
  for config in "${CONFIGS[@]}"; do
    IFS='|' read -r workload name launch command <<< "$config"
    log="$OUT/$workload/profiles/timing-$name-$launch-$STAMP.log"
    mkdir -p "$(dirname "$log")"
    echo "### $workload $name $launch (run $run of $RUNS)"
    {
      echo "### run $run: $command"
      flock "$GPU_LOCK" bash -c "$command"
    } >> "$log" 2>&1 || { echo "error: run failed, see $log" >&2; exit 1; }
  done
done
echo "### done; timing logs: $OUT/*/profiles/timing-*-$STAMP.log"
