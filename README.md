# cuda-elastic

Hand-written CUDA ports of the kernels in several Loom application workloads, each with a harness that runs the same work two ways, captured once into a **CUDA Graph** and replayed, or **launched directly** on every iteration, and checks the GPU result against a CPU reference. Each workload also has a **SOTA** version, the fastest implementation found for the RTX 3090 (see [SOTA versions](#sota-versions)). The repository also holds the Nsight Compute and Nsight Systems profiles of every run, on an RTX 3090 and an RTX 4090, and a spreadsheet built from them.

**Results:** [`results/README.md`](results/README.md) has the summary tables (kernel and whole-application time per workload, graph vs. direct vs. SOTA, and RTX 4090 vs. RTX 3090, predicted and measured), and [`results/cuda_performance.xlsx`](results/cuda_performance.xlsx) has the full ncu metrics. Every number links to the profile it was read from.

## Workloads

| Directory | Workload | Kernels | Largest size profiled |
|---|---|---|---|
| [`llama2c/`](llama2c) | The three llama2.c operations in Loom's kernel smoke workload | `rmsnorm`, `softmax`, `matmul` (naive and coalesced) | batch 1024, dim 16384, 8192 rows |
| [`loom_msa/`](loom_msa) | Multi-sensor attention: two sensor projections, attention, statistics reduction | frozen baseline (`msa_baseline.cu`) and optimized variant (`msa_optimized.cu`: one-pass online softmax, shared-memory context tiles, context split across blocks) | 25,000 queries x 250,000 context |
| [`pagerank/`](pagerank) | GAPBS PageRank on a generated regular CSR graph | `initialize_scores`, `pagerank_update` | 2^20 vertices, 16 edges/vertex, 20 iterations |
| [`mlperf-anomaly/`](mlperf-anomaly) | MLPerf Tiny anomaly detection: dequantize and mean squared error | `mse_partial`, `mse_finalize` | 64M elements |
| [`mlperf-classification/`](mlperf-classification) | MLPerf Tiny keyword spotting (dequantize + argmax) and visual wake words (center bytes) | `kws_postprocess`, `vww_preprocess` | 12 classes x 4M rows; 256M bytes |

[`vec_add.cu`](vec_add.cu) is a minimal vector add for checking that the toolchain and GPU work.

Every harness times its iterations with CUDA events, wraps each timed iteration in an NVTX range for nsys, and fails if the output does not match its reference. The loom_msa, PageRank and MLPerf harnesses time the whole application (H2D copies, kernels, D2H copies); llama2c times an rmsnorm + softmax + matmul iteration, with or without copies. The MLPerf ports also build Loom's own reference binary from upstream [mlcommons/tiny](https://github.com/mlcommons/tiny) (fetched at a pinned commit) and compare its output with the smoke oracle in `loom/`.

## Layout

Sources are shared at the top level; binaries and profiles go in a folder per GPU, so runs on different GPUs never overwrite each other:

```
<workload>/                  sources, Makefile or profile_cuda.sh
rtx3090/<workload>/build/    binaries (rtx4090/ likewise)
rtx3090/<workload>/profiles/ ncu-*.ncu-rep, nsys-*.nsys-rep and .sqlite exports, logs
results/                     summary README and spreadsheet, and the script that builds them
gpu.sh                       prints the current GPU's folder name (rtx3090) and arch (sm_86)
profile_ncu_all.sh           runs every ncu profile the spreadsheet uses
```

`gpu.sh` reads the first device in `CUDA_VISIBLE_DEVICES`; set `GPU=...` and `ARCH=...` to override it.

## Requirements

- An NVIDIA GPU and the CUDA toolkit (`nvcc`). The committed RTX 3090 runs used CUDA 13.3, the RTX 4090 runs CUDA 12.5.
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) for profiling. ncu needs access to the GPU performance counters; where that is admin-only, run it under `sudo` (see below).
- `clang++` and `git` for the MLPerf Loom reference builds.
- Python 3 for `loom_msa/benchmark.py` and the llama2c plots, and `openpyxl` for the spreadsheet.

## Build and check

loom_msa and the MLPerf ports build with `make`; `make check` builds everything, then runs every shape and input set and fails on a mismatch. Add `LAUNCH=direct` to check the direct-launch mode instead of graphs.

```bash
make -C loom_msa check
```

```bash
make -C mlperf-anomaly check
```

```bash
make -C mlperf-classification check
```

llama2c and PageRank are single files that their `profile_cuda.sh` builds on first use (and rebuilds when the source changes):

```bash
llama2c/profile_cuda.sh run graph auto 1024 16384 8192 1 100 copy coalesced
```

```bash
pagerank/profile_cuda.sh check
```

The llama2c arguments are `[graph|direct] [launches|auto] [batch] [size] [rows] [repeats] [warmup_ms] [copy|nocopy] [naive|coalesced]`, plus optional `random[=SEED]` (seeded inputs in [-1, 1), checked against a double-precision reference) and `--dump=FILE` (see `llama2c/llama2c_common.h`). PageRank is configured by environment variables: `PAGERANK_MODE` (`validate`, `bench` or `all`), `PAGERANK_LAUNCH` (`graph` or `direct`), `PAGERANK_BLOCK_SIZE`, `PAGERANK_PIN_HOST`, `PAGERANK_UPLOAD_CHUNKS` and `GRAPH_REPLAYS`.

loom_msa also has a repeated, interleaved baseline vs. optimized vs. SOTA benchmark that reports medians and spread:

```bash
make -C loom_msa benchmark
```

## SOTA versions

Each workload has a `<name>_sota.cu`: the fastest implementation found for the RTX 3090 after researching published libraries and kernels (each source's header lists what was tried and measured). They don't follow the Loom source, but they do the same work and produce the same output:

- **Same work per iteration.** Every timed iteration copies the full input from pinned host memory (by memcpy or zero-copy reads) and the full output back. llama2c is the exception: its inputs stay on the GPU in both versions, as weights would in real inference.
- **No caching across iterations.** Each SOTA binary except llama2c's rotates over 3 different inputs (`--rotate=K`, `PAGERANK_ROTATE` for PageRank) and checks every output against its own CPU reference, so a version that reused an earlier iteration's data would fail.
- **Same output.** The SOTA versions reuse the baselines' oracles and tolerances, and each workload's `compare_sota.sh` compares their dumped outputs with the graph and direct baselines on every input mode and several seeds: bitwise for the MLPerf ports, within the oracle tolerance elsewhere.
- **Launch mode.** Each one launches whichever way measured faster (graph or direct).

| Workload | SOTA source | Approach |
|---|---|---|
| llama2c | `llama2c_sota.cu` | One fused kernel per iteration, one block per 64 KiB row held in registers, so each input is read once |
| loom_msa | `msa_sota.cu` | Split-context attention with 8 queries per thread, projections folded into the loads, lazy softmax rescaling, the split merge and statistics in the last block, output written straight to host memory |
| PageRank | `pagerank_sota.cu` | Edges sorted by source sector on the GPU while the graph uploads, then a cooperative pull over destination parts that fit in shared memory |
| anomaly | `anomaly_mse_sota.cu` | Single-pass reduction overlapped with a chunked upload; zero-copy below 16M elements |
| KWS | `kws_postprocess_sota.cu` | One zero-copy kernel, one row per thread, coalesced through shared memory |
| VWW | `vww_preprocess_sota.cu` | 4 MiB chunks on separate copy-in, compute and copy-out streams |

Every workload's directory has a Makefile with `sota` and `check-sota` targets (`make check` in loom_msa and the MLPerf ports runs the SOTA checks too) and a `compare_sota.sh` (`make compare` in llama2c and PageRank, `make compare-sota` in loom_msa). `VARIANT=sota ./profile_cuda.sh ...` profiles the SOTA binary.

## Profile

Each workload's `profile_cuda.sh` takes `nsys`, `ncu` or `all` and writes to `<gpu>/<workload>/profiles/`. `LAUNCH=direct` (`PAGERANK_LAUNCH=direct` for PageRank) profiles the direct-launch mode; `TIMESTAMP=...` gives a graph and a direct run the same file suffix. Run the MLPerf scripts from their own directory, since they find `gpu.sh` by relative path:

```bash
cd mlperf-classification && APP=vww LAUNCH=direct ./profile_cuda.sh nsys
```

llama2c's script also runs launch-overhead sweeps (`sweep`, `sweep-fill`) and plots them (`plot-sweep`, `plot-nodes`); run it with no recognised command to see the usage.

To rerun every ncu profile the spreadsheet uses, in both launch modes, build all workloads for the current GPU and then run:

```bash
sudo env PATH="$PATH" CUDA_VISIBLE_DEVICES=0 ./profile_ncu_all.sh
```

Files written under `sudo` are handed back to the invoking user.

## Rebuild the results

The spreadsheet and the tables in `results/README.md` are generated from the committed profiles; nothing is re-run. The script needs `ncu` on `PATH` to read the `.ncu-rep` files:

```bash
python3 results/build_spreadsheet.py
```

Plain nsys traces are committed as `.nsys-rep` and `.sqlite`. GPU-metrics traces export over 100 MB of SQLite, so only their `.nsys-rep` and a zstd-compressed `.sqlite.zst` are committed; restore with `zstd -d`.
