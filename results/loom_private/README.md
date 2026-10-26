# Private Loom CUDA workloads: results v1

Timings and correctness checks for the six workloads in [`applications/loom_private`](../../applications/loom_private), ported from private Loom commit `4a75f8d`. Each workload has a baseline, the first direct port, and an optimized version of the same contract, which passes the same check. The older workloads have their own [results page](../README.md).

- Times are microseconds per iteration, averaged over a back-to-back batch. Lower is better.
- **Direct** launches every kernel on each iteration. **Graph** replays a CUDA Graph captured once. **Optimized** is the optimized version, replayed as a CUDA Graph.
- *pending* means no run has been selected yet.

## Workloads

| Workload | Large problem |
|---|---|
| PageRank | 1,048,576 vertices, 2,097,151 edges, 20 ordered Gauss–Seidel iterations |
| Multisensor attention | 25,000 queries × 250,000 context tokens |
| llama2c decoder | 1,024 sequences × 4 steps, width 16,384, vocabulary 8,192 |
| Anomaly detection | 104,858 samples × 640 values through the 10-layer int8 model |
| Vecadd memory | 1,048,576 integer additions |
| Vecadd product | 1,048,576 pairs of 64-byte operands, added byte by byte |

## End-to-end time (µs)

Includes copying inputs to the GPU and results back. The decoder keeps its weights on the GPU, and the anomaly model is loaded once, before timing. Each workload's `contract.json` defines the scope exactly.

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 3090 Optimized | RTX 4090 Direct | RTX 4090 Graph | RTX 4090 Optimized | RTX 5090 Direct | RTX 5090 Graph | RTX 5090 Optimized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PageRank | 5,343 | 5,059 | pending | 2,483 | 2,274 | pending | 1,699 | 1,569 | pending |
| Multisensor attention | 32,483 | 31,979 | pending | 13,232 | 13,430 | pending | 26,492 | 26,683 | pending |
| llama2c decoder | 147,573 | 149,914 | pending | 68,298 | 68,320 | pending | 52,141 | 52,354 | pending |
| Anomaly detection | 504,419 | 511,874 | pending | 239,661 | 239,625 | pending | 170,771 | 170,856 | pending |
| Vecadd memory | 536 | 534 | pending | 514 | 513 | pending | 401 | 423 | pending |
| Vecadd product | 8,300 | 8,300 | pending | 8,214 | 8,199 | pending | 4,750 | 5,088 | pending |

## Kernel-only time (µs)

The same work without the copies.

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 3090 Optimized | RTX 4090 Direct | RTX 4090 Graph | RTX 4090 Optimized | RTX 5090 Direct | RTX 5090 Graph | RTX 5090 Optimized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PageRank | 4,225 | 3,923 | pending | 1,469 | 1,263 | pending | 962 | 921 | pending |
| Multisensor attention | 32,259 | 31,877 | pending | 13,969 | 13,123 | pending | 26,382 | 26,475 | pending |
| llama2c decoder | 148,385 | 150,281 | pending | 68,763 | 68,812 | pending | 52,336 | 52,450 | pending |
| Anomaly detection | 494,544 | 498,308 | pending | 226,300 | 226,292 | pending | 163,177 | 163,141 | pending |
| Vecadd memory | 16.3 | 16.0 | pending | 3.9 | 3.7 | pending | 2.9 | 4.1 | pending |
| Vecadd product | 238 | 237 | pending | 213 | 212 | pending | 118 | 118 | pending |

## Optimized speedup

Graph time divided by Optimized time, so both replay a CUDA Graph and only the implementation differs.

| Workload | RTX 3090 end to end | RTX 3090 kernel only | RTX 4090 end to end | RTX 4090 kernel only | RTX 5090 end to end | RTX 5090 kernel only |
|---|--:|--:|--:|--:|--:|--:|
| PageRank | pending | pending | pending | pending | pending | pending |
| Multisensor attention | pending | pending | pending | pending | pending | pending |
| llama2c decoder | pending | pending | pending | pending | pending | pending |
| Anomaly detection | pending | pending | pending | pending | pending | pending |
| Vecadd memory | pending | pending | pending | pending | pending | pending |
| Vecadd product | pending | pending | pending | pending | pending | pending |

## Optimized versions

Each workload's README explains its optimized version and the parameters chosen for each GPU.

- **PageRank** ([details](../../applications/loom_private/pagerank/README.md#optimized-variant)): One persistent cooperative kernel runs all sweeps. Per-node sweep tags replace the per-level launches, and the ordered Gauss–Seidel updates stay bit for bit the same, over a level-ordered layout of the same graph.
- **Multisensor attention** ([details](../../applications/loom_private/attention/README.md#optimized-variant)): Blocks of 64 queries share key tiles staged in shared memory, which removes the L2 bottleneck behind the baseline's slow RTX 5090 time. A bound replaces the row-maximum pass, and the context upload overlaps the computation.
- **llama2c decoder** ([details](../../applications/loom_private/decoder/README.md#optimized-variant)): A register-tiled classifier with a stream-K tail keeps each logit's sequential sum, so its output matches the baseline bit for bit.
- **Anomaly detection** ([details](../../applications/loom_private/anomaly/README.md#optimized-variant)): Int8 tensor-core tiles of 16 samples run all ten layers on chip. Requantization uses binary32 paths proven equal to the baseline's, and chunked uploads overlap inference and downloads.
- **Vecadd memory** ([details](../../applications/loom_private/vecadd_memory/README.md#optimized-variant)): Chunked uploads, kernels, and downloads on several streams overlap the two copy directions, with chunk counts and vector widths chosen per GPU.
- **Vecadd product** ([details](../../applications/loom_private/vecadd_product/README.md#optimized-variant)): The same chunked multi-stream pipeline around the baseline kernel, which already runs near DRAM bandwidth.

## Correctness

Each check compares every output with an independent CPU reference. The GPU runs twice with direct launches and twice with a graph, each time starting from poisoned buffers, and all four outputs must be identical. Checks are not timed. Each cell links to its record.

| Workload | RTX 3090 baseline | RTX 3090 optimized | RTX 4090 baseline | RTX 4090 optimized | RTX 5090 baseline | RTX 5090 optimized |
|---|---|---|---|---|---|---|
| PageRank | [pass](../../rtx3090/loom_private/pagerank/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/pagerank/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/pagerank/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |
| Multisensor attention | [pass](../../rtx3090/loom_private/attention/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/attention/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/attention/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |
| llama2c decoder | [pass](../../rtx3090/loom_private/decoder/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/decoder/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/decoder/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |
| Anomaly detection | [pass](../../rtx3090/loom_private/anomaly/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/anomaly/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/anomaly/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |
| Vecadd memory | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |
| Vecadd product | [pass](../../rtx3090/loom_private/vecadd_product/results/20261025-234934-mono-39816759-check.json) | pending | [pass](../../rtx4090/loom_private/vecadd_product/results/20261025-235039-octacos-c1f83c8f-check.json) | pending | [pass](../../rtx5090/loom_private/vecadd_product/results/20261025-235201-kilia-b66c6f3a-check.json) | pending |

## Test machines

Taken from the recorder metadata (`<gpu>/runs/<run>/metadata.json`) of the selected runs. Source revision is the code the runs were built from, as stated in `inputs/<gpu>.json`.

| GPU | Host | Driver | CUDA | Source revision |
|---|---|---|---|---|
| RTX 3090 (sm_86, 24 GB) | mono | 610.43.02 | CUDA 13.3 | 296aa92 |
| RTX 4090 (sm_89, 24 GB) | octacos | 555.58.02 | CUDA 12.8 | 296aa92 |
| RTX 5090 (sm_120, 32 GB) | kilia | 615.71.10 | CUDA 13.4 | 296aa92 |

Generated by `results/loom_private/build_report.py` from the runs selected in `results/loom_private/inputs/<gpu>.json`. The workbook [`loom_private_v1.xlsx`](loom_private_v1.xlsx) has the same data plus single-iteration latencies, per-kernel times and Nsight Compute metrics from the selected profiles, contract hashes, and per-invocation check results.
