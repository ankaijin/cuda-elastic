# Private Loom CUDA workloads: results v1

Baseline timings and correctness checks for the six workloads in [`applications/loom_private`](../../applications/loom_private), ported from private Loom commit `4a75f8d`. These are the first, unoptimized implementations. The older workloads have their own [results page](../README.md).

- Times are microseconds per iteration, averaged over a back-to-back batch. Lower is better.
- **Direct** launches every kernel on each iteration. **Graph** replays a CUDA Graph captured once.
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

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 4090 Direct | RTX 4090 Graph | RTX 5090 Direct | RTX 5090 Graph |
|---|--:|--:|--:|--:|--:|--:|
| PageRank | 5,343 | 5,059 | 2,483 | 2,274 | 1,699 | 1,569 |
| Multisensor attention | 32,483 | 31,979 | 13,232 | 13,430 | 26,492 | 26,683 |
| llama2c decoder | 147,573 | 149,914 | 68,298 | 68,320 | 52,141 | 52,354 |
| Anomaly detection | 504,419 | 511,874 | 239,661 | 239,625 | 170,771 | 170,856 |
| Vecadd memory | 536 | 534 | 514 | 513 | 401 | 423 |
| Vecadd product | 8,300 | 8,300 | 8,214 | 8,199 | 4,750 | 5,088 |

## Kernel-only time (µs)

The same work without the copies.

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 4090 Direct | RTX 4090 Graph | RTX 5090 Direct | RTX 5090 Graph |
|---|--:|--:|--:|--:|--:|--:|
| PageRank | 4,225 | 3,923 | 1,469 | 1,263 | 962 | 921 |
| Multisensor attention | 32,259 | 31,877 | 13,969 | 13,123 | 26,382 | 26,475 |
| llama2c decoder | 148,385 | 150,281 | 68,763 | 68,812 | 52,336 | 52,450 |
| Anomaly detection | 494,544 | 498,308 | 226,300 | 226,292 | 163,177 | 163,141 |
| Vecadd memory | 16.3 | 16.0 | 3.9 | 3.7 | 2.9 | 4.1 |
| Vecadd product | 238 | 237 | 213 | 212 | 118 | 118 |

## Correctness

Each check compares every output with an independent CPU reference. The GPU runs twice with direct launches and twice with a graph, each time starting from poisoned buffers, and all four outputs must be identical. Checks are not timed. Each cell links to its record.

| Workload | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---|---|---|
| PageRank | [pass](../../rtx3090/loom_private/pagerank/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/pagerank/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/pagerank/results/20261025-235201-kilia-b66c6f3a-check.json) |
| Multisensor attention | [pass](../../rtx3090/loom_private/attention/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/attention/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/attention/results/20261025-235201-kilia-b66c6f3a-check.json) |
| llama2c decoder | [pass](../../rtx3090/loom_private/decoder/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/decoder/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/decoder/results/20261025-235201-kilia-b66c6f3a-check.json) |
| Anomaly detection | [pass](../../rtx3090/loom_private/anomaly/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/anomaly/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/anomaly/results/20261025-235201-kilia-b66c6f3a-check.json) |
| Vecadd memory | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261025-235201-kilia-b66c6f3a-check.json) |
| Vecadd product | [pass](../../rtx3090/loom_private/vecadd_product/results/20261025-234934-mono-39816759-check.json) | [pass](../../rtx4090/loom_private/vecadd_product/results/20261025-235039-octacos-c1f83c8f-check.json) | [pass](../../rtx5090/loom_private/vecadd_product/results/20261025-235201-kilia-b66c6f3a-check.json) |

## Test machines

Taken from the recorder metadata (`<gpu>/runs/<run>/metadata.json`) of the selected runs. Source revision is the last commit before the run that changed anything other than GPU results.

| GPU | Host | Driver | CUDA | Source revision |
|---|---|---|---|---|
| RTX 3090 (sm_86, 24 GB) | mono | 610.43.02 | CUDA 13.3 | 296aa92 |
| RTX 4090 (sm_89, 24 GB) | octacos | 555.58.02 | CUDA 12.8 | 296aa92 |
| RTX 5090 (sm_120, 32 GB) | kilia | 615.71.10 | CUDA 13.4 | 296aa92 |

Generated by `results/loom_private/build_report.py` from the runs selected in `results/loom_private/inputs/<gpu>.json`. The workbook [`loom_private_v1.xlsx`](loom_private_v1.xlsx) has the same data plus contract hashes and per-invocation check results.
