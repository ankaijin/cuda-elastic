# Private Loom CUDA workloads: results v1

Timings and correctness checks for the six workloads in [`applications/loom_private`](../../applications/loom_private), ported from private Loom commit `4a75f8d`. Each workload has a baseline, the first direct port, and an optimized version of the same contract, which passes the same check. The older workloads have their own [results page](../README.md).

- Times are microseconds per iteration, averaged over a back-to-back batch, then the median of 3 recorded runs. Lower is better.
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
| PageRank | 5,232 | 4,973 | 1,986 | 2,415 | 2,223 | 1,560 | 1,645 | 1,566 | 1,158 |
| Multisensor attention | 32,704 | 32,204 | 4,280 | 13,266 | 13,163 | 1,779 | 26,588 | 26,560 | 1,381 |
| llama2c decoder | 150,771 | 150,908 | 79,997 | 68,739 | 68,787 | 32,808 | 52,319 | 52,542 | 25,428 |
| Anomaly detection | 511,880 | 511,886 | 11,338 | 240,611 | 240,826 | 11,086 | 170,978 | 171,275 | 7,630 |
| Vecadd memory | 537 | 537 | 422 | 512 | 511 | 412 | 422 | 393 | 361 |
| Vecadd product | 8,307 | 8,313 | 6,160 | 8,187 | 8,188 | 6,134 | 4,951 | 4,711 | 4,615 |

## Kernel-only time (µs)

The same work without the copies.

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 3090 Optimized | RTX 4090 Direct | RTX 4090 Graph | RTX 4090 Optimized | RTX 5090 Direct | RTX 5090 Graph | RTX 5090 Optimized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PageRank | 4,197 | 3,934 | 948 | 1,408 | 1,216 | 549 | 961 | 921 | 436 |
| Multisensor attention | 32,806 | 32,074 | 4,341 | 14,309 | 14,183 | 1,798 | 26,508 | 26,410 | 1,502 |
| llama2c decoder | 151,122 | 151,220 | 80,002 | 69,028 | 69,187 | 32,882 | 52,490 | 52,707 | 25,693 |
| Anomaly detection | 498,304 | 498,307 | 1,167 | 227,312 | 227,511 | 508 | 163,290 | 163,717 | 402 |
| Vecadd memory | 16.3 | 16.0 | 15.1 | 4.0 | 3.8 | 3.9 | 2.9 | 4.1 | 4.1 |
| Vecadd product | 238 | 237 | 238 | 211 | 211 | 211 | 118 | 118 | 117 |

## Optimized speedup

Graph time divided by Optimized time, so both replay a CUDA Graph and only the implementation differs.

| Workload | RTX 3090 end to end | RTX 3090 kernel only | RTX 4090 end to end | RTX 4090 kernel only | RTX 5090 end to end | RTX 5090 kernel only |
|---|--:|--:|--:|--:|--:|--:|
| PageRank | 2.5× | 4.2× | 1.4× | 2.2× | 1.4× | 2.1× |
| Multisensor attention | 7.5× | 7.4× | 7.4× | 7.9× | 19.2× | 17.6× |
| llama2c decoder | 1.9× | 1.9× | 2.1× | 2.1× | 2.1× | 2.1× |
| Anomaly detection | 45.1× | 427.1× | 21.7× | 448.0× | 22.4× | 407.7× |
| Vecadd memory | 1.3× | 1.1× | 1.2× | 1.0× | 1.1× | 1.0× |
| Vecadd product | 1.3× | 1.0× | 1.3× | 1.0× | 1.0× | 1.0× |

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
| PageRank | [pass](../../rtx3090/loom_private/pagerank/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/pagerank/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/pagerank/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/pagerank/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/pagerank/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/pagerank/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |
| Multisensor attention | [pass](../../rtx3090/loom_private/attention/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/attention/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/attention/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/attention/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/attention/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/attention/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |
| llama2c decoder | [pass](../../rtx3090/loom_private/decoder/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/decoder/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/decoder/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/decoder/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/decoder/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/decoder/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |
| Anomaly detection | [pass](../../rtx3090/loom_private/anomaly/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/anomaly/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/anomaly/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/anomaly/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/anomaly/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/anomaly/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |
| Vecadd memory | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |
| Vecadd product | [pass](../../rtx3090/loom_private/vecadd_product/results/20261026-104210-mono-87ab43c6-check.json) | [pass](../../rtx3090/loom_private/vecadd_product/results/20261026-104210-mono-87ab43c6-check-optimized.json) | [pass](../../rtx4090/loom_private/vecadd_product/results/20261026-104211-octacos-49a84b20-check.json) | [pass](../../rtx4090/loom_private/vecadd_product/results/20261026-104211-octacos-49a84b20-check-optimized.json) | [pass](../../rtx5090/loom_private/vecadd_product/results/20261026-104211-kilia-876ceb8a-check.json) | [pass](../../rtx5090/loom_private/vecadd_product/results/20261026-104211-kilia-876ceb8a-check-optimized.json) |

## Test machines

Taken from the recorder metadata (`<gpu>/runs/<run>/metadata.json`) of the selected runs. Source revision is the code the runs were built from, as stated in `inputs/<gpu>.json`.

| GPU | Host | Driver | CUDA | Source revision |
|---|---|---|---|---|
| RTX 3090 (sm_86, 24 GB) | mono | 610.43.02 | CUDA 13.3 | b17b341 |
| RTX 4090 (sm_89, 24 GB) | octacos | 555.58.02 | CUDA 12.8 | b17b341 |
| RTX 5090 (sm_120, 32 GB) | kilia | 615.71.10 | CUDA 13.4 | b17b341 |

Generated by `results/loom_private/build_report.py` from the runs selected in `results/loom_private/inputs/<gpu>.json`. The workbook [`loom_private_v1.xlsx`](loom_private_v1.xlsx) has the same data plus single-iteration latencies, per-kernel times and Nsight Compute metrics from the selected profiles, contract hashes, and per-invocation check results.
