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
| PageRank | 5,232 | 4,966 | 1,983 | 2,417 | 2,227 | 1,568 | 1,670 | 1,628 | 1,158 |
| Multisensor attention | 32,821 | 32,203 | 4,285 | 13,028 | 13,415 | 1,763 | 26,446 | 26,737 | 1,386 |
| llama2c decoder | 150,831 | 150,829 | 80,020 | 68,621 | 68,671 | 32,783 | 52,274 | 52,421 | 25,408 |
| Anomaly detection | 511,910 | 511,940 | 11,348 | 240,653 | 240,876 | 11,123 | 170,916 | 171,270 | 7,558 |
| Vecadd memory | 536 | 535 | 420 | 513 | 514 | 414 | 417 | 415 | 381 |
| Vecadd product | 8,303 | 8,303 | 6,178 | 8,214 | 8,213 | 6,154 | 4,985 | 4,729 | 4,633 |

## Kernel-only time (µs)

The same work without the copies.

| Workload | RTX 3090 Direct | RTX 3090 Graph | RTX 3090 Optimized | RTX 4090 Direct | RTX 4090 Graph | RTX 4090 Optimized | RTX 5090 Direct | RTX 5090 Graph | RTX 5090 Optimized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PageRank | 4,184 | 3,929 | 947 | 1,406 | 1,214 | 555 | 961 | 921 | 436 |
| Multisensor attention | 32,678 | 32,172 | 4,310 | 14,597 | 13,879 | 1,792 | 26,526 | 26,404 | 1,501 |
| llama2c decoder | 151,093 | 151,106 | 80,001 | 69,007 | 69,014 | 32,804 | 52,362 | 52,671 | 25,695 |
| Anomaly detection | 498,352 | 498,360 | 1,169 | 227,311 | 227,508 | 508 | 163,287 | 163,722 | 402 |
| Vecadd memory | 16.4 | 16.0 | 15.1 | 4.0 | 3.8 | 3.8 | 2.9 | 4.1 | 4.1 |
| Vecadd product | 238 | 238 | 237 | 211 | 210 | 211 | 118 | 118 | 117 |

## Optimized speedup

Graph time divided by Optimized time, so both replay a CUDA Graph and only the implementation differs.

| Workload | RTX 3090 end to end | RTX 3090 kernel only | RTX 4090 end to end | RTX 4090 kernel only | RTX 5090 end to end | RTX 5090 kernel only |
|---|--:|--:|--:|--:|--:|--:|
| PageRank | 2.5× | 4.2× | 1.4× | 2.2× | 1.4× | 2.1× |
| Multisensor attention | 7.5× | 7.5× | 7.6× | 7.7× | 19.3× | 17.6× |
| llama2c decoder | 1.9× | 1.9× | 2.1× | 2.1× | 2.1× | 2.0× |
| Anomaly detection | 45.1× | 426.3× | 21.7× | 448.1× | 22.7× | 407.4× |
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
| PageRank | [pass](../../rtx3090/loom_private/pagerank/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/pagerank/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/pagerank/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/pagerank/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/pagerank/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/pagerank/results/20261026-173721-kilia-a338eba7-check-optimized.json) |
| Multisensor attention | [pass](../../rtx3090/loom_private/attention/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/attention/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/attention/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/attention/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/attention/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/attention/results/20261026-173721-kilia-a338eba7-check-optimized.json) |
| llama2c decoder | [pass](../../rtx3090/loom_private/decoder/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/decoder/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/decoder/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/decoder/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/decoder/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/decoder/results/20261026-173721-kilia-a338eba7-check-optimized.json) |
| Anomaly detection | [pass](../../rtx3090/loom_private/anomaly/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/anomaly/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/anomaly/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/anomaly/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/anomaly/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/anomaly/results/20261026-173721-kilia-a338eba7-check-optimized.json) |
| Vecadd memory | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/vecadd_memory/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/vecadd_memory/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/vecadd_memory/results/20261026-173721-kilia-a338eba7-check-optimized.json) |
| Vecadd product | [pass](../../rtx3090/loom_private/vecadd_product/results/20261026-173720-mono-481c9abe-check.json) | [pass](../../rtx3090/loom_private/vecadd_product/results/20261026-173720-mono-481c9abe-check-optimized.json) | [pass](../../rtx4090/loom_private/vecadd_product/results/20261026-173721-octacos-7914e968-check.json) | [pass](../../rtx4090/loom_private/vecadd_product/results/20261026-173721-octacos-7914e968-check-optimized.json) | [pass](../../rtx5090/loom_private/vecadd_product/results/20261026-173721-kilia-a338eba7-check.json) | [pass](../../rtx5090/loom_private/vecadd_product/results/20261026-173721-kilia-a338eba7-check-optimized.json) |

## Test machines

Taken from the recorder metadata (`<gpu>/runs/<run>/metadata.json`) of the selected runs. Source revision is the code the runs were built from, as stated in `inputs/<gpu>.json`.

| GPU | Host | Driver | CUDA | Source revision |
|---|---|---|---|---|
| RTX 3090 (sm_86, 24 GB) | mono | 610.43.02 | CUDA 13.3 | 90b0b41 |
| RTX 4090 (sm_89, 24 GB) | octacos | 555.58.02 | CUDA 12.8 | 90b0b41 |
| RTX 5090 (sm_120, 32 GB) | kilia | 615.71.10 | CUDA 13.4 | 90b0b41 |

Generated by `results/loom_private/build_report.py` from the runs selected in `results/loom_private/inputs/<gpu>.json`. The workbook [`loom_private_v1.xlsx`](loom_private_v1.xlsx) has the same data plus single-iteration latencies, per-kernel times and Nsight Compute metrics from the selected profiles, contract hashes, and per-invocation check results.
