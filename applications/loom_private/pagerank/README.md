# loom-private/pagerank

Ordered Gauss–Seidel PageRank on a generated graph. This is the CUDA port of
private Loom's `gapbs-pagerank` application: the GAPBS `PageRankPullGS` kernel
(gapbs `src/pr.cc` at `2972aeb2703165bafd921222f4ed7196f542d3a8`), wrapped by
private Loom's generator and harness in `test/applications/gapbs-pagerank-smoke.h`
at commit `4a75f8d743ef7fe755822bab0ea0fb78009c145b`. The source paths and their
hashes are listed in [`contract.json`](contract.json); nothing from private Loom
is copied here.

It is a new workload identity. The legacy [`pagerank/`](../../../pagerank)
application at the repository root runs Jacobi sweeps on a different graph and
stays unchanged.

## What it computes

The large workload has $N = 2^{20} = 1{,}048{,}576$ vertices, 2,097,151 edges, and
20 iterations.

- **Graph.** Node $n$ has out-degree $1 + (n \bmod 3)$. Its out-neighbour $i$ is
  $(7n + 13i + 1) \bmod N$. Self-loops and targets shared by several sources
  are kept. Each node's incoming list is in ascending source order (a stable
  counting sort of the out-edges), and every sum follows that order.
- **Iteration.** Every score starts at $1/N$ and every contribution at
  $c_n = (1/N)/\deg^+(n)$. One sweep visits $u = 0, 1, \dots, N-1$ in order.
  It sums $c_v$ over the incoming sources $v$ (in float, in list order), sets
  $s_u = (1-0.85)/N + 0.85 \cdot \text{sum}$, and then sets
  $c_u = s_u / \deg^+(u)$ immediately. Later nodes in the same sweep therefore
  read the new $c_u$, while sources $v \ge u$ contribute the previous sweep's
  value. Epsilon is 0, so all 20 sweeps run.
- **Output.** The 1,048,576 float scores after the 20th sweep.

$N$ must be a power of two of at least 3, as upstream's static assertion
requires. Overrides here accept $4 \le N \le 2^{26}$ and 1 to 1,000 iterations;
anything else exits 2.

## How the GPU keeps the sequential order

Within one sweep, node $u$ depends only on its sources $v < u$. Graph
preparation on the host ([`graph.cpp`](graph.cpp), untimed, like the CSR
build) gives each node a dependency level: 0 when it has no source $v < u$,
otherwise 1 more than the highest level among those sources. The large graph
has **16 levels**, with 307,918 nodes in level 0 down to 1 node in level 15.

Each level is one kernel launch, so a sweep is 16 launches and the large run
is 320 update launches after one initialization launch. The kernel keeps two
contribution buffers. Sweep $k$ writes buffer $k \bmod 2$ and treats the
other buffer as the previous sweep. A thread updating $u$ reads $c_v$ from
the current buffer when $v < u$ (an earlier level wrote it in this sweep) and
from the previous buffer when $v \ge u$. One thread computes each node's sum
sequentially in list order with the same float operations as the definition.
The build uses `--fmad=false` and `-ffp-contract=off`, and no atomics are
used. The scores therefore equal the sequential loop bit for bit, and Direct
and Graph launches agree bitwise.

The schedule changes when work runs, not what it computes. Updates stay
Gauss–Seidel; this port does not replace them with Jacobi sweeps. The level
schedule exists to keep the check fast. It is not tuned, and the large size
does not by itself show that the GPU is saturated: each of the last five
levels holds 636 nodes or fewer, and the last holds one.

## Commands

```bash
make -C applications/loom_private/pagerank build
make -C applications/loom_private/pagerank check-large CHECK_JSON=/tmp/lp-pagerank-check.json
make -C applications/loom_private/pagerank test
```

`check-large` runs the oracle once, then two Direct and two Graph invocations.
Each invocation starts from poisoned outputs and zeroed device inputs (an
edgeless graph), so a skipped write, download, or upload fails the check.
`test` builds two test-only binaries and runs them with small `--set nodes=…
--set iterations=…` overrides. None of these overrides matches an upstream
smoke or validation configuration.

## Oracle and tolerance

[`oracle.cpp`](oracle.cpp) is an independent host implementation. It
regenerates the graph from the out-edge rule into per-node incoming lists
(sources pushed in ascending order) and runs the sequential in-place loop
over $u = 0..N-1$. It shares no code with `graph.cpp` or the kernels.

Every score must satisfy private Loom's per-node bound
$|a - e| \le 10^{-5}\,e$, evaluated in float, and must be finite. The check
also reports how many scores are bitwise identical to the oracle and the
largest relative error. At the large size all 1,048,576 scores are bitwise
identical and the largest relative error is 0.

## Tests (`make test`)

Host tests ([`tests/test_host.cpp`](tests/test_host.cpp)):

- The edge count formula gives 2,097,151 at $2^{20}$, and the generator
  produces it.
- Incoming lists are in ascending source order.
- The level schedule covers each node once and places it after all its
  sources $v < u$. The large schedule has 16 levels.
- A host emulation of the level schedule with double buffering matches the
  oracle bitwise at 16×3, 4096×6, and 8192×20 (nodes × iterations).
- At the same sizes a Jacobi host computation is outside the tolerance on
  most nodes. The tolerance therefore separates the two update orders.

GPU tests ([`tests/run_gpu_tests.sh`](tests/run_gpu_tests.sh), under the GPU
lock):

- The real binary passes at 16×3, 4096×6, 65536×20, and 8192×1 (Direct only).
  The result record carries the overrides and `schedule_levels`.
- **Gauss–Seidel regression.** `pagerank_jacobi_mutant` is the same source
  compiled with `-DLP_PAGERANK_MUTANT_JACOBI`, whose kernel reads every
  source from the previous sweep. It must exit 1 with mismatches reported at
  4096×6 and 65536×20. A GPU that ran Jacobi updates fails these tests.
- Invalid dimensions and usage exit 2: a non-power of two, nodes 2, 0, −8,
  or above $2^{26}$, zero iterations, a non-integer value, an unknown key,
  an unknown preset, and `--set` with bench.

## Timing scopes

- **Application:** uploads of the incoming CSR (offsets and sources), the
  out-degrees, and the level node list from pinned host memory; score and
  contribution initialization; 20 sweeps; and the score download to pinned
  host memory. The legacy pagerank harness also timed its graph upload.
- **Compute:** initialization and the 20 sweeps with the graph and schedule
  already resident.
- **Untimed host preprocessing:** graph generation, the incoming-CSR counting
  sort, level computation, copies into pinned buffers, and allocation. The
  harness also keeps CUDA Graph construction, warmups, the oracle, and checks
  out of every timed interval.

No timings are reported yet.

## Extension notes

- The level offsets are baked into the launch sequence when the work is
  enqueued or captured. A different graph needs a new setup, not just new
  data.
- A faster version could merge levels into one persistent kernel with
  grid-wide barriers, or process the small tail levels in a single block. Any
  such change must keep each node's sum in list order and its reads of
  $v < u$ from the current sweep.

## Optimized variant

`make … VARIANT=optimized` builds
[`pagerank_optimized.cu`](pagerank_optimized.cu) with
[`layout.cpp`](layout.cpp). It implements the same contract with the same
graph, oracle, check, `--set` overrides and timing scopes. It keeps ordered
Gauss–Seidel updates, and its scores are bitwise identical to the baseline
and the oracle. At the large size all 1,048,576 scores are bitwise identical
on all three GPUs, and the largest relative error is 0.

**Why.** In the baseline, `update_level` takes 99.6% of kernel time: 320
launches per invocation. Half of the 16 levels hold 2,778 nodes or fewer, so
those launches mostly pay launch latency and a few dependent memory accesses.
On the RTX 3090 the large levels also miss in L2 on most accesses (ncu: about
33% L2 hit rate).

**One persistent kernel.** Each invocation is one memset of a barrier counter
and one cooperative launch of `gauss_seidel`. The cooperative launch
guarantees that all blocks are resident, and it is captured into the CUDA
Graph like any other kernel. The kernel initializes the contributions,
passes a grid-wide barrier, and runs all sweeps, with one grid-wide barrier
between consecutive sweeps.

**Per-node readiness instead of per-level launches.** Inside a sweep there is
no barrier between levels. Every stored contribution carries a tag of its
sweep in the sign bit. Contributions are positive, so the stored word is +c
or −c and its magnitude is the exact value. Each buffer is rewritten every
second sweep, and the tag alternates between those writes (sweeps 0, 1, 4,
5, … are positive; 2, 3, 6, 7, … negative), so the stale value from two
sweeps earlier never looks ready. Updating u, a read of a source v < u
re-reads v's slot in the current buffer until it carries this sweep's tag.
A read of v ≥ u takes the previous buffer, which the barrier has completed.
The two buffers and the current/previous rule are the baseline's. Each
node's sum still runs in one thread over its incoming list in order, in
float, with the same operations, so no result changes.

**Progress.** Thread i of T visits positions i, i + T, i + 2T, … in increasing
order, which is level order. Every wait is for a source at an earlier
position, at a lower level. All blocks are resident, so the earliest
unfinished position can always complete. Lanes of one warp may wait for each
other, which relies on independent thread scheduling (sm_70 and later).

**Level-ordered layout.** At setup, untimed like the baseline's schedule,
[`layout.cpp`](layout.cpp) renumbers the nodes to positions in level order:
position p holds node `nodes[p]`, which is the baseline's level node list. The
incoming CSR offsets, the out-degrees and the contribution buffers are
indexed by position, so each sweep reads the graph arrays sequentially, and
contribution stores are coalesced. Each source entry is
`2 * position(v) + (v < u)`, which names the buffer the read uses. The
application scope uploads these four arrays (offsets N+1, sources E,
out-degrees N, nodes N), the same arrays and byte counts as the baseline's
upload. The kernel writes scores only in the last sweep, to `scores[nodes[p]]`.
Earlier sweeps' scores are overwritten without being read, as in the
sequential loop, and the contributions carry every sweep's state.

**GPU-specific choices.** These are selected at setup from the device's compute
capability ([`pagerank_optimized.cu`](pagerank_optimized.cu), `kTunings`):

| GPU | Threads × blocks per SM | Graph array loads | Reason |
|---|---|---|---|
| RTX 3090 (sm_86) | 768 × 2 (1,536 threads, full occupancy) | streaming (`ld.global.cs`, evict first) | The 6 MB L2 cannot hold the 16 MB of graph arrays read every sweep. Evict-first loads keep more of the 8 MB of contributions in L2: ncu shows 69% L2 hits and about 19 MB of DRAM reads per sweep, against about 85 MB per sweep in the baseline. |
| RTX 4090 (sm_89) | 512 × 3 (1,536 threads, full occupancy) | read-only cache (`ld.global.nc`) | Graph and contributions fit in the 72 MB L2. The streaming hint made no measurable difference, and full occupancy was the best of the shapes tried. |
| RTX 5090 (sm_120) | 512 × 2 (1,024 of 1,536 threads) | read-only cache | Everything stays in the 96 MB L2 (ncu: 97% L2 hit rate, DRAM throughput 3% of peak), so the kernel is bound by L2 latency and dependency waits. Two thirds occupancy was slightly better than full occupancy. |

The time left is per-sweep memory latency and the 16-level dependency chain:
the last levels hold few nodes, and each step of the chain is one store and
one observed load through L2. Two alternatives did not help on any of the three
GPUs. One kept two positions per thread in flight, which raised register use
and lowered occupancy. The other removed the barrier between sweeps, using
three tagged buffers and a barrier two sweeps behind. It was slower than the
per-sweep barrier, because the first level of each sweep reads the previous
sweep's last levels anyway.

**Tests** (`make test VARIANT=optimized`) run:

- The baseline's host tests.
- [`tests/test_layout.cpp`](tests/test_layout.cpp): the layout describes the
  same incoming lists in order, with the same array sizes as the baseline. Every
  current-sweep source is at an earlier position. A host computation over the
  layout matches the oracle bitwise at 16×3, 4096×6 and 8192×20.
- [`tests/run_gpu_tests.sh`](tests/run_gpu_tests.sh), unchanged, on the
  optimized binary and its Jacobi mutant. The mutant is
  `pagerank_optimized.cu` built with `-DLP_PAGERANK_MUTANT_JACOBI`, which
  reads every source from the previous sweep, and it must fail.
- [`tests/run_optimized_tests.sh`](tests/run_optimized_tests.sh) on a test-only
  build (`-DLP_PAGERANK_TEST_TUNING`) in which `LP_PAGERANK_TUNING` forces a
  tuning entry. This runs every GPU's launch shape, including the 3090's
  streamed loads, on whichever GPU runs the tests. It also runs a one-block,
  64-thread grid, in which most waits are between lanes of one warp, at 4×9,
  4096×6 and 65536×20, and a node count that caps the grid. On the large
  graph it runs 1 to 6 sweeps, which covers the tag cycle.
