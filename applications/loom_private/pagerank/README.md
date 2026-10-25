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
