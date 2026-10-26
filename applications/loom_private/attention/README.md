# Multisensor attention (`loom-private/multisensor-attention`)

Audio tokens attend over IMU tokens. Each audio sample (x, y, z) is projected
to a query (x, y, z, x + y). Each IMU sample is projected to a key, which is
also the value: (x, y, z, x − y). Every query row takes a softmax over its
scaled dot products with all context tokens. The outputs are the attention
result plus a residual of the query, and four statistics over the query rows.

## Large workload

| Quantity | Value |
|---|---|
| Query (audio) tokens $Q$ | 25,000 |
| Context (IMU) tokens $C$ | 250,000 |
| Sensor lanes / feature lanes / statistics | 3 / 4 / 4 |
| Input variant | 0 (scale 1.0) |
| Score pairs $Q \cdot C$ | 6.25 × 10⁹ |

Inputs come from upstream's sensor pattern, generalized to two extents:
audio token $t$ lane $l$ is `sample(3t + l, 0)` for $t < Q$, and IMU token $t$
lane $l$ is `sample(3t + l, 1)` for $t < C$, where
`sample(o, p) = ((int)((o·37 + p·101) mod 251) − 125) / 128` (32-bit unsigned
mixing; no wraparound below $3 \cdot 2^{24}$ ordinals).

For each query row $t$:

- $s_c = 0.5 \sum_l q_l k_{c,l}$, with the lane products summed in lane order
- $p_c = \exp(s_c - \max_c s_c)$
- $a_l = \sum_c p_c v_{c,l} / \sum_c p_c$
- $\text{output}[t][l] = a_l + 0.125\, q_l$

Statistics over the $Q \times 4$ elements, with $e = 4t + l$: projection
energy $\sum q^2$, attention sum $\sum a$, weighted sum $\sum (e + 1)\,a$, and
$\max |a|$. Upstream's combined checksum
$s_2 + 3 s_0 + 7 s_1 + 11 s_3$ is derived from them on the host in float32.

## Mapping to upstream

Upstream: `test/applications/loom-multisensor-attention/main.c` at private
Loom commit `4a75f8d`. Its stages map to these kernels in
[`attention.cu`](attention.cu):

| Upstream function | Kernel |
|---|---|
| `project_audio` | `project_audio_kernel` |
| `project_imu` | `project_imu_kernel` |
| `fuse_attention` | `attention_kernel` |
| `reduce_statistics` | `residual_statistics_kernel` |

Upstream uses one token count for both inputs. This workload keeps the same
generator and math but uses separate query and context extents; with equal
extents it is the upstream computation. Upstream's smoke (4 tokens, variant 0),
validation (64, variant 1), and validation-scale-eda (96, variant 2)
configurations are recorded in [`contract.json`](contract.json) for reference
only. No target runs them.

The attention kernel is adapted from the legacy streaming baseline
`loom_msa/msa_baseline.cu`, which is not modified. Each query row gets one
256-thread block, and the block makes two passes over the context. The first
pass finds the row maximum. The second computes each exponential once and
accumulates the denominator and the four weighted value lanes. No score
matrix is stored. All accumulation is float32, as in upstream. Each thread
walks its strided columns in increasing order, and the partial sums combine
through a fixed shared-memory tree. There are no atomics, so results are
bitwise identical across invocations and between Direct and Graph launches.

## Commands

```bash
make -C applications/loom_private/attention build
make -C applications/loom_private/attention check-large CHECK_JSON=/tmp/lp-attention-check.json
make -C applications/loom_private/attention test
```

`check-large` runs the CPU oracle once (multithreaded), then two
Direct and two Graph invocations. The binary also accepts
`--set query_tokens=N`, `--set context_tokens=N`, and `--set input_variant=0|1|2`
with `--action check`, for tests only. The limits are $1 \le Q \le 2^{22}$
and $1 \le C \le 2^{24}$. Values outside them exit with status 2.

`make test` ([`run_tests.sh`](run_tests.sh)) checks nine small extent sets
against the oracle: $Q \ne C$ in both directions, a single token, extents
just past the 256- and 1,024-thread strides, variants 1 and 2, and Direct-only
and Graph-only runs. It also checks nine invalid-dimension and usage cases
that must exit 2.

## Oracle and tolerance

[`oracle.cpp`](oracle.cpp) is an independent host implementation. It shares no
code with the kernels. It regenerates the inputs from the formula and
evaluates everything in double precision. For each query row it streams over
the context twice (maximum, then exponentials and sums), so memory stays
$O(Q + C)$. Rows are spread over `std::thread` workers, with each row computed
by a single thread, so the result does not depend on the thread count.

Every output element, every statistic, and the combined checksum must satisfy
upstream's bound, $|a - e| \le 10^{-4}(1 + |e|)$. NaN fails. The scores are
exact in float32 because every input lies on a 1/128 grid. The device result
therefore differs from the oracle only through `expf` rounding and float32
summation. At the large preset, the worst $|a-e|$ is about 0.5% of the
tolerance. That figure holds for the output elements, the attention sum, the
weighted sum, and the combined checksum. So the statistics keep upstream's
float32 accumulation in a fixed order, and no wider accumulator is needed
(details in `contract.json` → `tolerances`).

## Timing scopes

- **Application:** upload audio $[Q][3]$ and IMU $[C][3]$ from pinned host
  memory; both projections; attention; residual and statistics; download the
  output $[Q][4]$ and the statistics. This follows the legacy loom_msa
  convention, in which transfers are included.
- **Compute:** the same kernels with the inputs already on the device. No
  transfers.

Allocation, input generation, graph capture, warmups, the oracle, and checking
are outside both scopes. No timings are claimed here.

## Extension notes

- The attention matrix $[Q][4]$ stays on the device as an intermediate, as
  in upstream, and is covered through the residual output. Downloading it
  would change the application scope.
- The statistics kernel is one block. That is enough for $Q \cdot 4$ up to
  $2^{24}$ elements, but it would become a bottleneck for much larger $Q$.
  A fixed-order two-level reduction would keep determinism.
- Any faster attention variant (several rows per block, online softmax, or
  context splitting with a fixed-order merge) must keep a fixed reduction
  order to preserve the bitwise Direct/Graph agreement.

## Optimized variant

[`attention_optimized.cu`](attention_optimized.cu) implements the same
contract: the same inputs, oracle, check, `--set` limits, scopes, and
transfers. Build it with `VARIANT=optimized`. `make test VARIANT=optimized`
runs the baseline's cases ([`run_tests.sh`](run_tests.sh)) on it, then
[`run_tests_optimized.sh`](run_tests_optimized.sh).

```bash
make -C applications/loom_private/attention check-large VARIANT=optimized
make -C applications/loom_private/attention test VARIANT=optimized
make -C applications/loom_private/attention bench VARIANT=optimized LAUNCH=graph
```

### Why the baseline is slow

`attention_kernel` takes over 99.9% of the GPU time. Each of its 25,000
blocks reads the 4 MB key/value table from L2 twice, which comes to about
200 GB of L2 traffic per invocation. Nsight Compute shows L2 throughput at
85% of peak on the RTX 3090 and 99.7% on the RTX 5090, while SM throughput
is only 43% and 19%. The kernel is limited by L2 bandwidth.

### What the optimized variant does

- **Several query rows per block.** A block of 8 warps owns 64 query rows,
  8 per warp, and one contiguous context split. Key/value tiles of 1,024
  tokens are staged in shared memory with double-buffered `cp.async`. Each
  tile comes from L2 once per block, and each key read from shared memory
  serves 8 rows held in registers. L2 traffic falls to about 1.6 GB per
  invocation.
- **One pass instead of two.** The softmax is unchanged by its shift, which
  only has to keep the exponentials in range. So each row uses an upper
  bound of its scores in the split: $\sum_l |h_l| \max |k_l|$, rounded up to
  an integer. It needs no maximum pass. `project_imu` folds the per-split
  maxima $\max |k_l|$ in with integer `atomicMax` on the float bits, which is
  exact and does not depend on order. With this shift every $p \le 1$. If
  the bound is at most 20, the largest $p$ is at least $e^{-40}$, far from
  underflow. If any row of a block has a larger bound, that block first
  finds the exact row maxima, as the baseline does. The contract's scores
  never exceed 8 in magnitude, so this fallback is only exercised by a test
  hook.
- **Context splits** are merged in split order, each rescaled by
  $e^{\text{shift}_s - \max \text{shift}}$. The merge kernel also writes the
  residual output, using the baseline's expression. The statistics kernel is
  the baseline's, minus the residual, and keeps the same reduction order.
- **About 8.5 instructions per score pair** (Nsight Compute; see below).
  - Because $q_w = x + y$ and $k_w = a - b$, the score is
    $(x + \tfrac12 y)a - \tfrac12 x b + \tfrac12 z c$, three fused
    multiply-adds. Its weights are exact, so in the exact-maximum fallback
    pass this gives the baseline's score bit for bit.
  - The main pass does not form that exact score. It multiplies the weights
    and the shift by $\log_2 e$ (each rounded to float) and evaluates
    $t = \log_2 e\,(s - \text{shift})$ as one fused chain, so $t$ carries
    rounding error. The exponential is the hardware `ex2.approx` of $t$.
  - The weighted sums use fused multiply-adds, and the fourth value lane is
    $\sum p a - \sum p b$.
- **Application scope.** The IMU upload is chunked by context split. Each
  split's projection and attention start when its chunk lands. They run on
  two side streams that fork from the harness stream and join back to it
  through events, so the graph captures every branch. Later uploads overlap
  earlier splits' compute. The output download overlaps the statistics
  kernel. The same bytes cross PCIe as in the baseline. The compute scope
  runs the same kernels on one stream, with one attention launch over all
  splits.

Every reduction has a fixed shape: per-lane key order, an xor butterfly,
split order, and the baseline's statistics tree. The only atomics are
integer maxima. Direct and Graph outputs are therefore bitwise identical.

### Numerics

Four things now differ from the baseline:

- the shift is a bound rather than the row maximum;
- the exponent is computed from rounded $\log_2 e$-scaled weights, and the
  exponential is `ex2.approx`;
- the value sums are fused, and the fourth lane is a difference of sums;
- the summation order: each of 32 lanes sums its keys of a split in
  increasing order, the lanes combine through an xor butterfly, and the
  splits are then merged in split order with their rescaling factors. The
  baseline sums 256 strided partials per row through a binary tree, with no
  splits.

The check is unchanged. Worst |error|/tolerance at the large preset, input
variant 0:

| Quantity | Baseline | RTX 3090 | RTX 4090 | RTX 5090 |
|---|--:|--:|--:|--:|
| output elements | 0.0045 | 0.0040 | 0.0032 | 0.0032 |
| attention sum | 0.0052 | 0.0034 | 0.0009 | 0.0009 |
| weighted sum | 0.0052 | 0.0018 | 0.0039 | 0.0039 |
| max \|attention\| | 0.0006 | 0.0009 | 0.0014 | 0.0014 |
| combined checksum | 0.0045 | 0.0010 | 0.0047 | 0.0047 |

Projection energy is exact on all three, as in the baseline. The RTX 4090
and RTX 5090 columns are identical because they use the same split count.

Input variants 1 (scale 0.5) and 2 (scale 1.5) at the large preset, on the
RTX 3090:

| Quantity | Variant 1 baseline | Variant 1 optimized | Variant 2 baseline | Variant 2 optimized |
|---|--:|--:|--:|--:|
| output elements | 0.0007 | 0.0006 | 0.0106 | 0.0115 |
| attention sum | 0.0028 | 0.0127 | 0.0060 | 0.0014 |
| weighted sum | 0.0057 | 0.0095 | 0.0017 | 0.0003 |
| max \|attention\| | 0.0001 | 0.0001 | 0.0010 | 0.0027 |
| combined checksum | 0.0057 | 0.0091 | 0.0028 | 0.0008 |

The largest value in any of these runs is 0.0127, for variant 1's attention
sum, which leaves a margin of about 79x.

### GPU-specific choices

All three GPUs use the same kernel shape: 8 rows per warp, 8 warps,
1,024-key tiles, 99 registers, and two blocks per SM.

- **Shape.** Shapes with 4 warps or 512-key tiles measured within noise of
  this one or slower. So did 16 warps, 4 rows per thread, 12 or 16 rows per
  thread, and register caps for three or more blocks per SM. Offloading
  some exponentials to the FMA pipe as a polynomial was also slower.
- **Split count.** This is the only per-GPU choice, read from the compute
  capability at setup. The grid is 391 row groups × splits blocks.
  - RTX 3090 (82 SMs) uses 8 splits: 3,128 blocks over 164 resident slots,
    about 19.1 waves. More splits measured slightly slower.
  - RTX 4090 (128 SMs) and RTX 5090 (170 SMs) use 12 splits: 4,692 blocks
    over 256 and 340 resident slots, about 18.3 and 13.8 waves. On the
    RTX 5090, 8 splits (about 9.2 waves) measured slower.

### What limits it now

Nsight Compute on the RTX 3090 and RTX 5090:

- the kernel issues about 8.5 instructions per score pair;
- issue slots are 72 to 74% busy;
- the special-function (MUFU) pipe that runs `ex2` is 68% busy;
- the FMA pipe is 60 to 68% busy;
- L2 throughput is 15 to 17% of peak.

The kernel is bound by instruction issue and the exponential unit together.
The main stall reasons are MIO throttle (MUFU and shared loads share the MIO
queue), dispatch stalls, and waits on MUFU results. Each score pair needs one
exponential. The MUFU pipe completes one warp-wide `ex2` per eight cycles per
scheduler, so at 8.5 instructions per pair the issue and MUFU limits are
close together.

### Tests

[`run_tests_optimized.sh`](run_tests_optimized.sh) uses a hooks build,
`attention_optimized_hooks`, whose environment variables force:

- the kernel shape (`LP_ATTENTION_SHAPE`);
- the split count (`LP_ATTENTION_SPLITS`);
- the exact-maximum fallback (`LP_ATTENTION_EXACT_MAXIMUM`);
- running the compute path inside the checked invocation
  (`LP_ATTENTION_CHECK_COMPUTE`).

Hook values outside their ranges exit 2. The setup line on stderr names the
shape, split count, and forced paths, and the tests check it so that every
hook is known to have applied.

It checks every compiled shape with 1, 3, and many splits (partial tiles,
chunks, and row groups). It also checks the fallback in both launch paths,
the compute path, and the largest query extent, where the cap on partial
buffers allows two splits. Invalid hook values must exit 2. A mutant that
skips the last key of every tile must fail the check with exit 1.
