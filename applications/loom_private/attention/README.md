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

`check-large` runs the CPU oracle once (about 2 s on 32 cores), then two
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
