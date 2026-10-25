# llama2c decoder (`loom-private/llama2c-decoder`)

Greedy decoding through llama2.c's last-layer kernels (`rmsnorm`, `matmul`,
`softmax`, `sample_argmax`) on a synthetic model. The workload decodes 1,024
independent sequences, each for 4 dependent steps.

## What it computes

The synthetic model works as follows. The embedding row for token $t$ is 1.0
everywhere except 4.0 at column $t$. The RMSNorm weight is 1.0. Classifier
row $r$ is 0 except for a 1.0 at column $(r + V - 1) \bmod V$. Sequence $b$
starts from token $b \bmod V$. Each step does the following for every
sequence:

1. **rmsnorm** of the current token's embedding row, computed as llama2.c
   does it: $ss = \sum x^2$, then $ss/W + 10^{-5}$, then
   $\text{scale} = 1/\sqrt{ss}$, then $o_j = w_j(\text{scale}\cdot x_j)$.
2. **Dense classifier matmul.** All $V$ rows are multiplied against all $W$
   columns, giving $V$ logits.
3. **softmax**: subtract the maximum, apply `expf`, then divide by the sum.
4. **argmax**: take the first maximum. That index is the next token, and it
   is written back on the GPU for the next step.

For this model the token advances by one each step, modulo $V$. The GPU
nevertheless does the full dense computation at every step and never uses
that shortcut.

| | Large |
|---|---|
| batch $B$ | 1,024 |
| width $W$ | 16,384 |
| vocabulary $V$ (classifier rows) | 8,192 |
| steps | 4 (4,096 generated tokens) |
| model | 1 GiB of float32 (embedding + classifier), resident |
| classifier work | $2^{37}$ multiply-adds per step |

## Mapping to upstream

Upstream is private Loom `4a75f8d`, file `test/applications/llama2c-kernels-smoke.h`.
It wraps llama2.c `run.c` at `350e04fe`. The blob and SHA-256 hashes are in
[`contract.json`](contract.json). Upstream decodes a single sequence. This
port runs a batch of independent sequences, which makes it an extension. The
model, the per-sequence arithmetic, the token check, and the invariant
constants all follow upstream.

Upstream copies the embedding row into an activation buffer before calling
`rmsnorm`. Here the kernel reads the embedding row directly, which gives the
same values. The classifier kernel is a tiled batch matmul. Each output is
accumulated by a single thread, in increasing column order, starting from
`0.0f`, with a separate multiply and add. That is the same order llama2.c's
`matmul` uses.

The rmsnorm square sum and the softmax reductions use fixed tree orders, so
results are deterministic. For this model the square sum is $W + 15$, and the
binary limits $W \le 2^{24} - 16$, so every partial sum is an exact float
integer in any order. The upstream smoke configuration (width 128, 16 rows)
and validation configuration (width 1024, 64 rows) are recorded in the
contract as references only. No target runs them.

## Commands

```bash
make -C applications/loom_private/decoder build
make -C applications/loom_private/decoder check-large   # CHECK_JSON=path for the record
make -C applications/loom_private/decoder test
```

`test` runs [`tests/run_tests.sh`](tests/run_tests.sh) under one GPU lock. It
covers five correct shapes given with `--set batch=… width=… vocab=…
steps=…`. These include prompt and token wraparound, the minimum shape
$V = W = 2$, $V = W$, ragged tiles, and a Graph-only run. It also covers ten
invalid inputs that must exit 2, and three injected faults that must exit 1
and report the offending index with its expected and actual values. The fault
hook is `--set fault=1|2|3`: it corrupts one token, one activation, or one
probability on the host after download, in check mode only.

## Oracle and tolerances

The CPU oracle lives in [`oracle.cpp`](oracle.cpp), is host only, and shares
no code with the kernels. It checks four things:

- **Tokens.** Every generated token must exactly match the analytic
  progression $t \mapsto (t+1) \bmod V$.
- **Normalization invariants**, for every sequence at every step. Upstream
  checks only the final step. The mean of squares of the normalized
  activation must be within $2\times10^{-5} + 1.2\times10^{-7}W$ of 1. The
  probability sum must be within $10^{-6} + 1.2\times10^{-7}V$ of 1. Both sums
  are taken in double. A NaN fails.
- **Dense double evaluation.** For 8 sequences spread across the batch
  (including the first and last), the oracle evaluates every step densely in
  double: every classifier row against every column. GPU normalized values
  and logits must match within $2\times10^{-6}$ relative, and probabilities
  within $2\times10^{-5}$ relative. The contract gives the error budget. At the
  large shape, the worst observed relative errors are about $3\times10^{-8}$
  for normalized values and logits, and $2\times10^{-7}$ for probabilities.
- **Oracle self-check.** The oracle's dense argmax must reproduce the analytic
  progression.

## Timing scopes

- **Application scope:** prompt upload (pinned), device token reset, all
  steps, and the generated-token download (pinned).
- **Compute scope:** the token reset from the resident prompts, plus all
  steps.

Setup generates the weights on the device. Setup is untimed, and the weights
and working buffers stay resident. The per-step normalized, logit, and
probability buffers are check diagnostics. They are copied only in
`check_outputs`, which is untimed, and they are poisoned before every
invocation like every other output. No performance numbers are claimed.

## Extension notes

- The per-step diagnostic buffers take $4 \cdot \text{steps}\cdot B(W+2V)$
  bytes on both the device and the host (512 MiB at the large shape). A
  timing-only build could keep just the last step's buffers.
- The classifier kernel is a plain 64×64 shared-memory tile, with 4×4
  outputs per thread and no FMA. Faster variants must keep the
  per-output accumulation order, or change the contract.
- Real llama2.c weights would make the rmsnorm tree reduction differ from
  the sequential sum by rounding. The bitwise argument above holds only for
  this synthetic model.
