// SOTA CUDA variant for the RTX 3090 (sm_86): one fused attention kernel.
//
// Kernels per iteration, in launch order:
//   msa_sota_attention_kernel  x SOTA_CHUNKS (13 large, 2 medium, 1 4x4):
//                              projections and split-context attention for
//                              one chunk of the context and, in the last
//                              block of each query block, the split merge,
//                              output rows and statistics.
//   msa_sota_merge_kernel      only where SOTA_FUSED_MERGE=0 (the medium
//                              shape, whose many splits merge faster with a
//                              warp per row): merge, output and statistics.
// Copies: audio H2D, then the imu H2D in SOTA_CHUNKS pieces back to back
// (piece 0 on the audio stream, the rest on the imu stream); attention
// launch c runs as soon as imu piece c has landed (launch 0 on the audio
// stream, the last on the imu stream, the others on forked streams), so it
// overlaps the later pieces' copies. Output
// and statistics are written by the kernels straight into the pinned host
// buffers (zero-copy through UVA), so there is no D2H copy node.
//
// Stage timings (metrics line): attention_us runs from the end of the audio
// copy to the end of the last attention launch, so it includes the imu copy
// time that the first launches wait for; through_statistics_us adds the
// merge kernel where there is one (otherwise the two are the same point).
// With SOTA_CHUNKS > 1 an ncu run sees each attention launch alone on the
// GPU; build with -DSOTA_CHUNKS=1 to profile the attention as one kernel.
//
// Algebra (exact): with q = (a, b, c, a+b) and k = v = (x, y, z, x-y),
//   q.k = a x + b y + c z + (a+b)(x-y) = (2a+b) x - a y + c z,
// so each score is a 3-term dot of the raw imu row with the per-query vector
// q' = (2a+b, -a, c) * 0.5*log2(e) (log2-domain score), and lane 3 of the
// attention is lane 0 minus lane 1. The projections are never materialized:
// keys come straight from the raw [C,3] imu rows, and only three value lanes
// plus the denominator are accumulated. Per (query, key) pair the inner loop
// is 3 FFMA (score, with -m folded in as the addend), one MUFU.EX2, one FADD
// (denominator) and 3 FFMA (values): 8 issue slots and 1 SFU op.
//
// Softmax reference ("running max"), maintained lazily as in
// FlashAttention-4's conditional rescaling, with the rescale decision taken
// from each tile's exp-sum instead of a per-pair max:
//   * m starts as the exact score of the split's first key, and only ever
//     takes exact attained scores, so m <= (true running max) and the
//     denominator is >= 1: nothing that matters can underflow (terms flushed
//     by ex2.approx.ftz are < 2^-126 while the sum is >= 1).
//   * each tile is accumulated into tile-local sums with the current m. If a
//     query's tile sum exceeds 2^40 (or is inf/NaN), some exp2 argument was
//     > 40: the tile-local sums are discarded, m is raised to the tile's exact
//     max (an extra pass over the tile), the running sums are rescaled and
//     the tile is recomputed for that query. Otherwise every argument was
//     <= 40, so all terms are <= 2^40 and the running sums stay far below
//     the float range (<= C * 2^40).
// The final normalization uses the actual sums, so the result is the exact
// softmax (up to float rounding) for any finite input; only how often the
// rescan runs depends on the data. With inputs in [-1, 1) every |score| is
// <= 3.6 (log2 units), so it never runs. -DSOTA_EXACT_MAX=1 instead takes the
// exact max of every tile before accumulating it (classic online softmax),
// for comparison.
//
// Precision: fp32 attention with MUFU.EX2 (relative error ~2^-22). Sums are
// two-level (tile-local, then running), so rounding grows with sqrt(tile)
// + sqrt(tiles), not sqrt(keys per split). Per-pair rounding averages out
// over the context; what does not is a per-query error, which shifts a whole
// row like a perturbed query, so the score coefficients follow the
// reference's float projection and are rounded once (from exact hi/lo
// sums). The weighted-sum statistic is accumulated in double. Measured on
// 3 x 25000 rows (large, random): RMS error of the attention 1.95e-8
// (msa_optimized 1.7e-8; float rounding of the values alone is ~1.2e-8),
// max 2e-7. -DSOTA_POLY_QUERIES=n computes the
// exp2 of n of each thread's queries on the FMA pipe instead (FA4-style
// degree-5 polynomial, relative error 1e-7); it is slower here, because the
// kernel is issue-bound rather than SFU-bound, and is off by default.
//
// Parallelization: each thread owns SOTA_QPT queries in registers, and every
// thread of a block reads the same key from a shared-memory tile (broadcast:
// three LDS.128 bring four keys to all of a warp's queries). The context is
// split into SOTA_SPLITS ranges (flash-decoding / split-K); each
// (query block, split) block writes an unnormalized (A, L, m) partial. Merges
// combine partials in split order and the statistics are reduced in a fixed
// block order (atomic arrival counters, which reset themselves), so every
// result is deterministic.
#include <cuda_pipeline.h>

#ifndef TOKEN_COUNT
#define TOKEN_COUNT 25000
#endif
#ifndef CONTEXT_COUNT
#define CONTEXT_COUNT 250000
#endif
#ifndef GRAPH_REPLAYS
#define GRAPH_REPLAYS 50
#endif

#include "msa_harness.cuh"

// Tuning, per shape (each overridable with -D).
#if TOKEN_COUNT >= 10000
// 25 query blocks x 13 splits = 325 blocks of 128 threads: one wave at four
// blocks (128 registers per thread) per SM.
#  define SOTA_DEFAULT_QPT 8
#  define SOTA_DEFAULT_QPB 1000
#  define SOTA_DEFAULT_SPLITS 13
#  define SOTA_DEFAULT_FUSED_MERGE 1
#  define SOTA_DEFAULT_MIN_BLOCKS 4
#  define SOTA_DEFAULT_GROUP_UNROLL 4
#  define SOTA_DEFAULT_ZERO_COPY_IMU 0
#  define SOTA_DEFAULT_CHUNKS 13
#elif TOKEN_COUNT >= 512
#  define SOTA_DEFAULT_QPT 4
#  define SOTA_DEFAULT_QPB 256
#  define SOTA_DEFAULT_SPLITS 82
#  define SOTA_DEFAULT_FUSED_MERGE 0
#  define SOTA_DEFAULT_MIN_BLOCKS 1
#  define SOTA_DEFAULT_GROUP_UNROLL 2
#  define SOTA_DEFAULT_ZERO_COPY_IMU 0
#  define SOTA_DEFAULT_CHUNKS 2
#else
#  define SOTA_DEFAULT_QPT 4
#  define SOTA_DEFAULT_QPB 128
#  define SOTA_DEFAULT_SPLITS 1
#  define SOTA_DEFAULT_FUSED_MERGE 1
#  define SOTA_DEFAULT_MIN_BLOCKS 1
#  define SOTA_DEFAULT_GROUP_UNROLL 2
#  define SOTA_DEFAULT_ZERO_COPY_IMU 0
#  define SOTA_DEFAULT_CHUNKS 1
#endif
#ifndef SOTA_QPT  // queries per thread
#define SOTA_QPT SOTA_DEFAULT_QPT
#endif
#ifndef SOTA_QPB  // queries per block
#define SOTA_QPB SOTA_DEFAULT_QPB
#endif
#ifndef SOTA_SPLITS  // context splits
#define SOTA_SPLITS SOTA_DEFAULT_SPLITS
#endif
#ifndef SOTA_TILE  // keys per shared-memory tile (multiple of 4)
#define SOTA_TILE 256
#endif
#ifndef SOTA_GROUP_UNROLL  // 4-key groups per unrolled inner-loop iteration
#define SOTA_GROUP_UNROLL SOTA_DEFAULT_GROUP_UNROLL
#endif
#ifndef SOTA_MIN_BLOCKS  // __launch_bounds__ minimum resident blocks per SM
#define SOTA_MIN_BLOCKS SOTA_DEFAULT_MIN_BLOCKS
#endif
#ifndef SOTA_FUSED_MERGE  // last attention block merges (else a merge kernel)
#define SOTA_FUSED_MERGE SOTA_DEFAULT_FUSED_MERGE
#endif
#ifndef SOTA_ZERO_COPY_IMU  // tiles read straight from pinned host memory
#define SOTA_ZERO_COPY_IMU SOTA_DEFAULT_ZERO_COPY_IMU
#endif
#ifndef SOTA_ZERO_COPY_OUTPUT  // output/statistics written to host memory
#define SOTA_ZERO_COPY_OUTPUT 1
#endif
#ifndef SOTA_CHUNKS  // imu copy chunks, each with its own attention launch
#define SOTA_CHUNKS SOTA_DEFAULT_CHUNKS
#endif
#ifndef SOTA_EXACT_MAX
#define SOTA_EXACT_MAX 0
#endif
#ifndef SOTA_POLY_QUERIES
#define SOTA_POLY_QUERIES 0
#endif
#ifndef SOTA_DEFAULT_ROTATION
#define SOTA_DEFAULT_ROTATION 3
#endif

constexpr int kQpt = SOTA_QPT;
constexpr int kQpb = SOTA_QPB;
constexpr int kSplits = SOTA_SPLITS;
constexpr int kChunks = SOTA_CHUNKS;
constexpr int kTile = SOTA_TILE;
constexpr int kGroupUnroll = SOTA_GROUP_UNROLL;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kThreads = ((kQpb / kQpt) + kWarp - 1) / kWarp * kWarp;
[[maybe_unused]] constexpr int kWarps = kThreads / kWarp;
constexpr int kQueryBlocks = (TOKEN_COUNT + kQpb - 1) / kQpb;
// Keys are split on 4-row boundaries when possible, so every tile starts on
// a 16-byte boundary of the raw [C,3] float rows (4 rows = 48 bytes).
constexpr bool kVectorTiles = CONTEXT_COUNT % 4 == 0;
constexpr int kSplitUnit = kVectorTiles ? 4 : 1;
constexpr int kSplitUnits = CONTEXT_COUNT / kSplitUnit;
constexpr int kTileFloats = (kTile * SENSOR_LANE_COUNT + 3) / 4 * 4;
constexpr float kScoreScaleLog2 = 0.5f * 1.4426950408889634f;
constexpr float kTileSumLimit = 1099511627776.0f;  // 2^40
constexpr int kMergeThreads = 256;
constexpr int kMergeRowsPerBlock = kMergeThreads / kWarp;
constexpr int kMergeBlocks =
    (TOKEN_COUNT + kMergeRowsPerBlock - 1) / kMergeRowsPerBlock;
constexpr int kStatisticBlocks =
    SOTA_FUSED_MERGE ? kQueryBlocks : kMergeBlocks;

static_assert(kQpb % kQpt == 0,
              "queries per block must be a multiple of queries per thread");
static_assert(kTile % 4 == 0, "tile rows must be a multiple of 4");
static_assert(kChunks >= 1 && kChunks <= kSplits,
              "every chunk needs at least one split");
static_assert(kSplits >= 1 && kSplits <= kSplitUnits,
              "every split needs at least one key");

__host__ __device__ constexpr int split_begin(int split) {
  return static_cast<int>(static_cast<long long>(split) * kSplitUnits /
                          kSplits) *
         kSplitUnit;
}

// Partials: [split][batch * TOKEN_COUNT + token].
__device__ float4 g_partial_acc[kSplits * BATCH_COUNT * TOKEN_COUNT];  // A, L
__device__ float g_partial_max[kSplits * BATCH_COUNT * TOKEN_COUNT];
__device__ unsigned int g_split_arrivals[BATCH_COUNT * kQueryBlocks];
__device__ unsigned int g_block_arrivals[BATCH_COUNT];
#if !SOTA_ZERO_COPY_OUTPUT
__device__ float4 g_output[BATCH_COUNT * TOKEN_COUNT];
__device__ float g_statistics[BATCH_COUNT * STATISTIC_COUNT];
#endif

// 2^x on the SFU. Subnormal results flush to 0; they are < 2^-126 while
// every denominator is >= 1.
__device__ __forceinline__ float ex2(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

// 2^x on the FMA pipe (FlashAttention-4 style), x <= 64: Cody-Waite split
// x = j + f with j = round(x), |f| <= 1/2 (exact, by the 1.5*2^23 trick and
// Sterbenz), 2^f by a degree-5 polynomial (relative error 1.0e-7 on
// [-1/2, 1/2], about that of MUFU.EX2), and 2^j added into the exponent
// field. x is clamped at -126; below that the result is < 2^-126, as
// negligible as ex2.approx.ftz's 0.
__device__ __forceinline__ float ex2_poly(float x) {
  constexpr float kRound = 12582912.0f;  // 1.5 * 2^23
  x = fmaxf(x, -126.0f);
  const float r = __fadd_rn(x, kRound);
  const float f = __fsub_rn(x, __fsub_rn(r, kRound));
  float p = fmaf(0.0013292919611558318f, f, 0.009671509265899658f);
  p = fmaf(p, f, 0.05550636723637581f);
  p = fmaf(p, f, 0.24022242426872253f);
  p = fmaf(p, f, 0.6931470632553101f);
  p = fmaf(p, f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(r) << 23));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(kFullMask, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
  return value;
}

// (m, l, A) combination: m = max(m_a, m_b), sums rescaled by 2^(m_x - m).
__device__ __forceinline__ void merge_state(float om, float ol, float ox,
                                            float oy, float oz, float& m,
                                            float& l, float& ax, float& ay,
                                            float& az) {
  const float maximum = fmaxf(m, om);
  if (maximum == -INFINITY)
    return;  // Both empty.
  const float scale = ex2(m - maximum);
  const float other_scale = ex2(om - maximum);
  l = l * scale + ol * other_scale;
  ax = ax * scale + ox * other_scale;
  ay = ay * scale + oy * other_scale;
  az = az * scale + oz * other_scale;
  m = maximum;
}

__device__ __forceinline__ void merge_partial(int index, float& m, float& l,
                                              float& ax, float& ay,
                                              float& az) {
  const float4 acc = __ldcg(&g_partial_acc[index]);
  merge_state(__ldcg(&g_partial_max[index]), acc.w, acc.x, acc.y, acc.z, m, l,
              ax, ay, az);
}

// The weighted sum is accumulated in double (4 FMAs per row; FP64 is slow on
// GA102): its weights go up to 4T, so float partial sums of it lose ~1e-5
// of its value to rounding, enough to matter when its terms cancel. The
// other statistics are float, in a fixed reduction order.
struct RowStatistics {
  float projection_energy, attention_sum;
  double weighted_sum;
  float maximum_magnitude;
};

__device__ __forceinline__ double warp_sum(double value) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(kFullMask, value, offset);
  return value;
}

// Normalizes one merged row, writes out = a + p/8 and adds the row to the
// statistics (the double products (i+1) * a are exact).
__device__ __forceinline__ void finish_row(int token, float l, float ax,
                                           float ay, float az,
                                           const float* batch_audio,
                                           float4* batch_output,
                                           RowStatistics& st) {
  const float inverse = 1.0f / l;
  const float a[4] = {ax * inverse, ay * inverse, az * inverse,
                      ax * inverse - ay * inverse};
  const float x = batch_audio[token * SENSOR_LANE_COUNT + 0];
  const float y = batch_audio[token * SENSOR_LANE_COUNT + 1];
  const float z = batch_audio[token * SENSOR_LANE_COUNT + 2];
  const float p[4] = {x, y, z, x + y};
  float out[4];
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    st.projection_energy = fmaf(p[lane], p[lane], st.projection_energy);
    st.attention_sum += a[lane];
    st.weighted_sum = fma(static_cast<double>(token * 4 + lane + 1),
                          static_cast<double>(a[lane]), st.weighted_sum);
    st.maximum_magnitude = fmaxf(st.maximum_magnitude, fabsf(a[lane]));
    out[lane] = fmaf(p[lane], 0.125f, a[lane]);
  }
  batch_output[token] = make_float4(out[0], out[1], out[2], out[3]);
}

__device__ __forceinline__ RowStatistics warp_reduce(RowStatistics st) {
  st.projection_energy = warp_sum(st.projection_energy);
  st.attention_sum = warp_sum(st.attention_sum);
  st.weighted_sum = warp_sum(st.weighted_sum);
  st.maximum_magnitude = warp_max(st.maximum_magnitude);
  return st;
}

__device__ RowStatistics g_block_statistics[BATCH_COUNT * kStatisticBlocks];

// Block statistics in a fixed order (warp butterfly, then warp 0 over the
// warps), stored per block; the last block of the batch (arrival counter)
// reduces the blocks in index order and writes the batch's statistics.
// kLaneZero: only lane 0 of each warp holds statistics (skip the warp step).
template <int kBlockThreads, bool kLaneZero = false>
__device__ __forceinline__ void reduce_statistics(
    RowStatistics st, int batch, int block, int blocks, float* statistics,
    RowStatistics* warp_statistics, bool* last_block) {
  constexpr int kBlockWarps = kBlockThreads / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (!kLaneZero)
    st = warp_reduce(st);
  if (lane == 0)
    warp_statistics[warp] = st;
  __syncthreads();
  if (warp != 0)
    return;
  RowStatistics total{};
  if (lane < kBlockWarps)
    total = warp_statistics[lane];
  total = warp_reduce(total);
  if (lane == 0) {
    g_block_statistics[batch * kStatisticBlocks + block] = total;
    __threadfence();
    *last_block = atomicAdd(&g_block_arrivals[batch], 1u) == blocks - 1;
  }
  __syncwarp();
  if (!__shfl_sync(kFullMask, *last_block ? 1 : 0, 0))
    return;
  __threadfence();
  // Lane i takes blocks i, i+32, ... in order, then a fixed butterfly.
  RowStatistics sum{};
  for (int b = lane; b < blocks; b += kWarp) {
    const RowStatistics* v = &g_block_statistics[batch * kStatisticBlocks + b];
    sum.projection_energy += __ldcg(&v->projection_energy);
    sum.attention_sum += __ldcg(&v->attention_sum);
    sum.weighted_sum += __ldcg(&v->weighted_sum);
    sum.maximum_magnitude =
        fmaxf(sum.maximum_magnitude, __ldcg(&v->maximum_magnitude));
  }
  sum = warp_reduce(sum);
  if (lane == 0) {
    g_block_arrivals[batch] = 0;
    float* out = statistics + batch * STATISTIC_COUNT;
    out[0] = static_cast<float>(sum.projection_energy);
    out[1] = static_cast<float>(sum.attention_sum);
    out[2] = static_cast<float>(sum.weighted_sum);
    out[3] = sum.maximum_magnitude;
  }
}

// Knuth's TwoSum: sum = fl(a + b); returns the exact error a + b - sum.
__device__ __forceinline__ float two_sum(float a, float b, float& sum) {
  sum = a + b;
  const float bp = sum - a;
  const float ap = sum - bp;
  return (a - ap) + (b - bp);
}

// Per query: score coefficients, -(reference), running and tile-local sums.
struct QueryState {
  float qx, qy, qz;
  float negm;
  float l, ax, ay, az;
};

// Exact max of one query's scores over a tile (rare path).
__device__ __noinline__ float tile_max(const float* tile, int rows, float qx,
                                       float qy, float qz) {
  float best = -INFINITY;
  for (int r = 0; r < rows; ++r)
    best = fmaxf(best, fmaf(qx, tile[3 * r],
                            fmaf(qy, tile[3 * r + 1], qz * tile[3 * r + 2])));
  return best;
}

// One query's tile sums with reference -negm (rare path).
__device__ __noinline__ float4 tile_sums(const float* tile, int rows,
                                         float qx, float qy, float qz,
                                         float negm, float* l) {
  float sl = 0.0f, sx = 0.0f, sy = 0.0f, sz = 0.0f;
  for (int r = 0; r < rows; ++r) {
    const float x = tile[3 * r], y = tile[3 * r + 1], z = tile[3 * r + 2];
    const float p = ex2(fmaf(qx, x, fmaf(qy, y, fmaf(qz, z, negm))));
    sl += p;
    sx = fmaf(p, x, sx);
    sy = fmaf(p, y, sy);
    sz = fmaf(p, z, sz);
  }
  *l = sl;
  return make_float4(sx, sy, sz, 0.0f);
}

// Raises query q's reference to max(m, the tile's exact max) and rescales
// its running sums.
__device__ __forceinline__ void raise_reference(QueryState& st,
                                                const float* tile, int rows) {
  const float maximum = tile_max(tile, rows, st.qx, st.qy, st.qz);
  if (maximum > -st.negm) {
    const float scale = ex2(-st.negm - maximum);
    st.l *= scale;
    st.ax *= scale;
    st.ay *= scale;
    st.az *= scale;
    st.negm = -maximum;
  }
}

// Accumulates one tile for all of a thread's queries into tile-local sums
// (the hot loop), then checks each query's tile sum and adds it to the
// running sums (see the file comment).
template <int kRows>
__device__ __forceinline__ void accumulate_tile(QueryState (&st)[kQpt],
                                                const float* tile, int rows) {
  float l[kQpt], ax[kQpt], ay[kQpt], az[kQpt];
#pragma unroll
  for (int q = 0; q < kQpt; ++q)
    l[q] = ax[q] = ay[q] = az[q] = 0.0f;
  const float4* tile4 = reinterpret_cast<const float4*>(tile);
  const int groups = kRows > 0 ? kRows / 4 : rows / 4;
#pragma unroll kGroupUnroll
  for (int g = 0; g < groups; ++g) {
    const float4 f0 = tile4[3 * g + 0];
    const float4 f1 = tile4[3 * g + 1];
    const float4 f2 = tile4[3 * g + 2];
    const float kx[4] = {f0.x, f0.w, f1.z, f2.y};
    const float ky[4] = {f0.y, f1.x, f1.w, f2.z};
    const float kz[4] = {f0.z, f1.y, f2.x, f2.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int q = 0; q < kQpt; ++q) {
        const float t = fmaf(st[q].qx, kx[j],
                             fmaf(st[q].qy, ky[j],
                                  fmaf(st[q].qz, kz[j], st[q].negm)));
        const float p = q < SOTA_POLY_QUERIES ? ex2_poly(t) : ex2(t);
        l[q] += p;
        ax[q] = fmaf(p, kx[j], ax[q]);
        ay[q] = fmaf(p, ky[j], ay[q]);
        az[q] = fmaf(p, kz[j], az[q]);
      }
    }
  }
  if (kRows <= 0) {
    for (int r = groups * 4; r < rows; ++r) {
      const float x = tile[3 * r], y = tile[3 * r + 1], z = tile[3 * r + 2];
#pragma unroll
      for (int q = 0; q < kQpt; ++q) {
        const float p = ex2(fmaf(st[q].qx, x,
                                 fmaf(st[q].qy, y,
                                      fmaf(st[q].qz, z, st[q].negm))));
        l[q] += p;
        ax[q] = fmaf(p, x, ax[q]);
        ay[q] = fmaf(p, y, ay[q]);
        az[q] = fmaf(p, z, az[q]);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < kQpt; ++q) {
    if (!(l[q] <= kTileSumLimit)) {  // Some argument > 40 (or inf/NaN).
      raise_reference(st[q], tile, rows);
      const float4 s = tile_sums(tile, rows, st[q].qx, st[q].qy, st[q].qz,
                                 st[q].negm, &l[q]);
      ax[q] = s.x;
      ay[q] = s.y;
      az[q] = s.z;
    }
    st[q].l += l[q];
    st[q].ax += ax[q];
    st[q].ay += ay[q];
    st[q].az += az[q];
  }
}

__global__ void __launch_bounds__(kThreads, SOTA_MIN_BLOCKS)
msa_sota_attention_kernel(const float* __restrict__ audio,
                          const float* __restrict__ imu,
                          float4* __restrict__ output,
                          float* __restrict__ statistics, int first_split) {
  __shared__ __align__(16) float key_tile[2][kTileFloats];
#if SOTA_FUSED_MERGE
  __shared__ RowStatistics warp_statistics[kWarps];
  __shared__ bool last_block;
#endif

  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int query_block = blockIdx.x % kQueryBlocks;
  const int batch = blockIdx.x / kQueryBlocks;
  const int split = first_split + blockIdx.y;
  const int block_query = query_block * kQpb;
  const int block_end = min(TOKEN_COUNT, block_query + kQpb);
  const int first_query = block_query + tid * kQpt;
  const float* batch_audio =
      audio + static_cast<size_t>(batch) * TOKEN_COUNT * SENSOR_LANE_COUNT;
  const float* batch_imu =
      imu + static_cast<size_t>(batch) * CONTEXT_COUNT * SENSOR_LANE_COUNT;
  // Warps without a single valid query only help load tiles.
  const bool warp_active = warp * kWarp * kQpt < kQpb &&
                           block_query + warp * kWarp * kQpt < block_end;

  QueryState st[kQpt];
#pragma unroll
  for (int q = 0; q < kQpt; ++q) {
    const int token = first_query + q;
    float a = 0.0f, b = 0.0f, c = 0.0f;
    if (token < block_end && tid * kQpt + q < kQpb) {
      a = batch_audio[token * SENSOR_LANE_COUNT + 0];
      b = batch_audio[token * SENSOR_LANE_COUNT + 1];
      c = batch_audio[token * SENSOR_LANE_COUNT + 2];
    }
    // The reference's projected lane is the float q3 = fl(a+b); with it the
    // score is (a + q3) x + (b - q3) y + c z (+ q3 times the key's rounding
    // of x-y, which only adds per-key noise). Each coefficient is rounded
    // once (the sums are kept exact as hi + lo pairs): a per-query
    // coefficient error biases a whole row (it acts like a perturbed query),
    // unlike per-pair rounding, which averages out over the context.
    const float q3 = a + b;
    float hi;
    float lo = two_sum(a, q3, hi);
    st[q].qx = fmaf(hi, kScoreScaleLog2, lo * kScoreScaleLog2);
    lo = two_sum(b, -q3, hi);
    st[q].qy = fmaf(hi, kScoreScaleLog2, lo * kScoreScaleLog2);
    st[q].qz = c * kScoreScaleLog2;
    st[q].l = st[q].ax = st[q].ay = st[q].az = 0.0f;
  }

  const int key_begin = split_begin(split);
  const int key_end = split_begin(split + 1);
  const int tiles = (key_end - key_begin + kTile - 1) / kTile;

  auto load_tile = [&](int tile, int buffer) {
    const int begin = key_begin + tile * kTile;
    const int rows = min(kTile, key_end - begin);
    const float* src =
        batch_imu + static_cast<size_t>(begin) * SENSOR_LANE_COUNT;
    if constexpr (kVectorTiles) {
      const int chunks = rows * SENSOR_LANE_COUNT / 4;
      for (int i = tid; i < chunks; i += kThreads)
        __pipeline_memcpy_async(&key_tile[buffer][4 * i], src + 4 * i, 16);
    } else {
      const int words = rows * SENSOR_LANE_COUNT;
      for (int i = tid; i < words; i += kThreads)
        __pipeline_memcpy_async(&key_tile[buffer][i], src + i, 4);
    }
    __pipeline_commit();
  };

  load_tile(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    const int buffer = tile & 1;
    const int rows = min(kTile, key_end - (key_begin + tile * kTile));
    if (tile + 1 < tiles) {
      load_tile(tile + 1, buffer ^ 1);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const float* keys = key_tile[buffer];
    if (warp_active) {
      if (tile == 0) {
        // Initial reference: the exact score of the split's first key.
#pragma unroll
        for (int q = 0; q < kQpt; ++q)
          st[q].negm = -fmaf(st[q].qx, keys[0],
                             fmaf(st[q].qy, keys[1], st[q].qz * keys[2]));
      }
#if SOTA_EXACT_MAX
#pragma unroll
      for (int q = 0; q < kQpt; ++q)
        raise_reference(st[q], keys, rows);
#endif
      if (kVectorTiles && rows == kTile)
        accumulate_tile<kTile>(st, keys, rows);
      else
        accumulate_tile<0>(st, keys, rows);
    }
    __syncthreads();  // Buffer consumed before it is refilled.
  }

  // Partial for this split.
#pragma unroll
  for (int q = 0; q < kQpt; ++q) {
    const int token = first_query + q;
    if (token < block_end && tid * kQpt + q < kQpb) {
      const int index =
          split * BATCH_COUNT * TOKEN_COUNT + batch * TOKEN_COUNT + token;
      g_partial_acc[index] =
          make_float4(st[q].ax, st[q].ay, st[q].az, st[q].l);
      g_partial_max[index] = -st[q].negm;
    }
  }

#if SOTA_FUSED_MERGE
  // The last block of this query block merges its splits.
  __threadfence();
  __syncthreads();
  const int counter = batch * kQueryBlocks + query_block;
  if (tid == 0)
    last_block = atomicAdd(&g_split_arrivals[counter], 1u) == kSplits - 1;
  __syncthreads();
  if (!last_block)
    return;
  __threadfence();
  if (tid == 0)
    g_split_arrivals[counter] = 0;

  RowStatistics row_statistics{};
  for (int j = 0; j < kQpt; ++j) {
    const int local = tid + j * kThreads;
    const int token = block_query + local;
    if (local >= kQpb || token >= block_end)
      continue;
    const int row = batch * TOKEN_COUNT + token;
    float m = -INFINITY, l = 0.0f, ax = 0.0f, ay = 0.0f, az = 0.0f;
#pragma unroll 4
    for (int s = 0; s < kSplits; ++s)
      merge_partial(s * BATCH_COUNT * TOKEN_COUNT + row, m, l, ax, ay, az);
    finish_row(token, l, ax, ay, az, batch_audio,
               output + batch * TOKEN_COUNT, row_statistics);
  }
  reduce_statistics<kThreads>(row_statistics, batch, query_block,
                              kQueryBlocks, statistics, warp_statistics,
                              &last_block);
#else
  (void)output;
  (void)statistics;
#endif
}

#if !SOTA_FUSED_MERGE
// Separate merge (for many splits): one warp per row, lanes over splits,
// then a fixed butterfly, so the merge order is fixed.
__global__ void __launch_bounds__(kMergeThreads) msa_sota_merge_kernel(
    const float* __restrict__ audio, float4* __restrict__ output,
    float* __restrict__ statistics) {
  __shared__ RowStatistics warp_statistics[kMergeThreads / kWarp];
  __shared__ bool last_block;
  const int lane = threadIdx.x % kWarp;
  const int block = blockIdx.x % kMergeBlocks;
  const int batch = blockIdx.x / kMergeBlocks;
  const int token = block * kMergeRowsPerBlock + threadIdx.x / kWarp;
  const float* batch_audio =
      audio + static_cast<size_t>(batch) * TOKEN_COUNT * SENSOR_LANE_COUNT;
  RowStatistics row_statistics{};
  if (token < TOKEN_COUNT) {
    const int row = batch * TOKEN_COUNT + token;
    float m = -INFINITY, l = 0.0f, ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int s = lane; s < kSplits; s += kWarp)
      merge_partial(s * BATCH_COUNT * TOKEN_COUNT + row, m, l, ax, ay, az);
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
      const float om = __shfl_xor_sync(kFullMask, m, offset);
      const float ol = __shfl_xor_sync(kFullMask, l, offset);
      const float ox = __shfl_xor_sync(kFullMask, ax, offset);
      const float oy = __shfl_xor_sync(kFullMask, ay, offset);
      const float oz = __shfl_xor_sync(kFullMask, az, offset);
      merge_state(om, ol, ox, oy, oz, m, l, ax, ay, az);
    }
    if (lane == 0)
      finish_row(token, l, ax, ay, az, batch_audio,
                 output + batch * TOKEN_COUNT, row_statistics);
  }
  reduce_statistics<kMergeThreads, true>(row_statistics, batch, block,
                                         kMergeBlocks, statistics,
                                         warp_statistics, &last_block);
}
#endif

// Context chunk c (a range of whole splits) is copied, and its attention
// blocks launched, separately, so the attention of the first chunks overlaps
// the copy of the later ones.
__host__ __device__ constexpr int chunk_first_split(int chunk) {
  return chunk * kSplits / kChunks;
}

static int enqueue_pipeline(const PipelineBuffers& b,
                            cudaStream_t audio_stream,
                            cudaStream_t imu_stream, cudaEvent_t imu_ready,
                            const StageEvents& stages) {
  // Streams and events of the chunk fan-out, created once (before the first
  // capture: the harness's first enqueue is a capture in graph mode, and
  // creating these objects is legal while capturing).
  static cudaStream_t chunk_streams[kChunks];
  static cudaEvent_t chunk_copied[kChunks];
  static cudaEvent_t chunk_done[kChunks];
  static cudaEvent_t audio_copied;
  static bool initialized = false;
  if (!initialized) {
    for (int c = 0; c < kChunks; ++c) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&chunk_streams[c],
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreateWithFlags(&chunk_copied[c],
                                          cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&chunk_done[c],
                                          cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&audio_copied, cudaEventDisableTiming));
    initialized = true;
  }

#if SOTA_ZERO_COPY_OUTPUT
  // Pinned host memory is device-accessible at the same address under UVA.
  float4* output = reinterpret_cast<float4*>(b.h_output);
  float* statistics = b.h_statistics;
#else
  float4* output = nullptr;
  float* statistics = nullptr;
  CUDA_CHECK(
      cudaGetSymbolAddress(reinterpret_cast<void**>(&output), g_output));
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&statistics),
                                  g_statistics));
#endif

  CUDA_CHECK(cudaMemcpyAsync(b.d_audio, b.h_audio,
                             kQuerySensorElements * sizeof(float),
                             cudaMemcpyHostToDevice, audio_stream));
  CUDA_CHECK(cudaEventRecord(audio_copied, audio_stream));
  CUDA_CHECK(record_stage(stages.attention_start, audio_stream));
#if SOTA_ZERO_COPY_IMU
  // Tiles are read from the pinned host buffer (UVA) by the kernel itself.
  const float* imu = b.h_imu;
#else
  const float* imu = b.d_imu;
#endif
  for (int c = 0; c < kChunks; ++c) {
    // Piece 0 is copied on audio_stream, right before launch 0; the other
    // pieces follow it back to back on imu_stream. Launch c > 0 runs on its
    // own stream (the last one on imu_stream, behind its piece) once piece
    // c and the audio have landed. (This beats copying every piece on
    // imu_stream: launch 0 then started ~8 us after its piece landed, and
    // graph mode lost ~10 us more when rotating over several graphs.) A
    // single piece is copied on imu_stream, concurrently with the audio.
    cudaStream_t copy_stream =
        c == 0 && kChunks > 1 ? audio_stream : imu_stream;
    if (c == 1)
      CUDA_CHECK(cudaStreamWaitEvent(imu_stream, chunk_copied[0], 0));
#if !SOTA_ZERO_COPY_IMU
    const size_t first = static_cast<size_t>(
        split_begin(chunk_first_split(c)) * SENSOR_LANE_COUNT);
    const size_t last = static_cast<size_t>(
        split_begin(chunk_first_split(c + 1)) * SENSOR_LANE_COUNT);
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
      const size_t offset =
          static_cast<size_t>(batch) * CONTEXT_COUNT * SENSOR_LANE_COUNT +
          first;
      CUDA_CHECK(cudaMemcpyAsync(b.d_imu + offset, b.h_imu + offset,
                                 (last - first) * sizeof(float),
                                 cudaMemcpyHostToDevice, copy_stream));
    }
#endif
    CUDA_CHECK(cudaEventRecord(chunk_copied[c], copy_stream));
    cudaStream_t stream = c == 0              ? audio_stream
                          : c == kChunks - 1 ? imu_stream
                                              : chunk_streams[c];
    if (stream != copy_stream) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, audio_copied, 0));
      CUDA_CHECK(cudaStreamWaitEvent(stream, chunk_copied[c], 0));
    }
    const int splits = chunk_first_split(c + 1) - chunk_first_split(c);
    msa_sota_attention_kernel<<<dim3(BATCH_COUNT * kQueryBlocks, splits),
                                kThreads, 0, stream>>>(
        b.d_audio, imu, output, statistics, chunk_first_split(c));
    CUDA_CHECK(cudaGetLastError());
  }
  // Join every chunk (and the copy stream) back into audio_stream.
  for (int c = 1; c < kChunks - 1; ++c) {
    CUDA_CHECK(cudaEventRecord(chunk_done[c], chunk_streams[c]));
    CUDA_CHECK(cudaStreamWaitEvent(audio_stream, chunk_done[c], 0));
  }
  CUDA_CHECK(cudaEventRecord(imu_ready, imu_stream));
  CUDA_CHECK(cudaStreamWaitEvent(audio_stream, imu_ready, 0));
  CUDA_CHECK(record_stage(stages.attention_stop, audio_stream));
#if !SOTA_FUSED_MERGE
  msa_sota_merge_kernel<<<BATCH_COUNT * kMergeBlocks, kMergeThreads, 0,
                          audio_stream>>>(b.d_audio, output, statistics);
  CUDA_CHECK(cudaGetLastError());
#endif
  CUDA_CHECK(record_stage(stages.statistics_stop, audio_stream));
#if !SOTA_ZERO_COPY_OUTPUT
  CUDA_CHECK(cudaMemcpyAsync(b.h_output, output,
                             kFeatureElements * sizeof(float),
                             cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaMemcpyAsync(b.h_statistics, statistics,
                             kStatisticsElements * sizeof(float),
                             cudaMemcpyDeviceToHost, audio_stream));
#endif
  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  return run_benchmark(argc, argv, "cuda_sota", enqueue_pipeline,
                       SOTA_DEFAULT_ROTATION);
}
