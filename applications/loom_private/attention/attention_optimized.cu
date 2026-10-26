// loom-private/multisensor-attention, optimized variant (same contract as
// attention.cu; README "Optimized variant" describes the changes).
//
// Pipeline per invocation:
//   upload audio [Q][3] and IMU [C][3] from pinned host memory (application
//                     scope only; the IMU in one chunk per context split)
//   project_audio  -> query     [Q][4] = (x, y, z, x + y)
//   project_imu    -> key/value [C][4] = (x, y, z, x - y), and for each
//                     context split the largest |x|, |y|, |z|
//   attention_partial -> for each (query row, context split): the softmax
//                     shift, the exponential sum, and the weighted value sums
//   attention_merge -> attention [Q][4] (the splits combined in split order)
//                     and the residual output [Q][4]
//   statistics     -> the four statistics, in the baseline's order
//   download output and statistics (application scope only)
//
// The baseline gives every query row its own block, so every block streams
// the whole key/value table from L2 twice: the kernel is L2-bandwidth-bound.
// Here a block owns kWarps x kRows query rows (kRows per warp) and one
// contiguous context split. Key/value tiles are staged in shared memory with
// cp.async double buffering, so each tile is read from L2 once per block, and
// each key read from shared memory serves kRows rows held in registers. The
// kernel is then bound by instruction issue and the exponential unit; the
// score, shift and exponential arithmetic below keep that at about 8.5
// instructions per score pair (Nsight Compute).
//
// Every reduction has a fixed shape (per-lane key order, xor butterfly, split
// order, the baseline's statistics tree), and the only atomics are integer
// maxima, so results are bitwise reproducible across invocations and between
// Direct and Graph launches.
#include "lp_harness.cuh"
#include "oracle.h"

#include <cmath>

namespace {

constexpr int kSensorLanes = 3;
constexpr int kFeatureLanes = 4;
constexpr int kStatistics = 4;

constexpr long long kLargeQueryTokens = 25000;
constexpr long long kLargeContextTokens = 250000;
// Query weights 4t + lane + 1 stay exact in float up to 2^24.
constexpr long long kMaxQueryTokens = 1ll << 22;
// Sensor ordinals 3t + lane stay far from 32-bit wraparound of ordinal * 37.
constexpr long long kMaxContextTokens = 1ll << 24;

constexpr int kProjectionThreads = 256;
constexpr int kMergeThreads = 256;
constexpr int kStatisticsThreads = 1024;
// Keys one split may hold. Each of a row's 32 lanes sums its split's keys in
// one float chain, so this bounds every chain to 2^16 terms: no longer than
// the baseline's longest (one of 256 threads over the largest context).
// Longer chains fail the tolerance at the largest contexts.
constexpr long long kMaxSplitKeys = 1ll << 21;
// Splits never make the partial buffers larger than this many (row, split)
// entries (24 bytes each); large query extents need no splitting to fill a
// GPU. The cap always leaves room for the splits kMaxSplitKeys requires.
constexpr long long kMaxPartials =
    (kMaxContextTokens + kMaxSplitKeys - 1) / kMaxSplitKeys * kMaxQueryTokens;
static_assert(kMaxPartials / kMaxQueryTokens * kMaxSplitKeys >= kMaxContextTokens,
              "the partial-buffer cap must allow the splits kMaxSplitKeys needs");
// Side streams the application scope spreads the context splits over.
constexpr int kSideStreams = 2;

float variant_scale(int variant) {
  return variant == 1 ? 0.5f : variant == 2 ? 1.5f : 1.0f;
}

// Host input generator (upstream's sensor_sample generalized to separate
// query and context extents).
float sensor_sample(unsigned ordinal, unsigned phase, float scale) {
  const unsigned mixed = (ordinal * 37u + phase * 101u) % 251u;
  return ((float)((int)mixed - 125) * (1.0f / 128.0f)) * scale;
}

__global__ void project_audio_kernel(const float *__restrict__ audio, int tokens,
                                     float4 *__restrict__ query) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const float x = audio[token * kSensorLanes + 0];
  const float y = audio[token * kSensorLanes + 1];
  const float z = audio[token * kSensorLanes + 2];
  query[token] = make_float4(x, y, z, x + y);
}

// Projects tokens [begin, end). Also folds max |x|, |y|, |z| of each block's
// tokens into key_bound[3 * split + lane], where split = token / split_keys
// (split_keys is a multiple of the block size, so a block never straddles two
// splits). The bounds are non-negative floats, whose bit patterns order as
// unsigned integers, so the integer atomic maximum is exact and independent
// of the order.
__global__ void __launch_bounds__(kProjectionThreads)
project_imu_kernel(const float *__restrict__ imu, int begin, int end,
                   int split_keys, float4 *__restrict__ key_value,
                   unsigned *__restrict__ key_bound) {
  __shared__ float warp_bounds[kSensorLanes][kProjectionThreads / 32];
  const int block_first = begin + blockIdx.x * kProjectionThreads;
  const int token = block_first + threadIdx.x;
  float bound[kSensorLanes] = {0.0f, 0.0f, 0.0f};
  if (token < end) {
    const float x = imu[token * kSensorLanes + 0];
    const float y = imu[token * kSensorLanes + 1];
    const float z = imu[token * kSensorLanes + 2];
    key_value[token] = make_float4(x, y, z, x - y);
    bound[0] = fabsf(x);
    bound[1] = fabsf(y);
    bound[2] = fabsf(z);
  }
  for (int lane = 0; lane < kSensorLanes; ++lane) {
    for (int offset = 16; offset > 0; offset /= 2)
      bound[lane] = fmaxf(bound[lane], __shfl_xor_sync(0xffffffffu, bound[lane], offset));
    if ((threadIdx.x & 31) == 0) warp_bounds[lane][threadIdx.x >> 5] = bound[lane];
  }
  __syncthreads();
  if (threadIdx.x < kSensorLanes) {
    float block_bound = 0.0f;
    for (int w = 0; w < kProjectionThreads / 32; ++w)
      block_bound = fmaxf(block_bound, warp_bounds[threadIdx.x][w]);
    atomicMax(&key_bound[kSensorLanes * (block_first / split_keys) + threadIdx.x],
              __float_as_uint(block_bound));
  }
}

__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void copy_async16(float4 *shared, const float4 *global) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address),
               "l"(global));
}
__device__ __forceinline__ void commit_async() {
  asm volatile("cp.async.commit_group;" ::);
}
template <int kPending> __device__ __forceinline__ void wait_async() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}

// h . (k.x, k.y, k.z) + bias. With the exact three-lane score weights below
// and bias 0 (the exact-maximum pass) this is the exact score; the
// exponential pass calls it with log2(e)-scaled, rounded weights and bias.
__device__ __forceinline__ float score3(const float3 h, const float4 k, float bias) {
  return fmaf(h.x, k.x, fmaf(h.y, k.y, fmaf(h.z, k.z, bias)));
}

// Streams keys [begin, end) through two shared-memory tiles and calls
// visit(key) for keys lane, lane + 32, ... of each tile, in increasing order.
// Every thread of the block loads; warps without rows (inactive) only load.
// kSkipLast (test mutant only) drops the last key of every tile.
template <int kTile, int kThreads, bool kSkipLast = false, typename Visit>
__device__ __forceinline__ void stream_keys(float4 (*tiles)[kTile],
                                            const float4 *__restrict__ key_value,
                                            int begin, int end, bool active,
                                            Visit &&visit) {
  const int lane = threadIdx.x & 31;
  const int tile_count = (end - begin + kTile - 1) / kTile;
  static_assert(kTile % kThreads == 0, "a tile is a whole number of block loads");
  auto load = [&](int tile) {
    const int first = begin + tile * kTile;
    if (first + kTile <= end) {
#pragma unroll
      for (int i = threadIdx.x; i < kTile; i += kThreads)
        copy_async16(&tiles[tile & 1][i], &key_value[first + i]);
    } else {
      for (int i = threadIdx.x; i < kTile; i += kThreads)
        if (first + i < end) copy_async16(&tiles[tile & 1][i], &key_value[first + i]);
    }
    commit_async();
  };
  load(0);
  for (int tile = 0; tile < tile_count; ++tile) {
    if (tile + 1 < tile_count) {
      load(tile + 1);
      wait_async<1>();
    } else {
      wait_async<0>();
    }
    __syncthreads();
    const float4 *keys = tiles[tile & 1];
    int length = min(kTile, end - (begin + tile * kTile));
    if (kSkipLast) --length;
    if (active) {
      if (length == kTile) {
#pragma unroll
        for (int j = 0; j < kTile / 32; ++j) visit(keys[lane + 32 * j]);
      } else {
        for (int j = lane; j < length; j += 32) visit(keys[j]);
      }
    }
    __syncthreads();
  }
}

// Largest per-row score bound used directly as the softmax shift. Above it
// the block finds the exact row maxima first (see attention_partial_kernel).
constexpr float kMaxBoundShift = 20.0f;

// One block per (group of kWarps x kRows query rows, context split
// first_split + blockIdx.y). Warp w owns rows first_row .. first_row + kRows
// - 1; lane i visits the split's keys i, i + 32, ... in increasing order and
// accumulates p = exp(s - shift), p a, p b and p c. Per-lane partials combine
// through a fixed xor butterfly; lane 0 writes (shift, sum p) and the
// weighted sums (with the fourth lane sum p (a - b) = sum p a - sum p b).
//
// Score. With q = (x, y, z, x + y) and k = (a, b, c, a - b) from the
// projections, 0.5 q.k = (x + 0.5 y) a - 0.5 x b + 0.5 z c. The weights
// h = (x + 0.5 y, -0.5 x, 0.5 z) are exact, and so is every product and sum
// for the contract's inputs (contract.json "exactness"), so with these
// weights the exact-maximum pass forms the baseline's score bit for bit with
// one product fewer. The exponential pass uses the weights times log2(e),
// rounded, so its exponents are not exact (see Exponential).
//
// Shift. The softmax is invariant to the shift; it only has to keep the
// exponentials in range. The shift is an upper bound of the row's scores in
// this split, sum_l |h_l| max_split |k_l|, rounded up to an integer. Every p
// is then at most 1, and while the bound is at most kMaxBoundShift the
// largest p is at least e^-40, far from underflow. If any row of the block
// has a larger bound (never for the contract's inputs, whose scores stay
// below 8 in magnitude), or force_exact_maximum is set (tests), the block
// first makes a separate pass for the exact row maxima, as the baseline does,
// and shifts by those.
//
// Exponential. p = 2^t with the hardware ex2.approx instruction, where
// t = log2(e) (s - shift) is one fused chain from the weights and the shift
// pre-multiplied by log2(e).
template <int kRows, int kWarps, int kTile>
__global__ void __launch_bounds__(kWarps * 32)
attention_partial_kernel(const float4 *__restrict__ query,
                         const float4 *__restrict__ key_value,
                         const unsigned *__restrict__ key_bound,
                         int query_tokens, int context_tokens, int split_keys,
                         int first_split, int force_exact_maximum,
                         float2 *__restrict__ partial_scale,
                         float4 *__restrict__ partial_values) {
  constexpr int kThreads = kWarps * 32;
  constexpr float kLog2e = 1.4426950408889634f;
  __shared__ __align__(16) float4 tiles[2][kTile];
  const int lane = threadIdx.x & 31;
  const int first_row = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * kRows;
  const bool active = first_row < query_tokens;
  const int split = first_split + blockIdx.y;
  const int begin = split * split_keys;
  const int end = min(context_tokens, begin + split_keys);
  const float bound_a = __uint_as_float(key_bound[kSensorLanes * split + 0]);
  const float bound_b = __uint_as_float(key_bound[kSensorLanes * split + 1]);
  const float bound_c = __uint_as_float(key_bound[kSensorLanes * split + 2]);

  float3 h[kRows];
  float shift[kRows];
  bool exact = force_exact_maximum != 0;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const float4 q = query[min(first_row + r, query_tokens - 1)];
    h[r] = make_float3(q.x + 0.5f * q.y, -0.5f * q.x, 0.5f * q.z);
    shift[r] = ceilf(fabsf(h[r].x) * bound_a + fabsf(h[r].y) * bound_b +
                     fabsf(h[r].z) * bound_c);
    exact = exact || !(shift[r] <= kMaxBoundShift);
  }
  if (__syncthreads_or(exact)) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) shift[r] = -INFINITY;
    stream_keys<kTile, kThreads>(tiles, key_value, begin, end, active,
                                 [&](const float4 k) {
#pragma unroll
                                   for (int r = 0; r < kRows; ++r)
                                     shift[r] = fmaxf(shift[r], score3(h[r], k, 0.0f));
                                 });
#pragma unroll
    for (int r = 0; r < kRows; ++r)
      for (int offset = 16; offset > 0; offset /= 2)
        shift[r] = fmaxf(shift[r], __shfl_xor_sync(0xffffffffu, shift[r], offset));
  }

  float3 h2[kRows], weighted[kRows];
  float bias[kRows], denominator[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    h2[r] = make_float3(h[r].x * kLog2e, h[r].y * kLog2e, h[r].z * kLog2e);
    bias[r] = -shift[r] * kLog2e;
    denominator[r] = 0.0f;
    weighted[r] = make_float3(0.0f, 0.0f, 0.0f);
  }
#ifdef LP_ATTENTION_MUTANT_SKIP_TILE_LAST_KEY
  constexpr bool kSkipLast = true;
#else
  constexpr bool kSkipLast = false;
#endif
  stream_keys<kTile, kThreads, kSkipLast>(
      tiles, key_value, begin, end, active, [&](const float4 k) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          const float p = exp2_approx(score3(h2[r], k, bias[r]));
          denominator[r] += p;
          weighted[r].x = fmaf(p, k.x, weighted[r].x);
          weighted[r].y = fmaf(p, k.y, weighted[r].y);
          weighted[r].z = fmaf(p, k.z, weighted[r].z);
        }
      });
  if (!active) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r)
    for (int offset = 16; offset > 0; offset /= 2) {
      denominator[r] += __shfl_xor_sync(0xffffffffu, denominator[r], offset);
      weighted[r].x += __shfl_xor_sync(0xffffffffu, weighted[r].x, offset);
      weighted[r].y += __shfl_xor_sync(0xffffffffu, weighted[r].y, offset);
      weighted[r].z += __shfl_xor_sync(0xffffffffu, weighted[r].z, offset);
    }
  if (lane == 0) {
    const long long base = (long long)split * query_tokens + first_row;
#pragma unroll
    for (int r = 0; r < kRows; ++r)
      if (first_row + r < query_tokens) {
        partial_scale[base + r] = make_float2(shift[r], denominator[r]);
        partial_values[base + r] =
            make_float4(weighted[r].x, weighted[r].y, weighted[r].z,
                        weighted[r].x - weighted[r].y);
      }
  }
}

// One thread per query row: combines the splits in increasing order, each
// rescaled by exp(shift_split - largest shift), divides by the total, and
// writes the attention row and the residual output row
// (attention + query * 0.125, the baseline's expression).
__global__ void attention_merge_kernel(const float2 *__restrict__ partial_scale,
                                       const float4 *__restrict__ partial_values,
                                       const float4 *__restrict__ query,
                                       int query_tokens, int splits,
                                       float4 *__restrict__ attention,
                                       float4 *__restrict__ output) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= query_tokens) return;
  float maximum = -INFINITY;
  for (int split = 0; split < splits; ++split)
    maximum = fmaxf(maximum, partial_scale[(long long)split * query_tokens + row].x);
  float denominator = 0.0f, v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (int split = 0; split < splits; ++split) {
    const long long index = (long long)split * query_tokens + row;
    const float2 scale = partial_scale[index];
    const float4 values = partial_values[index];
    const float factor = expf(scale.x - maximum);
    denominator += scale.y * factor;
    v0 += values.x * factor;
    v1 += values.y * factor;
    v2 += values.z * factor;
    v3 += values.w * factor;
  }
  const float4 a = make_float4(v0 / denominator, v1 / denominator,
                               v2 / denominator, v3 / denominator);
  const float4 q = query[row];
  attention[row] = a;
  output[row] = make_float4(a.x + q.x * 0.125f, a.y + q.y * 0.125f,
                            a.z + q.z * 0.125f, a.w + q.w * 0.125f);
}

// The baseline's residual_statistics_kernel without the residual (written by
// attention_merge_kernel above): one block, thread i visits flat elements
// i, i + T, ... (element = 4t + lane) in increasing order and accumulates the
// statistics in float as upstream does; partials combine through the same
// fixed binary tree, so the statistics match the baseline's order exactly.
__global__ void __launch_bounds__(kStatisticsThreads)
statistics_kernel(const float *__restrict__ query,
                  const float *__restrict__ attention, int elements,
                  float *__restrict__ statistics) {
  __shared__ float energy_partials[kStatisticsThreads];
  __shared__ float sum_partials[kStatisticsThreads];
  __shared__ float weighted_partials[kStatisticsThreads];
  __shared__ float maximum_partials[kStatisticsThreads];
  const int tid = threadIdx.x;
  float energy = 0.0f, sum = 0.0f, weighted = 0.0f;
  float maximum = 0.0f;
  for (int element = tid; element < elements; element += kStatisticsThreads) {
    const float projection = query[element];
    const float value = attention[element];
    const float magnitude = value < 0.0f ? -value : value;
    energy += projection * projection;
    sum += value;
    weighted += (float)(element + 1) * value;
    if (magnitude > maximum) maximum = magnitude;
  }
  energy_partials[tid] = energy;
  sum_partials[tid] = sum;
  weighted_partials[tid] = weighted;
  maximum_partials[tid] = maximum;
  __syncthreads();
  for (int stride = kStatisticsThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      energy_partials[tid] += energy_partials[tid + stride];
      sum_partials[tid] += sum_partials[tid + stride];
      weighted_partials[tid] += weighted_partials[tid + stride];
      if (maximum_partials[tid + stride] > maximum_partials[tid])
        maximum_partials[tid] = maximum_partials[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    statistics[0] = energy_partials[0];
    statistics[1] = sum_partials[0];
    statistics[2] = weighted_partials[0];
    statistics[3] = maximum_partials[0];
  }
}

// Attention kernel shapes: rows per warp, warps per block, keys per tile.
using PartialKernel = void (*)(const float4 *, const float4 *, const unsigned *,
                               int, int, int, int, int, float2 *, float4 *);

struct Shape {
  const char *name;
  int rows, warps;
  PartialKernel kernel;
};

const Shape kShapes[] = {
    {"rows8_warps8_tile1024", 8, 8, attention_partial_kernel<8, 8, 1024>},
    {"rows8_warps8_tile512", 8, 8, attention_partial_kernel<8, 8, 512>},
    {"rows8_warps4_tile512", 8, 4, attention_partial_kernel<8, 4, 512>},
    {"rows4_warps8_tile256", 4, 8, attention_partial_kernel<4, 8, 256>},
};
constexpr int kShapeCount = sizeof(kShapes) / sizeof(kShapes[0]);

// Per-GPU kernel shape and number of context splits at the large preset
// (README "Optimized variant" gives the measurements behind each choice).
// Every GPU runs two 256-thread blocks per SM with the chosen shape, so the
// large preset has 391 row groups x splits blocks. 8 splits give the RTX 3090
// (82 SMs, 164 resident blocks) about 19.1 waves; 12 splits give the RTX 4090
// (128 SMs, 256 resident) about 18.3 and the RTX 5090 (170 SMs, 340 resident)
// about 13.8 waves, which keeps the last partial wave small.
struct Tuning {
  int shape;
  int splits;
  bool force_exact_maximum;
  bool check_compute;
};

// Largest split count a tuning hook may force (the grid's y extent).
constexpr int kMaxSplits = 1024;

#ifdef LP_ATTENTION_TUNING_HOOKS
// An integer hook value in [minimum, maximum] from the environment, or
// fallback when unset; anything else is a usage error (exit 2).
int hook_value(const char *name, int minimum, int maximum, int fallback) {
  const char *text = std::getenv(name);
  if (!text) return fallback;
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value < minimum || value > maximum)
    lp::fail(lp::kExitUsage, "%s must be an integer in [%d, %d], got '%s'", name,
             minimum, maximum, text);
  return (int)value;
}
#endif

Tuning tuning(const cudaDeviceProp &properties) {
  Tuning chosen{0, 12, false, false};
  if (properties.major == 8 && properties.minor == 6) chosen.splits = 8;  // RTX 3090
#ifdef LP_ATTENTION_TUNING_HOOKS
  // Test and tuning builds only: force a shape, a split count, or the exact
  // row maxima, or make the checked invocation run enqueue_compute between
  // whole uploads and downloads (so the check covers the compute path too).
  chosen.shape = hook_value("LP_ATTENTION_SHAPE", 0, kShapeCount - 1, chosen.shape);
  chosen.splits = hook_value("LP_ATTENTION_SPLITS", 1, kMaxSplits, chosen.splits);
  chosen.force_exact_maximum = hook_value("LP_ATTENTION_EXACT_MAXIMUM", 0, 1, 0) != 0;
  chosen.check_compute = hook_value("LP_ATTENTION_CHECK_COMPUTE", 0, 1, 0) != 0;
#endif
  return chosen;
}

// Upstream's tolerance, evaluated in double against the double oracle; NaN
// fails.
bool within_tolerance(double actual, double expected) {
  return std::fabs(actual - expected) <= 1.0e-4 * (1.0 + std::fabs(expected));
}

class Attention final : public lp::Workload {
public:
  Attention(long long query_tokens, long long context_tokens, int variant)
      : Q_(query_tokens), C_(context_tokens), variant_(variant),
        scale_(variant_scale(variant)) {}

  ~Attention() override {
    for (cudaEvent_t event : events_) cudaEventDestroy(event);
    for (cudaStream_t side : side_)
      if (side) cudaStreamDestroy(side);
    cudaFreeHost(h_audio_);
    cudaFreeHost(h_imu_);
    cudaFreeHost(h_output_);
    cudaFreeHost(h_statistics_);
    cudaFree(d_audio_);
    cudaFree(d_imu_);
    cudaFree(d_query_);
    cudaFree(d_key_value_);
    cudaFree(d_key_bound_);
    cudaFree(d_partial_scale_);
    cudaFree(d_partial_values_);
    cudaFree(d_attention_);
    cudaFree(d_output_);
    cudaFree(d_statistics_);
  }

  std::string identity() const override {
    return "loom-private/multisensor-attention";
  }

  lp::JsonObject dimensions() const override {
    using lp::Json;
    return {{"query_tokens", Json::integer(Q_)},
            {"context_tokens", Json::integer(C_)},
            {"sensor_lanes", Json::integer(kSensorLanes)},
            {"feature_lanes", Json::integer(kFeatureLanes)},
            {"statistics", Json::integer(kStatistics)},
            {"input_variant", Json::integer(variant_)},
            {"input_scale", Json::number(scale_)},
            {"score_pairs", Json::integer(Q_ * C_)},
            {"output_elements", Json::integer(Q_ * kFeatureLanes)}};
  }

  std::string application_scope() const override {
    return "upload audio [Q][3] and IMU [C][3] float32 from pinned host memory; "
           "audio and IMU projections; streaming attention; residual and "
           "statistics; download output [Q][4] and the 4 statistics";
  }
  std::string compute_scope() const override {
    return "audio and IMU projections, streaming attention, residual and "
           "statistics with the sensor inputs resident on the device; no "
           "host-device transfers";
  }

  void setup(cudaStream_t) override {
    int device = 0;
    cudaDeviceProp properties;
    LP_CUDA(cudaGetDevice(&device));
    LP_CUDA(cudaGetDeviceProperties(&properties, device));
    const Tuning chosen = tuning(properties);
    if (chosen.shape < 0 || chosen.shape >= kShapeCount || chosen.splits < 1 ||
        chosen.splits > kMaxSplits)
      lp::fail(lp::kExitUsage, "invalid attention tuning");
    shape_ = chosen.shape;
    force_exact_maximum_ = chosen.force_exact_maximum ? 1 : 0;
    check_compute_ = chosen.check_compute;
    const Shape &shape = kShapes[shape_];
    row_groups_ = (int)((Q_ + shape.rows * shape.warps - 1) / (shape.rows * shape.warps));
    // Splits are contiguous, non-empty, a multiple of the projection block
    // long, and at most kMaxSplitKeys long (also when a hook forces fewer).
    long long splits = std::min<long long>(chosen.splits,
                                           (C_ + kProjectionThreads - 1) / kProjectionThreads);
    splits = std::max(1ll, std::min(splits, kMaxPartials / Q_));
    splits = std::max(splits, (C_ + kMaxSplitKeys - 1) / kMaxSplitKeys);
    const long long per_split = (C_ + splits - 1) / splits;
    split_keys_ = (int)((per_split + kProjectionThreads - 1) / kProjectionThreads *
                        kProjectionThreads);
    splits_ = (int)((C_ + split_keys_ - 1) / split_keys_);
    std::fprintf(stderr,
                 "attention: optimized shape %s, %d context splits of %d keys%s%s\n",
                 shape.name, splits_, split_keys_,
                 force_exact_maximum_ ? ", exact row maxima" : "",
                 check_compute_ ? ", application runs the compute path" : "");

    LP_CUDA(cudaMallocHost(&h_audio_, audio_bytes()));
    LP_CUDA(cudaMallocHost(&h_imu_, imu_bytes()));
    LP_CUDA(cudaMallocHost(&h_output_, output_bytes()));
    LP_CUDA(cudaMallocHost(&h_statistics_, statistics_bytes()));
    for (long long t = 0; t < Q_; ++t)
      for (int lane = 0; lane < kSensorLanes; ++lane)
        h_audio_[t * kSensorLanes + lane] =
            sensor_sample((unsigned)(t * kSensorLanes + lane), 0u, scale_);
    for (long long t = 0; t < C_; ++t)
      for (int lane = 0; lane < kSensorLanes; ++lane)
        h_imu_[t * kSensorLanes + lane] =
            sensor_sample((unsigned)(t * kSensorLanes + lane), 1u, scale_);
    LP_CUDA(cudaMalloc(&d_audio_, audio_bytes()));
    LP_CUDA(cudaMalloc(&d_imu_, imu_bytes()));
    LP_CUDA(cudaMalloc(&d_query_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_key_value_, key_value_bytes()));
    LP_CUDA(cudaMalloc(&d_key_bound_, key_bound_bytes()));
    LP_CUDA(cudaMalloc(&d_partial_scale_, partial_scale_bytes()));
    LP_CUDA(cudaMalloc(&d_partial_values_, partial_values_bytes()));
    LP_CUDA(cudaMalloc(&d_attention_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_output_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_statistics_, statistics_bytes()));
    for (cudaStream_t &side : side_)
      LP_CUDA(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    events_.resize(kEventsBeforeChunks + splits_);
    for (cudaEvent_t &event : events_)
      LP_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  // The IMU goes up in one chunk per context split. Each split's projection
  // and attention start as soon as its chunk has arrived, on side streams
  // that alternate between splits, so later chunks upload while earlier
  // splits compute, and neighbouring splits' kernels fill each other's tails.
  // The output download overlaps the statistics kernel. The side streams fork
  // from and join back to the harness stream through events, so a captured
  // graph contains every branch and Direct launches keep the same order.
  void enqueue_application(cudaStream_t stream) override {
    if (check_compute_) {
      LP_CUDA(cudaMemcpyAsync(d_audio_, h_audio_, audio_bytes(),
                              cudaMemcpyHostToDevice, stream));
      LP_CUDA(cudaMemcpyAsync(d_imu_, h_imu_, imu_bytes(), cudaMemcpyHostToDevice,
                              stream));
      enqueue_compute(stream);
      LP_CUDA(cudaMemcpyAsync(h_output_, d_output_, output_bytes(),
                              cudaMemcpyDeviceToHost, stream));
      LP_CUDA(cudaMemcpyAsync(h_statistics_, d_statistics_, statistics_bytes(),
                              cudaMemcpyDeviceToHost, stream));
      return;
    }
    const cudaEvent_t audio_ready = events_[0], query_ready = events_[1],
                      merged = events_[2], statistics_done = events_[3];
    const cudaEvent_t *side_done = &events_[4];
    const cudaEvent_t *chunk_ready = &events_[kEventsBeforeChunks];
    const int c = (int)C_;
    LP_CUDA(cudaMemsetAsync(d_key_bound_, 0, key_bound_bytes(), stream));
    LP_CUDA(cudaMemcpyAsync(d_audio_, h_audio_, audio_bytes(),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaEventRecord(audio_ready, stream));
    for (int split = 0; split < splits_; ++split) {
      const int begin = split * split_keys_, end = std::min(c, begin + split_keys_);
      const size_t first = (size_t)begin * kSensorLanes;
      LP_CUDA(cudaMemcpyAsync(d_imu_ + first, h_imu_ + first,
                              (size_t)(end - begin) * kSensorLanes * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
      LP_CUDA(cudaEventRecord(chunk_ready[split], stream));
    }
    LP_CUDA(cudaStreamWaitEvent(side_[0], audio_ready, 0));
    enqueue_project_audio(side_[0]);
    LP_CUDA(cudaEventRecord(query_ready, side_[0]));
    for (int i = 1; i < kSideStreams; ++i)
      LP_CUDA(cudaStreamWaitEvent(side_[i], query_ready, 0));
    for (int split = 0; split < splits_; ++split) {
      const cudaStream_t side = side_[split % kSideStreams];
      const int begin = split * split_keys_, end = std::min(c, begin + split_keys_);
      LP_CUDA(cudaStreamWaitEvent(side, chunk_ready[split], 0));
      enqueue_project_imu(side, begin, end);
      enqueue_attention(side, split, 1);
    }
    for (int i = 0; i < kSideStreams; ++i) {
      LP_CUDA(cudaEventRecord(side_done[i], side_[i]));
      LP_CUDA(cudaStreamWaitEvent(stream, side_done[i], 0));
    }
    enqueue_merge(stream);
    LP_CUDA(cudaEventRecord(merged, stream));
    LP_CUDA(cudaStreamWaitEvent(side_[0], merged, 0));
    enqueue_statistics(side_[0]);
    LP_CUDA(cudaEventRecord(statistics_done, side_[0]));
    LP_CUDA(cudaMemcpyAsync(h_output_, d_output_, output_bytes(),
                            cudaMemcpyDeviceToHost, stream));
    LP_CUDA(cudaStreamWaitEvent(stream, statistics_done, 0));
    LP_CUDA(cudaMemcpyAsync(h_statistics_, d_statistics_, statistics_bytes(),
                            cudaMemcpyDeviceToHost, stream));
  }

  // The same kernels on the harness stream, each over the whole extent.
  void enqueue_compute(cudaStream_t stream) override {
    LP_CUDA(cudaMemsetAsync(d_key_bound_, 0, key_bound_bytes(), stream));
    enqueue_project_audio(stream);
    enqueue_project_imu(stream, 0, (int)C_);
    enqueue_attention(stream, 0, splits_);
    enqueue_merge(stream);
    enqueue_statistics(stream);
  }

  // Every buffer the invocation writes, including the uploaded inputs and the
  // intermediates, is set to all-ones bytes (a NaN pattern); host outputs too.
  void poison_outputs(cudaStream_t stream) override {
    LP_CUDA(cudaMemsetAsync(d_audio_, 0xff, audio_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_imu_, 0xff, imu_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_query_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_key_value_, 0xff, key_value_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_key_bound_, 0xff, key_bound_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_partial_scale_, 0xff, partial_scale_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_partial_values_, 0xff, partial_values_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_attention_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_output_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_statistics_, 0xff, statistics_bytes(), stream));
    std::memset(h_output_, 0xff, output_bytes());
    std::memset(h_statistics_, 0xff, statistics_bytes());
  }

  void compute_reference() override {
    attention_oracle::compute(Q_, C_, (double)scale_, reference_);
  }

  bool check_outputs() override {
    static const char *const kNames[kStatistics] = {
        "projection_energy", "attention_sum", "weighted_sum", "max_abs_attention"};
    bool pass = true;
    // Residual output, every element.
    long long failures = 0;
    double worst_ratio = 0.0;
    for (long long i = 0; i < Q_ * kFeatureLanes; ++i) {
      const double actual = h_output_[i], expected = reference_.output[i];
      const double ratio =
          std::fabs(actual - expected) / (1.0e-4 * (1.0 + std::fabs(expected)));
      if (!(ratio <= worst_ratio)) worst_ratio = ratio;
      if (!within_tolerance(actual, expected)) {
        if (failures < 5)
          std::fprintf(stderr,
                       "attention: mismatch output[%lld][%lld] (flat %lld): "
                       "expected %.9g, actual %.9g\n",
                       i / kFeatureLanes, i % kFeatureLanes, i, expected, actual);
        ++failures;
      }
    }
    if (failures) {
      std::fprintf(stderr, "attention: %lld of %lld output elements mismatch\n",
                   failures, Q_ * kFeatureLanes);
      pass = false;
    }
    std::fprintf(stderr, "attention: output worst |error|/tolerance %.3g\n",
                 worst_ratio);
    // Statistics and upstream's combined checksum (float, upstream's order).
    const float *s = h_statistics_;
    const float combined = s[2] + 3.0f * s[0] + 7.0f * s[1] + 11.0f * s[3];
    for (int i = 0; i <= kStatistics; ++i) {
      const double actual = i < kStatistics ? s[i] : combined;
      const double expected =
          i < kStatistics ? reference_.statistics[i] : reference_.combined;
      const char *label = i < kStatistics ? kNames[i] : "combined_checksum";
      const double ratio =
          std::fabs(actual - expected) / (1.0e-4 * (1.0 + std::fabs(expected)));
      const bool ok = within_tolerance(actual, expected);
      std::fprintf(stderr,
                   "attention: %s%s: expected %.9g, actual %.9g, "
                   "|error|/tolerance %.3g\n",
                   ok ? "" : "mismatch ", label, expected, actual, ratio);
      pass = pass && ok;
    }
    return pass;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_statistics_, statistics_bytes(),
                       lp::fnv1a64(h_output_, output_bytes()));
  }

private:
  // events_: audio uploaded, query projected, merged, statistics done, one
  // per side stream (its splits done), then one per split (IMU chunk
  // uploaded).
  static constexpr int kEventsBeforeChunks = 4 + kSideStreams;

  void enqueue_project_audio(cudaStream_t stream) {
    const int q = (int)Q_;
    project_audio_kernel<<<(q + kProjectionThreads - 1) / kProjectionThreads,
                           kProjectionThreads, 0, stream>>>(d_audio_, q,
                                                            (float4 *)d_query_);
    LP_CUDA_LAUNCH();
  }

  void enqueue_project_imu(cudaStream_t stream, int begin, int end) {
    project_imu_kernel<<<(end - begin + kProjectionThreads - 1) / kProjectionThreads,
                         kProjectionThreads, 0, stream>>>(
        d_imu_, begin, end, split_keys_, (float4 *)d_key_value_, d_key_bound_);
    LP_CUDA_LAUNCH();
  }

  // Context splits [first_split, first_split + splits) for every row group.
  void enqueue_attention(cudaStream_t stream, int first_split, int splits) {
    const Shape &shape = kShapes[shape_];
    shape.kernel<<<dim3(row_groups_, splits), shape.warps * 32, 0, stream>>>(
        (const float4 *)d_query_, (const float4 *)d_key_value_, d_key_bound_,
        (int)Q_, (int)C_, split_keys_, first_split, force_exact_maximum_,
        d_partial_scale_, d_partial_values_);
    LP_CUDA_LAUNCH();
  }

  void enqueue_merge(cudaStream_t stream) {
    const int q = (int)Q_;
    attention_merge_kernel<<<(q + kMergeThreads - 1) / kMergeThreads,
                             kMergeThreads, 0, stream>>>(
        d_partial_scale_, d_partial_values_, (const float4 *)d_query_, q,
        splits_, (float4 *)d_attention_, (float4 *)d_output_);
    LP_CUDA_LAUNCH();
  }

  void enqueue_statistics(cudaStream_t stream) {
    statistics_kernel<<<1, kStatisticsThreads, 0, stream>>>(
        d_query_, d_attention_, (int)Q_ * kFeatureLanes, d_statistics_);
    LP_CUDA_LAUNCH();
  }

  size_t audio_bytes() const { return (size_t)Q_ * kSensorLanes * sizeof(float); }
  size_t imu_bytes() const { return (size_t)C_ * kSensorLanes * sizeof(float); }
  size_t key_value_bytes() const { return (size_t)C_ * kFeatureLanes * sizeof(float); }
  size_t output_bytes() const { return (size_t)Q_ * kFeatureLanes * sizeof(float); }
  size_t statistics_bytes() const { return kStatistics * sizeof(float); }
  size_t key_bound_bytes() const {
    return (size_t)splits_ * kSensorLanes * sizeof(unsigned);
  }
  size_t partial_scale_bytes() const { return (size_t)splits_ * Q_ * sizeof(float2); }
  size_t partial_values_bytes() const { return (size_t)splits_ * Q_ * sizeof(float4); }

  long long Q_, C_;
  int variant_;
  float scale_;
  int shape_ = 0, force_exact_maximum_ = 0;
  bool check_compute_ = false;
  int row_groups_ = 0, splits_ = 0, split_keys_ = 0;
  cudaStream_t side_[kSideStreams] = {};
  std::vector<cudaEvent_t> events_;
  float *h_audio_ = nullptr, *h_imu_ = nullptr, *h_output_ = nullptr,
        *h_statistics_ = nullptr;
  float *d_audio_ = nullptr, *d_imu_ = nullptr, *d_query_ = nullptr,
        *d_key_value_ = nullptr, *d_attention_ = nullptr, *d_output_ = nullptr,
        *d_statistics_ = nullptr;
  unsigned *d_key_bound_ = nullptr;
  float2 *d_partial_scale_ = nullptr;
  float4 *d_partial_values_ = nullptr;
  attention_oracle::Reference reference_;
};

std::unique_ptr<lp::Workload> make_workload(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long q = options.take_int("query_tokens", kLargeQueryTokens);
  const long long c = options.take_int("context_tokens", kLargeContextTokens);
  const long long variant = options.take_int("input_variant", 0);
  if (q < 1 || q > kMaxQueryTokens)
    lp::fail(lp::kExitUsage, "query_tokens must be in [1, %lld], got %lld",
             kMaxQueryTokens, q);
  if (c < 1 || c > kMaxContextTokens)
    lp::fail(lp::kExitUsage, "context_tokens must be in [1, %lld], got %lld",
             kMaxContextTokens, c);
  if (variant < 0 || variant > 2)
    lp::fail(lp::kExitUsage,
             "input_variant must be 0 (scale 1.0), 1 (0.5) or 2 (1.5), got %lld",
             variant);
  return std::make_unique<Attention>(q, c, (int)variant);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make_workload); }
