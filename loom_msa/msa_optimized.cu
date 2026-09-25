// Optimized CUDA variant.
//
//   attention:  one pass over the context with an online softmax. Each block
//               owns several query rows and streams the context through a
//               shared-memory tile that all of its query rows reuse.
//               With ATTENTION_CONTEXT_SPLITS > 1 the context is also split
//               across blocks (1000 queries alone cannot fill the GPU with
//               enough reuse per block); each block then writes an
//               unnormalized (m, l, acc) partial instead of the final row.
//   statistics: several warps per batch item, warp-shuffle reductions in a
//               fixed (deterministic) order. When the context is split it
//               first merges each row's partials in split order, so no extra
//               combine kernel or graph node is needed.
//   projections are intentionally left unfused and equivalent to the baseline
//   so the stages remain separately measurable.
#include <cuda_pipeline.h>

#include "msa_harness.cuh"

// Attention tile configuration (overridable for tuning sweeps). Defaults are
// the fastest measured on an RTX 3090 for 1000 x 50000: 63 query tiles x 8
// context splits = 504 blocks of 128 threads, 90 registers per thread.
#ifndef ATTENTION_QUERIES_PER_BLOCK
#define ATTENTION_QUERIES_PER_BLOCK 16
#endif
#ifndef ATTENTION_QUERIES_PER_THREAD
#define ATTENTION_QUERIES_PER_THREAD 4
#endif
#ifndef ATTENTION_THREADS
#define ATTENTION_THREADS 128
#endif
#ifndef ATTENTION_TILE_ROWS
#define ATTENTION_TILE_ROWS 256
#endif
#ifndef ATTENTION_CONTEXT_SPLITS
#define ATTENTION_CONTEXT_SPLITS 8
#endif

constexpr int kStatisticsThreads = 256;
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

// Scores are dot * 0.5 (main.c). Folding log2(e) into the same factor lets the
// softmax use exp2: exp(s - m) == exp2((s - m) * log2(e)).
constexpr float kScoreScaleLog2 = 0.5f * 1.4426950408889634f;

__global__ void project_audio_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = BATCH_COUNT * TOKEN_COUNT;
  if (item >= total_items)
    return;
  const int input_base = item * SENSOR_LANE_COUNT;
  const int output_base = item * FEATURE_LANE_COUNT;

  const float x = input[input_base + 0];
  const float y = input[input_base + 1];
  const float z = input[input_base + 2];
  output[output_base + 0] = x;
  output[output_base + 1] = y;
  output[output_base + 2] = z;
  output[output_base + 3] = x + y;
}

__global__ void project_imu_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = BATCH_COUNT * CONTEXT_COUNT;
  if (item >= total_items)
    return;
  const int input_base = item * SENSOR_LANE_COUNT;
  const int output_base = item * FEATURE_LANE_COUNT;

  const float x = input[input_base + 0];
  const float y = input[input_base + 1];
  const float z = input[input_base + 2];
  output[output_base + 0] = x;
  output[output_base + 1] = y;
  output[output_base + 2] = z;
  output[output_base + 3] = x - y;
}

// 2^x on the SFU. Same instruction and accuracy as exp2f for normal results;
// exp2f adds a denormal-result fixup (3 extra instructions per call). Every
// argument here is score - running maximum <= 0, so a flushed result is below
// 2^-126 while the denominator is >= 1: it cannot change any float sum.
__device__ __forceinline__ float exp2_flush_denormal(float x) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

// Running softmax state for one query row, in the log2 domain:
//   maximum     = max score seen so far
//   denominator = sum of exp2(score - maximum)
//   value       = sum of exp2(score - maximum) * key_value
struct SoftmaxState {
  float maximum;
  float denominator;
  float4 value;
};

#if ATTENTION_CONTEXT_SPLITS > 1
// Per-split partial softmax state, indexed [split][batch * TOKEN_COUNT + token].
__device__ float4 g_partial_value[ATTENTION_CONTEXT_SPLITS * BATCH_COUNT *
                                  TOKEN_COUNT];
__device__ float2 g_partial_softmax[ATTENTION_CONTEXT_SPLITS * BATCH_COUNT *
                                    TOKEN_COUNT];  // (maximum, denominator)
#endif

// (m, l, acc) combination:
//   m = max(m_a, m_b); l = l_a*2^(m_a-m) + l_b*2^(m_b-m); acc likewise.
__device__ __forceinline__ void merge_softmax_state(
    SoftmaxState& state, float other_maximum, float other_denominator,
    float4 other_value) {
  const float maximum = fmaxf(state.maximum, other_maximum);
  if (maximum == -INFINITY)
    return;  // Both sides are empty.
  const float scale = exp2_flush_denormal(state.maximum - maximum);
  const float other_scale = exp2_flush_denormal(other_maximum - maximum);
  state.denominator = state.denominator * scale + other_denominator * other_scale;
  state.value.x = state.value.x * scale + other_value.x * other_scale;
  state.value.y = state.value.y * scale + other_value.y * other_scale;
  state.value.z = state.value.z * scale + other_value.z * other_scale;
  state.value.w = state.value.w * scale + other_value.w * other_scale;
  state.maximum = maximum;
}

// Writes a finished row: the normalized output, or the raw partial state when
// the context is split across blocks.
__device__ __forceinline__ void store_attention_row(
    const SoftmaxState& state, int row, int split, float4* output) {
#if ATTENTION_CONTEXT_SPLITS > 1
  const int index = split * BATCH_COUNT * TOKEN_COUNT + row;
  g_partial_value[index] = state.value;
  g_partial_softmax[index] = make_float2(state.maximum, state.denominator);
  (void)output;
#else
  (void)split;
  const float denominator = state.denominator;
  output[row] = make_float4(
      state.value.x / denominator, state.value.y / denominator,
      state.value.z / denominator, state.value.w / denominator);
#endif
}

// Block layout: QUERIES_PER_BLOCK query rows split into groups of
// QUERIES_PER_THREAD; each group is a set of whole warps whose threads stride
// over the rows of the shared context tile. Every context row is loaded from
// global memory once per block and applied to all of the block's queries.
//
// Within a tile each thread first computes its scores and the tile maximum,
// rescales its running state at most once, then accumulates with exactly one
// exp2 per (query, key) pair.
template <int QUERIES_PER_BLOCK, int QUERIES_PER_THREAD, int THREADS,
          int TILE_ROWS>
__global__ void __launch_bounds__(THREADS) fuse_attention_kernel(
    const float4* __restrict__ query, const float4* __restrict__ key_value,
    float4* __restrict__ output) {
  constexpr int kGroups = QUERIES_PER_BLOCK / QUERIES_PER_THREAD;
  constexpr int kGroupThreads = THREADS / kGroups;
  constexpr int kGroupWarps = kGroupThreads / kWarpSize;
  constexpr int kRowsPerThread = TILE_ROWS / kGroupThreads;
  constexpr int kBlocksPerBatch =
      (TOKEN_COUNT + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
  static_assert(QUERIES_PER_BLOCK % QUERIES_PER_THREAD == 0,
                "queries per block must be a multiple of queries per thread");
  static_assert(THREADS % kGroups == 0 && kGroupThreads % kWarpSize == 0,
                "each query group must be made of whole warps");
  static_assert(TILE_ROWS % kGroupThreads == 0,
                "tile rows must divide evenly across a query group");

  __shared__ float4 key_tile[2][TILE_ROWS];
  __shared__ float merge_maximum[QUERIES_PER_BLOCK][kGroupWarps];
  __shared__ float merge_denominator[QUERIES_PER_BLOCK][kGroupWarps];
  __shared__ float4 merge_value[QUERIES_PER_BLOCK][kGroupWarps];

  constexpr int kSliceRows =
      (CONTEXT_COUNT + ATTENTION_CONTEXT_SPLITS - 1) / ATTENTION_CONTEXT_SPLITS;
  const int split = blockIdx.y;
  const int slice_begin = split * kSliceRows;
  const int slice_end = min(CONTEXT_COUNT, slice_begin + kSliceRows);
  const int batch = blockIdx.x / kBlocksPerBatch;
  const int block_query = (blockIdx.x % kBlocksPerBatch) * QUERIES_PER_BLOCK;
  const int group = threadIdx.x / kGroupThreads;
  const int group_thread = threadIdx.x % kGroupThreads;
  const int first_query = block_query + group * QUERIES_PER_THREAD;
  const float4* context = key_value + batch * CONTEXT_COUNT;

  float4 scaled_query[QUERIES_PER_THREAD];
  SoftmaxState state[QUERIES_PER_THREAD];
#pragma unroll
  for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
    const int token = first_query + q;
    const float4 value = token < TOKEN_COUNT
                             ? query[batch * TOKEN_COUNT + token]
                             : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    scaled_query[q] =
        make_float4(value.x * kScoreScaleLog2, value.y * kScoreScaleLog2,
                    value.z * kScoreScaleLog2, value.w * kScoreScaleLog2);
    state[q].maximum = -INFINITY;
    state[q].denominator = 0.0f;
    state[q].value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  // Double-buffered context tiles: tile t+1 streams in with cp.async while
  // tile t is consumed.
  auto load_tile = [&](int tile_start, int buffer) {
    const int tile_rows = min(TILE_ROWS, slice_end - tile_start);
    for (int row = threadIdx.x; row < tile_rows; row += THREADS)
      __pipeline_memcpy_async(&key_tile[buffer][row],
                              &context[tile_start + row], sizeof(float4));
    __pipeline_commit();
  };
  if (slice_begin < slice_end)
    load_tile(slice_begin, 0);

  int buffer = 0;
  for (int tile_start = slice_begin; tile_start < slice_end;
       tile_start += TILE_ROWS) {
    const int tile_rows = min(TILE_ROWS, slice_end - tile_start);
    if (tile_start + TILE_ROWS < slice_end) {
      load_tile(tile_start + TILE_ROWS, buffer ^ 1);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();  // Tile `buffer` is visible to every thread.
    const float4* tile = key_tile[buffer];

    float score[kRowsPerThread][QUERIES_PER_THREAD];
    float tile_maximum[QUERIES_PER_THREAD];
#pragma unroll
    for (int q = 0; q < QUERIES_PER_THREAD; ++q)
      tile_maximum[q] = -INFINITY;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const int row = group_thread + r * kGroupThreads;
      if (row < tile_rows) {
        const float4 key = tile[row];
#pragma unroll
        for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
          const float4 v = scaled_query[q];
          score[r][q] = v.x * key.x + v.y * key.y + v.z * key.z + v.w * key.w;
          tile_maximum[q] = fmaxf(tile_maximum[q], score[r][q]);
        }
      }
    }

#pragma unroll
    for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
      if (tile_maximum[q] > state[q].maximum) {
        const float scale = exp2_flush_denormal(state[q].maximum - tile_maximum[q]);
        state[q].denominator *= scale;
        state[q].value.x *= scale;
        state[q].value.y *= scale;
        state[q].value.z *= scale;
        state[q].value.w *= scale;
        state[q].maximum = tile_maximum[q];
      }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const int row = group_thread + r * kGroupThreads;
      if (row < tile_rows) {
        const float4 key = tile[row];
#pragma unroll
        for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
          const float probability = exp2_flush_denormal(score[r][q] - state[q].maximum);
          state[q].denominator += probability;
          state[q].value.x += probability * key.x;
          state[q].value.y += probability * key.y;
          state[q].value.z += probability * key.z;
          state[q].value.w += probability * key.w;
        }
      }
    }
    __syncthreads();  // Tile `buffer` fully consumed before it is refilled.
    buffer ^= 1;
  }

  // Merge the group's per-thread states: warp butterfly, then across warps.
#pragma unroll
  for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      const float other_maximum =
          __shfl_xor_sync(kFullWarpMask, state[q].maximum, offset);
      const float other_denominator =
          __shfl_xor_sync(kFullWarpMask, state[q].denominator, offset);
      float4 other_value;
      other_value.x = __shfl_xor_sync(kFullWarpMask, state[q].value.x, offset);
      other_value.y = __shfl_xor_sync(kFullWarpMask, state[q].value.y, offset);
      other_value.z = __shfl_xor_sync(kFullWarpMask, state[q].value.z, offset);
      other_value.w = __shfl_xor_sync(kFullWarpMask, state[q].value.w, offset);
      merge_softmax_state(state[q], other_maximum, other_denominator,
                          other_value);
    }
  }

  const int lane = threadIdx.x % kWarpSize;
  if constexpr (kGroupWarps == 1) {
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
        const int token = first_query + q;
        if (token < TOKEN_COUNT)
          store_attention_row(state[q], batch * TOKEN_COUNT + token, split,
                              output);
      }
    }
  } else {
    const int group_warp = group_thread / kWarpSize;
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
        const int local_query = group * QUERIES_PER_THREAD + q;
        merge_maximum[local_query][group_warp] = state[q].maximum;
        merge_denominator[local_query][group_warp] = state[q].denominator;
        merge_value[local_query][group_warp] = state[q].value;
      }
    }
    __syncthreads();
    if (threadIdx.x < QUERIES_PER_BLOCK) {
      const int local_query = threadIdx.x;
      const int token = block_query + local_query;
      SoftmaxState merged{merge_maximum[local_query][0],
                          merge_denominator[local_query][0],
                          merge_value[local_query][0]};
#pragma unroll
      for (int w = 1; w < kGroupWarps; ++w)
        merge_softmax_state(merged, merge_maximum[local_query][w],
                            merge_denominator[local_query][w],
                            merge_value[local_query][w]);
      if (token < TOKEN_COUNT)
        store_attention_row(merged, batch * TOKEN_COUNT + token, split,
                            output);
    }
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  return value;
}

__device__ __forceinline__ float warp_maximum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const float other = __shfl_xor_sync(kFullWarpMask, value, offset);
    if (other > value)
      value = other;
  }
  return value;
}

// One block per batch item; each thread handles whole tokens (one float4 of
// projection and attention), then a fixed-order warp/block reduction. With a
// split context, attention rows are first merged from their partials.
__global__ void __launch_bounds__(kStatisticsThreads) reduce_statistics_kernel(
    const float4* __restrict__ projected_audio,
    const float4* __restrict__ attention, float4* __restrict__ output,
    float* __restrict__ statistics) {
  constexpr int kWarps = kStatisticsThreads / kWarpSize;
  __shared__ float partials[STATISTIC_COUNT][kWarps];

  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  const int base = batch * TOKEN_COUNT;

  float projection_energy = 0.0f;
  float attention_sum = 0.0f;
  float weighted_sum = 0.0f;
  float maximum_magnitude = 0.0f;
  for (int token = tid; token < TOKEN_COUNT; token += kStatisticsThreads) {
    const float4 p = projected_audio[base + token];
#if ATTENTION_CONTEXT_SPLITS > 1
    SoftmaxState merged{-INFINITY, 0.0f, make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    for (int split = 0; split < ATTENTION_CONTEXT_SPLITS; ++split) {
      const int index = split * BATCH_COUNT * TOKEN_COUNT + base + token;
      const float2 softmax = g_partial_softmax[index];
      merge_softmax_state(merged, softmax.x, softmax.y,
                          g_partial_value[index]);
    }
    const float4 a = make_float4(
        merged.value.x / merged.denominator, merged.value.y / merged.denominator,
        merged.value.z / merged.denominator, merged.value.w / merged.denominator);
    (void)attention;
#else
    const float4 a = attention[base + token];
#endif
    const float projection[FEATURE_LANE_COUNT] = {p.x, p.y, p.z, p.w};
    const float value[FEATURE_LANE_COUNT] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      const float magnitude = value[lane] < 0.0f ? -value[lane] : value[lane];
      projection_energy += projection[lane] * projection[lane];
      attention_sum += value[lane];
      weighted_sum +=
          static_cast<float>(token * FEATURE_LANE_COUNT + lane + 1) *
          value[lane];
      if (magnitude > maximum_magnitude)
        maximum_magnitude = magnitude;
    }
    output[base + token] =
        make_float4(a.x + p.x * 0.125f, a.y + p.y * 0.125f,
                    a.z + p.z * 0.125f, a.w + p.w * 0.125f);
  }

  projection_energy = warp_sum(projection_energy);
  attention_sum = warp_sum(attention_sum);
  weighted_sum = warp_sum(weighted_sum);
  maximum_magnitude = warp_maximum(maximum_magnitude);
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  if (lane == 0) {
    partials[0][warp] = projection_energy;
    partials[1][warp] = attention_sum;
    partials[2][warp] = weighted_sum;
    partials[3][warp] = maximum_magnitude;
  }
  __syncthreads();

  if (warp == 0) {
    // Magnitudes are nonnegative, so 0 is the identity for the maximum too.
    projection_energy = warp_sum(lane < kWarps ? partials[0][lane] : 0.0f);
    attention_sum = warp_sum(lane < kWarps ? partials[1][lane] : 0.0f);
    weighted_sum = warp_sum(lane < kWarps ? partials[2][lane] : 0.0f);
    maximum_magnitude =
        warp_maximum(lane < kWarps ? partials[3][lane] : 0.0f);
    if (lane == 0) {
      statistics[batch * STATISTIC_COUNT + 0] = projection_energy;
      statistics[batch * STATISTIC_COUNT + 1] = attention_sum;
      statistics[batch * STATISTIC_COUNT + 2] = weighted_sum;
      statistics[batch * STATISTIC_COUNT + 3] = maximum_magnitude;
    }
  }
}

// Same graph topology as the baseline; only the attention and statistics
// kernels differ.
static int enqueue_pipeline(const PipelineBuffers& b,
                            cudaStream_t audio_stream,
                            cudaStream_t imu_stream, cudaEvent_t imu_ready) {
  CUDA_CHECK(cudaMemcpyAsync(b.d_audio, b.h_audio,
                             kQuerySensorElements * sizeof(float),
                             cudaMemcpyHostToDevice, audio_stream));
  constexpr int projection_threads = 128;
  const int audio_projection_blocks =
      (BATCH_COUNT * TOKEN_COUNT + projection_threads - 1) /
      projection_threads;
  const int imu_projection_blocks =
      (BATCH_COUNT * CONTEXT_COUNT + projection_threads - 1) /
      projection_threads;
  project_audio_kernel<<<audio_projection_blocks, projection_threads, 0,
                         audio_stream>>>(b.d_audio, b.d_projected_audio);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(b.d_imu, b.h_imu,
                             kContextSensorElements * sizeof(float),
                             cudaMemcpyHostToDevice, imu_stream));
  project_imu_kernel<<<imu_projection_blocks, projection_threads, 0,
                       imu_stream>>>(b.d_imu, b.d_projected_imu);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(imu_ready, imu_stream));
  CUDA_CHECK(cudaStreamWaitEvent(audio_stream, imu_ready, 0));
  constexpr int attention_blocks =
      BATCH_COUNT * ((TOKEN_COUNT + ATTENTION_QUERIES_PER_BLOCK - 1) /
                     ATTENTION_QUERIES_PER_BLOCK);
  fuse_attention_kernel<ATTENTION_QUERIES_PER_BLOCK,
                        ATTENTION_QUERIES_PER_THREAD, ATTENTION_THREADS,
                        ATTENTION_TILE_ROWS>
      <<<dim3(attention_blocks, ATTENTION_CONTEXT_SPLITS), ATTENTION_THREADS,
         0, audio_stream>>>(
          reinterpret_cast<const float4*>(b.d_projected_audio),
          reinterpret_cast<const float4*>(b.d_projected_imu),
          reinterpret_cast<float4*>(b.d_attention));
  CUDA_CHECK(cudaGetLastError());
  reduce_statistics_kernel<<<BATCH_COUNT, kStatisticsThreads, 0,
                             audio_stream>>>(
      reinterpret_cast<const float4*>(b.d_projected_audio),
      reinterpret_cast<const float4*>(b.d_attention),
      reinterpret_cast<float4*>(b.d_output), b.d_statistics);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(b.h_output, b.d_output,
                             kFeatureElements * sizeof(float),
                             cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaMemcpyAsync(b.h_statistics, b.d_statistics,
                             kStatisticsElements * sizeof(float),
                             cudaMemcpyDeviceToHost, audio_stream));
  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  return run_benchmark(argc, argv, "cuda_optimized", enqueue_pipeline);
}
