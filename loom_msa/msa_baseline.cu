// Frozen CUDA baseline. The kernels below are copied verbatim from
// multi_sensor_attention_graph.cu; only the host harness is shared
// (msa_harness.cuh) so both variants are measured identically.
#include <cfloat>

#include "msa_harness.cuh"

enum {
  REDUCTION_THREADS = 32,
  ATTENTION_THREADS = 256,
};

__global__ void project_audio_kernel(
    const float* __restrict__ input, float* __restrict__ output) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_items = BATCH_COUNT * TOKEN_COUNT;
  if (item >= total_items)
    return;
  const int batch = item / TOKEN_COUNT;
  const int token = item % TOKEN_COUNT;
  const int input_base = item * SENSOR_LANE_COUNT;
  const int output_base = (batch * TOKEN_COUNT + token) * FEATURE_LANE_COUNT;

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
  const int batch = item / CONTEXT_COUNT;
  const int token = item % CONTEXT_COUNT;
  const int input_base = item * SENSOR_LANE_COUNT;
  const int output_base =
      (batch * CONTEXT_COUNT + token) * FEATURE_LANE_COUNT;

  const float x = input[input_base + 0];
  const float y = input[input_base + 1];
  const float z = input[input_base + 2];
  output[output_base + 0] = x;
  output[output_base + 1] = y;
  output[output_base + 2] = z;
  output[output_base + 3] = x - y;
}

// One block handles one query row. Threads cooperate over the full context
// length for the max, softmax denominator, and weighted value reductions.
__global__ void fuse_attention_kernel(
    const float* __restrict__ query, const float* __restrict__ key_value,
    float* __restrict__ output) {
  __shared__ float query_shared[FEATURE_LANE_COUNT];
  __shared__ float max_partials[ATTENTION_THREADS];
  __shared__ float sum_partials[ATTENTION_THREADS];
  __shared__ float value_partials[FEATURE_LANE_COUNT][ATTENTION_THREADS];

  const int query_index = blockIdx.x;
  const int batch = query_index / TOKEN_COUNT;
  const int tid = threadIdx.x;
  const int query_base = query_index * FEATURE_LANE_COUNT;
  const int context_base = batch * CONTEXT_COUNT * FEATURE_LANE_COUNT;

  if (tid < FEATURE_LANE_COUNT)
    query_shared[tid] = query[query_base + tid];
  __syncthreads();

  float maximum = -FLT_MAX;
  for (int column = tid; column < CONTEXT_COUNT;
       column += ATTENTION_THREADS) {
    float dot = 0.0f;
    const float4 key = reinterpret_cast<const float4*>(key_value)[
        context_base / FEATURE_LANE_COUNT + column];
    dot += query_shared[0] * key.x + query_shared[1] * key.y +
           query_shared[2] * key.z + query_shared[3] * key.w;
    maximum = fmaxf(maximum, dot * 0.5f);
  }
  max_partials[tid] = maximum;
  __syncthreads();
  for (int stride = ATTENTION_THREADS / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      max_partials[tid] = fmaxf(max_partials[tid], max_partials[tid + stride]);
    __syncthreads();
  }
  const float row_maximum = max_partials[0];

  float denominator = 0.0f;
  for (int column = tid; column < CONTEXT_COUNT;
       column += ATTENTION_THREADS) {
    float dot = 0.0f;
    const float4 key = reinterpret_cast<const float4*>(key_value)[
        context_base / FEATURE_LANE_COUNT + column];
    dot += query_shared[0] * key.x + query_shared[1] * key.y +
           query_shared[2] * key.z + query_shared[3] * key.w;
    denominator += expf(dot * 0.5f - row_maximum);
  }
  sum_partials[tid] = denominator;
  __syncthreads();
  for (int stride = ATTENTION_THREADS / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      sum_partials[tid] += sum_partials[tid + stride];
    __syncthreads();
  }
  const float row_denominator = sum_partials[0];

  for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
    float value = 0.0f;
    for (int column = tid; column < CONTEXT_COUNT;
         column += ATTENTION_THREADS) {
      float dot = 0.0f;
      const float4 key = reinterpret_cast<const float4*>(key_value)[
          context_base / FEATURE_LANE_COUNT + column];
      dot += query_shared[0] * key.x + query_shared[1] * key.y +
             query_shared[2] * key.z + query_shared[3] * key.w;
      const float* key_lanes = reinterpret_cast<const float*>(&key);
      value += expf(dot * 0.5f - row_maximum) * key_lanes[lane];
    }
    value_partials[lane][tid] = value;
  }
  __syncthreads();
  for (int stride = ATTENTION_THREADS / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
        value_partials[lane][tid] += value_partials[lane][tid + stride];
    }
    __syncthreads();
  }
  if (tid < FEATURE_LANE_COUNT)
    output[query_base + tid] = value_partials[tid][0] / row_denominator;
}

__global__ void reduce_statistics_kernel(
    const float* __restrict__ projected_audio,
    const float* __restrict__ attention, float* __restrict__ output,
    float* __restrict__ statistics) {
  if (blockIdx.x >= BATCH_COUNT)
    return;

  const int batch = blockIdx.x;
  const int tid = threadIdx.x;
  const int base = batch * TOKEN_COUNT * FEATURE_LANE_COUNT;
  const int statistics_base = batch * STATISTIC_COUNT;
  const int element_count = TOKEN_COUNT * FEATURE_LANE_COUNT;

  __shared__ float energy_partials[REDUCTION_THREADS];
  __shared__ float sum_partials[REDUCTION_THREADS];
  __shared__ float weighted_partials[REDUCTION_THREADS];
  __shared__ float maximum_partials[REDUCTION_THREADS];

  // Each thread accumulates a strided subset of this batch item's values. This
  // remains correct if TOKEN_COUNT grows beyond REDUCTION_THREADS.
  float projection_energy = 0.0f;
  float attention_sum = 0.0f;
  float weighted_sum = 0.0f;
  float maximum_magnitude = 0.0f;

  for (int index = tid; index < element_count; index += REDUCTION_THREADS) {
    const int global_index = base + index;
    const float projection = projected_audio[global_index];
    const float value = attention[global_index];
    const float magnitude = value < 0.0f ? -value : value;
    projection_energy += projection * projection;
    attention_sum += value;
    weighted_sum += static_cast<float>(index + 1) * value;
    if (magnitude > maximum_magnitude)
      maximum_magnitude = magnitude;
    output[global_index] = value + projection * 0.125f;
  }

  energy_partials[tid] = projection_energy;
  sum_partials[tid] = attention_sum;
  weighted_partials[tid] = weighted_sum;
  maximum_partials[tid] = maximum_magnitude;
  __syncthreads();

  for (int stride = REDUCTION_THREADS / 2; stride > 0; stride /= 2) {
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
    statistics[statistics_base + 0] = energy_partials[0];
    statistics[statistics_base + 1] = sum_partials[0];
    statistics[statistics_base + 2] = weighted_partials[0];
    statistics[statistics_base + 3] = maximum_partials[0];
  }
}

// Same stream topology as the original capture: two sensor branches joined
// before attention.
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
  fuse_attention_kernel<<<BATCH_COUNT * TOKEN_COUNT, ATTENTION_THREADS, 0,
                          audio_stream>>>(
      b.d_projected_audio, b.d_projected_imu, b.d_attention);
  CUDA_CHECK(cudaGetLastError());
  reduce_statistics_kernel<<<BATCH_COUNT, REDUCTION_THREADS, 0,
                             audio_stream>>>(
      b.d_projected_audio, b.d_attention, b.d_output, b.d_statistics);
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
  return run_benchmark(argc, argv, "cuda_baseline", enqueue_pipeline);
}
