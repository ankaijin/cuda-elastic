#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iostream>

enum {
  // This benchmark uses 1,000 query tokens and a 50,000-token key/value
  // context. The graph is specialized to these dimensions.
  BATCH_COUNT = 1,
  TOKEN_COUNT = 1000,
  CONTEXT_COUNT = 50000,
  SENSOR_LANE_COUNT = 3,
  FEATURE_LANE_COUNT = 4,
  STATISTIC_COUNT = 4,
  REDUCTION_THREADS = 32,
  ATTENTION_THREADS = 256,
  GRAPH_REPLAYS = 5000,
};

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__)             \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';           \
      return EXIT_FAILURE;                                                    \
    }                                                                         \
  } while (0)

__global__ void project_audio_kernel(
    const float* input, float* output) {
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
    const float* input, float* output) {
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
    const float* query, const float* key_value, float* output) {
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
    const int key_base = context_base + column * FEATURE_LANE_COUNT;
    for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
      dot += query_shared[lane] * key_value[key_base + lane];
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
    const int key_base = context_base + column * FEATURE_LANE_COUNT;
    for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
      dot += query_shared[lane] * key_value[key_base + lane];
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
      const int key_base = context_base + column * FEATURE_LANE_COUNT;
      for (int feature = 0; feature < FEATURE_LANE_COUNT; ++feature)
        dot += query_shared[feature] * key_value[key_base + feature];
      value += expf(dot * 0.5f - row_maximum) * key_value[key_base + lane];
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
    const float* projected_audio, const float* attention, float* output,
    float* statistics) {
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

int main() {
  const float audio_seed[4][SENSOR_LANE_COUNT] = {
      {1.0f, 0.0f, 0.5f},
      {0.5f, 1.0f, -0.5f},
      {-1.0f, 0.5f, 1.0f},
      {0.25f, -0.75f, 0.5f},
  };
  const float imu_seed[4][SENSOR_LANE_COUNT] = {
      {0.5f, 1.0f, 0.0f},
      {-0.5f, 0.25f, 1.0f},
      {1.0f, -0.5f, 0.5f},
      {0.0f, 0.75f, -1.0f},
  };

  const size_t sensor_elements =
      static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * SENSOR_LANE_COUNT;
  const size_t query_sensor_elements =
      static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * SENSOR_LANE_COUNT;
  const size_t feature_elements =
      static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * FEATURE_LANE_COUNT;
  const size_t context_feature_elements =
      static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * FEATURE_LANE_COUNT;
  const size_t statistics_elements =
      static_cast<size_t>(BATCH_COUNT) * STATISTIC_COUNT;
  const size_t sensor_bytes = sensor_elements * sizeof(float);
  const size_t query_sensor_bytes =
      query_sensor_elements * sizeof(float);
  const size_t feature_bytes = feature_elements * sizeof(float);
  const size_t context_feature_bytes =
      context_feature_elements * sizeof(float);
  const size_t statistics_bytes = statistics_elements * sizeof(float);

  // host memory allocation
  float* h_audio = nullptr;
  float* h_imu = nullptr;
  float* h_output = nullptr;
  float* h_statistics = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_audio, query_sensor_bytes));
  CUDA_CHECK(cudaMallocHost(&h_imu, sensor_bytes));
  CUDA_CHECK(cudaMallocHost(&h_output, feature_bytes));
  CUDA_CHECK(cudaMallocHost(&h_statistics, statistics_bytes));
  for (int batch = 0; batch < BATCH_COUNT; ++batch) {
    for (int token = 0; token < TOKEN_COUNT; ++token)
      std::memcpy(h_audio +
                      (batch * TOKEN_COUNT + token) * SENSOR_LANE_COUNT,
                  audio_seed[token % 4], sizeof(audio_seed[0]));
    for (int token = 0; token < CONTEXT_COUNT; ++token)
      std::memcpy(h_imu +
                      (batch * CONTEXT_COUNT + token) * SENSOR_LANE_COUNT,
                  imu_seed[token % 4], sizeof(imu_seed[0]));
  }

  // device memory allocation
  float* d_audio = nullptr;
  float* d_imu = nullptr;
  float* d_projected_audio = nullptr;
  float* d_projected_imu = nullptr;
  float* d_attention = nullptr;
  float* d_output = nullptr;
  float* d_statistics = nullptr;
  CUDA_CHECK(cudaMalloc(&d_audio, query_sensor_bytes));
  CUDA_CHECK(cudaMalloc(&d_imu, sensor_bytes));
  CUDA_CHECK(cudaMalloc(&d_projected_audio, feature_bytes));
  CUDA_CHECK(cudaMalloc(&d_projected_imu, context_feature_bytes));
  CUDA_CHECK(cudaMalloc(&d_attention, feature_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, feature_bytes));
  CUDA_CHECK(cudaMalloc(&d_statistics, statistics_bytes));

  cudaStream_t audio_stream = nullptr;
  cudaStream_t imu_stream = nullptr;
  cudaEvent_t capture_start = nullptr;
  cudaEvent_t imu_ready = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&audio_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&imu_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&imu_ready, cudaEventDisableTiming));

  // Capture two independent sensor branches, then join them before attention.
  // audio_stream is the capture origin; the event wait joins imu_stream to the
  // same capture graph without imposing a dependency on the audio work.
  CUDA_CHECK(cudaStreamBeginCapture(audio_stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaEventRecord(capture_start, audio_stream));
  CUDA_CHECK(cudaStreamWaitEvent(imu_stream, capture_start, 0));

  CUDA_CHECK(cudaMemcpyAsync(d_audio, h_audio, query_sensor_bytes,
                             cudaMemcpyHostToDevice, audio_stream));
  constexpr int projection_threads = 128;
  const int audio_projection_blocks =
      (BATCH_COUNT * TOKEN_COUNT + projection_threads - 1) /
      projection_threads;
  const int imu_projection_blocks =
      (BATCH_COUNT * CONTEXT_COUNT + projection_threads - 1) /
      projection_threads;
  project_audio_kernel<<<audio_projection_blocks, projection_threads, 0,
                         audio_stream>>>(
      d_audio, d_projected_audio);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(d_imu, h_imu, sensor_bytes,
                             cudaMemcpyHostToDevice, imu_stream));
  project_imu_kernel<<<imu_projection_blocks, projection_threads, 0, imu_stream>>>(
      d_imu, d_projected_imu);
  CUDA_CHECK(cudaGetLastError());

  // The wait on the origin stream makes attention depend on both branches:
  // audio_stream is already ordered after project_audio, and imu_ready is
  // recorded only after project_imu has completed.
  CUDA_CHECK(cudaEventRecord(imu_ready, imu_stream));
  CUDA_CHECK(cudaStreamWaitEvent(audio_stream, imu_ready, 0));
  fuse_attention_kernel<<<BATCH_COUNT * TOKEN_COUNT, ATTENTION_THREADS, 0,
                         audio_stream>>>(
      d_projected_audio, d_projected_imu, d_attention);
  CUDA_CHECK(cudaGetLastError());
  reduce_statistics_kernel<<<BATCH_COUNT, REDUCTION_THREADS, 0, audio_stream>>>(
      d_projected_audio, d_attention, d_output, d_statistics);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(
      h_output, d_output,
      feature_bytes,
      cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaMemcpyAsync(h_statistics, d_statistics, statistics_bytes,
                             cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaStreamEndCapture(audio_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, audio_stream));
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));

  for (int replay = 0; replay < GRAPH_REPLAYS; ++replay) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, audio_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));

  for (size_t index = 0; index < feature_elements; ++index) {
    if (!std::isfinite(h_output[index])) {
      std::cerr << "Verification failed: non-finite output at index "
                << index << '\n';
      return EXIT_FAILURE;
    }
  }
  for (size_t index = 0; index < statistics_elements; ++index) {
    if (!std::isfinite(h_statistics[index])) {
      std::cerr << "Verification failed: non-finite statistic at index "
                << index << '\n';
      return EXIT_FAILURE;
    }
  }

  const float* statistics = h_statistics;
  const float combined = statistics[2] + 3.0f * statistics[0] +
                         7.0f * statistics[1] + 11.0f * statistics[3];

  std::cout << "attention checksum: " << h_statistics[2] << '\n'
            << "projection energy: " << h_statistics[0] << '\n'
            << "attention sum: " << h_statistics[1] << '\n'
            << "attention max: " << h_statistics[3] << '\n'
            << "combined checksum: " << combined << '\n'
            << "PASSED\n";

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(imu_ready));
  CUDA_CHECK(cudaEventDestroy(capture_start));
  CUDA_CHECK(cudaStreamDestroy(imu_stream));
  CUDA_CHECK(cudaStreamDestroy(audio_stream));
  CUDA_CHECK(cudaFree(d_statistics));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_attention));
  CUDA_CHECK(cudaFree(d_projected_imu));
  CUDA_CHECK(cudaFree(d_projected_audio));
  CUDA_CHECK(cudaFree(d_imu));
  CUDA_CHECK(cudaFree(d_audio));
  CUDA_CHECK(cudaFreeHost(h_statistics));
  CUDA_CHECK(cudaFreeHost(h_output));
  CUDA_CHECK(cudaFreeHost(h_imu));
  CUDA_CHECK(cudaFreeHost(h_audio));
  return EXIT_SUCCESS;
}
