#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

enum {
  TOKEN_COUNT = 4,
  SENSOR_LANE_COUNT = 3,
  FEATURE_LANE_COUNT = 4,
  STATISTIC_COUNT = 4,
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
  const int token = threadIdx.x;
  if (token >= TOKEN_COUNT)
    return;

  const float x = input[token * SENSOR_LANE_COUNT + 0];
  const float y = input[token * SENSOR_LANE_COUNT + 1];
  const float z = input[token * SENSOR_LANE_COUNT + 2];
  output[token * FEATURE_LANE_COUNT + 0] = x;
  output[token * FEATURE_LANE_COUNT + 1] = y;
  output[token * FEATURE_LANE_COUNT + 2] = z;
  output[token * FEATURE_LANE_COUNT + 3] = x + y;
}

__global__ void project_imu_kernel(
    const float* input, float* output) {
  const int token = threadIdx.x;
  if (token >= TOKEN_COUNT)
    return;

  const float x = input[token * SENSOR_LANE_COUNT + 0];
  const float y = input[token * SENSOR_LANE_COUNT + 1];
  const float z = input[token * SENSOR_LANE_COUNT + 2];
  output[token * FEATURE_LANE_COUNT + 0] = x;
  output[token * FEATURE_LANE_COUNT + 1] = y;
  output[token * FEATURE_LANE_COUNT + 2] = z;
  output[token * FEATURE_LANE_COUNT + 3] = x - y;
}

// One block is sufficient for this deliberately small fixed-size attention
// problem. Each thread owns one query row and writes that row's output.
__global__ void fuse_attention_kernel(
    const float* query, const float* key_value, float* output) {
  __shared__ float query_shared[TOKEN_COUNT][FEATURE_LANE_COUNT];
  __shared__ float key_value_shared[TOKEN_COUNT][FEATURE_LANE_COUNT];

  const int row = threadIdx.x;
  if (row >= TOKEN_COUNT)
    return;

  for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
    query_shared[row][lane] = query[row * FEATURE_LANE_COUNT + lane];
    key_value_shared[row][lane] =
        key_value[row * FEATURE_LANE_COUNT + lane];
  }
  __syncthreads();

  float scores[TOKEN_COUNT];
  float probabilities[TOKEN_COUNT];
  float maximum = 0.0f;

  for (int column = 0; column < TOKEN_COUNT; ++column) {
    float dot = 0.0f;
    for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
      dot += query_shared[row][lane] * key_value_shared[column][lane];
    scores[column] = dot * 0.5f;
    if (column == 0 || scores[column] > maximum)
      maximum = scores[column];
  }

  float denominator = 0.0f;
  for (int column = 0; column < TOKEN_COUNT; ++column) {
    probabilities[column] = expf(scores[column] - maximum);
    denominator += probabilities[column];
  }

  for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
    float value = 0.0f;
    for (int column = 0; column < TOKEN_COUNT; ++column)
      value += probabilities[column] * key_value_shared[column][lane];
    output[row * FEATURE_LANE_COUNT + lane] = value / denominator;
  }
}

__global__ void reduce_statistics_kernel(
    const float* projected_audio, const float* attention, float* output,
    float* statistics) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  float projection_energy = 0.0f;
  float attention_sum = 0.0f;
  float weighted_sum = 0.0f;
  float maximum_magnitude = 0.0f;

  for (int token = 0; token < TOKEN_COUNT; ++token) {
    for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      const int index = token * FEATURE_LANE_COUNT + lane;
      const float projection = projected_audio[index];
      const float value = attention[index];
      const float magnitude = value < 0.0f ? -value : value;
      projection_energy += projection * projection;
      attention_sum += value;
      weighted_sum += static_cast<float>(index + 1) * value;
      if (magnitude > maximum_magnitude)
        maximum_magnitude = magnitude;
      output[index] = value + projection * 0.125f;
    }
  }

  statistics[0] = projection_energy;
  statistics[1] = attention_sum;
  statistics[2] = weighted_sum;
  statistics[3] = maximum_magnitude;
}

static bool outside_tolerance(float actual, float expected) {
  return std::fabs(actual - expected) > 1.0e-4f;
}

int main() {
  const float audio[TOKEN_COUNT][SENSOR_LANE_COUNT] = {
      {1.0f, 0.0f, 0.5f},
      {0.5f, 1.0f, -0.5f},
      {-1.0f, 0.5f, 1.0f},
      {0.25f, -0.75f, 0.5f},
  };
  const float imu[TOKEN_COUNT][SENSOR_LANE_COUNT] = {
      {0.5f, 1.0f, 0.0f},
      {-0.5f, 0.25f, 1.0f},
      {1.0f, -0.5f, 0.5f},
      {0.0f, 0.75f, -1.0f},
  };

  // host memory allocation
  float* h_audio = nullptr;
  float* h_imu = nullptr;
  float* h_output = nullptr;
  float* h_statistics = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_audio, sizeof(audio)));
  CUDA_CHECK(cudaMallocHost(&h_imu, sizeof(imu)));
  CUDA_CHECK(cudaMallocHost(
      &h_output, TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_statistics, STATISTIC_COUNT * sizeof(float)));
  std::memcpy(h_audio, audio, sizeof(audio));
  std::memcpy(h_imu, imu, sizeof(imu));

  // device memory allocation
  float* d_audio = nullptr;
  float* d_imu = nullptr;
  float* d_projected_audio = nullptr;
  float* d_projected_imu = nullptr;
  float* d_attention = nullptr;
  float* d_output = nullptr;
  float* d_statistics = nullptr;
  CUDA_CHECK(cudaMalloc(&d_audio, sizeof(audio)));
  CUDA_CHECK(cudaMalloc(&d_imu, sizeof(imu)));
  CUDA_CHECK(cudaMalloc(&d_projected_audio,
                        TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_projected_imu,
                        TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attention,
                        TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output,
                        TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_statistics, STATISTIC_COUNT * sizeof(float)));

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

  CUDA_CHECK(cudaMemcpyAsync(d_audio, h_audio, sizeof(audio),
                             cudaMemcpyHostToDevice, audio_stream));
  project_audio_kernel<<<1, TOKEN_COUNT, 0, audio_stream>>>(
      d_audio, d_projected_audio);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(d_imu, h_imu, sizeof(imu),
                             cudaMemcpyHostToDevice, imu_stream));
  project_imu_kernel<<<1, TOKEN_COUNT, 0, imu_stream>>>(d_imu, d_projected_imu);
  CUDA_CHECK(cudaGetLastError());

  // The wait on the origin stream makes attention depend on both branches:
  // audio_stream is already ordered after project_audio, and imu_ready is
  // recorded only after project_imu has completed.
  CUDA_CHECK(cudaEventRecord(imu_ready, imu_stream));
  CUDA_CHECK(cudaStreamWaitEvent(audio_stream, imu_ready, 0));
  fuse_attention_kernel<<<1, TOKEN_COUNT, 0, audio_stream>>>(
      d_projected_audio, d_projected_imu, d_attention);
  CUDA_CHECK(cudaGetLastError());
  reduce_statistics_kernel<<<1, 1, 0, audio_stream>>>(
      d_projected_audio, d_attention, d_output, d_statistics);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(
      h_output, d_output,
      TOKEN_COUNT * FEATURE_LANE_COUNT * sizeof(float),
      cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaMemcpyAsync(h_statistics, d_statistics,
                             STATISTIC_COUNT * sizeof(float),
                             cudaMemcpyDeviceToHost, audio_stream));
  CUDA_CHECK(cudaStreamEndCapture(audio_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  CUDA_CHECK(cudaGraphLaunch(graph_exec, audio_stream));
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));

  const float combined = h_statistics[2] + 3.0f * h_statistics[0] +
                         7.0f * h_statistics[1] + 11.0f * h_statistics[3];
  if (outside_tolerance(h_statistics[0], 9.62500f) ||
      outside_tolerance(h_statistics[1], 3.93196f) ||
      outside_tolerance(h_statistics[2], 24.58165f) ||
      outside_tolerance(h_statistics[3], 0.73098f) ||
      outside_tolerance(combined, 89.02110f)) {
    std::cerr << "Verification failed\n";
    return EXIT_FAILURE;
  }

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
