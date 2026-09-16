// Illustrative CUDA Graph for the llama2.c kernel smoke workload.
//
// The upstream run.c contains a much larger transformer runtime.  This file
// models only the three operations selected by
// test/applications/llama2c-kernels-smoke.h: rmsnorm, softmax, and matmul.
// It is a standalone executable example, not a replacement for run.c.

#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error__ = (call);                                              \
    if (error__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s failed: %s\n", #call,                        \
                   cudaGetErrorString(error__));                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

constexpr int kSize = 16384;
// Batch independent vectors so the reduction kernels can occupy the GPU.
// Raise this toward 512 if the goal is to stress memory bandwidth further.
constexpr int kBatch = 512;
constexpr int kRows = 8192;
constexpr int kDefaultLaunchCount = 180;
constexpr float kRmsEpsilon = 1.0e-5f;

// One block handles one llama2.c vector. Shared-memory reductions expose the
// parallelism that the smoke implementation's scalar loop leaves implicit.
__global__ void rmsnorm_kernel(float* output, const float* input,
                               const float* weight, int size, int batch,
                               float epsilon) {
  extern __shared__ float reduction[];
  const int lane = threadIdx.x;
  const int vector = blockIdx.x;
  if (vector >= batch) return;
  const size_t offset = static_cast<size_t>(vector) * size;
  float square_sum = 0.0f;
  for (int index = lane; index < size; index += blockDim.x)
    square_sum += input[offset + index] * input[offset + index];
  reduction[lane] = square_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(reduction[0] / size + epsilon);
  for (int index = lane; index < size; index += blockDim.x)
    output[offset + index] = input[offset + index] * scale * weight[index];
}

// Max and sum are block reductions; each element then participates in the
// exp/normalization phases independently.
__global__ void softmax_kernel(float* values, const float* logits, int size,
                               int batch) {
  extern __shared__ float reduction[];
  const int lane = threadIdx.x;
  const int vector = blockIdx.x;
  if (vector >= batch) return;
  const size_t offset = static_cast<size_t>(vector) * size;
  float maximum = -FLT_MAX;
  for (int index = lane; index < size; index += blockDim.x)
    maximum = fmaxf(maximum, logits[offset + index]);
  reduction[lane] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    __syncthreads();
  }
  maximum = reduction[0];
  // All warps must read the maximum before reusing reduction for the sum.
  __syncthreads();
  float sum = 0.0f;
  for (int index = lane; index < size; index += blockDim.x) {
    values[offset + index] = expf(logits[offset + index] - maximum);
    sum += values[offset + index];
  }
  reduction[lane] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  sum = reduction[0];
  for (int index = lane; index < size; index += blockDim.x)
    values[offset + index] /= sum;
}

// Matrix storage follows llama2.c's matmul(output, input, weight, columns,
// rows) convention: each output row has `columns` contiguous weights.
__global__ void matmul_kernel(float* output, const float* input,
                              const float* matrix, int columns, int rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  float sum = 0.0f;
  for (int column = 0; column < columns; ++column)
    sum += input[column] * matrix[row * columns + column];
  output[row] = sum;
}

static bool approximately_equal(float actual, float expected, float tolerance) {
  return std::fabs(actual - expected) <= tolerance;
}

int main(int argc, char **argv) {
  const int launches = argc > 1 ? std::atoi(argv[1]) : kDefaultLaunchCount;
  const size_t vector_elements = static_cast<size_t>(kBatch) * kSize;
  std::vector<float> h_input(vector_elements), h_weight(kSize, 1.0f), h_probabilities(vector_elements), h_matrix(static_cast<size_t>(kRows) * kSize);
  for (size_t i = 0; i < vector_elements; ++i) { h_input[i] = 1.0f + (i % 4); h_probabilities[i] = 1.0f + (i % 4); }
  for (int row = 0; row < kRows; ++row) for (int col = 0; col < kSize; ++col) h_matrix[static_cast<size_t>(row) * kSize + col] = (col == (row % kSize)) ? 1.0f : 0.0f;

  float *d_input = nullptr, *d_weight = nullptr, *d_probabilities = nullptr;
  float* d_softmax = nullptr;
  float *d_matrix = nullptr, *d_normalized = nullptr, *d_product = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probabilities, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_softmax, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_matrix, h_matrix.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMalloc(&d_product, sizeof(float) * kRows));

  // Inputs are prepared before capture, just as host-side model setup in
  // run.c precedes execution of its kernels.
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_probabilities, h_probabilities.data(), h_probabilities.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix.data(), h_matrix.size() * sizeof(float), cudaMemcpyHostToDevice));

  float *h_normalized = nullptr, *h_softmax = nullptr, *h_product = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_normalized, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMallocHost(&h_softmax, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMallocHost(&h_product, sizeof(float) * kRows));

  cudaStream_t rms_stream = nullptr, softmax_stream = nullptr;
  cudaStream_t matmul_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&rms_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&softmax_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&matmul_stream, cudaStreamNonBlocking));
  cudaEvent_t capture_start = nullptr;
  cudaEvent_t softmax_branch_done = nullptr, matmul_branch_done = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&softmax_branch_done, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&matmul_branch_done, cudaEventDisableTiming));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  // Global mode does not automatically capture work on other streams.
  // Fork capture through an event before any kernel so the independent
  // branches can overlap. Each D2H copy follows its kernel in stream order;
  // the completion events rejoin both side branches before ending capture.
  CUDA_CHECK(cudaStreamBeginCapture(rms_stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaEventRecord(capture_start, rms_stream));
  CUDA_CHECK(cudaStreamWaitEvent(softmax_stream, capture_start, 0));
  CUDA_CHECK(cudaStreamWaitEvent(matmul_stream, capture_start, 0));

  rmsnorm_kernel<<<kBatch, 256, 256 * sizeof(float), rms_stream>>>(d_normalized, d_input, d_weight,
                                            kSize, kBatch, kRmsEpsilon);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h_normalized, d_normalized, sizeof(float) * vector_elements,
                             cudaMemcpyDeviceToHost, rms_stream));

  // Preserve the logits so every graph replay computes the same softmax.
  softmax_kernel<<<kBatch, 256, 256 * sizeof(float), softmax_stream>>>(d_softmax, d_probabilities, kSize, kBatch);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h_softmax, d_softmax, sizeof(float) * vector_elements,
                             cudaMemcpyDeviceToHost, softmax_stream));
  CUDA_CHECK(cudaEventRecord(softmax_branch_done, softmax_stream));

  matmul_kernel<<<(kRows + 31) / 32, 32, 0, matmul_stream>>>(d_product, d_input, d_matrix,
                                              kSize, kRows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h_product, d_product, sizeof(float) * kRows,
                             cudaMemcpyDeviceToHost, matmul_stream));
  CUDA_CHECK(cudaEventRecord(matmul_branch_done, matmul_stream));
  CUDA_CHECK(cudaStreamWaitEvent(rms_stream, softmax_branch_done, 0));
  CUDA_CHECK(cudaStreamWaitEvent(rms_stream, matmul_branch_done, 0));
  CUDA_CHECK(cudaStreamEndCapture(rms_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

  auto start = std::chrono::steady_clock::now();
  for (int launch = 0; launch < launches; ++launch)
    CUDA_CHECK(cudaGraphLaunch(executable, rms_stream));
  // Synchronize every participating stream before inspecting pinned host
  // buffers.  This also makes the example explicit about all graph branches.
  CUDA_CHECK(cudaStreamSynchronize(rms_stream));
  CUDA_CHECK(cudaStreamSynchronize(softmax_stream));
  CUDA_CHECK(cudaStreamSynchronize(matmul_stream));
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("graph_launches=%d elapsed_seconds=%.3f\n", launches, seconds);

  const float square_mean = (h_normalized[0] * h_normalized[0] + h_normalized[1] * h_normalized[1]) / 2.0f;
  double probability_sum = 0.0;
  for (int index = 0; index < kSize; ++index)
    probability_sum += h_softmax[index];

  bool valid = std::isfinite(square_mean) &&
               std::fabs(probability_sum - 1.0) <= 1.0e-6 &&
               h_softmax[0] < h_softmax[1] && h_softmax[1] < h_softmax[2] &&
               h_softmax[2] < h_softmax[3] &&
               approximately_equal(h_product[0], h_input[0], 1.0e-4f) &&
               approximately_equal(h_product[1], h_input[1], 1.0e-4f);

  for (int row = 0; row < kRows; ++row)
    valid = approximately_equal(h_product[row], h_input[row % kSize], 1.0e-4f) && valid;
  if (!valid)
    std::fprintf(stderr, "Output validation failed\n");

  std::printf("normalized0=%.6f softmax_sum=%.6f matmul0=%.6f\n", h_normalized[0], probability_sum, h_product[0]);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(capture_start));
  CUDA_CHECK(cudaEventDestroy(matmul_branch_done));
  CUDA_CHECK(cudaEventDestroy(softmax_branch_done));
  CUDA_CHECK(cudaStreamDestroy(matmul_stream));
  CUDA_CHECK(cudaStreamDestroy(softmax_stream));
  CUDA_CHECK(cudaStreamDestroy(rms_stream));
  CUDA_CHECK(cudaFreeHost(h_product));
  CUDA_CHECK(cudaFreeHost(h_softmax));
  CUDA_CHECK(cudaFreeHost(h_normalized));
  CUDA_CHECK(cudaFree(d_product));
  CUDA_CHECK(cudaFree(d_normalized));
  CUDA_CHECK(cudaFree(d_matrix));
  CUDA_CHECK(cudaFree(d_probabilities));
  CUDA_CHECK(cudaFree(d_softmax));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_input));
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
