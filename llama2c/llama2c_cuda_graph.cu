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
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstring>
#include <limits>

#include "llama2c_common.h"

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define LLAMA2C_HAS_NVTX 1
#  endif
#endif

#ifndef LLAMA2C_HAS_NVTX
#  define LLAMA2C_HAS_NVTX 0
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error__ = (call);                                              \
    if (error__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s failed: %s\n", #call,                        \
                   cudaGetErrorString(error__));                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

constexpr int kDefaultSize = 16384;
// Batch independent vectors so the reduction kernels can occupy the GPU.
constexpr int kDefaultBatch = 512;
constexpr int kDefaultRows = 8192;
constexpr int kDefaultLaunchCount = 180;
constexpr int kDefaultRepeats = 1;
constexpr int kDefaultWarmupMs = 100;
constexpr double kAutoTargetMs = 50.0;
constexpr int kAutoMinLaunches = 20;
constexpr int kAutoMaxLaunches = 5000;
constexpr int kSingleIterations = 20;
constexpr int kReductionThreads = 256;

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
    sum += input[column] * matrix[static_cast<size_t>(row) * columns + column];
  output[row] = sum;
}

// Coalesced variant: one warp per output row. Consecutive lanes read
// consecutive float4s, so each warp load is one contiguous 512-byte request
// instead of 32 loads from 32 different rows. The weights are streamed
// (read once, evict first) so the reused input vector stays cached.
constexpr int kMatmulWarpsPerBlock = 8;
__global__ void matmul_coalesced_kernel(float* output, const float* input,
                                        const float* matrix, int columns, int rows) {
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * kMatmulWarpsPerBlock + threadIdx.x / 32;
  if (row >= rows)
    return;
  const float* weights = matrix + static_cast<size_t>(row) * columns;
  float sum = 0.0f;
  if (columns % 4 == 0) {
    // cudaMalloc returns 256-byte-aligned memory and each row starts at a
    // multiple of 16 bytes when columns % 4 == 0, so float4 loads are aligned.
    const float4* weights4 = reinterpret_cast<const float4*>(weights);
    const float4* input4 = reinterpret_cast<const float4*>(input);
#pragma unroll 4
    for (int index = lane; index < columns / 4; index += 32) {
      const float4 w = __ldcs(weights4 + index);
      const float4 x = __ldg(input4 + index);
      sum += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
    }
  } else {
    for (int index = lane; index < columns; index += 32)
      sum += __ldcs(weights + index) * __ldg(input + index);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  if (lane == 0)
    output[row] = sum;
}

static void nvtx_push(const char *label) {
#if LLAMA2C_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

static void nvtx_pop() {
#if LLAMA2C_HAS_NVTX
  nvtxRangePop();
#endif
}

struct Options {
  bool graph = true;
  int launches = kDefaultLaunchCount;
  int batch = kDefaultBatch;
  int size = kDefaultSize;
  int rows = kDefaultRows;
  int repeats = kDefaultRepeats;
  int warmup_ms = kDefaultWarmupMs;
  bool copy_outputs = true;
  bool coalesced_matmul = false;
  // Pick the launch count after warm-up so every timed loop lasts about
  // kAutoTargetMs, whatever the per-iteration cost.
  bool auto_launches = false;
  // Input mode and --dump (see llama2c_common.h).
  InputOptions inputs;
};

static void usage(const char *program) {
  std::fprintf(stderr,
      "Usage: %s [graph|direct] [launches|auto] [batch] [size] [rows] [repeats] [warmup_ms]"
      " [copy|nocopy] [naive|coalesced] [deterministic|random[=SEED]] [--dump=FILE]\n",
      program);
}

static bool positive_int(const char *text, int *value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (*text == '\0' || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
    return false;
  *value = static_cast<int>(parsed);
  return true;
}

static bool parse_options(int argc, char **argv, Options *options) {
  int positional = 0;
  for (int index = 1; index < argc; ++index) {
    if (parse_input_option(argv[index], &options->inputs))
      continue;
    if (std::strcmp(argv[index], "graph") == 0) {
      options->graph = true;
      continue;
    }
    if (std::strcmp(argv[index], "direct") == 0) {
      options->graph = false;
      continue;
    }
    if (std::strcmp(argv[index], "nocopy") == 0) {
      options->copy_outputs = false;
      continue;
    }
    if (std::strcmp(argv[index], "copy") == 0) {
      options->copy_outputs = true;
      continue;
    }
    if (std::strcmp(argv[index], "naive") == 0) {
      options->coalesced_matmul = false;
      continue;
    }
    if (std::strcmp(argv[index], "coalesced") == 0) {
      options->coalesced_matmul = true;
      continue;
    }
    if (positional == 0 && std::strcmp(argv[index], "auto") == 0) {
      options->auto_launches = true;
      ++positional;
      continue;
    }
    int *destination = nullptr;
    if (positional == 0) destination = &options->launches;
    if (positional == 1) destination = &options->batch;
    if (positional == 2) destination = &options->size;
    if (positional == 3) destination = &options->rows;
    if (positional == 4) destination = &options->repeats;
    if (positional == 5) destination = &options->warmup_ms;
    if (destination == nullptr || !positive_int(argv[index], destination))
      return false;
    ++positional;
  }
  input_options_from_env(&options->inputs);
  return true;
}

int main(int argc, char **argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  const size_t vector_elements = static_cast<size_t>(options.batch) * options.size;
  const size_t matrix_elements = static_cast<size_t>(options.rows) * options.size;
  std::vector<float> h_input, h_weight, h_probabilities, h_matrix;
  fill_inputs(options.inputs, options.batch, options.size, options.rows,
              &h_input, &h_probabilities, &h_weight, &h_matrix);

  float *d_input = nullptr, *d_weight = nullptr, *d_probabilities = nullptr;
  float* d_softmax = nullptr;
  float *d_matrix = nullptr, *d_normalized = nullptr, *d_product = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probabilities, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_softmax, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_matrix, h_matrix.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMalloc(&d_product, sizeof(float) * options.rows));

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
  CUDA_CHECK(cudaMallocHost(&h_product, sizeof(float) * options.rows));

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
  double graph_capture_ms = 0.0;
  double graph_instantiate_ms = 0.0;
  double graph_upload_ms = 0.0;

  const int matmul_block = options.coalesced_matmul ? 32 * kMatmulWarpsPerBlock : 32;
  const int matmul_grid = options.coalesced_matmul
      ? (options.rows + kMatmulWarpsPerBlock - 1) / kMatmulWarpsPerBlock
      : (options.rows + 31) / 32;

  // Waves = grid blocks / (resident blocks per SM x SMs). At >= 1 every SM
  // holds its maximum number of blocks at the start of the kernel, so the
  // kernel fills the GPU rather than leaving SMs idle.
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  auto waves = [&](const void* kernel, int grid, int block, size_t shared_bytes) {
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block, shared_bytes));
    return static_cast<double>(grid) / (static_cast<double>(blocks_per_sm) * sm_count);
  };
  const double rms_waves = waves(reinterpret_cast<const void*>(rmsnorm_kernel), options.batch,
                                 kReductionThreads, kReductionThreads * sizeof(float));
  const double softmax_waves = waves(reinterpret_cast<const void*>(softmax_kernel), options.batch,
                                     kReductionThreads, kReductionThreads * sizeof(float));
  const double matmul_waves = waves(options.coalesced_matmul
                                        ? reinterpret_cast<const void*>(matmul_coalesced_kernel)
                                        : reinterpret_cast<const void*>(matmul_kernel),
                                    matmul_grid, matmul_block, 0);
  const double min_waves = std::fmin(rms_waves, std::fmin(softmax_waves, matmul_waves));
  // Minimum device-memory traffic of one iteration's kernels: each kernel
  // reads its inputs and writes its outputs once. Host copies use PCIe and
  // are excluded. Achieved GB/s against this shows how full the GPU is kept.
  const double vector_bytes = static_cast<double>(vector_elements) * sizeof(float);
  const double min_bytes_per_iter = options.repeats * (
      2.0 * vector_bytes + options.size * sizeof(float) +                 // rmsnorm
      2.0 * vector_bytes +                                                 // softmax
      static_cast<double>(matrix_elements) * sizeof(float) +
      (static_cast<double>(options.size) + options.rows) * sizeof(float)); // matmul

  auto enqueue_iteration = [&]() {
    // Every direct iteration repeats the same fork/join dependencies used by
    // graph capture. This prevents side streams from running ahead of the
    // next iteration and keeps the comparison apples-to-apples.
    CUDA_CHECK(cudaEventRecord(capture_start, rms_stream));
    CUDA_CHECK(cudaStreamWaitEvent(softmax_stream, capture_start, 0));
    CUDA_CHECK(cudaStreamWaitEvent(matmul_stream, capture_start, 0));
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      rmsnorm_kernel<<<options.batch, kReductionThreads, kReductionThreads * sizeof(float), rms_stream>>>(
          d_normalized, d_input, d_weight, options.size, options.batch, kRmsEpsilon);
      CUDA_CHECK(cudaGetLastError());
      softmax_kernel<<<options.batch, kReductionThreads, kReductionThreads * sizeof(float), softmax_stream>>>(
          d_softmax, d_probabilities, options.size, options.batch);
      CUDA_CHECK(cudaGetLastError());
      if (options.coalesced_matmul)
        matmul_coalesced_kernel<<<matmul_grid, matmul_block, 0, matmul_stream>>>(
            d_product, d_input, d_matrix, options.size, options.rows);
      else
        matmul_kernel<<<matmul_grid, matmul_block, 0, matmul_stream>>>(
            d_product, d_input, d_matrix, options.size, options.rows);
      CUDA_CHECK(cudaGetLastError());
      if (repeat + 1 == options.repeats && options.copy_outputs) {
        CUDA_CHECK(cudaMemcpyAsync(h_normalized, d_normalized, sizeof(float) * vector_elements,
                                   cudaMemcpyDeviceToHost, rms_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_softmax, d_softmax, sizeof(float) * vector_elements,
                                   cudaMemcpyDeviceToHost, softmax_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_product, d_product, sizeof(float) * options.rows,
                                   cudaMemcpyDeviceToHost, matmul_stream));
      }
    }
    CUDA_CHECK(cudaEventRecord(softmax_branch_done, softmax_stream));
    CUDA_CHECK(cudaEventRecord(matmul_branch_done, matmul_stream));
    CUDA_CHECK(cudaStreamWaitEvent(rms_stream, softmax_branch_done, 0));
    CUDA_CHECK(cudaStreamWaitEvent(rms_stream, matmul_branch_done, 0));
  };

  if (options.graph) {
    const auto capture_start_time = std::chrono::steady_clock::now();
    nvtx_push("llama2c.graph_capture");
    CUDA_CHECK(cudaStreamBeginCapture(rms_stream, cudaStreamCaptureModeGlobal));
    enqueue_iteration();
    CUDA_CHECK(cudaStreamEndCapture(rms_stream, &graph));
    nvtx_pop();
    graph_capture_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - capture_start_time).count();
    const auto instantiate_start_time = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    graph_instantiate_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - instantiate_start_time).count();
    // Exclude graph upload from the measured replay interval.
    const auto upload_start_time = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphUpload(executable, rms_stream));
    CUDA_CHECK(cudaStreamSynchronize(rms_stream));
    graph_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_start_time).count();
  }

  auto synchronize_all = [&]() {
    CUDA_CHECK(cudaStreamSynchronize(rms_stream));
    CUDA_CHECK(cudaStreamSynchronize(softmax_stream));
    CUDA_CHECK(cudaStreamSynchronize(matmul_stream));
  };
  // Keep the GPU busy for a fixed interval so tiny samples are not measured
  // at idle clocks. This is warm-up only and is excluded from all metrics.
  const auto warmup_start_time = std::chrono::steady_clock::now();
  int warmup_iterations = 0;
  do {
    nvtx_push("llama2c.warmup");
    if (options.graph)
      CUDA_CHECK(cudaGraphLaunch(executable, rms_stream));
    else
      enqueue_iteration();
    nvtx_pop();
    synchronize_all();
    ++warmup_iterations;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - warmup_start_time).count() < options.warmup_ms);
  if (options.auto_launches) {
    // Warm-up iterations each end with a synchronize, so this per-iteration
    // estimate is slightly pessimistic; it only sizes the timed loop.
    const double warmup_ms_per_iter = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - warmup_start_time).count() / warmup_iterations;
    const double planned = kAutoTargetMs / warmup_ms_per_iter;
    options.launches = static_cast<int>(std::fmin(kAutoMaxLaunches, std::fmax(kAutoMinLaunches, planned)));
  }

  cudaEvent_t timing_start = nullptr, timing_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&timing_start));
  CUDA_CHECK(cudaEventCreate(&timing_stop));
  CUDA_CHECK(cudaEventRecord(timing_start, rms_stream));
  const auto wall_start = std::chrono::steady_clock::now();
  const auto submit_start = std::chrono::steady_clock::now();
  for (int launch = 0; launch < options.launches; ++launch) {
    nvtx_push("llama2c.iteration");
    if (options.graph)
      CUDA_CHECK(cudaGraphLaunch(executable, rms_stream));
    else
      enqueue_iteration();
    nvtx_pop();
  }
  const auto submit_end = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(timing_stop, rms_stream));
  // Synchronize every participating stream before inspecting pinned host
  // buffers.  This also makes the example explicit about all graph branches.
  synchronize_all();
  const auto wall_end = std::chrono::steady_clock::now();
  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, timing_start, timing_stop));
  const double submit_us = std::chrono::duration<double, std::micro>(submit_end - submit_start).count();
  const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  // Single iterations: each starts on an idle device, so its launch cost is
  // exposed instead of overlapping the previous iteration's work as in the
  // back-to-back loop above. single_us is their median.
  std::vector<double> single_samples;
  for (int iteration = 0; iteration < kSingleIterations; ++iteration) {
    synchronize_all();
    nvtx_push("llama2c.single");
    CUDA_CHECK(cudaEventRecord(timing_start, rms_stream));
    if (options.graph)
      CUDA_CHECK(cudaGraphLaunch(executable, rms_stream));
    else
      enqueue_iteration();
    CUDA_CHECK(cudaEventRecord(timing_stop, rms_stream));
    nvtx_pop();
    CUDA_CHECK(cudaEventSynchronize(timing_stop));
    float single_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&single_ms, timing_start, timing_stop));
    single_samples.push_back(1000.0 * single_ms);
  }
  synchronize_all();
  std::sort(single_samples.begin(), single_samples.end());
  const double single_us = (single_samples[(kSingleIterations - 1) / 2] +
                            single_samples[kSingleIterations / 2]) / 2.0;
  auto copy_outputs_to_host = [&]() {
    CUDA_CHECK(cudaMemcpyAsync(h_normalized, d_normalized, sizeof(float) * vector_elements,
                               cudaMemcpyDeviceToHost, rms_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_softmax, d_softmax, sizeof(float) * vector_elements,
                               cudaMemcpyDeviceToHost, softmax_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_product, d_product, sizeof(float) * options.rows,
                               cudaMemcpyDeviceToHost, matmul_stream));
    synchronize_all();
  };
  if (!options.copy_outputs)
    copy_outputs_to_host();
  std::vector<float> product_tolerance;
  std::vector<double> product_reference;
  bool valid = validate_all(options.inputs, h_input, h_probabilities, h_weight, h_matrix,
                            std::vector<float>(h_normalized, h_normalized + vector_elements),
                            std::vector<float>(h_softmax, h_softmax + vector_elements),
                            std::vector<float>(h_product, h_product + options.rows),
                            options.batch, options.size, options.rows, &product_tolerance,
                            &product_reference);
  if (!valid)
    std::fprintf(stderr, "Output validation failed\n");
  if (!options.inputs.dump_path.empty() &&
      !write_dump(options.inputs.dump_path, options.inputs, options.batch, options.size,
                  options.rows, h_normalized, h_softmax, h_product, product_tolerance,
                  product_reference))
    valid = false;
  const double gpu_ms_per_iter = gpu_ms / options.launches;
  const double achieved_gbps = min_bytes_per_iter / (gpu_ms_per_iter * 1.0e6);
  std::printf("mode=%s matmul=%s launches=%d batch=%d size=%d rows=%d repeats=%d copies=%d warmup_ms=%d warmup_iterations=%d gpu_ms=%.6f gpu_ms_per_iter=%.6f single_iterations=%d single_us=%.3f wall_ms=%.6f wall_ms_per_iter=%.6f submit_us=%.3f submit_us_per_iter=%.3f graph_capture_ms=%.6f graph_instantiate_ms=%.6f graph_upload_ms=%.6f rms_waves=%.3f softmax_waves=%.3f matmul_waves=%.3f min_waves=%.3f min_bytes_per_iter=%.0f achieved_gbps=%.1f valid=%d inputs=%s seed=%llu\n",
              options.graph ? "graph" : "direct", options.coalesced_matmul ? "coalesced" : "naive",
              options.launches, options.batch, options.size,
              options.rows, options.repeats, options.copy_outputs ? 1 : 0, options.warmup_ms,
              warmup_iterations, gpu_ms, gpu_ms_per_iter, kSingleIterations, single_us,
              wall_ms, wall_ms / options.launches,
              submit_us, submit_us / options.launches, graph_capture_ms, graph_instantiate_ms,
              graph_upload_ms, rms_waves, softmax_waves, matmul_waves, min_waves, min_bytes_per_iter,
              achieved_gbps, valid ? 1 : 0, options.inputs.random ? "random" : "deterministic",
              static_cast<unsigned long long>(options.inputs.seed));

  if (executable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(executable));
  if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(timing_stop));
  CUDA_CHECK(cudaEventDestroy(timing_start));
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
