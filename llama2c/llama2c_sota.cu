// SOTA llama2c iteration: rmsnorm + softmax + matmul (GEMV) on the RTX 3090.
//
// Same work, inputs and outputs as llama2c_cuda_graph.cu (see its header and
// llama2c_common.h); same command line and key=value output line.
//
// Design (times measured on an RTX 3090 at the sheet shape, 1024 x 16384
// vectors and an 8192 x 16384 matrix):
// * All three operations are memory-bound streams, so the floor is the DRAM
//   traffic: each input read once and each output written once (768 MiB at
//   the sheet shape). The baseline moves ~1024 MiB because its rmsnorm reads
//   every row twice and its softmax reads three times and writes twice.
// * Every operation is "one block per 64 KiB row": the block loads its row
//   once into registers with 16-byte streaming loads (__ldcs), reduces it
//   (warp shuffles plus one shared-memory step), and for rmsnorm/softmax
//   writes the result from the same registers with streaming stores. A GEMV
//   row is a matrix row dotted with x (the first `size` floats of input).
// * The three operations are horizontally fused into one kernel launch: the
//   grid is [rmsnorm, softmax interleaved | matmul rows], so the GPU never
//   drains between operations and the short matmul blocks fill the tail.
//   Measured on an RTX 3090 at the sheet shape this is ~924 us per iteration
//   against ~935 us for the same kernels on three streams and ~951 us serial;
//   cuBLAS sgemv and warp-per-row GEMV variants reach the same ~890 GB/s read
//   peak as this kernel's matmul rows, but not inside a fused launch.
// * copy mode: outputs go to the host every iteration, as in the baseline.
//   The rmsnorm rows then run first as their own launch so the normalized
//   copy starts while softmax + matmul (a second launch on a second stream)
//   still compute; the iteration joins on all copies before the next starts.
// * Shapes with size % 4 != 0, or rows too long for registers, take a
//   generic path (multi-pass, scalar loads) in the same kernel.

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
constexpr int kDefaultBatch = 512;
constexpr int kDefaultRows = 8192;
constexpr int kDefaultLaunchCount = 180;
constexpr int kDefaultRepeats = 1;
constexpr int kDefaultWarmupMs = 100;
constexpr double kAutoTargetMs = 50.0;
constexpr int kAutoMinLaunches = 20;
constexpr int kAutoMaxLaunches = 5000;
constexpr int kSingleIterations = 20;
constexpr int kMaxThreads = 512;
constexpr int kGenericThreads = 256;

// One launch's share of the iteration. Blocks map to work items in order:
// 2 * min(rms_rows, softmax_rows) interleaved rmsnorm/softmax rows, then the
// leftover rmsnorm rows, then the leftover softmax rows, then matmul rows.
struct Work {
  float *normalized;
  const float *input;
  const float *weight;
  float *softmax;
  const float *logits;
  float *product;
  const float *matrix;
  int size;
  int rms_rows;
  int softmax_rows;
  int matmul_rows;
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  return value;
}

// Block-wide reductions; every thread gets the result. Each call site passes
// its own 32-float scratch slot, so no trailing barrier is needed (a block
// handles one work item).
__device__ __forceinline__ float block_sum(float value, float *scratch) {
  value = warp_sum(value);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : 0.0f;
  return warp_sum(value);
}

__device__ __forceinline__ float block_max(float value, float *scratch) {
  value = warp_max(value);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : -FLT_MAX;
  return warp_max(value);
}

// Register-resident row: thread t holds float4 columns t, t + T, ..., t + (V-1) T.
// EXACT: T * V == size / 4, so no bounds checks.
template <int V, bool EXACT>
__device__ __forceinline__ bool column_valid(int i, int columns4) {
  return EXACT || static_cast<int>(threadIdx.x + i * blockDim.x) < columns4;
}

template <int V, bool EXACT>
__device__ __forceinline__ void load_row(float4 (&x)[V], const float4 *row, int columns4) {
#pragma unroll
  for (int i = 0; i < V; ++i)
    x[i] = column_valid<V, EXACT>(i, columns4) ? __ldcs(row + threadIdx.x + i * blockDim.x)
                                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template <int V, bool EXACT>
__device__ __forceinline__ void rmsnorm_row(const Work &w, int vector, float *scratch) {
  const int columns4 = w.size / 4;
  const size_t offset = static_cast<size_t>(vector) * w.size;
  float4 x[V];
  load_row<V, EXACT>(x, reinterpret_cast<const float4 *>(w.input + offset), columns4);
  float square_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < V; ++i)
    square_sum += x[i].x * x[i].x + x[i].y * x[i].y + x[i].z * x[i].z + x[i].w * x[i].w;
  square_sum = block_sum(square_sum, scratch);
  const float scale = rsqrtf(square_sum / w.size + kRmsEpsilon);
  const float4 *weight4 = reinterpret_cast<const float4 *>(w.weight);
  float4 *out4 = reinterpret_cast<float4 *>(w.normalized + offset);
#pragma unroll
  for (int i = 0; i < V; ++i) {
    if (!column_valid<V, EXACT>(i, columns4)) continue;
    const int column = threadIdx.x + i * blockDim.x;
    const float4 g = __ldg(weight4 + column);
    float4 r;
    r.x = x[i].x * scale * g.x;
    r.y = x[i].y * scale * g.y;
    r.z = x[i].z * scale * g.z;
    r.w = x[i].w * scale * g.w;
    __stcs(out4 + column, r);
  }
}

template <int V, bool EXACT>
__device__ __forceinline__ void softmax_row(const Work &w, int vector, float *scratch) {
  const int columns4 = w.size / 4;
  const size_t offset = static_cast<size_t>(vector) * w.size;
  float4 x[V];
  load_row<V, EXACT>(x, reinterpret_cast<const float4 *>(w.logits + offset), columns4);
  float maximum = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < V; ++i)
    if (column_valid<V, EXACT>(i, columns4))
      maximum = fmaxf(maximum, fmaxf(fmaxf(x[i].x, x[i].y), fmaxf(x[i].z, x[i].w)));
  maximum = block_max(maximum, scratch);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < V; ++i) {
    if (!column_valid<V, EXACT>(i, columns4)) continue;
    x[i].x = expf(x[i].x - maximum);
    x[i].y = expf(x[i].y - maximum);
    x[i].z = expf(x[i].z - maximum);
    x[i].w = expf(x[i].w - maximum);
    sum += x[i].x + x[i].y + x[i].z + x[i].w;
  }
  sum = block_sum(sum, scratch + 32);
  float4 *out4 = reinterpret_cast<float4 *>(w.softmax + offset);
#pragma unroll
  for (int i = 0; i < V; ++i) {
    if (!column_valid<V, EXACT>(i, columns4)) continue;
    float4 r;
    r.x = x[i].x / sum;
    r.y = x[i].y / sum;
    r.z = x[i].z / sum;
    r.w = x[i].w / sum;
    __stcs(out4 + threadIdx.x + i * blockDim.x, r);
  }
}

template <int V, bool EXACT>
__device__ __forceinline__ void matmul_row(const Work &w, int row, float *scratch) {
  const int columns4 = w.size / 4;
  float4 m[V];
  load_row<V, EXACT>(m, reinterpret_cast<const float4 *>(w.matrix + static_cast<size_t>(row) * w.size),
                     columns4);
  const float4 *x4 = reinterpret_cast<const float4 *>(w.input);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < V; ++i) {
    if (!column_valid<V, EXACT>(i, columns4)) continue;
    const float4 x = __ldg(x4 + threadIdx.x + i * blockDim.x);
    sum += m[i].x * x.x + m[i].y * x.y + m[i].z * x.z + m[i].w * x.w;
  }
  sum = block_sum(sum, scratch);
  if (threadIdx.x == 0) w.product[row] = sum;
}

// Generic path (V == 0): any size, scalar loads, rows re-read from memory.
__device__ void rmsnorm_row_generic(const Work &w, int vector, float *scratch) {
  const size_t offset = static_cast<size_t>(vector) * w.size;
  float square_sum = 0.0f;
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    square_sum += w.input[offset + i] * w.input[offset + i];
  square_sum = block_sum(square_sum, scratch);
  const float scale = rsqrtf(square_sum / w.size + kRmsEpsilon);
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    w.normalized[offset + i] = w.input[offset + i] * scale * w.weight[i];
}

__device__ void softmax_row_generic(const Work &w, int vector, float *scratch) {
  const size_t offset = static_cast<size_t>(vector) * w.size;
  float maximum = -FLT_MAX;
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    maximum = fmaxf(maximum, w.logits[offset + i]);
  maximum = block_max(maximum, scratch);
  float sum = 0.0f;
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    sum += expf(w.logits[offset + i] - maximum);
  sum = block_sum(sum, scratch + 32);
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    w.softmax[offset + i] = expf(w.logits[offset + i] - maximum) / sum;
}

__device__ void matmul_row_generic(const Work &w, int row, float *scratch) {
  const float *m = w.matrix + static_cast<size_t>(row) * w.size;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < w.size; i += blockDim.x)
    sum += m[i] * w.input[i];
  sum = block_sum(sum, scratch);
  if (threadIdx.x == 0) w.product[row] = sum;
}

template <int V, bool EXACT>
__global__ void __launch_bounds__(kMaxThreads)
llama2c_sota_kernel(Work w) {
  __shared__ float scratch[64];
  int item = blockIdx.x;
  int kind, row;
  const int pairs = min(w.rms_rows, w.softmax_rows);
  if (item < 2 * pairs) {
    kind = item & 1;
    row = item >> 1;
  } else {
    item -= 2 * pairs;
    if (item < w.rms_rows - pairs) {
      kind = 0;
      row = pairs + item;
    } else {
      item -= w.rms_rows - pairs;
      if (item < w.softmax_rows - pairs) {
        kind = 1;
        row = pairs + item;
      } else {
        kind = 2;
        row = item - (w.softmax_rows - pairs);
      }
    }
  }
  if constexpr (V == 0) {
    if (kind == 0) rmsnorm_row_generic(w, row, scratch);
    else if (kind == 1) softmax_row_generic(w, row, scratch);
    else matmul_row_generic(w, row, scratch);
  } else {
    if (kind == 0) rmsnorm_row<V, EXACT>(w, row, scratch);
    else if (kind == 1) softmax_row<V, EXACT>(w, row, scratch);
    else matmul_row<V, EXACT>(w, row, scratch);
  }
}

using KernelFn = void (*)(Work);

struct Config {
  KernelFn kernel;
  int threads;
  int vectors;  // float4 per thread (0 = generic path)
  bool exact;
};

// Smallest float4-per-thread count V in {1,2,4,8,16} whose row fits in
// registers with at most 512 threads; threads = ceil(size / 4 / V) rounded up
// to a warp.
static Config choose_config(int size) {
  if (size % 4 == 0) {
    const int columns4 = size / 4;
    static const KernelFn exact[] = {llama2c_sota_kernel<1, true>, llama2c_sota_kernel<2, true>,
                                     llama2c_sota_kernel<4, true>, llama2c_sota_kernel<8, true>,
                                     llama2c_sota_kernel<16, true>};
    static const KernelFn bounded[] = {llama2c_sota_kernel<1, false>, llama2c_sota_kernel<2, false>,
                                       llama2c_sota_kernel<4, false>, llama2c_sota_kernel<8, false>,
                                       llama2c_sota_kernel<16, false>};
    for (int index = 0, vectors = 1; index < 5; ++index, vectors *= 2) {
      if (columns4 > kMaxThreads * vectors) continue;
      const int threads = std::max(32, (columns4 + vectors - 1) / vectors + 31) / 32 * 32;
      const bool is_exact = threads * vectors == columns4;
      return {is_exact ? exact[index] : bounded[index], std::min(threads, kMaxThreads), vectors, is_exact};
    }
  }
  return {llama2c_sota_kernel<0, false>, kGenericThreads, 0, false};
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
  bool auto_launches = false;
  InputOptions inputs;
};

static void usage(const char *program) {
  std::fprintf(stderr,
      "Usage: %s [graph|direct] [launches|auto] [batch] [size] [rows] [repeats] [warmup_ms]"
      " [copy|nocopy] [naive|coalesced] [deterministic|random[=SEED]] [--dump=FILE]\n"
      "  naive|coalesced are accepted for command-line compatibility and ignored.\n",
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
    const char *arg = argv[index];
    if (parse_input_option(arg, &options->inputs)) continue;
    if (std::strcmp(arg, "graph") == 0) { options->graph = true; continue; }
    if (std::strcmp(arg, "direct") == 0) { options->graph = false; continue; }
    if (std::strcmp(arg, "nocopy") == 0) { options->copy_outputs = false; continue; }
    if (std::strcmp(arg, "copy") == 0) { options->copy_outputs = true; continue; }
    if (std::strcmp(arg, "naive") == 0 || std::strcmp(arg, "coalesced") == 0) continue;
    if (positional == 0 && std::strcmp(arg, "auto") == 0) {
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
    if (destination == nullptr || !positive_int(arg, destination)) return false;
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

  float *d_input = nullptr, *d_weight = nullptr, *d_probabilities = nullptr, *d_softmax = nullptr;
  float *d_matrix = nullptr, *d_normalized = nullptr, *d_product = nullptr;
  // Allocation order (inputs, then outputs) measured ~0.3% faster than the
  // baseline's interleaved order on an RTX 3090; it only moves physical pages.
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probabilities, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_matrix, h_matrix.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMalloc(&d_softmax, h_probabilities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_product, sizeof(float) * options.rows));
  // Inputs are uploaded once, before timing (the workload is device-resident).
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_probabilities, h_probabilities.data(), h_probabilities.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix.data(), h_matrix.size() * sizeof(float), cudaMemcpyHostToDevice));

  float *h_normalized = nullptr, *h_softmax = nullptr, *h_product = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_normalized, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMallocHost(&h_softmax, sizeof(float) * vector_elements));
  CUDA_CHECK(cudaMallocHost(&h_product, sizeof(float) * options.rows));

  cudaStream_t main_stream = nullptr, side_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking));
  cudaEvent_t fork = nullptr, join = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

  const Config config = choose_config(options.size);
  const Work all_work{d_normalized, d_input, d_weight, d_softmax, d_probabilities, d_product, d_matrix,
                      options.size, options.batch, options.batch, options.rows};
  Work rms_work = all_work;
  rms_work.softmax_rows = 0;
  rms_work.matmul_rows = 0;
  Work rest_work = all_work;
  rest_work.rms_rows = 0;
  auto grid_of = [](const Work &w) { return w.rms_rows + w.softmax_rows + w.matmul_rows; };
  auto launch = [&](const Work &w, cudaStream_t stream) {
    config.kernel<<<grid_of(w), config.threads, 0, stream>>>(w);
    CUDA_CHECK(cudaGetLastError());
  };

  int sm_count = 0, blocks_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, reinterpret_cast<const void *>(config.kernel), config.threads, 0));
  auto waves = [&](int grid) { return static_cast<double>(grid) / (static_cast<double>(blocks_per_sm) * sm_count); };
  // One fused launch covers all three operations, so they share one wave count.
  const double fused_waves = waves(grid_of(all_work));
  const double vector_bytes = static_cast<double>(vector_elements) * sizeof(float);
  const double min_bytes_per_iter = options.repeats * (
      2.0 * vector_bytes + options.size * sizeof(float) +                 // rmsnorm
      2.0 * vector_bytes +                                                 // softmax
      static_cast<double>(matrix_elements) * sizeof(float) +
      (static_cast<double>(options.size) + options.rows) * sizeof(float)); // matmul

  // Kernel launches per iteration: one fused launch per repeat; in copy mode
  // the last repeat is split in two (rmsnorm | softmax + matmul).
  const int kernels_per_iter = options.repeats + (options.copy_outputs ? 1 : 0);
  auto enqueue_iteration = [&]() {
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      if (!(options.copy_outputs && repeat + 1 == options.repeats)) {
        launch(all_work, main_stream);
        continue;
      }
      CUDA_CHECK(cudaEventRecord(fork, main_stream));
      CUDA_CHECK(cudaStreamWaitEvent(side_stream, fork, 0));
      launch(rms_work, main_stream);
      CUDA_CHECK(cudaMemcpyAsync(h_normalized, d_normalized, sizeof(float) * vector_elements,
                                 cudaMemcpyDeviceToHost, main_stream));
      launch(rest_work, side_stream);
      CUDA_CHECK(cudaMemcpyAsync(h_softmax, d_softmax, sizeof(float) * vector_elements,
                                 cudaMemcpyDeviceToHost, side_stream));
      CUDA_CHECK(cudaMemcpyAsync(h_product, d_product, sizeof(float) * options.rows,
                                 cudaMemcpyDeviceToHost, side_stream));
      CUDA_CHECK(cudaEventRecord(join, side_stream));
      CUDA_CHECK(cudaStreamWaitEvent(main_stream, join, 0));
    }
  };

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  double graph_capture_ms = 0.0, graph_instantiate_ms = 0.0, graph_upload_ms = 0.0;
  if (options.graph) {
    const auto capture_start_time = std::chrono::steady_clock::now();
    nvtx_push("llama2c.graph_capture");
    CUDA_CHECK(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal));
    enqueue_iteration();
    CUDA_CHECK(cudaStreamEndCapture(main_stream, &graph));
    nvtx_pop();
    graph_capture_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - capture_start_time).count();
    const auto instantiate_start_time = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    graph_instantiate_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - instantiate_start_time).count();
    const auto upload_start_time = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphUpload(executable, main_stream));
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    graph_upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_start_time).count();
  }
  auto run_iteration = [&]() {
    if (options.graph)
      CUDA_CHECK(cudaGraphLaunch(executable, main_stream));
    else
      enqueue_iteration();
  };
  auto synchronize_all = [&]() {
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    CUDA_CHECK(cudaStreamSynchronize(side_stream));
  };

  const auto warmup_start_time = std::chrono::steady_clock::now();
  int warmup_iterations = 0;
  do {
    nvtx_push("llama2c.warmup");
    run_iteration();
    nvtx_pop();
    synchronize_all();
    ++warmup_iterations;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - warmup_start_time).count() < options.warmup_ms);
  if (options.auto_launches) {
    const double warmup_ms_per_iter = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - warmup_start_time).count() / warmup_iterations;
    const double planned = kAutoTargetMs / warmup_ms_per_iter;
    options.launches = static_cast<int>(std::fmin(kAutoMaxLaunches, std::fmax(kAutoMinLaunches, planned)));
  }

  cudaEvent_t timing_start = nullptr, timing_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&timing_start));
  CUDA_CHECK(cudaEventCreate(&timing_stop));
  CUDA_CHECK(cudaEventRecord(timing_start, main_stream));
  const auto wall_start = std::chrono::steady_clock::now();
  const auto submit_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < options.launches; ++iteration) {
    nvtx_push("llama2c.iteration");
    run_iteration();
    nvtx_pop();
  }
  const auto submit_end = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(timing_stop, main_stream));
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
    CUDA_CHECK(cudaEventRecord(timing_start, main_stream));
    run_iteration();
    CUDA_CHECK(cudaEventRecord(timing_stop, main_stream));
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

  // Validate what the last timed iteration produced: in copy mode it already
  // copied its outputs to the host; in nocopy mode copy them now.
  if (!options.copy_outputs) {
    CUDA_CHECK(cudaMemcpy(h_normalized, d_normalized, sizeof(float) * vector_elements, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_softmax, d_softmax, sizeof(float) * vector_elements, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_product, d_product, sizeof(float) * options.rows, cudaMemcpyDeviceToHost));
  }
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
  std::printf("mode=%s matmul=sota launches=%d batch=%d size=%d rows=%d repeats=%d copies=%d warmup_ms=%d warmup_iterations=%d gpu_ms=%.6f gpu_ms_per_iter=%.6f single_iterations=%d single_us=%.3f wall_ms=%.6f wall_ms_per_iter=%.6f submit_us=%.3f submit_us_per_iter=%.3f graph_capture_ms=%.6f graph_instantiate_ms=%.6f graph_upload_ms=%.6f rms_waves=%.3f softmax_waves=%.3f matmul_waves=%.3f min_waves=%.3f min_bytes_per_iter=%.0f achieved_gbps=%.1f valid=%d inputs=%s seed=%llu kernels_per_iter=%d threads=%d vectors=%d\n",
              options.graph ? "graph" : "direct", options.launches, options.batch, options.size,
              options.rows, options.repeats, options.copy_outputs ? 1 : 0, options.warmup_ms,
              warmup_iterations, gpu_ms, gpu_ms_per_iter, kSingleIterations, single_us,
              wall_ms, wall_ms / options.launches,
              submit_us, submit_us / options.launches, graph_capture_ms, graph_instantiate_ms,
              graph_upload_ms, fused_waves, fused_waves, fused_waves, fused_waves, min_bytes_per_iter,
              achieved_gbps, valid ? 1 : 0, options.inputs.random ? "random" : "deterministic",
              static_cast<unsigned long long>(options.inputs.seed), kernels_per_iter,
              config.threads, config.vectors);

  if (executable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(executable));
  if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(timing_stop));
  CUDA_CHECK(cudaEventDestroy(timing_start));
  CUDA_CHECK(cudaEventDestroy(join));
  CUDA_CHECK(cudaEventDestroy(fork));
  CUDA_CHECK(cudaStreamDestroy(side_stream));
  CUDA_CHECK(cudaStreamDestroy(main_stream));
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
