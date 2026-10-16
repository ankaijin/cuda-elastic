// SOTA CUDA version of mlperf_tiny_kws_postprocess_kernel (see
// kws_postprocess.cu for the port it is checked against):
//
//   for i in [0, count): p[i] = (float)(q[i] - zeroPoint) * scale
//   best = first index of the maximum of p
//
// for --batch rows of --count int8 logits. Same options, inputs, NVTX ranges
// and output lines as kws_postprocess, plus:
//   --rotate=K   rotate the timed iterations over K input sets (default 3),
//                each with its own pinned host input and output buffers, so
//                every iteration works on new data; every set's outputs are
//                checked after timing.
//   --mode=M     how an end-to-end iteration moves data:
//                  zerocopy  (default) one kernel reads the int8 rows from
//                            mapped pinned host memory and writes the
//                            probabilities and indices straight back to it
//                            over PCIe; no copies, no device buffers.
//                  pipeline  chunked H2D / kernel / D2H on three streams
//                            (CopyComputePipeline) through device buffers.
//   --chunk=R    rows per pipeline chunk (default 262144).
//   --lookahead=L  pipeline only: H2D may run at most L chunks ahead of D2H
//                (default 0, unlimited).
// --launch defaults to direct here. A zero-copy iteration is one kernel
// launch either way, and graph replay measured the same.
//
// Why zero-copy: at 12 x 4M rows an iteration moves 50 MB in and 218 MB out,
// so it is bound by the device-to-host direction of PCIe. On the RTX 3090
// the zero-copy kernel (8.67 ms) runs within 1% of two plain cudaMemcpyAsync
// calls of the same sizes on separate streams with nothing to wait for
// (8.60 ms), while the chunked pipeline loses to its chunk dependencies
// (8.79 ms launched directly, 9.27 ms as a graph). The zero-copy kernel's
// own duration therefore includes all of the iteration's PCIe time.
//
// The 12-category kernel (the model's kCategoryCount) gives each thread one
// row and stages a 256-row tile through shared memory, so every global
// access is a coalesced 16 B vector: 192 threads load the tile's 3 KB of
// int8 (one uint4 each), each thread dequantizes its 12 values and finds the
// first maximum, then the block writes the 12 KB of probabilities as 768
// uint4 and the 256 indices as one int per thread. Over PCIe that means
// every request is a full 128 B line. Other --count values use a generic
// thread-per-row kernel. The math is the baseline's: an int subtract, an
// int->float conversion and one fp32 multiply (--fmad=false), so the outputs
// are bitwise equal to the host loop.
//
// --dump=FILE writes input set 0's outputs (set 0 is the baseline's input):
// batch x count float32 probabilities, row-major, then batch int32 indices.
#include <cfloat>
#include <climits>
#include <cstdint>
#include <random>
#include <vector>

#include "classification_harness.cuh"

namespace {

constexpr size_t kSmokeCount = 12;  // kCategoryCount
constexpr int8_t kSmokeQuantized[kSmokeCount] = {-8, -4, 1, 3, 7, 12,
                                                 5,  2,  0, 9, 4, 6};
constexpr float kSmokeScale = 0.125f;
constexpr int kSmokeZeroPoint = 0;
constexpr float kRandomScale = 0.0390625f;
constexpr int kRandomZeroPoint = -7;
constexpr int kTiesRange = 3;

constexpr int kCategories = 12;
constexpr int kTileRows = 256;  // one row per thread
constexpr int kTileInVectors = kTileRows * kCategories / 16;       // 192
constexpr int kTileOutVectors = kTileRows * kCategories * 4 / 16;  // 768
constexpr int kGenericThreads = 256;
constexpr size_t kDefaultChunkRows = 262144;

float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint) {
  return static_cast<float>(value - zeroPoint) * scale;
}

// Verbatim loop of mlperf_tiny_kws_postprocess_kernel.
int loom_postprocess(const int8_t* quantized, float* probabilities, int count,
                     float scale, int zeroPoint) {
  int best = 0;
  for (int index = 0; index < count; ++index) {
    probabilities[index] =
        DequantizeInt8ToFloat(quantized[index], scale, zeroPoint);
    if (probabilities[index] > probabilities[best])
      best = index;
  }
  return best;
}

// Row r is the smoke vector rotated left by r + offset (offset 0 is the
// baseline's smoke input; other input sets use offset = set index).
void fill_smoke(int8_t* quantized, size_t count, size_t batch,
                size_t offset) {
  for (size_t row = 0; row < batch; ++row)
    for (size_t index = 0; index < count; ++index)
      quantized[row * count + index] =
          kSmokeQuantized[(index + row + offset) % kSmokeCount];
}

void fill_random(int8_t* quantized, size_t elements, int low, int high,
                 unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(low, high);
  for (size_t index = 0; index < elements; ++index)
    quantized[index] = static_cast<int8_t>(distribution(generator));
}

__device__ __forceinline__ float dequantize(uint32_t word, int byte,
                                            float scale, int zeroPoint) {
  const int value = static_cast<int8_t>(word >> (8 * byte));
  return static_cast<float>(value - zeroPoint) * scale;
}

// count == 12. Block b handles rows [256 b, 256 b + 256); pointers may be
// device memory or mapped pinned host memory (zero-copy). quantized,
// probabilities and best must be 16 B aligned.
__global__ void __launch_bounds__(kTileRows)
    kws_postprocess_sota_kernel(const int8_t* __restrict__ quantized,
                                float* __restrict__ probabilities,
                                int* __restrict__ best, size_t batch,
                                float scale, int zeroPoint) {
  __shared__ uint4 inTile[kTileInVectors];
  __shared__ uint4 outTile[kTileOutVectors];
  const int thread = threadIdx.x;
  const size_t tileRow = static_cast<size_t>(blockIdx.x) * kTileRows;
  const size_t rows = min(static_cast<size_t>(kTileRows), batch - tileRow);

  if (rows == kTileRows) {
    if (thread < kTileInVectors)
      inTile[thread] = reinterpret_cast<const uint4*>(
          quantized + tileRow * kCategories)[thread];
    __syncthreads();
    // Words 3t..3t+2: stride 3 is conflict-free.
    const uint32_t* words = reinterpret_cast<const uint32_t*>(inTile) +
                            3 * thread;
    const uint32_t w[3] = {words[0], words[1], words[2]};
    float p[kCategories];
#pragma unroll
    for (int index = 0; index < kCategories; ++index)
      p[index] = dequantize(w[index / 4], index % 4, scale, zeroPoint);
    int bestIndex = 0;
#pragma unroll
    for (int index = 1; index < kCategories; ++index)
      if (p[index] > p[bestIndex])
        bestIndex = index;
    // float4s 3t..3t+2: a quarter warp's 48 B stride covers all banks.
    float4* out = reinterpret_cast<float4*>(outTile) + 3 * thread;
    out[0] = make_float4(p[0], p[1], p[2], p[3]);
    out[1] = make_float4(p[4], p[5], p[6], p[7]);
    out[2] = make_float4(p[8], p[9], p[10], p[11]);
    best[tileRow + thread] = bestIndex;
    __syncthreads();
    uint4* probabilities16 =
        reinterpret_cast<uint4*>(probabilities + tileRow * kCategories);
#pragma unroll
    for (int vector = thread; vector < kTileOutVectors; vector += kTileRows)
      probabilities16[vector] = outTile[vector];
  } else if (thread < rows) {
    // The last, partial tile: one row per thread straight from global.
    const size_t row = tileRow + thread;
    const int8_t* q = quantized + row * kCategories;
    float* p = probabilities + row * kCategories;
    float bestValue = 0.0f;
    int bestIndex = 0;
    for (int index = 0; index < kCategories; ++index) {
      const float value =
          static_cast<float>(q[index] - zeroPoint) * scale;
      p[index] = value;
      if (index == 0 || value > bestValue) {
        bestValue = value;
        bestIndex = index;
      }
    }
    best[row] = bestIndex;
  }
}

// Any count: one thread per row, the Loom loop.
__global__ void __launch_bounds__(kGenericThreads)
    kws_postprocess_sota_generic_kernel(const int8_t* __restrict__ quantized,
                                        float* __restrict__ probabilities,
                                        int* __restrict__ best, int count,
                                        size_t batch, float scale,
                                        int zeroPoint) {
  const size_t row =
      static_cast<size_t>(blockIdx.x) * kGenericThreads + threadIdx.x;
  if (row >= batch)
    return;
  const int8_t* q = quantized + row * count;
  float* p = probabilities + row * count;
  float bestValue = 0.0f;
  int bestIndex = 0;
  for (int index = 0; index < count; ++index) {
    const float value = static_cast<float>(q[index] - zeroPoint) * scale;
    p[index] = value;
    if (index == 0 || value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  best[row] = bestIndex;
}

struct HostSet {
  int8_t* quantized = nullptr;
  float* probabilities = nullptr;
  int* best = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  options.count = kSmokeCount;
  options.launch = "direct";  // measured fastest; see the header
  size_t rotate = 3, lookahead = 0, chunkRows = kDefaultChunkRows;
  std::string mode = "zerocopy";
  const auto extra = [&](const std::string& argument) {
    if (argument.compare(0, 7, "--mode=") == 0) {
      mode = argument.substr(7);
      return true;
    }
    return parse_size_option(argument, "rotate", rotate) ||
           parse_size_option(argument, "lookahead", lookahead) ||
           parse_size_option(argument, "chunk", chunkRows);
  };
  if (!parse_options(argc, argv, options, /*allowBatch=*/true,
                     "smoke random ties", extra,
                     " [--rotate=K] [--mode={zerocopy,pipeline}] "
                     "[--chunk=ROWS] [--lookahead=L]"))
    return EXIT_FAILURE;
  if (options.count > INT_MAX || rotate == 0 || rotate > 64 ||
      chunkRows == 0 || (mode != "zerocopy" && mode != "pipeline")) {
    std::fprintf(stderr, "invalid options\n");
    return EXIT_FAILURE;
  }
  // Keep pipeline chunks on tile boundaries so every chunk stays aligned.
  chunkRows = (chunkRows + kTileRows - 1) / kTileRows * kTileRows;
  const int count = static_cast<int>(options.count);
  const size_t batch = options.batch;
  const size_t elements = options.count * batch;
  const bool smoke = options.inputs == "smoke";
  const float scale = smoke ? kSmokeScale : kRandomScale;
  const int zeroPoint = smoke ? kSmokeZeroPoint : kRandomZeroPoint;
  const bool tiled = count == kCategories;
  const bool zeroCopy = mode == "zerocopy";
  const size_t chunks = zeroCopy ? 1 : (batch + chunkRows - 1) / chunkRows;

  // Input set s: smoke rotated by s more, or seed + s.
  std::vector<HostSet> sets(rotate);
  for (size_t set = 0; set < rotate; ++set) {
    HostSet& h = sets[set];
    CUDA_CHECK(cudaMallocHost(&h.quantized, elements * sizeof(int8_t)));
    CUDA_CHECK(cudaMallocHost(&h.probabilities, elements * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h.best, batch * sizeof(int)));
    const unsigned seed = options.seed + static_cast<unsigned>(set);
    if (smoke)
      fill_smoke(h.quantized, options.count, batch, set);
    else if (options.inputs == "random")
      fill_random(h.quantized, elements, -128, 127, seed);
    else
      fill_random(h.quantized, elements, -kTiesRange, kTiesRange, seed);
    std::memset(h.probabilities, 0xff, elements * sizeof(float));
    std::memset(h.best, 0xff, batch * sizeof(int));
  }

  // Device buffers: the kernels-only variant's resident data, and the
  // pipeline's staging.
  int8_t* deviceQuantized = nullptr;
  float* deviceProbabilities = nullptr;
  int* deviceBest = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceQuantized, elements * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&deviceProbabilities, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceBest, batch * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(deviceQuantized, sets[0].quantized,
                        elements * sizeof(int8_t), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CopyComputePipeline pipeline;
  if (!zeroCopy && pipeline.create(chunks, lookahead))
    return EXIT_FAILURE;

  // Rows [first, first + rows) of q/p/best (device or mapped host pointers).
  const auto launch_kernel = [&](const int8_t* q, float* p, int* b,
                                 size_t first, size_t rows,
                                 cudaStream_t s) {
    if (tiled)
      kws_postprocess_sota_kernel<<<
          static_cast<unsigned>((rows + kTileRows - 1) / kTileRows),
          kTileRows, 0, s>>>(q + first * kCategories,
                             p + first * kCategories, b + first, rows, scale,
                             zeroPoint);
    else
      kws_postprocess_sota_generic_kernel<<<
          static_cast<unsigned>((rows + kGenericThreads - 1) /
                                kGenericThreads),
          kGenericThreads, 0, s>>>(q + first * options.count,
                                   p + first * options.count, b + first,
                                   count, rows, scale, zeroPoint);
    CUDA_CHECK(cudaGetLastError());
    return EXIT_SUCCESS;
  };
  // endToEnd 0: kernel only, on the device buffers. 1: input set `set`
  // from and back to its pinned host buffers.
  const auto enqueue = [&](int endToEnd, size_t set) {
    if (!endToEnd)
      return launch_kernel(deviceQuantized, deviceProbabilities, deviceBest,
                           0, batch, stream);
    const HostSet& h = sets[set];
    if (zeroCopy)
      return launch_kernel(h.quantized, h.probabilities, h.best, 0, batch,
                           stream);
    const auto first = [&](size_t chunk) { return chunk * chunkRows; };
    const auto rows = [&](size_t chunk) {
      return std::min(chunkRows, batch - chunk * chunkRows);
    };
    return pipeline.enqueue(
        stream,
        [&](size_t chunk, cudaStream_t s) {
          CUDA_CHECK(cudaMemcpyAsync(
              deviceQuantized + first(chunk) * options.count,
              h.quantized + first(chunk) * options.count,
              rows(chunk) * options.count, cudaMemcpyHostToDevice, s));
          return EXIT_SUCCESS;
        },
        [&](size_t chunk, cudaStream_t s) {
          return launch_kernel(deviceQuantized, deviceProbabilities,
                               deviceBest, first(chunk), rows(chunk), s);
        },
        [&](size_t chunk, cudaStream_t s) {
          CUDA_CHECK(cudaMemcpyAsync(
              h.probabilities + first(chunk) * options.count,
              deviceProbabilities + first(chunk) * options.count,
              rows(chunk) * options.count * sizeof(float),
              cudaMemcpyDeviceToHost, s));
          CUDA_CHECK(cudaMemcpyAsync(h.best + first(chunk),
                                     deviceBest + first(chunk),
                                     rows(chunk) * sizeof(int),
                                     cudaMemcpyDeviceToHost, s));
          return EXIT_SUCCESS;
        });
  };
  const bool graph = options.launch == "graph";
  // Graph mode: one kernels-only graph and one end-to-end graph per set.
  std::vector<cudaGraphExec_t> execs(graph ? 1 + rotate : 0);
  for (size_t index = 0; index < execs.size(); ++index)
    if (capture_graph(
            [&] { return enqueue(index > 0, index > 0 ? index - 1 : 0); },
            stream, &execs[index]))
      return EXIT_FAILURE;
  size_t iteration = 0;  // end-to-end iterations so far; picks the set
  const auto launch = [&](int endToEnd) {
    const size_t set = endToEnd ? iteration++ % rotate : 0;
    if (!graph)
      return enqueue(endToEnd, set);
    CUDA_CHECK(cudaGraphLaunch(execs[endToEnd ? 1 + set : 0], stream));
    return EXIT_SUCCESS;
  };

  // Every end-to-end iteration (warm-up included) writes into the host
  // outputs poisoned above; at least `rotate` of them run, so every set is
  // written, and every set is checked afterwards.
  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  if (time_launches([&] { return launch(0); }, stream, options.replays,
                    "kws.kernels", kernelMicroseconds) ||
      time_launches([&] { return launch(1); }, stream, options.replays,
                    "kws.end_to_end", endToEndMicroseconds,
                    static_cast<int>(rotate)))
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> expected(options.count);
  size_t mismatchedRows = 0, tiedRows = 0;
  for (size_t set = 0; set < rotate; ++set) {
    const HostSet& h = sets[set];
    for (size_t row = 0; row < batch; ++row) {
      const size_t base = row * options.count;
      const int expectedBest = loom_postprocess(
          h.quantized + base, expected.data(), count, scale, zeroPoint);
      const bool match = h.best[row] == expectedBest &&
                         std::memcmp(h.probabilities + base, expected.data(),
                                     options.count * sizeof(float)) == 0;
      int maximumCount = 0;
      for (int index = 0; index < count; ++index)
        maximumCount += expected[index] == expected[expectedBest];
      tiedRows += maximumCount > 1;
      if (!match && mismatchedRows++ < 5)
        std::fprintf(stderr, "set %zu row %zu: gpu best %d, loom best %d\n",
                     set, row, h.best[row], expectedBest);
    }
  }
  const HostSet& first = sets[0];
  const bool smokeShape = smoke && options.count == kSmokeCount;
  const bool smokeOracle = !smokeShape || (first.best[0] == 5 &&
                                           first.probabilities[5] == 1.5f &&
                                           first.probabilities[0] == -1.0f);
  if (!options.dump.empty() &&
      !write_dump(options.dump,
                  {{first.probabilities, elements * sizeof(float)},
                   {first.best, batch * sizeof(int)}}))
    return EXIT_FAILURE;

  const char* how = graph ? "graph replay" : "direct launch";
  std::printf("mlperf-classification kws postprocess sota: count=%d "
              "batch=%zu inputs=%s scale=%g zero_point=%d launch=%s "
              "replays=%d rotate=%zu mode=%s\n",
              count, batch, options.inputs.c_str(), scale, zeroPoint,
              options.launch.c_str(), options.replays, rotate, mode.c_str());
  std::printf("  row 0        best=%d p[best]=%g\n", first.best[0],
              first.probabilities[first.best[0]]);
  if (smokeShape)
    std::printf("  smoke oracle %s (best == 5, p[5] == 1.5, p[0] == -1)\n",
                smokeOracle ? "pass" : "FAIL");
  std::printf("  check        %s (%zu/%zu rows bitwise equal to loom loop "
              "over %zu input sets; %zu rows with a tied maximum)\n",
              mismatchedRows == 0 ? "pass" : "FAIL",
              rotate * batch - mismatchedRows, rotate * batch, rotate,
              tiedRows);
  std::printf("  kernel       %.3f us/iter (%s)\n", kernelMicroseconds, how);
  if (zeroCopy)
    std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H as one "
                "zero-copy kernel on pinned host memory, %s)\n",
                endToEndMicroseconds, how);
  else
    std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H, %zu-chunk "
                "pipeline, %s)\n",
                endToEndMicroseconds, chunks, how);
  std::printf("  throughput   %.1f M rows/s (kernel)\n",
              static_cast<double>(batch) / kernelMicroseconds);
  std::printf("  launches     kernels-only 1 x %s; end-to-end %zu x %s\n",
              tiled ? "kws_postprocess_sota_kernel"
                    : "kws_postprocess_sota_generic_kernel",
              chunks,
              tiled ? "kws_postprocess_sota_kernel"
                    : "kws_postprocess_sota_generic_kernel");
  print_timed_total(options.replays, kernelMicroseconds,
                    endToEndMicroseconds);

  for (cudaGraphExec_t exec : execs)
    CUDA_CHECK(cudaGraphExecDestroy(exec));
  if (!zeroCopy && pipeline.destroy())
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceQuantized));
  CUDA_CHECK(cudaFree(deviceProbabilities));
  CUDA_CHECK(cudaFree(deviceBest));
  for (HostSet& h : sets) {
    CUDA_CHECK(cudaFreeHost(h.quantized));
    CUDA_CHECK(cudaFreeHost(h.probabilities));
    CUDA_CHECK(cudaFreeHost(h.best));
  }
  return mismatchedRows == 0 && smokeOracle ? EXIT_SUCCESS : EXIT_FAILURE;
}
