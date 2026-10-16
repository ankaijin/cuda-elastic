// SOTA CUDA version of mlperf_tiny_vww_preprocess_kernel (see
// vww_preprocess.cu for the port it is checked against):
//
//   centered[i] = static_cast<int8_t>(static_cast<int>(input[i]) - 128)
//
// i.e. flip the top bit of every byte. Same options, inputs, NVTX ranges and
// output lines as vww_preprocess, plus:
//   --rotate=K   rotate the timed iterations over K input sets (default 3),
//                each with its own pinned host input and output buffers, so
//                every iteration works on new data; every set's output is
//                checked after timing.
//   --mode=M     how an end-to-end iteration moves data:
//                  pipeline  (default) chunked H2D / kernel / D2H on three
//                            streams (CopyComputePipeline): H2D of chunk k+1,
//                            the kernel on chunk k and D2H of chunk k-1 run
//                            at once on the two copy engines and the SMs.
//                  zerocopy  one kernel reads mapped pinned host memory and
//                            writes the result straight back over PCIe.
//   --chunk=B    bytes per pipeline chunk (default 4 MiB, rounded up to a
//                multiple of 16).
//   --lookahead=L  pipeline only: H2D may run at most L chunks ahead of D2H
//                (default 0, unlimited; throttling measured no gain).
// --launch defaults to direct: the pipeline's 64 chunks replayed as a graph
// measured about 0.5 ms slower per iteration than enqueued directly.
//
// The work is PCIe-bound: 256 MB each way. Measured on the RTX 3090 (PCIe
// 4.0 x16): pinned cudaMemcpyAsync moves 24.3 GB/s H2D and 26.4 GB/s D2H
// alone, and 45.4 GB/s in total with one 256 MB copy each way at once
// (11.85 ms, the floor for this shape). This pipeline reaches 12.2 ms; the
// zero-copy kernel, also bidirectional, 12.45-12.6 ms; chunk sizes from
// 1 to 8 MiB, ramped first/last chunks and H2D throttling all measured
// within noise or worse. The kernel gives each thread four
// 16 B vectors (all loads issued before the stores), one 16 KB tile per
// block, with a scalar tail for counts that are not a multiple of 16.
//
// --dump=FILE writes input set 0's output (set 0 is the baseline's input):
// count int8 bytes.
#include <cstdint>
#include <random>
#include <vector>

#include "classification_harness.cuh"

namespace {

constexpr size_t kSmokeCount = 16;  // kVwwInputSize
constexpr uint8_t kSmokeInput[kSmokeCount] = {0,   1,   7,   31,  63,  95,
                                              126, 127, 128, 129, 160, 192,
                                              224, 248, 254, 255};

constexpr int kThreads = 256;
constexpr int kVectorsPerThread = 4;
constexpr size_t kTileVectors = kThreads * kVectorsPerThread;  // 16 KB
constexpr size_t kDefaultChunkBytes = 4u << 20;

// Verbatim loop of mlperf_tiny_vww_preprocess_kernel.
void loom_preprocess(const uint8_t* input, int8_t* centered, size_t count) {
  for (size_t index = 0; index < count; ++index)
    centered[index] = static_cast<int8_t>(static_cast<int>(input[index]) - 128);
}

// The smoke vector, starting at element offset, repeated to fill count
// (offset 0 is the baseline's smoke input; other sets use the set index).
void fill_smoke(uint8_t* input, size_t count, size_t offset) {
  for (size_t index = 0; index < count; ++index)
    input[index] = kSmokeInput[(index + offset) % kSmokeCount];
}

void fill_random(uint8_t* input, size_t count, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);
  for (size_t index = 0; index < count; ++index)
    input[index] = static_cast<uint8_t>(distribution(generator));
}

__device__ __forceinline__ uint4 center(uint4 bytes) {
  bytes.x ^= 0x80808080u;
  bytes.y ^= 0x80808080u;
  bytes.z ^= 0x80808080u;
  bytes.w ^= 0x80808080u;
  return bytes;
}

// Block b handles vectors [kTileVectors b, kTileVectors (b + 1)); the bytes
// past the last whole vector are done by the first threads of block 0.
// input and centered must be 16 B aligned; they may be device memory or
// mapped pinned host memory (zero-copy).
__global__ void __launch_bounds__(kThreads)
    vww_preprocess_sota_kernel(const uint8_t* __restrict__ input,
                               int8_t* __restrict__ centered, size_t count) {
  const size_t vectors = count / 16;
  const uint4* input16 = reinterpret_cast<const uint4*>(input);
  uint4* centered16 = reinterpret_cast<uint4*>(centered);
  const size_t base =
      static_cast<size_t>(blockIdx.x) * kTileVectors + threadIdx.x;
  if (base + (kVectorsPerThread - 1) * kThreads < vectors) {
    uint4 bytes[kVectorsPerThread];
#pragma unroll
    for (int v = 0; v < kVectorsPerThread; ++v)
      bytes[v] = input16[base + v * kThreads];
#pragma unroll
    for (int v = 0; v < kVectorsPerThread; ++v)
      centered16[base + v * kThreads] = center(bytes[v]);
  } else {
#pragma unroll
    for (int v = 0; v < kVectorsPerThread; ++v)
      if (base + v * kThreads < vectors)
        centered16[base + v * kThreads] =
            center(input16[base + v * kThreads]);
  }
  if (blockIdx.x == 0 && vectors * 16 + threadIdx.x < count) {
    const size_t index = vectors * 16 + threadIdx.x;
    centered[index] =
        static_cast<int8_t>(static_cast<int>(input[index]) - 128);
  }
}

unsigned blocks_for(size_t count) {
  return static_cast<unsigned>(
      std::max<size_t>((count / 16 + kTileVectors - 1) / kTileVectors, 1));
}

struct HostSet {
  uint8_t* input = nullptr;
  int8_t* centered = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  options.count = kSmokeCount;
  options.launch = "direct";  // measured fastest; see the header
  size_t rotate = 3, lookahead = 0, chunkBytes = kDefaultChunkBytes;
  std::string mode = "pipeline";
  const auto extra = [&](const std::string& argument) {
    if (argument.compare(0, 7, "--mode=") == 0) {
      mode = argument.substr(7);
      return true;
    }
    return parse_size_option(argument, "rotate", rotate) ||
           parse_size_option(argument, "lookahead", lookahead) ||
           parse_size_option(argument, "chunk", chunkBytes);
  };
  if (!parse_options(argc, argv, options, /*allowBatch=*/false,
                     "smoke random", extra,
                     " [--rotate=K] [--mode={pipeline,zerocopy}] "
                     "[--chunk=BYTES] [--lookahead=L]"))
    return EXIT_FAILURE;
  if (rotate == 0 || rotate > 64 || chunkBytes == 0 ||
      (mode != "zerocopy" && mode != "pipeline")) {
    std::fprintf(stderr, "invalid options\n");
    return EXIT_FAILURE;
  }
  chunkBytes = (chunkBytes + 15) / 16 * 16;  // chunks stay 16 B aligned
  const size_t count = options.count;
  const bool smoke = options.inputs == "smoke";
  const bool zeroCopy = mode == "zerocopy";
  const size_t chunks = zeroCopy ? 1 : (count + chunkBytes - 1) / chunkBytes;

  std::vector<HostSet> sets(rotate);
  for (size_t set = 0; set < rotate; ++set) {
    HostSet& h = sets[set];
    CUDA_CHECK(cudaMallocHost(&h.input, count));
    CUDA_CHECK(cudaMallocHost(&h.centered, count));
    if (smoke)
      fill_smoke(h.input, count, set);
    else
      fill_random(h.input, count, options.seed + static_cast<unsigned>(set));
    std::memset(h.centered, 0x55, count);
  }

  uint8_t* deviceInput = nullptr;
  int8_t* deviceCentered = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInput, count));
  CUDA_CHECK(cudaMalloc(&deviceCentered, count));
  CUDA_CHECK(cudaMemcpy(deviceInput, sets[0].input, count,
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CopyComputePipeline pipeline;
  if (!zeroCopy && pipeline.create(chunks, lookahead))
    return EXIT_FAILURE;

  const auto launch_kernel = [&](const uint8_t* in, int8_t* out, size_t n,
                                 cudaStream_t s) {
    vww_preprocess_sota_kernel<<<blocks_for(n), kThreads, 0, s>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
    return EXIT_SUCCESS;
  };
  // endToEnd 0: kernel only, on the device buffers. 1: input set `set`
  // from and back to its pinned host buffers.
  const auto enqueue = [&](int endToEnd, size_t set) {
    if (!endToEnd)
      return launch_kernel(deviceInput, deviceCentered, count, stream);
    const HostSet& h = sets[set];
    if (zeroCopy)
      return launch_kernel(h.input, h.centered, count, stream);
    const auto bytes = [&](size_t chunk) {
      return std::min(chunkBytes, count - chunk * chunkBytes);
    };
    return pipeline.enqueue(
        stream,
        [&](size_t chunk, cudaStream_t s) {
          const size_t offset = chunk * chunkBytes;
          CUDA_CHECK(cudaMemcpyAsync(deviceInput + offset, h.input + offset,
                                     bytes(chunk), cudaMemcpyHostToDevice,
                                     s));
          return EXIT_SUCCESS;
        },
        [&](size_t chunk, cudaStream_t s) {
          const size_t offset = chunk * chunkBytes;
          return launch_kernel(deviceInput + offset, deviceCentered + offset,
                               bytes(chunk), s);
        },
        [&](size_t chunk, cudaStream_t s) {
          const size_t offset = chunk * chunkBytes;
          CUDA_CHECK(cudaMemcpyAsync(h.centered + offset,
                                     deviceCentered + offset, bytes(chunk),
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
                    "vww.kernels", kernelMicroseconds) ||
      time_launches([&] { return launch(1); }, stream, options.replays,
                    "vww.end_to_end", endToEndMicroseconds,
                    static_cast<int>(rotate)))
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<int8_t> expected(count);
  size_t mismatches = 0;
  for (size_t set = 0; set < rotate; ++set) {
    const HostSet& h = sets[set];
    loom_preprocess(h.input, expected.data(), count);
    for (size_t index = 0; index < count; ++index)
      if (h.centered[index] != expected[index] && mismatches++ < 5)
        std::fprintf(stderr, "set %zu element %zu: input %d, gpu %d, loom %d\n",
                     set, index, h.input[index], h.centered[index],
                     expected[index]);
  }
  if (!options.dump.empty() &&
      !write_dump(options.dump, {{sets[0].centered, count}}))
    return EXIT_FAILURE;

  const char* how = graph ? "graph replay" : "direct launch";
  std::printf("mlperf-classification vww preprocess sota: count=%zu "
              "inputs=%s blocks=%u launch=%s replays=%d rotate=%zu "
              "mode=%s\n",
              count, options.inputs.c_str(), blocks_for(count),
              options.launch.c_str(), options.replays, rotate, mode.c_str());
  std::printf("  check        %s (%zu/%zu bytes equal to loom loop over %zu "
              "input sets)\n",
              mismatches == 0 ? "pass" : "FAIL", rotate * count - mismatches,
              rotate * count, rotate);
  std::printf("  kernel       %.3f us/iter (%s)\n", kernelMicroseconds, how);
  if (zeroCopy)
    std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H as one "
                "zero-copy kernel on pinned host memory, %s)\n",
                endToEndMicroseconds, how);
  else
    std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H, %zu-chunk "
                "pipeline, %s)\n",
                endToEndMicroseconds, chunks, how);
  std::printf("  bandwidth    %.1f GB/s (kernel, 2 B/element)\n",
              2.0 * static_cast<double>(count) / (kernelMicroseconds * 1.0e3));
  std::printf("  launches     kernels-only 1 x vww_preprocess_sota_kernel; "
              "end-to-end %zu x vww_preprocess_sota_kernel\n",
              chunks);
  print_timed_total(options.replays, kernelMicroseconds,
                    endToEndMicroseconds);

  for (cudaGraphExec_t exec : execs)
    CUDA_CHECK(cudaGraphExecDestroy(exec));
  if (!zeroCopy && pipeline.destroy())
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceCentered));
  for (HostSet& h : sets) {
    CUDA_CHECK(cudaFreeHost(h.input));
    CUDA_CHECK(cudaFreeHost(h.centered));
  }
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
