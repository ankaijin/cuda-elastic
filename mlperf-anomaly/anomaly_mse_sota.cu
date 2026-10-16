// Fastest-known CUDA version of mlperf_tiny_anomaly_mse_kernel on the RTX 3090
// (see anomaly_mse.cu for the baseline port and the Loom source it follows):
//
//   mse = sum_i ((float)(output[i] - zeroPoint) * scale - input[i])^2 / count
//
// The end-to-end iteration moves count * 5 bytes from pinned host memory, so
// it is bound by PCIe H2D bandwidth (~24.3 GB/s measured on this RTX 3090 for
// one 320 MB cudaMemcpyAsync pair, the same for chunked copies of 20 MB and
// up and for cudaMemcpyBatchAsync; the reduction itself runs at ~880 GB/s
// from DRAM). The design hides the reduction behind the copy and adds as
// little as possible after the last byte arrives:
//
//   * One kernel, mse_sota_kernel, replaces mse_partial + mse_finalize: each
//     block writes its fp32 partial and takes an atomic ticket; the last block
//     of the iteration sums all partials in double precision and writes the
//     result. The end-to-end result goes straight to the pinned host result
//     slot (mapped through UVA), so there is no D2H memcpy. A single-block
//     launch (small counts) skips the ticket.
//   * pipeline path (count >= --zerocopy-below, default 16M): the int8 array
//     is copied in one piece, then the float array in geometrically shrinking
//     chunks (each leaves 1/--ratio of what remains, down to --min-chunk
//     elements; 48M, 12M, 3M, 1M at 64M). A second stream runs
//     mse_sota_kernel on each chunk as soon as its copy lands, so only the
//     last 1M-element chunk's kernel is exposed after the copy engine
//     finishes. (--zerocopy-tail=N instead reduces the last ~N elements with
//     a zero-copy launch at the end of the copy stream. It is off by default:
//     it is no faster in direct mode, and in graph mode with rotation, graphs
//     that mix copies with a zero-copy kernel ran ~200-300 us slower.)
//   * zerocopy path (count < --zerocopy-below, or --path=zerocopy): one
//     mse_sota_kernel reads the mapped pinned host buffers directly over
//     PCIe, with no copies at all. It is the lowest-latency path for small
//     counts (16, 640) and streams at ~24.25 GB/s, which wins up to ~16M
//     elements; at 64M the copy engine's slightly higher rate (~0.3%) wins.
//
// Every timed end-to-end iteration reads the full inputs from pinned host
// memory; nothing is kept on the device between iterations. --rotate=K
// (default 3) cycles the timed iterations through K distinct input sets (set
// k uses seed + k, or the smoke pattern shifted by k), each with its own host
// buffers and result slot, and every set's result is re-checked after timing
// against its own double-precision reference. Set 0 is the baseline's input.
//
// The kernels-only variant (NVTX anomaly.kernels) runs mse_sota_kernel once
// over device-resident copies of the K sets, rotating the same way.
//
// --launch=graph (default) captures one graph per (variant, set) and replays
// it; --launch=direct enqueues the same work directly. --dump=<file> writes
// count and each set's float32 result after the correctness pass, in the same
// format as anomaly_mse --dump.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define ANOMALY_HAS_NVTX 1
#  endif
#endif
#ifndef ANOMALY_HAS_NVTX
#  define ANOMALY_HAS_NVTX 0
#endif

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                   cudaGetErrorString(error__), __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (0)

namespace {

constexpr size_t kSmokeCount = 16;
constexpr float kSmokeScale = 0.25f;
constexpr int kSmokeZeroPoint = 0;
constexpr float kSmokeExpected = 0.015625f;
constexpr float kSmokeTolerance = 1.0e-6f;

constexpr float kRandomScale = 0.0625f;
constexpr int kRandomZeroPoint = -3;

constexpr double kRelativeTolerance = 1.0e-5;
constexpr double kAbsoluteTolerance = 1.0e-7;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlocksPerSm = 8;
// Chunk boundaries are multiples of this many elements, so every chunk but
// the last starts and ends on a float4/char4 boundary.
constexpr size_t kChunkAlign = 4096;

// ---------------------------------------------------------------------------
// Host oracles (same as anomaly_mse.cu).

float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint) {
  return static_cast<float>(value - zeroPoint) * scale;
}

float loom_mse(const int8_t* output, const float* input, size_t count,
               float scale, int zeroPoint) {
  float differenceSum = 0.0f;
  for (size_t index = 0; index < count; ++index) {
    const float converted =
        DequantizeInt8ToFloat(output[index], scale, zeroPoint);
    const float difference = converted - input[index];
    differenceSum += difference * difference;
  }
  return differenceSum / static_cast<float>(count);
}

double reference_mse(const int8_t* output, const float* input, size_t count,
                     float scale, int zeroPoint) {
  double differenceSum = 0.0;
  for (size_t index = 0; index < count; ++index) {
    const double difference =
        static_cast<double>(DequantizeInt8ToFloat(output[index], scale,
                                                  zeroPoint)) -
        static_cast<double>(input[index]);
    differenceSum += difference * difference;
  }
  return differenceSum / static_cast<double>(count);
}

// The baseline's smoke pattern, shifted by offset (offset 0 is the baseline).
void fill_smoke(int8_t* output, float* input, size_t count, size_t offset) {
  for (size_t position = 0; position < count; ++position) {
    const size_t index = position + offset;
    const int value = static_cast<int>(index % 17) - 8;
    input[position] = static_cast<float>(value) * 0.25f;
    output[position] = static_cast<int8_t>(value + ((index & 3U) == 0));
  }
}

void fill_random(int8_t* output, float* input, size_t count,
                 unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> quantized(-128, 127);
  std::uniform_real_distribution<float> noise(-4.0f, 4.0f);
  for (size_t index = 0; index < count; ++index) {
    output[index] = static_cast<int8_t>(quantized(generator));
    input[index] =
        DequantizeInt8ToFloat(output[index], kRandomScale, kRandomZeroPoint) +
        noise(generator) * kRandomScale;
  }
}

// ---------------------------------------------------------------------------
// Device code.

__device__ __forceinline__ void accumulate(float& sum, int8_t output,
                                           float input, float scale,
                                           int zeroPoint) {
  const float difference =
      static_cast<float>(output - zeroPoint) * scale - input;
  sum += difference * difference;
}

__device__ __forceinline__ void accumulate4(float& sum, char4 o, float4 x,
                                            float scale, int zeroPoint) {
  accumulate(sum, o.x, x.x, scale, zeroPoint);
  accumulate(sum, o.y, x.y, scale, zeroPoint);
  accumulate(sum, o.z, x.z, scale, zeroPoint);
  accumulate(sum, o.w, x.w, scale, zeroPoint);
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// Block-wide sum, valid in thread 0. Requires kThreads threads.
template <typename T>
__device__ __forceinline__ T block_sum(T value) {
  __shared__ T warp_partials[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0)
    warp_partials[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? warp_partials[lane] : T(0);
    value = warp_sum(value);
  }
  return value;
}

// Reduces elements [0, length) of output/input (a chunk of the whole array,
// or all of it; device memory or mapped pinned host memory) into one fp32
// partial per block, stored at partials[partialBase + blockIdx.x]. Every block
// then takes a ticket; the block that takes ticket totalBlocks - 1 (the last
// of all launches in the iteration) sums the totalBlocks partials in double,
// writes sum / count to *result and resets the ticket for the next iteration.
// The chunk must start on a 16-byte boundary of input and 4-byte boundary of
// output; length % 4 tail elements are handled by the first threads.
__global__ void __launch_bounds__(kThreads)
    mse_sota_kernel(const int8_t* __restrict__ output,
                    const float* __restrict__ input, size_t length,
                    float scale, int zeroPoint, float* partials,
                    unsigned partialBase, unsigned totalBlocks,
                    unsigned* ticket, size_t count, float* result) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * kThreads +
                        threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  const size_t vectorCount = length / 4;
  const char4* output4 = reinterpret_cast<const char4*>(output);
  const float4* input4 = reinterpret_cast<const float4*>(input);

  // Two independent fp32 accumulators with two vector loads in flight.
  float sum0 = 0.0f, sum1 = 0.0f;
  size_t index = thread;
  for (; index + stride < vectorCount; index += 2 * stride) {
    const char4 o0 = output4[index];
    const float4 x0 = input4[index];
    const char4 o1 = output4[index + stride];
    const float4 x1 = input4[index + stride];
    accumulate4(sum0, o0, x0, scale, zeroPoint);
    accumulate4(sum1, o1, x1, scale, zeroPoint);
  }
  if (index < vectorCount)
    accumulate4(sum0, output4[index], input4[index], scale, zeroPoint);
  for (size_t tail = vectorCount * 4 + thread; tail < length; tail += stride)
    accumulate(sum1, output[tail], input[tail], scale, zeroPoint);

  const float sum = block_sum(sum0 + sum1);
  if (totalBlocks == 1) {  // small counts: one block, no ticket needed
    if (threadIdx.x == 0)
      *result = static_cast<float>(static_cast<double>(sum) /
                                   static_cast<double>(count));
    return;
  }
  __shared__ bool lastBlock;
  if (threadIdx.x == 0) {
    partials[partialBase + blockIdx.x] = sum;
    __threadfence();
    lastBlock = atomicAdd(ticket, 1u) == totalBlocks - 1;
  }
  __syncthreads();
  if (!lastBlock)
    return;
  __threadfence();
  double total = 0.0;
  for (unsigned partial = threadIdx.x; partial < totalBlocks;
       partial += kThreads)
    total += static_cast<double>(__ldcg(partials + partial));
  total = block_sum(total);
  if (threadIdx.x == 0) {
    *result = static_cast<float>(total / static_cast<double>(count));
    *ticket = 0;
  }
}

// ---------------------------------------------------------------------------
// Host harness.

struct Options {
  size_t count = kSmokeCount;
  std::string inputs = "smoke";
  std::string launch = "graph";  // graph | direct
  std::string path = "auto";     // auto | pipeline | zerocopy
  int replays = 10000;
  unsigned seed = 1;
  int rotate = 3;
  // Pipeline tuning (elements): the zero-copy tail, the smallest copied
  // float chunk, the chunk ratio and the blocks per SM of copied chunks.
  size_t zerocopyTail = 0;
  size_t minChunk = size_t(1) << 20;
  int ratio = 4;
  int chunkBlocksPerSm = 2;
  size_t zerocopyBelow = size_t(1) << 24;   // auto: zerocopy below this
  std::string dump;
};

bool parse_options(int argc, char** argv, Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char* prefix) -> const char* {
      const size_t length = std::strlen(prefix);
      return argument.compare(0, length, prefix) == 0
                 ? argument.c_str() + length
                 : nullptr;
    };
    if (const char* v = value("--count="))
      options.count = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--inputs="))
      options.inputs = v;
    else if (const char* v = value("--launch="))
      options.launch = v;
    else if (const char* v = value("--path="))
      options.path = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else if (const char* v = value("--rotate="))
      options.rotate = std::atoi(v);
    else if (const char* v = value("--zerocopy-tail="))
      options.zerocopyTail = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--min-chunk="))
      options.minChunk = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--chunk-blocks-per-sm="))
      options.chunkBlocksPerSm = std::atoi(v);
    else if (const char* v = value("--ratio="))
      options.ratio = std::atoi(v);
    else if (const char* v = value("--zerocopy-below="))
      options.zerocopyBelow = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--dump="))
      options.dump = v;
    else {
      std::fprintf(stderr,
                   "usage: %s [--count=N] [--inputs=smoke|random] "
                   "[--launch=graph|direct] [--replays=R] [--seed=S] "
                   "[--rotate=K] [--dump=FILE] [--path=auto|pipeline|zerocopy] "
                   "[--zerocopy-tail=N] [--min-chunk=N] [--ratio=R] "
                   "[--chunk-blocks-per-sm=B] [--zerocopy-below=N]\n",
                   argv[0]);
      return false;
    }
  }
  if (options.count == 0 || options.replays <= 0 || options.rotate <= 0 ||
      options.ratio < 2 || options.minChunk == 0 ||
      options.chunkBlocksPerSm <= 0 ||
      (options.inputs != "smoke" && options.inputs != "random") ||
      (options.launch != "graph" && options.launch != "direct") ||
      (options.path != "auto" && options.path != "pipeline" &&
       options.path != "zerocopy")) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

void nvtx_push(const char* label) {
#if ANOMALY_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

void nvtx_pop() {
#if ANOMALY_HAS_NVTX
  nvtxRangePop();
#endif
}

template <typename Enqueue>
int capture_graph(Enqueue enqueue, cudaStream_t stream,
                  cudaGraphExec_t* exec) {
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  if (enqueue() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(exec, graph, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return EXIT_SUCCESS;
}

// Average microseconds per iteration, after warm-up. launch(replay) enqueues
// iteration replay on stream; before_timed() runs between warm-up and the
// timed loop. Each timed iteration is an NVTX range named range.
template <typename Launch, typename BeforeTimed>
int time_launches(Launch launch, BeforeTimed before_timed,
                  cudaStream_t stream, int replays, const char* range,
                  double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int replay = 0; replay < std::min(replays, 100); ++replay)
    if (launch(replay) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  before_timed();
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int replay = 0; replay < replays; ++replay) {
    nvtx_push(range);
    const int status = launch(replay);
    nvtx_pop();
    if (status != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  microseconds = 1000.0 * milliseconds / replays;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return EXIT_SUCCESS;
}

struct Chunk {
  size_t begin, length;
  int blocks;
  unsigned partialBase;
  bool zerocopy;  // read straight from pinned host memory
};

int blocks_for(size_t length, int maxBlocks) {
  const size_t work = std::max<size_t>(length / 4, 1);
  return static_cast<int>(std::min<size_t>((work + kThreads - 1) / kThreads,
                                           static_cast<size_t>(maxBlocks)));
}

// End-to-end plan. Elements [0, copied) are copied to the device (int8 in one
// piece, float in chunks that each leave 1/ratio of what remains, rounded to
// kChunkAlign, until at most minChunk elements are left for the last one) and
// reduced chunk by chunk on the compute stream; the last chunk also takes any
// count % 4 remainder. [copied, count) is reduced by one zero-copy launch at
// the end of the copy stream. copied is count (default), count minus
// zerocopyTail rounded down to kChunkAlign, or 0 for the zero-copy path.
std::vector<Chunk> plan_chunks(size_t count, size_t copied, size_t minChunk,
                               int ratio, int chunkBlocks, int zerocopyBlocks,
                               unsigned& totalBlocks) {
  std::vector<Chunk> chunks;
  size_t begin = 0;
  totalBlocks = 0;
  while (begin < copied) {
    const size_t remaining = copied - begin;
    size_t length = remaining;
    if (remaining > minChunk) {
      const size_t left = (remaining / ratio + kChunkAlign - 1) /
                          kChunkAlign * kChunkAlign;
      if (left < remaining && remaining - left >= kChunkAlign)
        length = (remaining - left) / kChunkAlign * kChunkAlign;
    }
    const int blocks = blocks_for(length, chunkBlocks);
    chunks.push_back({begin, length, blocks, totalBlocks, false});
    totalBlocks += blocks;
    begin += length;
  }
  if (copied < count) {
    const int blocks = blocks_for(count - copied, zerocopyBlocks);
    chunks.push_back({copied, count - copied, blocks, totalBlocks, true});
    totalBlocks += blocks;
  }
  return chunks;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options))
    return EXIT_FAILURE;
  const size_t count = options.count;
  const int sets = options.rotate;
  const bool smoke = options.inputs == "smoke";
  const float scale = smoke ? kSmokeScale : kRandomScale;
  const int zeroPoint = smoke ? kSmokeZeroPoint : kRandomZeroPoint;
  const bool zerocopy =
      options.path == "zerocopy" ||
      (options.path == "auto" && count < options.zerocopyBelow);

  int device = 0, smCount = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
  const int maxBlocks = smCount * kBlocksPerSm;
  const int blocks = blocks_for(count, maxBlocks);  // one launch over all

  const size_t copiedCount =
      zerocopy || count <= options.zerocopyTail ? 0
      : options.zerocopyTail == 0
          ? count
          : (count - options.zerocopyTail) / kChunkAlign * kChunkAlign;
  unsigned endToEndBlocks = 0;
  const std::vector<Chunk> chunks = plan_chunks(
      count, copiedCount, options.minChunk, options.ratio,
      smCount * options.chunkBlocksPerSm, maxBlocks, endToEndBlocks);

  // Pinned host inputs, one pair per set, and one result slot per set.
  std::vector<int8_t*> hostOutput(sets, nullptr);
  std::vector<float*> hostInput(sets, nullptr);
  float* hostResult = nullptr;
  for (int set = 0; set < sets; ++set) {
    CUDA_CHECK(cudaMallocHost(&hostOutput[set], count * sizeof(int8_t)));
    CUDA_CHECK(cudaMallocHost(&hostInput[set], count * sizeof(float)));
    if (smoke)
      fill_smoke(hostOutput[set], hostInput[set], count, set);
    else
      fill_random(hostOutput[set], hostInput[set], count, options.seed + set);
  }
  CUDA_CHECK(cudaMallocHost(&hostResult, sets * sizeof(float)));
  // With UVA, cudaMallocHost memory is mapped at the same address on the
  // device; zero-copy reads and the result write use the host pointers.
  int unifiedAddressing = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&unifiedAddressing,
                                    cudaDevAttrUnifiedAddressing, device));
  if (!unifiedAddressing) {
    std::fprintf(stderr, "unified addressing required\n");
    return EXIT_FAILURE;
  }

  // Device-resident sets for the kernels-only variant, and (pipeline) one
  // staging pair the end-to-end copies land in.
  std::vector<int8_t*> deviceOutput(sets, nullptr);
  std::vector<float*> deviceInput(sets, nullptr);
  for (int set = 0; set < sets; ++set) {
    CUDA_CHECK(cudaMalloc(&deviceOutput[set], count * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&deviceInput[set], count * sizeof(float)));
  }
  int8_t* stageOutput = nullptr;
  float* stageInput = nullptr;
  if (copiedCount) {
    CUDA_CHECK(cudaMalloc(&stageOutput, copiedCount * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&stageInput, copiedCount * sizeof(float)));
  }
  float* devicePartials = nullptr;
  float* deviceResult = nullptr;
  unsigned* deviceTicket = nullptr;
  CUDA_CHECK(cudaMalloc(&devicePartials,
                        std::max<unsigned>(endToEndBlocks, blocks) *
                            sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceResult, sets * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceTicket, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(deviceTicket, 0, sizeof(unsigned)));

  cudaStream_t stream, computeStream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
  std::vector<cudaEvent_t> copied(chunks.size());
  for (cudaEvent_t& event : copied)
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  cudaEvent_t joined;
  CUDA_CHECK(cudaEventCreateWithFlags(&joined, cudaEventDisableTiming));

  // endToEnd 0: kernels only, set already resident on the device.
  // endToEnd 1: full H2D of the set's inputs (copies or zero-copy reads),
  //             reduction, result written to the set's pinned host slot.
  const auto enqueue = [&](int endToEnd, int set) {
    if (!endToEnd) {
      mse_sota_kernel<<<blocks, kThreads, 0, stream>>>(
          deviceOutput[set], deviceInput[set], count, scale, zeroPoint,
          devicePartials, 0, blocks, deviceTicket, count,
          deviceResult + set);
      CUDA_CHECK(cudaGetLastError());
      return EXIT_SUCCESS;
    }
    if (copiedCount)
      CUDA_CHECK(cudaMemcpyAsync(stageOutput, hostOutput[set],
                                 copiedCount * sizeof(int8_t),
                                 cudaMemcpyHostToDevice, stream));
    for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
      const Chunk& c = chunks[chunk];
      if (c.zerocopy) {
        mse_sota_kernel<<<c.blocks, kThreads, 0, stream>>>(
            hostOutput[set] + c.begin, hostInput[set] + c.begin, c.length,
            scale, zeroPoint, devicePartials, c.partialBase, endToEndBlocks,
            deviceTicket, count, hostResult + set);
        CUDA_CHECK(cudaGetLastError());
        continue;
      }
      CUDA_CHECK(cudaMemcpyAsync(stageInput + c.begin,
                                 hostInput[set] + c.begin,
                                 c.length * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaEventRecord(copied[chunk], stream));
      CUDA_CHECK(cudaStreamWaitEvent(computeStream, copied[chunk], 0));
      mse_sota_kernel<<<c.blocks, kThreads, 0, computeStream>>>(
          stageOutput + c.begin, stageInput + c.begin, c.length, scale,
          zeroPoint, devicePartials, c.partialBase, endToEndBlocks,
          deviceTicket, count, hostResult + set);
      CUDA_CHECK(cudaGetLastError());
    }
    if (copiedCount) {
      CUDA_CHECK(cudaEventRecord(joined, computeStream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, joined, 0));
    }
    return EXIT_SUCCESS;
  };
  const bool graph = options.launch == "graph";
  std::vector<cudaGraphExec_t> execs[2] = {
      std::vector<cudaGraphExec_t>(sets), std::vector<cudaGraphExec_t>(sets)};
  for (int endToEnd = 0; graph && endToEnd < 2; ++endToEnd)
    for (int set = 0; set < sets; ++set)
      if (capture_graph([&] { return enqueue(endToEnd, set); }, stream,
                        &execs[endToEnd][set]))
        return EXIT_FAILURE;
  const auto launch = [&](int endToEnd, int set) {
    if (!graph)
      return enqueue(endToEnd, set);
    CUDA_CHECK(cudaGraphLaunch(execs[endToEnd][set], stream));
    return EXIT_SUCCESS;
  };

  // References for every set.
  std::vector<double> reference(sets);
  for (int set = 0; set < sets; ++set)
    reference[set] = reference_mse(hostOutput[set], hostInput[set], count,
                                   scale, zeroPoint);
  const auto within = [&](float gpu, int set) {
    const double error =
        std::fabs(static_cast<double>(gpu) - reference[set]);
    return error <= kRelativeTolerance * std::fabs(reference[set]) +
                        kAbsoluteTolerance;
  };

  // Correctness: one end-to-end iteration per set from poisoned results.
  for (int set = 0; set < sets; ++set)
    hostResult[set] = NAN;
  for (int set = 0; set < sets; ++set)
    if (launch(1, set))
      return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> correctness(hostResult, hostResult + sets);
  bool correctnessPass = true;
  for (int set = 0; set < sets; ++set)
    correctnessPass = correctnessPass && within(correctness[set], set);
  const float gpu = correctness[0];
  const float loom =
      loom_mse(hostOutput[0], hostInput[0], count, scale, zeroPoint);
  const double error = std::fabs(static_cast<double>(gpu) - reference[0]);
  const bool smokeShape = smoke && count == kSmokeCount;
  const bool smokeOracle =
      !smokeShape || std::fabs(gpu - kSmokeExpected) <= kSmokeTolerance;

  if (!options.dump.empty()) {
    FILE* file = std::fopen(options.dump.c_str(), "w");
    if (!file) {
      std::fprintf(stderr, "cannot write %s\n", options.dump.c_str());
      return EXIT_FAILURE;
    }
    for (int set = 0; set < sets; ++set) {
      uint32_t bits;
      std::memcpy(&bits, &correctness[set], sizeof(bits));
      std::fprintf(file, "count=%zu set=%d result=%.9g bits=0x%08x\n", count,
                   set, correctness[set], bits);
    }
    std::fclose(file);
  }

  // Kernels-only inputs: device-resident copies of every set (untimed).
  for (int set = 0; set < sets; ++set) {
    CUDA_CHECK(cudaMemcpy(deviceOutput[set], hostOutput[set],
                          count * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInput[set], hostInput[set],
                          count * sizeof(float), cudaMemcpyHostToDevice));
  }

  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  if (time_launches([&](int replay) { return launch(0, replay % sets); },
                    [&] {
                      // All-ones bits are a NaN: poison the kernels-only
                      // results so the post-timing check sees timed output.
                      (void)cudaMemsetAsync(deviceResult, 0xff,
                                            sets * sizeof(float), stream);
                      (void)cudaStreamSynchronize(stream);
                    },
                    stream, options.replays, "anomaly.kernels",
                    kernelMicroseconds) ||
      time_launches([&](int replay) { return launch(1, replay % sets); },
                    [&] {
                      for (int set = 0; set < sets; ++set)
                        hostResult[set] = NAN;
                    },
                    stream, options.replays, "anomaly.end_to_end",
                    endToEndMicroseconds))
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Every set's result from the timed loop, against its own reference.
  std::vector<float> kernelResults(sets);
  CUDA_CHECK(cudaMemcpy(kernelResults.data(), deviceResult,
                        sets * sizeof(float), cudaMemcpyDeviceToHost));
  bool timedPass = true;
  const int timedSets = std::min(sets, options.replays);
  for (int set = 0; set < timedSets; ++set)
    timedPass = timedPass && within(hostResult[set], set) &&
                within(kernelResults[set], set);
  const bool withinTolerance = correctnessPass && timedPass;
  const double bytes = static_cast<double>(count) * (sizeof(int8_t) +
                                                     sizeof(float));

  std::printf("mlperf-anomaly mse: count=%zu inputs=%s scale=%g zero_point=%d "
              "blocks=%d launch=%s replays=%d path=%s chunks=%zu rotate=%d\n",
              count, options.inputs.c_str(), scale, zeroPoint, blocks,
              options.launch.c_str(), options.replays,
              copiedCount ? "pipeline" : "zerocopy", chunks.size(), sets);
  std::printf("  gpu          %.9g\n", gpu);
  std::printf("  loom loop    %.9g (host, sequential float)%s\n", loom,
              gpu == loom ? ", bitwise equal to gpu" : "");
  std::printf("  reference    %.9g (host, double)\n", reference[0]);
  if (smokeShape)
    std::printf("  smoke oracle %s (|mse - %g| <= %g)\n",
                smokeOracle ? "pass" : "FAIL", kSmokeExpected,
                kSmokeTolerance);
  std::printf("  rotation     %d input sets; correctness pass %s, timed "
              "results of %d sets %s\n",
              sets, correctnessPass ? "pass" : "FAIL", timedSets,
              timedPass ? "pass" : "FAIL");
  std::printf("  check        %s (abs err %.3e vs double)\n",
              withinTolerance ? "pass" : "FAIL", error);
  std::printf("  kernels      %.3f us/iter (%s)\n", kernelMicroseconds,
              graph ? "graph replay" : "direct launch");
  std::printf("  end-to-end   %.3f us/iter (H2D + kernels + D2H, %s)\n",
              endToEndMicroseconds, graph ? "graph replay" : "direct launch");
  std::printf("  bandwidth    %.1f GB/s (kernels, %zu B/element)\n",
              bytes / (kernelMicroseconds * 1.0e3),
              sizeof(int8_t) + sizeof(float));
  std::printf("  timed total  %.3f s (replays x (kernels + end-to-end), "
              "excluding warm-up)\n",
              options.replays * (kernelMicroseconds + endToEndMicroseconds) /
                  1.0e6);

  for (auto& list : execs)
    for (cudaGraphExec_t exec : list)
      if (exec)
        CUDA_CHECK(cudaGraphExecDestroy(exec));
  for (cudaEvent_t event : copied)
    CUDA_CHECK(cudaEventDestroy(event));
  CUDA_CHECK(cudaEventDestroy(joined));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaStreamDestroy(computeStream));
  for (int set = 0; set < sets; ++set) {
    CUDA_CHECK(cudaFree(deviceOutput[set]));
    CUDA_CHECK(cudaFree(deviceInput[set]));
    CUDA_CHECK(cudaFreeHost(hostOutput[set]));
    CUDA_CHECK(cudaFreeHost(hostInput[set]));
  }
  if (stageOutput)
    CUDA_CHECK(cudaFree(stageOutput));
  if (stageInput)
    CUDA_CHECK(cudaFree(stageInput));
  CUDA_CHECK(cudaFree(devicePartials));
  CUDA_CHECK(cudaFree(deviceResult));
  CUDA_CHECK(cudaFree(deviceTicket));
  CUDA_CHECK(cudaFreeHost(hostResult));
  return withinTolerance && smokeOracle ? EXIT_SUCCESS : EXIT_FAILURE;
}
