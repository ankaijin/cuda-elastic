// Shared host harness for the mlperf-tiny classification kernel ports
// (kws_postprocess.cu, vww_preprocess.cu): error checking, option parsing,
// and launch timing, so both kernels are measured the same way.
//
// Both binaries run two pieces of work: kernels only (inputs resident) and
// end to end (H2D + kernels + D2H). With --launch=graph (the default) each is
// captured once into a CUDA graph and replayed; with --launch=direct the same
// stream operations are enqueued directly on every iteration. Either way each
// is warmed up, then run --replays times between CUDA events, each iteration
// inside an NVTX range.
//
// --dump=FILE makes a binary write its host output buffers to FILE after its
// correctness check, as raw little-endian arrays (each binary documents its
// layout), so compare_sota.sh can cmp the outputs of different builds.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define CLASSIFICATION_HAS_NVTX 1
#  endif
#endif
#ifndef CLASSIFICATION_HAS_NVTX
#  define CLASSIFICATION_HAS_NVTX 0
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

constexpr int kWarmupReplays = 100;

struct Options {
  size_t count = 0;  // set by each binary to its smoke size
  size_t batch = 1;
  std::string inputs = "smoke";
  std::string launch = "graph";  // graph | direct
  int replays = 10000;
  unsigned seed = 1;
  std::string dump;  // --dump=FILE: write the host outputs here
};

// Parses --count, --batch (if allowBatch), --inputs (one of validInputs,
// space separated), --launch, --replays, --seed and --dump. A binary with
// options of its own passes extra, which returns true for an argument it
// consumed, and describes them in extraUsage.
inline bool parse_options(
    int argc, char** argv, Options& options, bool allowBatch,
    const char* validInputs,
    const std::function<bool(const std::string&)>& extra = nullptr,
    const char* extraUsage = "") {
  const std::string valid = std::string(" ") + validInputs + " ";
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
    else if (const char* v = allowBatch ? value("--batch=") : nullptr)
      options.batch = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--inputs="))
      options.inputs = v;
    else if (const char* v = value("--launch="))
      options.launch = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else if (const char* v = value("--dump="))
      options.dump = v;
    else if (!extra || !extra(argument)) {
      std::fprintf(stderr,
                   "usage: %s [--count=N]%s [--inputs={%s}] "
                   "[--launch={graph,direct}] [--replays=R] [--seed=S] "
                   "[--dump=FILE]%s\n",
                   argv[0], allowBatch ? " [--batch=B]" : "", validInputs,
                   extraUsage);
      return false;
    }
  }
  if (options.count == 0 || options.batch == 0 || options.replays <= 0 ||
      valid.find(" " + options.inputs + " ") == std::string::npos ||
      (options.launch != "graph" && options.launch != "direct")) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

inline void nvtx_push(const char* label) {
#if CLASSIFICATION_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

inline void nvtx_pop() {
#if CLASSIFICATION_HAS_NVTX
  nvtxRangePop();
#endif
}

// Captures the stream work of enqueue() (an int() callable returning
// EXIT_SUCCESS) into an executable graph.
template <typename Enqueue>
inline int capture_graph(Enqueue enqueue, cudaStream_t stream,
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

// Average microseconds per iteration, after warm-up. launch() enqueues one
// iteration on stream (a graph replay or the direct stream operations) and
// returns EXIT_SUCCESS; each timed iteration is an NVTX range named range.
// Warm-up runs min(replays, kWarmupReplays) iterations, or minWarmup if more.
template <typename Launch>
inline int time_launches(Launch launch, cudaStream_t stream, int replays,
                         const char* range, double& microseconds,
                         int minWarmup = 0) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const int warmup = std::max(std::min(replays, kWarmupReplays), minWarmup);
  for (int replay = 0; replay < warmup; ++replay)
    if (launch() != EXIT_SUCCESS)
      return EXIT_FAILURE;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int replay = 0; replay < replays; ++replay) {
    nvtx_push(range);
    const int status = launch();
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

// Measured time of both timed loops, excluding warm-up.
inline void print_timed_total(int replays, double kernelMicroseconds,
                              double endToEndMicroseconds) {
  std::printf("  timed total  %.3f s (replays x (kernel + end-to-end), "
              "excluding warm-up)\n",
              replays * (kernelMicroseconds + endToEndMicroseconds) / 1.0e6);
}

inline int sm_count() {
  int device = 0, count = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess)
    return 1;
  return count;
}

// Writes size bytes at each of the parts, in order, to path (for --dump).
struct DumpPart {
  const void* data;
  size_t size;
};
inline bool write_dump(const std::string& path,
                       std::initializer_list<DumpPart> parts) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  bool ok = true;
  for (const DumpPart& part : parts)
    ok = ok && std::fwrite(part.data, 1, part.size, file) == part.size;
  ok = std::fclose(file) == 0 && ok;
  if (!ok)
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
  return ok;
}

// Chunked copy/compute pipeline for the SOTA binaries. One iteration forks
// three streams off the caller's stream: copy-in runs the chunks' H2Ds in
// order, compute runs chunk k's kernel once its H2D is done, and copy-out
// runs chunk k's D2H once its kernel is done, so H2D of chunk k+1, compute of
// chunk k and D2H of chunk k-1 overlap (the RTX 3090 has one copy engine per
// direction). With lookahead L > 0, the H2D of chunk k also waits for the
// D2H of chunk k - L, so copy-in cannot run more than L chunks ahead of
// copy-out: when both directions share the link, whatever copy-out still
// has to move after copy-in finishes runs alone, so it pays to keep them
// level. The iteration joins back into the caller's stream, so consecutive
// iterations never overlap. The callbacks enqueue one chunk's work on the
// stream they are given and return EXIT_SUCCESS; the whole iteration can be
// captured into a graph like any other stream work.
struct CopyComputePipeline {
  cudaStream_t copyIn = nullptr, compute = nullptr, copyOut = nullptr;
  cudaEvent_t forked = nullptr, computeDone = nullptr, copyOutDone = nullptr;
  std::vector<cudaEvent_t> copied, computed, drained;  // per chunk
  size_t lookahead = 0;

  int create(size_t chunks, size_t lookaheadChunks = 0) {
    lookahead = lookaheadChunks;
    for (cudaStream_t* stream : {&copyIn, &compute, &copyOut})
      CUDA_CHECK(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
    for (cudaEvent_t* event : {&forked, &computeDone, &copyOutDone})
      CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    for (std::vector<cudaEvent_t>* events : {&copied, &computed, &drained}) {
      events->resize(chunks);
      for (cudaEvent_t& event : *events)
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    return EXIT_SUCCESS;
  }

  // h2d, kernel and d2h are int(size_t chunk, cudaStream_t) callables.
  template <typename H2D, typename Kernel, typename D2H>
  int enqueue(cudaStream_t stream, H2D h2d, Kernel kernel, D2H d2h) {
    CUDA_CHECK(cudaEventRecord(forked, stream));
    for (cudaStream_t s : {copyIn, compute, copyOut})
      CUDA_CHECK(cudaStreamWaitEvent(s, forked, 0));
    for (size_t chunk = 0; chunk < copied.size(); ++chunk) {
      if (lookahead > 0 && chunk >= lookahead)
        CUDA_CHECK(cudaStreamWaitEvent(copyIn, drained[chunk - lookahead], 0));
      if (h2d(chunk, copyIn) != EXIT_SUCCESS)
        return EXIT_FAILURE;
      CUDA_CHECK(cudaEventRecord(copied[chunk], copyIn));
      CUDA_CHECK(cudaStreamWaitEvent(compute, copied[chunk], 0));
      if (kernel(chunk, compute) != EXIT_SUCCESS)
        return EXIT_FAILURE;
      CUDA_CHECK(cudaEventRecord(computed[chunk], compute));
      CUDA_CHECK(cudaStreamWaitEvent(copyOut, computed[chunk], 0));
      if (d2h(chunk, copyOut) != EXIT_SUCCESS)
        return EXIT_FAILURE;
      if (lookahead > 0)
        CUDA_CHECK(cudaEventRecord(drained[chunk], copyOut));
    }
    // Copy-in finished before the last kernel started, so joining compute
    // and copy-out covers all three streams.
    CUDA_CHECK(cudaEventRecord(computeDone, compute));
    CUDA_CHECK(cudaEventRecord(copyOutDone, copyOut));
    CUDA_CHECK(cudaStreamWaitEvent(stream, computeDone, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream, copyOutDone, 0));
    return EXIT_SUCCESS;
  }

  int destroy() {
    for (std::vector<cudaEvent_t>* events : {&copied, &computed, &drained}) {
      for (cudaEvent_t event : *events)
        CUDA_CHECK(cudaEventDestroy(event));
      events->clear();
    }
    for (cudaEvent_t event : {forked, computeDone, copyOutDone})
      if (event)
        CUDA_CHECK(cudaEventDestroy(event));
    for (cudaStream_t stream : {copyIn, compute, copyOut})
      if (stream)
        CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
  }
};

// Parses "--name=<unsigned>" into value; true if argument was that option.
inline bool parse_size_option(const std::string& argument, const char* name,
                              size_t& value) {
  const std::string prefix = std::string("--") + name + "=";
  if (argument.compare(0, prefix.size(), prefix) != 0)
    return false;
  value = std::strtoull(argument.c_str() + prefix.size(), nullptr, 10);
  return true;
}
