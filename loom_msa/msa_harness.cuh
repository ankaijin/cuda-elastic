// Shared benchmark harness for the multi-sensor attention CUDA variants.
//
// Every variant (msa_baseline.cu, msa_optimized.cu) provides only its kernels
// and an enqueue function. Everything that must be held constant for a fair
// comparison lives here: problem shape, input data, H2D/D2H inclusion, graph
// capture, warm-up policy, replay count, timing, and the correctness oracle.
//
// --launch=graph (the default) captures the enqueue function once into a CUDA
// graph and replays it; --launch=direct calls it on every iteration, with the
// same fork/join of the two streams that capture uses. Each timed end-to-end
// iteration is an NVTX range named msa.end_to_end.
//
// The oracle is a direct port of main.c's loops, generalized to
// TOKEN_COUNT x CONTEXT_COUNT and accumulated in double precision.
//
// Further flags:
//   --seed=N    xorshift32 seed of the random inputs (decimal or 0x hex;
//               default 0x9e3779b9, the original inputs). Ignored by repeat.
//   --rotate=K  the timed loops rotate over K distinct input sets, each with
//               its own pinned host input and output buffers (and, in graph
//               mode, its own captured graphs); iteration i uses set i % K.
//               Set 0 is exactly the single-set input; set k > 0 uses the
//               seed mixed with k (random) or the seed tables offset by k
//               rows (repeat). After timing, every set's output and
//               statistics are checked against that set's own CPU
//               reference, so a variant that reused data from an earlier
//               iteration fails. The default comes from the variant
//               (1 for baseline and optimized, so their numbers are
//               unchanged).
//   --dump=F    after the correctness pass, writes raw little-endian float32
//               to F, for each set k = 0..K-1 in order:
//                 output[BATCH_COUNT][TOKEN_COUNT][4]   (host output)
//                 statistics[BATCH_COUNT][4]            (projection_energy,
//                                                        attention_sum,
//                                                        weighted_sum,
//                                                        max |attention|)
//               i.e. K * (B*T*4 + B*4) floats, no header.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define MSA_HAS_NVTX 1
#  endif
#endif
#ifndef MSA_HAS_NVTX
#  define MSA_HAS_NVTX 0
#endif

#ifndef BATCH_COUNT
#define BATCH_COUNT 1
#endif
#ifndef TOKEN_COUNT
#define TOKEN_COUNT 1000
#endif
#ifndef CONTEXT_COUNT
#define CONTEXT_COUNT 50000
#endif
#ifndef GRAPH_REPLAYS
#define GRAPH_REPLAYS 5000
#endif

enum {
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

// Tolerances for changed summation order against the double reference.
constexpr double kOutputAbsoluteTolerance = 1.0e-4;
constexpr double kOutputRelativeTolerance = 1.0e-4;
constexpr double kStatisticAbsoluteTolerance = 1.0e-4;
constexpr double kStatisticRelativeTolerance = 1.0e-4;

constexpr size_t kQuerySensorElements =
    static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * SENSOR_LANE_COUNT;
constexpr size_t kContextSensorElements =
    static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * SENSOR_LANE_COUNT;
constexpr size_t kFeatureElements =
    static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * FEATURE_LANE_COUNT;
constexpr size_t kContextFeatureElements =
    static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * FEATURE_LANE_COUNT;
constexpr size_t kStatisticsElements =
    static_cast<size_t>(BATCH_COUNT) * STATISTIC_COUNT;

struct PipelineBuffers {
  float* h_audio;
  float* h_imu;
  float* h_output;
  float* h_statistics;
  float* d_audio;
  float* d_imu;
  float* d_projected_audio;
  float* d_projected_imu;
  float* d_attention;
  float* d_output;
  float* d_statistics;
};

// Stage-timing events. They are recorded (as event-record nodes, in graph
// mode) only in a separate instrumented pass; in the end-to-end pass every
// member is null and record_stage() is a no-op, so it is not perturbed.
//   attention_start  -> attention_stop:  attention kernel
//   attention_start  -> statistics_stop: attention through statistics, which
//                       for a split context includes merging the partials
//                       into the final attention rows
struct StageEvents {
  cudaEvent_t attention_start;
  cudaEvent_t attention_stop;
  cudaEvent_t statistics_stop;
};

// During capture the event becomes an external event-record node, so it is
// still recorded on every replay; launched directly it is an ordinary record
// (cudaEventRecordExternal is only valid while capturing).
inline cudaError_t record_stage(cudaEvent_t event, cudaStream_t stream) {
  if (event == nullptr)
    return cudaSuccess;
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaError_t error = cudaStreamIsCapturing(stream, &status);
  if (error != cudaSuccess)
    return error;
  return status == cudaStreamCaptureStatusActive
             ? cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal)
             : cudaEventRecord(event, stream);
}

// Enqueues one full pipeline iteration (H2D, kernels, D2H), during capture
// or directly.
// audio_stream is the capture origin; imu_stream has already been forked from
// it and must be joined back before returning. imu_ready is a spare event for
// that join. Variants call record_stage() at the three StageEvents points.
using EnqueuePipeline = int (*)(const PipelineBuffers& buffers,
                                cudaStream_t audio_stream,
                                cudaStream_t imu_stream,
                                cudaEvent_t imu_ready,
                                const StageEvents& stages);

// Algorithmic attention FLOPs, counted from main.c's math rather than from
// either kernel's instruction stream, so the numerator is identical for every
// variant (effective attention throughput):
//   per (query, key) pair: score 4 mul + 3 add + 1 scale (8), subtract max
//   (1), exp (1), denominator add (1), weighted value 4 mul + 4 add (8) = 19
//   per query row: 4 normalizing divisions
// Max comparisons are not counted.
constexpr double kAttentionFlopsPerPair = 19.0;
constexpr double kAttentionFlopsPerRow = 4.0;
constexpr double kAttentionFlops =
    static_cast<double>(BATCH_COUNT) * TOKEN_COUNT *
    (kAttentionFlopsPerPair * CONTEXT_COUNT + kAttentionFlopsPerRow);

struct Summary {
  double median;
  double minimum;
  double maximum;
};

inline Summary summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t count = samples.size();
  const double median = count % 2 == 1
                            ? samples[count / 2]
                            : 0.5 * (samples[count / 2 - 1] + samples[count / 2]);
  return {median, samples.front(), samples.back()};
}

inline void nvtx_push(const char* label) {
#if MSA_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

inline void nvtx_pop() {
#if MSA_HAS_NVTX
  nvtxRangePop();
#endif
}

// Forks imu_stream from audio_stream, then enqueues one pipeline iteration,
// which joins imu_stream back. Graph capture and direct launches both go
// through here, so they see the same dependencies; in direct mode the fork
// also keeps imu_stream from running ahead of the previous iteration.
inline int enqueue_iteration(EnqueuePipeline enqueue_pipeline,
                             const PipelineBuffers& buffers,
                             cudaStream_t audio_stream, cudaStream_t imu_stream,
                             cudaEvent_t fork, cudaEvent_t imu_ready,
                             const StageEvents& stages) {
  CUDA_CHECK(cudaEventRecord(fork, audio_stream));
  CUDA_CHECK(cudaStreamWaitEvent(imu_stream, fork, 0));
  return enqueue_pipeline(buffers, audio_stream, imu_stream, imu_ready, stages);
}

inline int capture_pipeline(EnqueuePipeline enqueue_pipeline,
                            const PipelineBuffers& buffers,
                            cudaStream_t audio_stream, cudaStream_t imu_stream,
                            cudaEvent_t capture_start, cudaEvent_t imu_ready,
                            const StageEvents& stages, cudaGraph_t* graph,
                            cudaGraphExec_t* graph_exec) {
  CUDA_CHECK(cudaStreamBeginCapture(audio_stream, cudaStreamCaptureModeGlobal));
  if (enqueue_iteration(enqueue_pipeline, buffers, audio_stream, imu_stream,
                        capture_start, imu_ready, stages) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamEndCapture(audio_stream, graph));
  CUDA_CHECK(cudaGraphInstantiate(graph_exec, *graph, nullptr, nullptr, 0));
  return EXIT_SUCCESS;
}

enum class InputMode { kRepeat, kRandom };

constexpr uint32_t kDefaultSeed = 0x9e3779b9u;
constexpr int kMaxRotation = 16;

// Seed of input set `set`: set 0 uses the seed unchanged; later sets mix in
// the set index (never 0, which would make xorshift32 stick at 0).
inline uint32_t set_seed(uint32_t seed, int set) {
  uint32_t mixed = seed;
  if (set > 0) {
    mixed ^= static_cast<uint32_t>(set) * 0x85ebca6bu;
    mixed = (mixed ^ (mixed >> 16)) * 0x7feb352du;
    mixed ^= mixed >> 15;
  }
  return mixed == 0 ? 1u : mixed;
}

// Input set `set` (see --rotate). Set 0 with the default seed is the
// original single-set input.
inline void fill_inputs(InputMode mode, uint32_t seed, int set, float* audio,
                        float* imu) {
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
  if (mode == InputMode::kRepeat) {
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
      for (int token = 0; token < TOKEN_COUNT; ++token)
        std::memcpy(audio + (batch * TOKEN_COUNT + token) * SENSOR_LANE_COUNT,
                    audio_seed[(token + set) % 4], sizeof(audio_seed[0]));
      for (int token = 0; token < CONTEXT_COUNT; ++token)
        std::memcpy(imu + (batch * CONTEXT_COUNT + token) * SENSOR_LANE_COUNT,
                    imu_seed[(token + 3 * set) % 4], sizeof(imu_seed[0]));
    }
    return;
  }

  // Deterministic, nonrepeating values in [-1, 1).
  uint32_t state = set_seed(seed, set);
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  for (size_t index = 0; index < kQuerySensorElements; ++index)
    audio[index] = next();
  for (size_t index = 0; index < kContextSensorElements; ++index)
    imu[index] = next();
}

// Direct port of main.c semantics: projections in float exactly as written,
// attention and statistics accumulated in double.
inline void compute_reference(const float* audio, const float* imu,
                              std::vector<double>& output,
                              std::vector<double>& statistics) {
  output.assign(kFeatureElements, 0.0);
  statistics.assign(kStatisticsElements, 0.0);
  std::vector<float> projected_audio(TOKEN_COUNT * FEATURE_LANE_COUNT);
  std::vector<float> projected_imu(CONTEXT_COUNT * FEATURE_LANE_COUNT);
  std::vector<double> attention(TOKEN_COUNT * FEATURE_LANE_COUNT);
  std::vector<double> scores(CONTEXT_COUNT);

  for (int batch = 0; batch < BATCH_COUNT; ++batch) {
    for (int token = 0; token < TOKEN_COUNT; ++token) {
      const float* in =
          audio + (batch * TOKEN_COUNT + token) * SENSOR_LANE_COUNT;
      float* out = &projected_audio[token * FEATURE_LANE_COUNT];
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[2];
      out[3] = in[0] + in[1];
    }
    for (int token = 0; token < CONTEXT_COUNT; ++token) {
      const float* in =
          imu + (batch * CONTEXT_COUNT + token) * SENSOR_LANE_COUNT;
      float* out = &projected_imu[token * FEATURE_LANE_COUNT];
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[2];
      out[3] = in[0] - in[1];
    }

    // Rows are independent; parallelized so large shapes stay checkable
    // (a no-op without -fopenmp).
#pragma omp parallel for schedule(static) firstprivate(scores)
    for (int row = 0; row < TOKEN_COUNT; ++row) {
      const float* query = &projected_audio[row * FEATURE_LANE_COUNT];
      double maximum = 0.0;
      for (int column = 0; column < CONTEXT_COUNT; ++column) {
        const float* key = &projected_imu[column * FEATURE_LANE_COUNT];
        double dot = 0.0;
        for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
          dot += static_cast<double>(query[lane]) * key[lane];
        scores[column] = dot * 0.5;
        if (column == 0 || scores[column] > maximum)
          maximum = scores[column];
      }
      double denominator = 0.0;
      for (int column = 0; column < CONTEXT_COUNT; ++column) {
        scores[column] = std::exp(scores[column] - maximum);
        denominator += scores[column];
      }
      for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
        double value = 0.0;
        for (int column = 0; column < CONTEXT_COUNT; ++column)
          value += scores[column] * projected_imu[column * FEATURE_LANE_COUNT +
                                                  lane];
        attention[row * FEATURE_LANE_COUNT + lane] = value / denominator;
      }
    }

    double projection_energy = 0.0;
    double attention_sum = 0.0;
    double weighted_sum = 0.0;
    double maximum_magnitude = 0.0;
    for (int token = 0; token < TOKEN_COUNT; ++token) {
      for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
        const int index = token * FEATURE_LANE_COUNT + lane;
        const double projection = projected_audio[index];
        const double value = attention[index];
        const double magnitude = value < 0.0 ? -value : value;
        projection_energy += projection * projection;
        attention_sum += value;
        weighted_sum += static_cast<double>(index + 1) * value;
        if (magnitude > maximum_magnitude)
          maximum_magnitude = magnitude;
        output[batch * TOKEN_COUNT * FEATURE_LANE_COUNT + index] =
            value + projection * 0.125;
      }
    }
    statistics[batch * STATISTIC_COUNT + 0] = projection_energy;
    statistics[batch * STATISTIC_COUNT + 1] = attention_sum;
    statistics[batch * STATISTIC_COUNT + 2] = weighted_sum;
    statistics[batch * STATISTIC_COUNT + 3] = maximum_magnitude;
  }
}

inline bool within_tolerance(double actual, double expected,
                             double absolute_tolerance,
                             double relative_tolerance) {
  // Written so that NaN fails.
  return std::fabs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::fabs(expected);
}

// check_golden: also compare with main.c's constants (4x4 shape, repeat
// inputs, set 0 only). label prefixes the error lines (empty for one set).
inline bool verify(InputMode mode, bool check_golden, const char* label,
                   const float* audio, const float* imu,
                   const float* output, const float* statistics) {
  std::vector<double> expected_output;
  std::vector<double> expected_statistics;
  compute_reference(audio, imu, expected_output, expected_statistics);

  bool passed = true;
  double worst_output_error = 0.0;
  size_t failures = 0;
  for (size_t index = 0; index < kFeatureElements; ++index) {
    const double error = std::fabs(output[index] - expected_output[index]);
    if (!(error <= worst_output_error))
      worst_output_error = error;
    if (!within_tolerance(output[index], expected_output[index],
                          kOutputAbsoluteTolerance,
                          kOutputRelativeTolerance)) {
      if (failures++ < 5)
        std::cerr << label << "output[" << index << "] = " << output[index]
                  << ", expected " << expected_output[index] << '\n';
      passed = false;
    }
  }
  double worst_statistic_error = 0.0;
  for (size_t index = 0; index < kStatisticsElements; ++index) {
    const double expected = expected_statistics[index];
    const double relative =
        std::fabs(statistics[index] - expected) /
        (std::fabs(expected) > 1.0 ? std::fabs(expected) : 1.0);
    if (!(relative <= worst_statistic_error))
      worst_statistic_error = relative;
    if (!within_tolerance(statistics[index], expected,
                          kStatisticAbsoluteTolerance,
                          kStatisticRelativeTolerance)) {
      std::cerr << label << "statistics[" << index << "] = " << statistics[index]
                << ", expected " << expected << '\n';
      passed = false;
    }
  }
  std::cout << label << "reference max |output error|: " << worst_output_error
            << '\n'
            << label << "reference max statistic error (rel): "
            << worst_statistic_error << '\n';

#if BATCH_COUNT == 1 && TOKEN_COUNT == 4 && CONTEXT_COUNT == 4
  // Regression against the constants hard-coded in main.c, with its
  // tolerance. Only meaningful for main.c's own inputs.
  if (mode == InputMode::kRepeat && check_golden) {
    const float combined = statistics[2] + 3.0f * statistics[0] +
                           7.0f * statistics[1] + 11.0f * statistics[3];
    const float actual[5] = {statistics[0], statistics[1], statistics[2],
                             statistics[3], combined};
    const float golden[5] = {9.62500f, 3.93196f, 24.58165f, 0.73098f,
                             89.02110f};
    for (int index = 0; index < 5; ++index) {
      if (!(std::fabs(actual[index] - golden[index]) <= 1.0e-4f)) {
        std::cerr << "main.c constant " << index << ": " << actual[index]
                  << ", expected " << golden[index] << '\n';
        passed = false;
      }
    }
    std::cout << "main.c 4x4 constants: " << (passed ? "match" : "MISMATCH")
              << '\n';
  }
#else
  (void)mode;
  (void)check_golden;
#endif
  return passed;
}

inline int run_benchmark(int argc, char** argv, const char* variant_name,
                         EnqueuePipeline enqueue_pipeline,
                         int default_rotation = 1) {
  InputMode mode = InputMode::kRepeat;
  bool use_graph = true;
  uint32_t seed = kDefaultSeed;
  bool seed_given = false;
  int rotation = default_rotation;
  const char* dump_path = nullptr;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--inputs=random") == 0) {
      mode = InputMode::kRandom;
    } else if (std::strcmp(arg, "--inputs=repeat") == 0) {
      mode = InputMode::kRepeat;
    } else if (std::strcmp(arg, "--launch=graph") == 0) {
      use_graph = true;
    } else if (std::strcmp(arg, "--launch=direct") == 0) {
      use_graph = false;
    } else if (std::strncmp(arg, "--seed=", 7) == 0 && arg[7] != '\0') {
      char* end = nullptr;
      const unsigned long long value = std::strtoull(arg + 7, &end, 0);
      if (*end != '\0' || value == 0 || value > 0xffffffffull) {
        std::cerr << "--seed must be a nonzero 32-bit integer\n";
        return EXIT_FAILURE;
      }
      seed = static_cast<uint32_t>(value);
      seed_given = true;
    } else if (std::strncmp(arg, "--rotate=", 9) == 0) {
      char* end = nullptr;
      const long value = std::strtol(arg + 9, &end, 10);
      if (*end != '\0' || value < 1 || value > kMaxRotation) {
        std::cerr << "--rotate must be 1.." << kMaxRotation << '\n';
        return EXIT_FAILURE;
      }
      rotation = static_cast<int>(value);
    } else if (std::strncmp(arg, "--dump=", 7) == 0 && arg[7] != '\0') {
      dump_path = arg + 7;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--inputs=repeat|random] [--launch=graph|direct]"
                   " [--seed=N] [--rotate=K] [--dump=FILE]\n";
      return EXIT_FAILURE;
    }
  }

  // One PipelineBuffers per input set: own pinned host buffers, shared
  // device buffers (iterations are serialized on audio_stream).
  std::vector<PipelineBuffers> sets(rotation);
  PipelineBuffers& buffers = sets[0];
  for (int set = 0; set < rotation; ++set) {
    PipelineBuffers& b = sets[set];
    CUDA_CHECK(cudaMallocHost(&b.h_audio,
                              kQuerySensorElements * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b.h_imu,
                              kContextSensorElements * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b.h_output, kFeatureElements * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b.h_statistics,
                              kStatisticsElements * sizeof(float)));
    fill_inputs(mode, seed, set, b.h_audio, b.h_imu);
    // Poison the outputs so a set that was never written cannot pass.
    for (size_t index = 0; index < kFeatureElements; ++index)
      b.h_output[index] = NAN;
    for (size_t index = 0; index < kStatisticsElements; ++index)
      b.h_statistics[index] = NAN;
  }

  CUDA_CHECK(cudaMalloc(&buffers.d_audio,
                        kQuerySensorElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_imu,
                        kContextSensorElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_projected_audio,
                        kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_projected_imu,
                        kContextFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_attention,
                        kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_output, kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_statistics,
                        kStatisticsElements * sizeof(float)));
  for (int set = 1; set < rotation; ++set) {
    sets[set].d_audio = buffers.d_audio;
    sets[set].d_imu = buffers.d_imu;
    sets[set].d_projected_audio = buffers.d_projected_audio;
    sets[set].d_projected_imu = buffers.d_projected_imu;
    sets[set].d_attention = buffers.d_attention;
    sets[set].d_output = buffers.d_output;
    sets[set].d_statistics = buffers.d_statistics;
  }

  cudaStream_t audio_stream = nullptr;
  cudaStream_t imu_stream = nullptr;
  cudaEvent_t capture_start = nullptr;
  cudaEvent_t imu_ready = nullptr;
  cudaEvent_t timing_start = nullptr;
  cudaEvent_t timing_stop = nullptr;
  std::vector<cudaGraph_t> graphs(rotation, nullptr);
  std::vector<cudaGraphExec_t> graph_execs(rotation, nullptr);
  std::vector<cudaGraph_t> stage_graphs(rotation, nullptr);
  std::vector<cudaGraphExec_t> stage_graph_execs(rotation, nullptr);
  StageEvents stages{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&audio_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&imu_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&imu_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreate(&timing_start));
  CUDA_CHECK(cudaEventCreate(&timing_stop));
  CUDA_CHECK(cudaEventCreate(&stages.attention_start));
  CUDA_CHECK(cudaEventCreate(&stages.attention_stop));
  CUDA_CHECK(cudaEventCreate(&stages.statistics_stop));

  // In graph mode each set gets its own pair of graphs (its host buffers are
  // baked into the copy nodes).
  if (use_graph) {
    for (int set = 0; set < rotation; ++set) {
      if (capture_pipeline(enqueue_pipeline, sets[set], audio_stream,
                           imu_stream, capture_start, imu_ready, StageEvents{},
                           &graphs[set], &graph_execs[set]) != EXIT_SUCCESS ||
          capture_pipeline(enqueue_pipeline, sets[set], audio_stream,
                           imu_stream, capture_start, imu_ready, stages,
                           &stage_graphs[set],
                           &stage_graph_execs[set]) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    }
  }
  // One iteration on input set `set`, uninstrumented (instrumented when
  // staged): a graph replay, or the same work enqueued directly.
  const auto launch = [&](bool staged, int set) {
    if (!use_graph)
      return enqueue_iteration(enqueue_pipeline, sets[set], audio_stream,
                               imu_stream, capture_start, imu_ready,
                               staged ? stages : StageEvents{});
    CUDA_CHECK(cudaGraphLaunch(
        staged ? stage_graph_execs[set] : graph_execs[set], audio_stream));
    return EXIT_SUCCESS;
  };

  // End-to-end: uninstrumented. Warm-up policy: run for at least
  // kWarmupSeconds (so GPU clocks leave their idle state; one launch is not
  // enough at small shapes), then GRAPH_REPLAYS back-to-back timed launches.
  // Every loop rotates over the input sets.
  constexpr double kWarmupSeconds = 0.25;
  const auto warmup_begin = std::chrono::steady_clock::now();
  int warmup = 0;
  do {
    if (launch(false, warmup++ % rotation) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    CUDA_CHECK(cudaStreamSynchronize(audio_stream));
  } while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         warmup_begin)
               .count() < kWarmupSeconds);
  CUDA_CHECK(cudaEventRecord(timing_start, audio_stream));
  for (int replay = 0; replay < GRAPH_REPLAYS; ++replay) {
    nvtx_push("msa.end_to_end");
    const int status = launch(false, replay % rotation);
    nvtx_pop();
    if (status != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaEventRecord(timing_stop, audio_stream));
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timing_start, timing_stop));

  // Stages: instrumented, one warm-up launch, then GRAPH_REPLAYS launches
  // each synchronized so every replay's stage events can be read. They
  // cover every set at least once, so each set's host output comes from its
  // own latest run when it is verified.
  const int stage_replays = std::max(GRAPH_REPLAYS, rotation);
  std::vector<double> attention_us;
  std::vector<double> through_statistics_us;
  attention_us.reserve(stage_replays);
  through_statistics_us.reserve(stage_replays);
  if (launch(true, 0) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));
  for (int replay = 0; replay < stage_replays; ++replay) {
    if (launch(true, replay % rotation) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    CUDA_CHECK(cudaStreamSynchronize(audio_stream));
    float stage_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&stage_ms, stages.attention_start,
                                    stages.attention_stop));
    attention_us.push_back(1000.0 * stage_ms);
    CUDA_CHECK(cudaEventElapsedTime(&stage_ms, stages.attention_start,
                                    stages.statistics_stop));
    through_statistics_us.push_back(1000.0 * stage_ms);
  }
  const Summary attention = summarize(attention_us);
  const Summary through_statistics = summarize(through_statistics_us);
  const double end_to_end_us = 1000.0 * elapsed_ms / GRAPH_REPLAYS;

  const float* statistics = buffers.h_statistics;
  const float combined = statistics[2] + 3.0f * statistics[0] +
                         7.0f * statistics[1] + 11.0f * statistics[3];
  // Seed and rotation are only printed when they differ from the original
  // single-set run, so existing variants' output is unchanged.
  char seed_text[32] = "";
  if (seed_given)
    std::snprintf(seed_text, sizeof(seed_text), ", seed=0x%08x", seed);
  char rotation_text[32] = "";
  if (rotation != 1)
    std::snprintf(rotation_text, sizeof(rotation_text), ", rotate=%d",
                  rotation);
  std::cout << "variant: " << variant_name << " (" << BATCH_COUNT << "x"
            << TOKEN_COUNT << "x" << CONTEXT_COUNT << ", replays="
            << GRAPH_REPLAYS << ", inputs="
            << (mode == InputMode::kRepeat ? "repeat" : "random")
            << seed_text
            << ", launch=" << (use_graph ? "graph" : "direct")
            << rotation_text << ")\n"
            << "attention checksum: " << statistics[2] << '\n'
            << "projection energy: " << statistics[0] << '\n'
            << "attention sum: " << statistics[1] << '\n'
            << "attention max: " << statistics[3] << '\n'
            << "combined checksum: " << combined << '\n'
            << (use_graph ? "graph replay" : "direct launch")
            << " average (us): " << end_to_end_us << '\n'
            << "attention kernel median (us): " << attention.median
            << " [min " << attention.minimum << ", max " << attention.maximum
            << "]\n"
            << "attention through statistics median (us): "
            << through_statistics.median << " [min "
            << through_statistics.minimum << ", max "
            << through_statistics.maximum << "]\n"
            << "effective attention throughput (TFLOP/s): kernel "
            << kAttentionFlops / (attention.median * 1.0e6)
            << ", through statistics "
            << kAttentionFlops / (through_statistics.median * 1.0e6) << '\n'
            << "metrics: shape=" << BATCH_COUNT << "x" << TOKEN_COUNT << "x"
            << CONTEXT_COUNT << " replays=" << GRAPH_REPLAYS
            << " launch=" << (use_graph ? "graph" : "direct")
            << " end_to_end_us=" << end_to_end_us
            << " attention_us=" << attention.median
            << " attention_min_us=" << attention.minimum
            << " attention_max_us=" << attention.maximum
            << " through_statistics_us=" << through_statistics.median
            << " through_statistics_min_us=" << through_statistics.minimum
            << " through_statistics_max_us=" << through_statistics.maximum
            << " attention_flops=" << kAttentionFlops;
  if (rotation != 1)
    std::cout << " rotate=" << rotation;
  std::cout << '\n';

  // Every set against its own reference; main.c's constants only for set 0.
  bool passed = true;
  for (int set = 0; set < rotation; ++set) {
    char label[32] = "";
    if (rotation != 1) {
      std::snprintf(label, sizeof(label), "set %d: ", set);
      const float* s = sets[set].h_statistics;
      std::cout << label << "attention checksum " << s[2]
                << ", projection energy " << s[0] << ", attention sum "
                << s[1] << ", attention max " << s[3] << '\n';
    }
    const auto verify_begin = std::chrono::steady_clock::now();
    const bool set_passed =
        verify(mode, set == 0, label, sets[set].h_audio, sets[set].h_imu,
               sets[set].h_output, sets[set].h_statistics);
    if (rotation != 1)
      std::cout << label << (set_passed ? "passed" : "FAILED")
                << " (CPU reference "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - verify_begin)
                       .count()
                << " s)\n";
    passed = passed && set_passed;
  }

  if (dump_path != nullptr) {
    FILE* file = std::fopen(dump_path, "wb");
    bool written = file != nullptr;
    for (int set = 0; written && set < rotation; ++set)
      written = std::fwrite(sets[set].h_output, sizeof(float),
                            kFeatureElements, file) == kFeatureElements &&
                std::fwrite(sets[set].h_statistics, sizeof(float),
                            kStatisticsElements, file) == kStatisticsElements;
    if (file != nullptr && std::fclose(file) != 0)
      written = false;
    if (!written) {
      std::cerr << "could not write " << dump_path << '\n';
      passed = false;
    }
  }

  if (use_graph) {
    for (int set = 0; set < rotation; ++set) {
      CUDA_CHECK(cudaGraphExecDestroy(stage_graph_execs[set]));
      CUDA_CHECK(cudaGraphDestroy(stage_graphs[set]));
      CUDA_CHECK(cudaGraphExecDestroy(graph_execs[set]));
      CUDA_CHECK(cudaGraphDestroy(graphs[set]));
    }
  }
  CUDA_CHECK(cudaEventDestroy(stages.statistics_stop));
  CUDA_CHECK(cudaEventDestroy(stages.attention_stop));
  CUDA_CHECK(cudaEventDestroy(stages.attention_start));
  CUDA_CHECK(cudaEventDestroy(timing_stop));
  CUDA_CHECK(cudaEventDestroy(timing_start));
  CUDA_CHECK(cudaEventDestroy(imu_ready));
  CUDA_CHECK(cudaEventDestroy(capture_start));
  CUDA_CHECK(cudaStreamDestroy(imu_stream));
  CUDA_CHECK(cudaStreamDestroy(audio_stream));
  CUDA_CHECK(cudaFree(buffers.d_statistics));
  CUDA_CHECK(cudaFree(buffers.d_output));
  CUDA_CHECK(cudaFree(buffers.d_attention));
  CUDA_CHECK(cudaFree(buffers.d_projected_imu));
  CUDA_CHECK(cudaFree(buffers.d_projected_audio));
  CUDA_CHECK(cudaFree(buffers.d_imu));
  CUDA_CHECK(cudaFree(buffers.d_audio));
  for (int set = rotation - 1; set >= 0; --set) {
    CUDA_CHECK(cudaFreeHost(sets[set].h_statistics));
    CUDA_CHECK(cudaFreeHost(sets[set].h_output));
    CUDA_CHECK(cudaFreeHost(sets[set].h_imu));
    CUDA_CHECK(cudaFreeHost(sets[set].h_audio));
  }

  std::cout << (passed ? "PASSED" : "FAILED") << '\n';
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
