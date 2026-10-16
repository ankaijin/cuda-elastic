// CUDA Graph representation of the GAPBS PageRank workload.
//
// This is an executable example of the dataflow represented by
// externals/gapbs/src/pr.cc and gapbs_pagerank_kernel(...).  It is not a
// replacement for Loom's source-level application or its generated kernels.
// The benchmark graph is specialized to a generated regular CSR fixture:
// 2^20 vertices, sixteen incoming and outgoing edges per vertex, and twenty
// fixed PageRank iterations.
//
// PAGERANK_MODE selects what runs: "validate" checks captured graphs against
// a CPU reference, "bench" replays the benchmark graph, and "all" (the
// default) validates first and benchmarks only if validation passes.
// PAGERANK_BLOCK_SIZE, PAGERANK_PIN_HOST and PAGERANK_UPLOAD_CHUNKS override
// the defaults in Settings; validation runs with the same settings.
// PAGERANK_LAUNCH=direct enqueues the same stream operations directly on
// every run instead of capturing them once into a graph and replaying it
// (PAGERANK_LAUNCH=graph, the default).  Each timed run is an NVTX range
// (pagerank.full, pagerank.compute).
//
// PAGERANK_GRAPH, PAGERANK_START, PAGERANK_ROTATE, PAGERANK_DUMP and
// PAGERANK_DUMP_REF (see InputOptions in pagerank_common.h) select the
// benchmark input, rotate the timed replays between distinct inputs (default
// 1, the original benchmark) and dump the benchmark input's final scores.

#include "pagerank_common.h"

// Threads per block for every kernel; PAGERANK_BLOCK_SIZE overrides it.
// With gathered contributions, 32 through 512 measure within ~4% of each
// other (0.234-0.247 ms per iteration; 1024 is ~25% slower) because the
// kernel is bound by L2 bandwidth for the random contribution reads, not by
// occupancy.  32 was marginally fastest.
constexpr int kDefaultBlockSize = 32;

// What each outgoing edge of a vertex carries: its score divided by its
// out-degree.  Dangling vertices (out-degree zero) are never read as sources,
// so they get zero instead of a division by zero.
__device__ float contribution(float score, int out_degree) {
  return out_degree > 0 ? score / out_degree : 0.0f;
}

// One thread owns one destination vertex.  Each update gathers the previous
// iteration's contributions and writes a distinct element of the score vector
// and of the next contribution vector, so no atomic operation (and no
// artificial cross-vertex dependency) is required.  Dividing once per vertex
// here, instead of once per edge in the next iteration, leaves one gathered
// array per edge instead of two.  A launch covers destinations
// [vertex_begin, vertex_end), so the first iteration can run one upload chunk
// at a time.
__global__ void pagerank_update(const int* row_offsets, const int* columns,
                                const int* out_degrees, const float* contributions,
                                float* scores, float* next_contributions,
                                int vertex_begin, int vertex_end, int vertices,
                                float damping) {
  int dst = vertex_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= vertex_end) return;

  float incoming = 0.0f;
  for (int edge = row_offsets[dst]; edge < row_offsets[dst + 1]; ++edge)
    incoming += contributions[columns[edge]];
  const float score = (1.0f - damping) / vertices + damping * incoming;
  scores[dst] = score;
  next_contributions[dst] = contribution(score, out_degrees[dst]);
}

// Starts every score at 1/V when `uniform` is set, otherwise keeps the scores
// already copied in, and computes the first contributions from them.
__global__ void initialize_scores(float* scores, float* contributions,
                                  const int* out_degrees, int vertices, bool uniform) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertices) return;
  const float score = uniform ? 1.0f / vertices : scores[i];
  if (uniform) scores[i] = score;
  contributions[i] = contribution(score, out_degrees[i]);
}

struct DeviceGraph {
  int *row_offsets = nullptr, *columns = nullptr, *out_degrees = nullptr;
  float* scores = nullptr;
  // Read and written alternately by successive iterations.
  float *contributions_a = nullptr, *contributions_b = nullptr;
};

DeviceGraph allocate_device_graph(const HostGraph& host) {
  DeviceGraph device;
  CUDA_CHECK(cudaMalloc(&device.row_offsets, host.row_offsets.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device.columns, host.columns.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device.out_degrees, host.out_degrees.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device.scores, sizeof(float) * host.vertices));
  CUDA_CHECK(cudaMalloc(&device.contributions_a, sizeof(float) * host.vertices));
  CUDA_CHECK(cudaMalloc(&device.contributions_b, sizeof(float) * host.vertices));
  return device;
}

void free_device_graph(DeviceGraph& device) {
  CUDA_CHECK(cudaFree(device.contributions_b));
  CUDA_CHECK(cudaFree(device.contributions_a));
  CUDA_CHECK(cudaFree(device.scores));
  CUDA_CHECK(cudaFree(device.columns));
  CUDA_CHECK(cudaFree(device.out_degrees));
  CUDA_CHECK(cudaFree(device.row_offsets));
  device = DeviceGraph{};
}

struct Streams {
  cudaStream_t compute = nullptr, upload = nullptr;
  cudaEvent_t capture_start = nullptr, metadata_ready = nullptr;
};

Streams create_streams() {
  Streams s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.upload, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&s.capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&s.metadata_ready, cudaEventDisableTiming));
  return s;
}

void destroy_streams(Streams& s) {
  CUDA_CHECK(cudaEventDestroy(s.metadata_ready));
  CUDA_CHECK(cudaEventDestroy(s.capture_start));
  CUDA_CHECK(cudaStreamDestroy(s.upload));
  CUDA_CHECK(cudaStreamDestroy(s.compute));
  s = Streams{};
}

// Run-wide settings, read from the environment in main.
struct Settings {
  // PAGERANK_LAUNCH: graph replays a captured graph, direct enqueues the
  // same stream operations on every run.
  bool graph = true;
  int block_size = kDefaultBlockSize;  // PAGERANK_BLOCK_SIZE
  // PAGERANK_PIN_HOST: page-lock the host buffers the graph copies from and
  // to, so the copies run by DMA instead of through the driver's pageable
  // staging buffer.  Measured 22.5 GB/s against 13-14 GB/s pageable.
  bool pin_host = true;
  // PAGERANK_UPLOAD_CHUNKS: how many pieces the column upload is split into;
  // the first iteration starts on each piece as soon as it arrives.  8 and 32
  // both saved ~0.2 ms per replay against 1 (about one iteration hidden
  // behind the copy); more chunks add nodes without saving more.
  int upload_chunks = 8;
};

struct RunOptions {
  // nullptr starts the scores uniform (initialize_scores, as in the
  // benchmark); otherwise they are copied from the host.
  const float* initial_scores = nullptr;
  int iterations = kIterations;
  int block_size = kDefaultBlockSize;
  // false leaves out the graph upload and result download, for timing the
  // computation alone against data already on the device.
  bool include_transfers = true;
  int upload_chunks = 1;
};

// Splits the vertices into at most `chunks` contiguous ranges holding about
// equal numbers of incoming edges, and returns the range boundaries.
std::vector<int> split_by_edges(const HostGraph& graph, int chunks) {
  std::vector<int> bounds{0};
  const long long edges = static_cast<long long>(graph.columns.size());
  for (int chunk = 1; chunk < chunks; ++chunk) {
    const int target = static_cast<int>(edges * chunk / chunks);
    const int vertex = static_cast<int>(
        std::lower_bound(graph.row_offsets.begin(), graph.row_offsets.end(), target) -
        graph.row_offsets.begin());
    if (vertex > bounds.back() && vertex < graph.vertices) bounds.push_back(vertex);
  }
  bounds.push_back(graph.vertices);
  return bounds;
}

// One PageRank run (the upload, initialization, the updates and the result
// download), either captured into an executable graph or, when executable is
// null, enqueued directly on every launch.  Validation and the benchmark
// share it so validation exercises the same work that is timed.
struct PagerankRun {
  const Streams* streams = nullptr;
  const HostGraph* host = nullptr;
  const DeviceGraph* device = nullptr;
  RunOptions options;
  float* result = nullptr;
  // Upload chunk boundaries and, for direct launches, the reusable event
  // that marks each chunk's arrival.
  std::vector<int> bounds;
  std::vector<cudaEvent_t> chunk_ready;
  cudaGraphExec_t executable = nullptr;
};

// Enqueues one run on the streams.  Every run begins by forking the upload
// stream from the compute stream and ends with the compute stream joined to
// every upload, so graph capture and direct launches see the same
// dependencies and the upload stream never runs ahead of the previous run.
void enqueue_pagerank(const PagerankRun& run) {
  const Streams& s = *run.streams;
  const HostGraph& host = *run.host;
  const DeviceGraph& device = *run.device;
  const RunOptions& options = run.options;
  const std::vector<int>& bounds = run.bounds;
  const std::vector<cudaEvent_t>& chunk_ready = run.chunk_ready;
  const int vertices = host.vertices;

  const int block_size = options.block_size;
  const auto launch_update = [&](int begin, int end, const float* contributions,
                                 float* next_contributions) {
    const int blocks = (end - begin + block_size - 1) / block_size;
    pagerank_update<<<blocks, block_size, 0, s.compute>>>(
        device.row_offsets, device.columns, device.out_degrees, contributions,
        device.scores, next_contributions, begin, end, vertices, kDamping);
    CUDA_CHECK(cudaGetLastError());
  };

  // The compute stream is the origin.  The GPU has one host-to-device copy
  // engine, so all uploads go in order on one stream: row offsets and
  // out-degrees first (initialization needs the degrees), then the columns
  // in chunks of whole rows.  Each chunk's first-iteration update waits only
  // for its own chunk, so it overlaps the copies of the chunks after it.
  // Later iterations remain serial because iteration i+1 depends on every
  // contribution written by iteration i.
  if (options.include_transfers) {
    // During capture, waiting on an event recorded by the capturing stream
    // joins the upload stream to the same graph capture.
    CUDA_CHECK(cudaEventRecord(s.capture_start, s.compute));
    CUDA_CHECK(cudaStreamWaitEvent(s.upload, s.capture_start, 0));
    CUDA_CHECK(cudaMemcpyAsync(device.row_offsets, host.row_offsets.data(),
                               host.row_offsets.size() * sizeof(int),
                               cudaMemcpyHostToDevice, s.upload));
    CUDA_CHECK(cudaMemcpyAsync(device.out_degrees, host.out_degrees.data(),
                               host.out_degrees.size() * sizeof(int),
                               cudaMemcpyHostToDevice, s.upload));
    CUDA_CHECK(cudaEventRecord(s.metadata_ready, s.upload));
    for (size_t chunk = 0; chunk < chunk_ready.size(); ++chunk) {
      const int first_edge = host.row_offsets[bounds[chunk]];
      const int last_edge = host.row_offsets[bounds[chunk + 1]];
      if (last_edge > first_edge)
        CUDA_CHECK(cudaMemcpyAsync(device.columns + first_edge,
                                   host.columns.data() + first_edge,
                                   sizeof(int) * (last_edge - first_edge),
                                   cudaMemcpyHostToDevice, s.upload));
      CUDA_CHECK(cudaEventRecord(chunk_ready[chunk], s.upload));
    }
    CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.metadata_ready, 0));
  }

  if (options.initial_scores)
    CUDA_CHECK(cudaMemcpyAsync(device.scores, options.initial_scores,
                               sizeof(float) * vertices, cudaMemcpyHostToDevice,
                               s.compute));
  initialize_scores<<<(vertices + block_size - 1) / block_size, block_size, 0, s.compute>>>(
      device.scores, device.contributions_a, device.out_degrees, vertices,
      options.initial_scores == nullptr);
  CUDA_CHECK(cudaGetLastError());

  float* contributions = device.contributions_a;
  float* next_contributions = device.contributions_b;
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    if (iteration == 0) {
      for (size_t chunk = 0; chunk + 1 < bounds.size(); ++chunk) {
        if (options.include_transfers)
          CUDA_CHECK(cudaStreamWaitEvent(s.compute, chunk_ready[chunk], 0));
        launch_update(bounds[chunk], bounds[chunk + 1], contributions, next_contributions);
      }
    } else {
      launch_update(0, vertices, contributions, next_contributions);
    }
    float* temporary = contributions;
    contributions = next_contributions;
    next_contributions = temporary;
  }
  // With no iterations the chunks still have to rejoin the capture.
  if (options.include_transfers && options.iterations == 0)
    for (cudaEvent_t event : chunk_ready) CUDA_CHECK(cudaStreamWaitEvent(s.compute, event, 0));

  // The final copy is ordered after all dependent updates.
  if (options.include_transfers)
    CUDA_CHECK(cudaMemcpyAsync(run.result, device.scores, sizeof(float) * vertices,
                               cudaMemcpyDeviceToHost, s.compute));
}

// Prepares a run: captures and instantiates its graph when `graph` is set,
// otherwise keeps the events that direct launches record on every run.
PagerankRun prepare_pagerank(const Streams& s, const HostGraph& host,
                             const DeviceGraph& device, const RunOptions& options,
                             float* result, bool graph) {
  PagerankRun run;
  run.streams = &s;
  run.host = &host;
  run.device = &device;
  run.options = options;
  run.result = result;
  run.bounds = options.include_transfers ? split_by_edges(host, options.upload_chunks)
                                         : std::vector<int>{0, host.vertices};
  run.chunk_ready.resize(run.bounds.size() - 1);
  for (cudaEvent_t& event : run.chunk_ready)
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  if (graph) {
    cudaGraph_t captured = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(s.compute, cudaStreamCaptureModeGlobal));
    enqueue_pagerank(run);
    CUDA_CHECK(cudaStreamEndCapture(s.compute, &captured));
    // The captured graph keeps the dependencies, not the events.
    for (cudaEvent_t event : run.chunk_ready) CUDA_CHECK(cudaEventDestroy(event));
    run.chunk_ready.clear();
    CUDA_CHECK(cudaGraphInstantiate(&run.executable, captured, 0));
    CUDA_CHECK(cudaGraphDestroy(captured));
  }
  return run;
}

// Enqueues one run on the compute stream: a graph replay or the direct work.
void launch_pagerank(const PagerankRun& run) {
  if (run.executable)
    CUDA_CHECK(cudaGraphLaunch(run.executable, run.streams->compute));
  else
    enqueue_pagerank(run);
}

void destroy_pagerank(PagerankRun& run) {
  if (run.executable) CUDA_CHECK(cudaGraphExecDestroy(run.executable));
  for (cudaEvent_t event : run.chunk_ready) CUDA_CHECK(cudaEventDestroy(event));
  run = PagerankRun{};
}

// Runs PageRank once and compares every score with the CPU reference
// (compare_scores in pagerank_common.h).
bool validate_case(const Streams& streams, const Settings& settings, const char* name,
                   const HostGraph& graph, const std::vector<float>* initial,
                   int iterations) {
  if (!check_case_graph(name, graph)) return false;
  const int vertices = graph.vertices;
  std::vector<float> actual(vertices);
  std::vector<void*> pinned = pin_host_buffers(settings.pin_host, graph, initial, actual);
  DeviceGraph device = allocate_device_graph(graph);
  RunOptions options;
  options.initial_scores = initial ? initial->data() : nullptr;
  options.iterations = iterations;
  options.block_size = settings.block_size;
  options.upload_chunks = settings.upload_chunks;
  PagerankRun run =
      prepare_pagerank(streams, graph, device, options, actual.data(), settings.graph);
  launch_pagerank(run);
  CUDA_CHECK(cudaStreamSynchronize(streams.compute));
  destroy_pagerank(run);
  free_device_graph(device);
  unpin_host_buffers(pinned);

  const std::vector<double> expected = reference_pagerank(
      graph, initial ? *initial : std::vector<float>(vertices, 1.0f / vertices), iterations);
  return compare_scores(name, graph, iterations, actual, expected);
}

bool run_validation(const Streams& streams, const Settings& settings,
                    const HostGraph& fixture) {
  const bool ok = run_validation_cases(
      fixture, [&](const char* name, const HostGraph& graph, const std::vector<float>* initial,
                   int iterations) {
        return validate_case(streams, settings, name, graph, initial, iterations);
      });
  std::printf("validation %s (launch=%s block_size=%d pin_host=%d upload_chunks=%d)\n",
              ok ? "passed" : "FAILED", settings.graph ? "graph" : "direct",
              settings.block_size, settings.pin_host, settings.upload_chunks);
  return ok;
}

// Average milliseconds per run over `replays` back-to-back runs, each an
// NVTX range named `range`.  Replay r runs runs[r % runs.size()], so with
// PAGERANK_ROTATE > 1 consecutive replays use distinct inputs.
float time_replays(const Streams& streams, const std::vector<PagerankRun>& runs, int replays,
                   const char* range) {
  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, streams.compute));
  for (int replay = 0; replay < replays; ++replay) {
    nvtx_push(range);
    launch_pagerank(runs[replay % runs.size()]);
    nvtx_pop();
  }
  CUDA_CHECK(cudaEventRecord(stop, streams.compute));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms / replays;
}

// Times the benchmark run (upload, twenty iterations, download) and a
// compute-only run (initialization and twenty iterations on data already on
// the device).  The full run goes first, so a profiler limited to the first
// launches sees it and the compute-only run finds the data uploaded.  With
// PAGERANK_ROTATE=K > 1 the full replays cycle through K inputs (one captured
// graph each), every input's last result is checked against its own CPU
// reference afterwards, and the compute-only run uses input 0's data.
bool run_benchmark(const Streams& streams, const Settings& settings,
                   const HostGraph& fixture, const InputOptions& input_options) {
  std::vector<BenchInput> inputs = make_bench_inputs(input_options, fixture);
  const bool rotating = inputs.size() > 1;
  std::vector<std::vector<void*>> pinned;
  for (BenchInput& input : inputs) {
    if (rotating) std::fill(input.result.begin(), input.result.end(), std::nanf(""));
    pinned.push_back(pin_host_buffers(settings.pin_host, input.graph, input.initial_or_null(),
                                      input.result));
  }
  DeviceGraph device = allocate_device_graph(inputs[0].graph);
  RunOptions options;
  options.block_size = settings.block_size;
  options.upload_chunks = settings.upload_chunks;
  std::vector<PagerankRun> full;
  for (BenchInput& input : inputs) {
    options.initial_scores = input.initial.empty() ? nullptr : input.initial.data();
    full.push_back(prepare_pagerank(streams, input.graph, device, options, input.result.data(),
                                    settings.graph));
  }
  options.include_transfers = false;
  std::vector<PagerankRun> compute_only{
      prepare_pagerank(streams, inputs[0].graph, device, options, nullptr, settings.graph)};
  // Uploading ahead keeps one-time graph setup out of the timings.
  if (settings.graph) {
    for (PagerankRun& run : full) CUDA_CHECK(cudaGraphUpload(run.executable, streams.compute));
    CUDA_CHECK(cudaGraphUpload(compute_only[0].executable, streams.compute));
  }

  int graph_replays = 0;
  if (const char* value = std::getenv("GRAPH_REPLAYS"))
    graph_replays = std::atoi(value);
  if (graph_replays <= 0) {
    const float replay_ms = time_replays(streams, full, 1, "pagerank.calibrate");
    graph_replays = replay_ms > 0.0f
                        ? static_cast<int>(10000.0f / replay_ms) + 1
                        : 1;
  }
  // Every input runs at least once, so every input's result can be checked.
  if (rotating) graph_replays = std::max<int>(graph_replays, static_cast<int>(inputs.size()));
  const float full_ms = time_replays(streams, full, graph_replays, "pagerank.full");
  // The compute-only run works on input 0, whatever input ran last.
  if (rotating) {
    launch_pagerank(full[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams.compute));
  }
  const float compute_ms =
      time_replays(streams, compute_only, graph_replays, "pagerank.compute");

  const std::vector<float>& h_result = inputs[0].result;
  const char* launch = settings.graph ? "graph" : "direct";
  std::printf("PageRank scores after %d iterations (launch=%s replays=%d): %.6f %.6f %.6f %.6f\n",
              kIterations, launch, graph_replays, h_result[0], h_result[1],
              h_result[2], h_result[3]);
  const bool ok = check_bench_inputs(input_options, inputs, rotating);
  std::printf("launch=%s block_size=%d pin_host=%d upload_chunks=%d: %.3f ms per replay with "
              "transfers, %.3f ms compute only (%.4f ms per iteration)\n",
              launch, settings.block_size, settings.pin_host, settings.upload_chunks,
              full_ms, compute_ms, compute_ms / kIterations);

  destroy_pagerank(compute_only[0]);
  for (PagerankRun& run : full) destroy_pagerank(run);
  free_device_graph(device);
  for (std::vector<void*>& buffers : pinned) unpin_host_buffers(buffers);
  return ok;
}

int main() {
  const char* mode_value = std::getenv("PAGERANK_MODE");
  const std::string mode = mode_value ? mode_value : "all";
  if (mode != "all" && mode != "validate" && mode != "bench") {
    std::fprintf(stderr, "PAGERANK_MODE must be all, validate or bench (got \"%s\")\n",
                 mode.c_str());
    return 2;
  }
  Settings settings;
  if (const char* value = std::getenv("PAGERANK_LAUNCH")) {
    const std::string launch = value;
    if (launch != "graph" && launch != "direct") {
      std::fprintf(stderr, "PAGERANK_LAUNCH must be graph or direct (got \"%s\")\n", value);
      return 2;
    }
    settings.graph = launch == "graph";
  }
  const char* block_size_value = std::getenv("PAGERANK_BLOCK_SIZE");
  if (block_size_value) settings.block_size = std::atoi(block_size_value);
  if (settings.block_size < 32 || settings.block_size > 1024 || settings.block_size % 32 != 0) {
    std::fprintf(stderr,
                 "PAGERANK_BLOCK_SIZE must be a multiple of 32 from 32 to 1024 (got \"%s\")\n",
                 block_size_value);
    return 2;
  }
  if (const char* value = std::getenv("PAGERANK_PIN_HOST")) {
    const std::string pin_host = value;
    if (pin_host != "0" && pin_host != "1") {
      std::fprintf(stderr, "PAGERANK_PIN_HOST must be 0 or 1 (got \"%s\")\n", value);
      return 2;
    }
    settings.pin_host = pin_host == "1";
  }
  if (const char* value = std::getenv("PAGERANK_UPLOAD_CHUNKS")) {
    settings.upload_chunks = std::atoi(value);
    if (settings.upload_chunks < 1 || settings.upload_chunks > 1024) {
      std::fprintf(stderr, "PAGERANK_UPLOAD_CHUNKS must be from 1 to 1024 (got \"%s\")\n",
                   value);
      return 2;
    }
  }

  const InputOptions input_options = read_input_options(1);

  Streams streams = create_streams();
  const HostGraph fixture = make_benchmark_fixture();
  bool ok = true;
  if (mode != "bench") ok = run_validation(streams, settings, fixture);
  if (ok && mode != "validate") ok = run_benchmark(streams, settings, fixture, input_options);
  destroy_streams(streams);
  return ok ? 0 : 1;
}
