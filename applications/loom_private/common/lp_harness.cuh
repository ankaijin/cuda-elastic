// Shared harness for the loom_private CUDA applications.
//
// Every application implements lp::Workload and calls lp::run_main from main().
// The harness owns the common command line, the Direct and CUDA Graph
// executors, the correctness protocol, timing, and the JSON result record.
//
//   --preset large            the only preset outside tests (default)
//   --launch direct|graph|both  both is the default and only valid for check
//   --action check|bench      check is the default; bench must be explicit
//   --iterations N --warmup W bench only (defaults 20 and 3)
//   --set key=value           dimension override, check only, for tests
//   --asset name=path         application asset path override
//   --json-out PATH           also write the result record to PATH
//
// Exit codes: 0 pass, 1 output mismatch, 2 usage or invalid dimensions,
// 3 missing or corrupt asset, 70 CUDA error.
//
// Check protocol: setup and the CPU oracle run once. Each selected launch mode
// then runs two invocations. Before each one the outputs are poisoned, and
// after it the host output is checked against the oracle and digested. Both
// invocations of a mode must agree, and Direct must agree with Graph
// bitwise. Nothing in check mode is timed.
//
// Bench protocol: setup, then warmups, then a timed batch of back-to-back
// invocations and the same number of single invocations, each bracketed by
// CUDA events. Application scope replays enqueue_application; compute scope
// replays enqueue_compute after one application invocation made the inputs
// resident. Allocation, graph construction, warmups, the CPU oracle, and
// output checking stay outside every timed interval. Bench does not run the
// oracle; a recorded run validates with check first.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef LP_CONTRACT_SHA256
#error "build with -DLP_CONTRACT_SHA256=\"<sha256 of contract.json>\" (see common.mk)"
#endif
#ifndef LP_APP
#error "build with -DLP_APP=\"<application directory name>\" (see common.mk)"
#endif

#define LP_SUITE_VERSION "v1"
#define LP_LOOM_COMMIT "4a75f8d743ef7fe755822bab0ea0fb78009c145b"

#define LP_CUDA(call)                                                          \
  do {                                                                         \
    cudaError_t lp_error_ = (call);                                            \
    if (lp_error_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,       \
                   #call, cudaGetErrorString(lp_error_));                      \
      std::exit(lp::kExitCuda);                                                \
    }                                                                          \
  } while (0)

// Check the last kernel launch.
#define LP_CUDA_LAUNCH() LP_CUDA(cudaGetLastError())

namespace lp {

constexpr int kExitPass = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitUsage = 2;
constexpr int kExitAsset = 3;
constexpr int kExitCuda = 70;

__attribute__((format(printf, 2, 3))) [[noreturn]] inline void
fail(int code, const char *format, ...) {
  std::fprintf(stderr, "%s: ", LP_APP);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::exit(code);
}

// FNV-1a over bytes; the digest of an output compares Direct with Graph and
// repeated invocations. It is not a cryptographic hash.
inline uint64_t fnv1a64(const void *data, size_t bytes,
                        uint64_t state = 1469598103934665603ull) {
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; ++i) {
    state ^= p[i];
    state *= 1099511628211ull;
  }
  return state;
}

inline std::string hex64(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof text, "%016llx", (unsigned long long)value);
  return text;
}

// Minimal JSON value for the result record.
class Json {
public:
  static std::string quote(const std::string &text) {
    std::string out = "\"";
    for (char c : text) {
      switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char escape[7];
          std::snprintf(escape, sizeof escape, "\\u%04x", c);
          out += escape;
        } else {
          out += c;
        }
      }
    }
    return out + "\"";
  }
  static std::string number(double value) {
    char text[64];
    std::snprintf(text, sizeof text, "%.9g", value);
    return text;
  }
  static std::string integer(long long value) { return std::to_string(value); }
  static std::string boolean(bool value) { return value ? "true" : "false"; }
};

// Ordered key -> raw JSON text.
using JsonObject = std::vector<std::pair<std::string, std::string>>;

inline std::string render(const JsonObject &object) {
  std::string out = "{";
  for (size_t i = 0; i < object.size(); ++i) {
    if (i) out += ", ";
    out += Json::quote(object[i].first) + ": " + object[i].second;
  }
  return out + "}";
}

enum class Launch { Direct, Graph, Both };
enum class Action { Check, Bench };

inline const char *name(Launch launch) {
  return launch == Launch::Direct ? "direct"
         : launch == Launch::Graph ? "graph"
                                   : "both";
}

struct Options {
  std::string preset = "large";
  Launch launch = Launch::Both;
  bool launch_given = false;
  Action action = Action::Check;
  int iterations = 20;
  int warmup = 3;
  std::map<std::string, std::string> overrides; // --set
  std::map<std::string, std::string> assets;    // --asset
  std::string json_out;

  // Integer dimension from --set, removing it from the unconsumed set.
  long long take_int(const std::string &key, long long fallback) {
    auto it = overrides.find(key);
    if (it == overrides.end()) return fallback;
    char *end = nullptr;
    errno = 0;
    const long long value = std::strtoll(it->second.c_str(), &end, 0);
    if (errno || end == it->second.c_str() || *end)
      fail(kExitUsage, "--set %s=%s is not an integer", key.c_str(),
           it->second.c_str());
    overrides.erase(it);
    consumed_.push_back({key, std::to_string(value)});
    return value;
  }
  // Asset path from --asset, else the fallback (normally from the cache).
  std::string asset(const std::string &key, const std::string &fallback) const {
    auto it = assets.find(key);
    return it == assets.end() ? fallback : it->second;
  }
  const std::vector<std::pair<std::string, std::string>> &consumed() const {
    return consumed_;
  }

private:
  std::vector<std::pair<std::string, std::string>> consumed_;
};

// Default asset cache, outside tracked source. Fetch with
// applications/loom_private/tools/fetch_assets.py.
inline std::string asset_cache() {
  if (const char *env = std::getenv("LOOM_PRIVATE_ASSET_CACHE"))
    if (*env) return env;
  const char *home = std::getenv("HOME");
  return std::string(home ? home : ".") +
         "/.cache/cuda-elastic/loom-private-assets";
}

// One application. Buffers that cross PCIe in enqueue_application must be
// pinned (cudaMallocHost) so the work can be captured into a CUDA Graph.
// enqueue_* must be capture-safe: no synchronization, allocation, or host
// logic that depends on device results.
class Workload {
public:
  virtual ~Workload() = default;
  // Stable identity, e.g. "loom-private/pagerank".
  virtual std::string identity() const = 0;
  // Dimensions, as ordered JSON values, for the result record.
  virtual JsonObject dimensions() const = 0;
  // Asset name -> verified sha256; empty when the application has none.
  virtual JsonObject asset_hashes() const { return {}; }
  // What the two timed scopes include, in words.
  virtual std::string application_scope() const = 0;
  virtual std::string compute_scope() const = 0;
  // Allocate, generate or load inputs, upload resident state. Not timed.
  virtual void setup(cudaStream_t stream) = 0;
  // One complete invocation at application scope.
  virtual void enqueue_application(cudaStream_t stream) = 0;
  // The compute part alone, with inputs already resident.
  virtual void enqueue_compute(cudaStream_t stream) = 0;
  // Overwrite device and host outputs with a sentinel. Not timed.
  virtual void poison_outputs(cudaStream_t stream) = 0;
  // Independent CPU oracle, run once in check mode. Not timed.
  virtual void compute_reference() = 0;
  // Compare the host outputs with the oracle; print the first mismatches and
  // a summary to stderr. Return true when every output passes.
  virtual bool check_outputs() = 0;
  // Digest of every host output byte.
  virtual uint64_t output_digest() const = 0;
};

using Factory = std::unique_ptr<Workload> (*)(Options &options);

// Runs enqueue directly, or as one captured and instantiated graph.
class Executor {
public:
  Executor(Launch launch, cudaStream_t stream, Workload &workload,
           bool compute_only)
      : launch_(launch), stream_(stream), workload_(workload),
        compute_only_(compute_only) {
    if (launch_ == Launch::Graph) {
      cudaGraph_t graph = nullptr;
      LP_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      enqueue();
      LP_CUDA(cudaStreamEndCapture(stream_, &graph));
      LP_CUDA(cudaGraphInstantiate(&exec_, graph, 0));
      LP_CUDA(cudaGraphDestroy(graph));
      LP_CUDA(cudaGraphUpload(exec_, stream_));
      LP_CUDA(cudaStreamSynchronize(stream_));
    }
  }
  ~Executor() {
    if (exec_) cudaGraphExecDestroy(exec_);
  }
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  void launch() {
    if (launch_ == Launch::Graph)
      LP_CUDA(cudaGraphLaunch(exec_, stream_));
    else
      enqueue();
  }

private:
  void enqueue() {
    if (compute_only_)
      workload_.enqueue_compute(stream_);
    else
      workload_.enqueue_application(stream_);
  }

  Launch launch_;
  cudaStream_t stream_;
  Workload &workload_;
  bool compute_only_;
  cudaGraphExec_t exec_ = nullptr;
};

[[noreturn]] inline void usage(int code) {
  std::fprintf(
      code ? stderr : stdout,
      "usage: %s [--preset large] [--launch direct|graph|both]\n"
      "          [--action check|bench] [--iterations N] [--warmup W]\n"
      "          [--set key=value]... [--asset name=path]... [--json-out PATH]\n"
      "Defaults: --preset large --action check --launch both.\n"
      "bench needs an explicit --launch direct or --launch graph.\n",
      LP_APP);
  std::exit(code);
}

inline std::pair<std::string, std::string> split_pair(const char *flag,
                                                      const std::string &text) {
  const size_t equals = text.find('=');
  if (equals == std::string::npos || equals == 0)
    fail(kExitUsage, "%s expects key=value, got '%s'", flag, text.c_str());
  return {text.substr(0, equals), text.substr(equals + 1)};
}

inline int parse_count(const char *flag, const std::string &text, int minimum) {
  char *end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end || value < minimum || value > 1000000)
    fail(kExitUsage, "%s expects an integer >= %d, got '%s'", flag, minimum,
         text.c_str());
  return (int)value;
}

inline Options parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") usage(0);
    if (i + 1 >= argc) fail(kExitUsage, "%s needs a value", flag.c_str());
    const std::string value = argv[++i];
    if (flag == "--preset") {
      options.preset = value;
    } else if (flag == "--launch") {
      options.launch_given = true;
      if (value == "direct") options.launch = Launch::Direct;
      else if (value == "graph") options.launch = Launch::Graph;
      else if (value == "both") options.launch = Launch::Both;
      else fail(kExitUsage, "--launch must be direct, graph or both");
    } else if (flag == "--action") {
      if (value == "check") options.action = Action::Check;
      else if (value == "bench") options.action = Action::Bench;
      else fail(kExitUsage, "--action must be check or bench");
    } else if (flag == "--iterations") {
      options.iterations = parse_count("--iterations", value, 1);
    } else if (flag == "--warmup") {
      options.warmup = parse_count("--warmup", value, 0);
    } else if (flag == "--set") {
      auto kv = split_pair("--set", value);
      options.overrides[kv.first] = kv.second;
    } else if (flag == "--asset") {
      auto kv = split_pair("--asset", value);
      options.assets[kv.first] = kv.second;
    } else if (flag == "--json-out") {
      options.json_out = value;
    } else {
      fail(kExitUsage, "unknown option %s (see --help)", flag.c_str());
    }
  }
  if (options.action == Action::Bench) {
    if (!options.launch_given || options.launch == Launch::Both)
      fail(kExitUsage, "bench needs --launch direct or --launch graph");
    if (!options.overrides.empty())
      fail(kExitUsage, "--set is only valid with --action check");
  }
  return options;
}

inline double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

struct Timing {
  double batch_total_ms = 0.0;
  double batch_per_iteration_us = 0.0;
  std::vector<double> single_us;
};

inline Timing time_scope(Launch launch, cudaStream_t stream, Workload &workload,
                         bool compute_only, const Options &options) {
  Executor executor(launch, stream, workload, compute_only);
  for (int i = 0; i < options.warmup; ++i) executor.launch();
  LP_CUDA(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  LP_CUDA(cudaEventCreate(&start));
  LP_CUDA(cudaEventCreate(&stop));
  Timing timing;
  LP_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < options.iterations; ++i) executor.launch();
  LP_CUDA(cudaEventRecord(stop, stream));
  LP_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  LP_CUDA(cudaEventElapsedTime(&ms, start, stop));
  timing.batch_total_ms = ms;
  timing.batch_per_iteration_us = 1000.0 * ms / options.iterations;
  for (int i = 0; i < options.iterations; ++i) {
    LP_CUDA(cudaDeviceSynchronize());
    LP_CUDA(cudaEventRecord(start, stream));
    executor.launch();
    LP_CUDA(cudaEventRecord(stop, stream));
    LP_CUDA(cudaEventSynchronize(stop));
    LP_CUDA(cudaEventElapsedTime(&ms, start, stop));
    timing.single_us.push_back(1000.0 * ms);
  }
  LP_CUDA(cudaEventDestroy(start));
  LP_CUDA(cudaEventDestroy(stop));
  return timing;
}

inline std::string timing_json(const Timing &t, const std::string &scope) {
  std::string singles = "[";
  for (size_t i = 0; i < t.single_us.size(); ++i)
    singles += (i ? ", " : "") + Json::number(t.single_us[i]);
  singles += "]";
  return render({{"scope", Json::quote(scope)},
                 {"batch_total_ms", Json::number(t.batch_total_ms)},
                 {"batch_per_iteration_us", Json::number(t.batch_per_iteration_us)},
                 {"single_median_us", Json::number(median(t.single_us))},
                 {"single_us", singles}});
}

inline double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

inline int run_main(int argc, char **argv, Factory factory) {
  Options options = parse(argc, argv);
  std::unique_ptr<Workload> workload = factory(options);
  if (!workload)
    fail(kExitUsage, "unknown preset '%s' (available: large)",
         options.preset.c_str());
  if (!options.overrides.empty())
    fail(kExitUsage, "unsupported --set key '%s'",
         options.overrides.begin()->first.c_str());

  int device = 0;
  cudaDeviceProp properties{};
  LP_CUDA(cudaGetDevice(&device));
  LP_CUDA(cudaGetDeviceProperties(&properties, device));
  int runtime_version = 0, driver_version = 0;
  LP_CUDA(cudaRuntimeGetVersion(&runtime_version));
  LP_CUDA(cudaDriverGetVersion(&driver_version));

  cudaStream_t stream;
  LP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto clock = std::chrono::steady_clock::now();
  workload->setup(stream);
  LP_CUDA(cudaStreamSynchronize(stream));
  std::fprintf(stderr, "%s: setup %.2f s\n", LP_APP, seconds_since(clock));

  std::string overrides = "{";
  for (size_t i = 0; i < options.consumed().size(); ++i)
    overrides += (i ? ", " : "") + Json::quote(options.consumed()[i].first) +
                 ": " + options.consumed()[i].second;
  overrides += "}";

  JsonObject record = {
      {"schema", Json::quote("loom_private.result/1")},
      {"suite_version", Json::quote(LP_SUITE_VERSION)},
      {"identity", Json::quote(workload->identity())},
      {"loom_commit", Json::quote(LP_LOOM_COMMIT)},
      {"contract_sha256", Json::quote(LP_CONTRACT_SHA256)},
      {"preset", Json::quote(options.preset)},
      {"overrides", overrides},
      {"dimensions", render(workload->dimensions())},
      {"asset_sha256", render(workload->asset_hashes())},
      {"action", Json::quote(options.action == Action::Check ? "check" : "bench")},
      {"launch", Json::quote(name(options.launch))},
      {"device", render({{"name", Json::quote(properties.name)},
                         {"cc", Json::quote(std::to_string(properties.major) +
                                            "." + std::to_string(properties.minor))},
                         {"cuda_runtime", Json::integer(runtime_version)},
                         {"cuda_driver", Json::integer(driver_version)}})},
      {"timing_scope", render({{"application", Json::quote(workload->application_scope())},
                               {"compute", Json::quote(workload->compute_scope())}})},
  };

  int code = kExitPass;
  if (options.action == Action::Check) {
    clock = std::chrono::steady_clock::now();
    workload->compute_reference();
    std::fprintf(stderr, "%s: CPU oracle %.2f s\n", LP_APP, seconds_since(clock));
    std::vector<Launch> modes;
    if (options.launch != Launch::Graph) modes.push_back(Launch::Direct);
    if (options.launch != Launch::Direct) modes.push_back(Launch::Graph);
    std::string checks = "[";
    std::vector<uint64_t> digests;
    bool all_pass = true;
    for (Launch mode : modes) {
      Executor executor(mode, stream, *workload, false);
      for (int repetition = 1; repetition <= 2; ++repetition) {
        workload->poison_outputs(stream);
        LP_CUDA(cudaStreamSynchronize(stream));
        clock = std::chrono::steady_clock::now();
        executor.launch();
        LP_CUDA(cudaStreamSynchronize(stream));
        LP_CUDA(cudaGetLastError());
        const double seconds = seconds_since(clock);
        const bool pass = workload->check_outputs();
        const uint64_t digest = workload->output_digest();
        digests.push_back(digest);
        all_pass = all_pass && pass;
        std::fprintf(stderr, "%s: %s invocation %d: %s, digest %s (%.2f s, untimed)\n",
                     LP_APP, name(mode), repetition, pass ? "PASS" : "FAIL",
                     hex64(digest).c_str(), seconds);
        if (checks.size() > 1) checks += ", ";
        checks += render({{"launch", Json::quote(name(mode))},
                          {"invocation", Json::integer(repetition)},
                          {"pass", Json::boolean(pass)},
                          {"digest", Json::quote(hex64(digest))}});
      }
    }
    checks += "]";
    bool consistent = true;
    for (uint64_t digest : digests) consistent = consistent && digest == digests[0];
    if (!consistent)
      std::fprintf(stderr, "%s: FAIL: invocations or launch modes disagree\n", LP_APP);
    all_pass = all_pass && consistent;
    record.push_back({"checks", checks});
    record.push_back({"outputs_agree", Json::boolean(consistent)});
    record.push_back({"timing", "null"});
    record.push_back({"status", Json::quote(all_pass ? "pass" : "fail")});
    code = all_pass ? kExitPass : kExitMismatch;
  } else {
    const Timing application =
        time_scope(options.launch, stream, *workload, false, options);
    const Timing compute =
        time_scope(options.launch, stream, *workload, true, options);
    LP_CUDA(cudaStreamSynchronize(stream));
    record.push_back({"iterations", Json::integer(options.iterations)});
    record.push_back({"warmup", Json::integer(options.warmup)});
    record.push_back({"timing", render({{"application", timing_json(application, workload->application_scope())},
                                        {"compute", timing_json(compute, workload->compute_scope())}})});
    record.push_back({"output_digest", Json::quote(hex64(workload->output_digest()))});
    record.push_back({"status", Json::quote("unvalidated_bench")});
  }

  const std::string line = render(record);
  std::printf("RESULT %s\n", line.c_str());
  std::fflush(stdout);
  if (!options.json_out.empty()) {
    FILE *file = std::fopen(options.json_out.c_str(), "w");
    if (!file || std::fprintf(file, "%s\n", line.c_str()) < 0 || std::fclose(file))
      fail(kExitUsage, "cannot write %s", options.json_out.c_str());
  }
  LP_CUDA(cudaStreamDestroy(stream));
  return code;
}

} // namespace lp
