// Experiment A (experiments/blackwell/packed), candidate add_u8x4: the
// production optimized variant (applications/loom_private/vecadd_product/
// vecadd_product_optimized.cu) with one change, add_bytes as the native packed
// byte add (PTX add.u8x4, which needs sm_120f or sm_120a). Everything else,
// including the chunked pipeline and the mutants, is the production code;
// add_u8x4.diff is the exact difference.
#include "lp_harness.cuh"

#include "fixtures.h"
#include "oracle.h"

#include <cuda_runtime.h>

namespace {

namespace fx = vecadd_fixtures;

constexpr long long kDefaultRecords = 1048576;
// Largest record count accepted by --set records: 2^24 records are a 2 GiB
// input and a 1 GiB output (plus the host copies and the oracle's output).
constexpr long long kMaxRecords = 1ll << 24;
constexpr int kBlock = 256;
constexpr size_t kGuardBytes = 256; // canary bytes before and after the output

// Chunks per invocation. The sweep on the RTX 3090, 4090 and 5090 put each on
// the same plateau (see the README), so one count serves all three; to give a
// GPU its own, select on `properties.major * 10 + properties.minor` here.
constexpr int kChunks = 16;

#ifdef LP_VECADD_TEST_TUNING
// Test builds only (`make test`): LP_VECADD_CHUNKS (1 to 64) overrides the
// count, so small problems cross chunk boundaries. The result does not depend
// on it. A value out of range exits 2. The suite binary has none of this.
int chunks_for() {
  const char *text = std::getenv("LP_VECADD_CHUNKS");
  if (!text) return kChunks;
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end || value < 1 || value > 64)
    lp::fail(lp::kExitUsage, "LP_VECADD_CHUNKS must be an integer from 1 to 64, got '%s'",
             text);
  return (int)value;
}
#else
int chunks_for() { return kChunks; }
#endif

// Registered in applications/loom_private/assets.json; the sizes and hashes
// below must match it.
constexpr const char *kOperandsAsset = "vecadd-product-operands";
constexpr size_t kOperandsBytes = 128;
constexpr const char *kOperandsSha256 =
    "89712b453849a8d420cce0c7a5afff97c7c5f5a84509abef8fd47b67075815fb";
constexpr const char *kReferenceAsset = "vecadd-product-upstream-expected-smoke";
constexpr size_t kReferenceBytes = 154;
constexpr const char *kReferenceSha256 =
    "39a9a7849bacd718ad79ac23efb85bdc6dcd7a1e18fcf0a5302ccf04a914b39b";

// Test-only variants, compiled by `make test` into separate binaries that
// the tests expect to fail the check (never part of the suite build):
//   1 saturating byte add (no wraparound)   2 one word written past the output
//   3 each record's sum lands in its neighbour's slot   4 some slots skipped
//   5 the last chunk is never uploaded   6 the last chunk is never downloaded
//   7 the harness stream never joins the chunks
#ifndef LP_VECADD_MUTANT
#define LP_VECADD_MUTANT 0
#endif

// Byte-lane sum modulo 256 of four packed bytes: one native packed add,
// which wraps within each lane (no carry leaves a lane).
__device__ __forceinline__ uint32_t add_bytes(uint32_t a, uint32_t b) {
#if LP_VECADD_MUTANT == 1
  return __vaddus4(a, b);
#else
  uint32_t sum;
  asm("add.u8x4 %0, %1, %2;" : "=r"(sum) : "r"(a), "r"(b));
  return sum;
#endif
}

// One thread per 4 output bytes, for the words [first, words) of the whole
// output. Word w is lane group (w mod 16) of record (w / 16); the record's left
// operand is input words 32r .. 32r+15 and its right operand words 32r+16 ..
// 32r+31. A chunk launches only its own range; indices stay absolute.
__global__ void add_records(const uint32_t *input, uint32_t *output,
                            uint64_t first, uint64_t words) {
  const uint64_t w = first + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= words) return;
#if LP_VECADD_MUTANT == 4
  if ((w / 16) % 256 == 255) return;
#endif
  const uint64_t record = w / 16, group = w % 16;
  const uint32_t sum =
      add_bytes(input[record * 32 + group], input[record * 32 + 16 + group]);
#if LP_VECADD_MUTANT == 3
  output[w ^ 16] = sum;
#else
  output[w] = sum;
#endif
#if LP_VECADD_MUTANT == 2
  if (w == words - 1) output[words] = sum;
#endif
}

#if LP_VECADD_MUTANT == 7
// Test-only (no-join mutant): hold the last chunk for 20 ms, so that without
// the join the harness stream is certain to finish first.
__global__ void spin_20ms() {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < 20000000ull);
}
#endif

#ifdef LP_VECADD_TEST_SLOW_WARMUP
// Test-only: the first three application invocations (the default bench
// warmups) also spin the GPU for 300 ms, so the bench batch time shows whether
// they fell inside the timed interval.
__global__ void spin_300ms() {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < 300000000ull);
}
int application_invocations = 0;
#endif

uint8_t poison_byte(size_t i) { return (uint8_t)(0xA5 ^ (i * 29)); }
uint8_t guard_byte(size_t i) { return (uint8_t)(0x3C + 7 * i); }

class VecaddProduct : public lp::Workload {
public:
  VecaddProduct(long long records, std::string operands_path,
                std::string reference_path)
      : records_((uint64_t)records), operands_path_(std::move(operands_path)),
        reference_path_(std::move(reference_path)) {}

  ~VecaddProduct() override {
    for (cudaStream_t s : streams_) cudaStreamDestroy(s);
    for (cudaEvent_t e : joins_) cudaEventDestroy(e);
    if (fork_) cudaEventDestroy(fork_);
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFreeHost(h_guard_);
    cudaFree(d_input_);
    cudaFree(d_output_base_);
  }

  std::string identity() const override { return "loom-private/vecadd-product"; }

  lp::JsonObject dimensions() const override {
    return {{"records", lp::Json::integer((long long)records_)},
            {"operand_bytes", lp::Json::integer(64)},
            {"input_bytes", lp::Json::integer((long long)(records_ * 128))},
            {"output_bytes", lp::Json::integer((long long)(records_ * 64))}};
  }

  lp::JsonObject asset_hashes() const override {
    return {{kOperandsAsset, lp::Json::quote(kOperandsSha256)},
            {kReferenceAsset, lp::Json::quote(kReferenceSha256)}};
  }

  std::string application_scope() const override {
    return "upload of the 128-byte-per-record inputs from pinned host memory, "
           "the lane-wise byte-sum kernel over all records, and the download "
           "of the 64-byte-per-record outputs to pinned host memory";
  }
  std::string compute_scope() const override {
    return "the lane-wise byte-sum kernel over all records with the inputs "
           "resident; no transfers";
  }

  void setup(cudaStream_t) override {
    load_asset(operands_path_, kOperandsAsset, kOperandsBytes, kOperandsSha256,
               fixture_);
    std::vector<uint8_t> reference;
    load_asset(reference_path_, kReferenceAsset, kReferenceBytes,
               kReferenceSha256, reference);
    const std::string problem = fx::parse_reference_sample(reference, upstream_slot_);
    if (!problem.empty())
      lp::fail(lp::kExitAsset, "asset '%s': %s", kReferenceAsset, problem.c_str());

    const size_t input_bytes = records_ * 128, output_bytes = records_ * 64;
    LP_CUDA(cudaMallocHost(&h_input_, input_bytes));
    LP_CUDA(cudaMallocHost(&h_output_, output_bytes));
    LP_CUDA(cudaMallocHost(&h_guard_, kGuardBytes));
    // Record r, operand byte j: (fixture[j] + (r mod 256)) mod 256.
    for (uint64_t r = 0; r < records_; ++r) {
      uint8_t *record = h_input_ + r * 128;
      const unsigned shift = (unsigned)(r & 255);
      for (int j = 0; j < 128; ++j)
        record[j] = (uint8_t)((fixture_[j] + shift) & 255);
    }
    for (size_t i = 0; i < kGuardBytes; ++i) h_guard_[i] = guard_byte(i);

    LP_CUDA(cudaMalloc(&d_input_, input_bytes));
    LP_CUDA(cudaMalloc(&d_output_base_, kGuardBytes + output_bytes + kGuardBytes));
    d_output_ = d_output_base_ + kGuardBytes;

    const uint64_t chunks = std::min<uint64_t>((uint64_t)chunks_for(), records_);
    // Chunk c holds records [begin_[c], begin_[c + 1]). One stream per chunk,
    // joined back through events, so Direct and the captured Graph both fork
    // and join the same pipeline.
    for (uint64_t c = 0; c <= chunks; ++c) begin_.push_back(records_ * c / chunks);
    LP_CUDA(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    for (uint64_t c = 0; c < chunks; ++c) {
      cudaStream_t stream;
      cudaEvent_t join;
      LP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      LP_CUDA(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
      streams_.push_back(stream);
      joins_.push_back(join);
    }
  }

  void enqueue_application(cudaStream_t stream) override {
#ifdef LP_VECADD_TEST_SLOW_WARMUP
    if (application_invocations++ < 3) {
      spin_300ms<<<1, 1, 0, stream>>>();
      LP_CUDA_LAUNCH();
    }
#endif
    LP_CUDA(cudaEventRecord(fork_, stream));
    for (size_t c = 0; c < streams_.size(); ++c) {
      cudaStream_t side = streams_[c];
      const uint64_t first = begin_[c], count = begin_[c + 1] - begin_[c];
      LP_CUDA(cudaStreamWaitEvent(side, fork_, 0));
#if LP_VECADD_MUTANT == 5
      if (c + 1 < streams_.size())
#endif
        LP_CUDA(cudaMemcpyAsync(d_input_ + first * 128, h_input_ + first * 128,
                                count * 128, cudaMemcpyHostToDevice, side));
      launch(first, first + count, side);
#if LP_VECADD_MUTANT == 7
      if (c + 1 == streams_.size()) {
        spin_20ms<<<1, 1, 0, side>>>();
        LP_CUDA_LAUNCH();
      }
#endif
#if LP_VECADD_MUTANT == 6
      if (c + 1 < streams_.size())
#endif
        LP_CUDA(cudaMemcpyAsync(h_output_ + first * 64, d_output_ + first * 64,
                                count * 64, cudaMemcpyDeviceToHost, side));
      LP_CUDA(cudaEventRecord(joins_[c], side));
    }
#if LP_VECADD_MUTANT != 7
    for (size_t c = 0; c < streams_.size(); ++c)
      LP_CUDA(cudaStreamWaitEvent(stream, joins_[c], 0));
#endif
  }

  void enqueue_compute(cudaStream_t stream) override {
    launch(0, records_, stream);
  }

  void poison_outputs(cudaStream_t stream) override {
    // Distinct poison per byte position; the canary guards on both sides get
    // their own pattern. Host poison is the same bytes the device receives.
    const size_t bytes = records_ * 64;
    // Device input: every left operand complemented (staged through
    // h_output_), right operands as uploaded. If an invocation skips the
    // upload, each output byte is (255 - a + b) mod 256, which never equals
    // (a + b) mod 256 because 2a is even, so every byte fails the check.
    for (uint64_t r = 0; r < records_; ++r)
      for (int j = 0; j < 64; ++j) h_output_[r * 64 + j] = (uint8_t)~h_input_[r * 128 + j];
    LP_CUDA(cudaMemcpy2DAsync(d_input_, 128, h_output_, 64, 64, records_,
                              cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpy2DAsync(d_input_ + 64, 128, h_input_ + 64, 128, 64, records_,
                              cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaStreamSynchronize(stream)); // before h_output_ is reused
    for (size_t i = 0; i < bytes; ++i) h_output_[i] = poison_byte(i);
    LP_CUDA(cudaMemcpyAsync(d_output_, h_output_, bytes, cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_output_base_, h_guard_, kGuardBytes,
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_output_ + bytes, h_guard_, kGuardBytes,
                            cudaMemcpyHostToDevice, stream));
  }

  void compute_reference() override {
    expected_ = vecadd_oracle::all_outputs(fixture_.data(), records_);
    // Record 0 is upstream's single sample: compare bytes with the reference
    // file (this does not run the upstream smoke configuration).
    if (std::memcmp(expected_.data(), upstream_slot_, 64) != 0)
      lp::fail(lp::kExitMismatch,
               "oracle record 0 differs from asset '%s' (upstream sample 0)",
               kReferenceAsset);
  }

  bool check_outputs() override {
    const size_t bytes = records_ * 64;
    long long mismatches = 0;
    for (size_t i = 0; i < bytes; ++i) {
      if (h_output_[i] == expected_[i]) continue;
      if (mismatches < 8)
        std::fprintf(stderr,
                     "%s: mismatch at output byte %zu (record %zu, lane %zu): "
                     "expected 0x%02x, actual 0x%02x\n",
                     LP_APP, i, i / 64, i % 64, expected_[i], h_output_[i]);
      ++mismatches;
    }
    // Canary: the bytes just outside the output extent must be untouched.
    uint8_t before[kGuardBytes], after[kGuardBytes];
    LP_CUDA(cudaMemcpy(before, d_output_base_, kGuardBytes, cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(after, d_output_ + bytes, kGuardBytes, cudaMemcpyDeviceToHost));
    long long guard_damage = 0;
    for (size_t i = 0; i < kGuardBytes; ++i) {
      if (before[i] != guard_byte(i)) {
        if (guard_damage < 4)
          std::fprintf(stderr, "%s: canary before the output overwritten at offset %zu\n",
                       LP_APP, i);
        ++guard_damage;
      }
      if (after[i] != guard_byte(i)) {
        if (guard_damage < 4)
          std::fprintf(stderr, "%s: canary after the output overwritten at offset %zu\n",
                       LP_APP, i);
        ++guard_damage;
      }
    }
    std::fprintf(stderr,
                 "%s: %lld of %zu output bytes differ from the oracle; %lld "
                 "canary bytes damaged\n",
                 LP_APP, mismatches, bytes, guard_damage);
    return mismatches == 0 && guard_damage == 0;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_output_, records_ * 64);
  }

private:
  // The kernel for records [first, end).
  void launch(uint64_t first, uint64_t end, cudaStream_t stream) const {
    const uint64_t words = (end - first) * 16;
    const uint64_t blocks = (words + kBlock - 1) / kBlock;
    add_records<<<(unsigned)blocks, kBlock, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(d_input_),
        reinterpret_cast<uint32_t *>(d_output_), first * 16, end * 16);
    LP_CUDA_LAUNCH();
  }

  static void load_asset(const std::string &path, const char *name, size_t bytes,
                         const char *sha256, std::vector<uint8_t> &out) {
    const std::string problem = fx::read_verified(path, bytes, sha256, out);
    if (!problem.empty())
      lp::fail(lp::kExitAsset, "asset '%s': %s", name, problem.c_str());
  }

  uint64_t records_;
  std::string operands_path_, reference_path_;
  std::vector<uint8_t> fixture_, expected_;
  uint8_t upstream_slot_[64] = {};
  std::vector<uint64_t> begin_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> joins_;
  cudaEvent_t fork_ = nullptr;
  uint8_t *h_input_ = nullptr, *h_output_ = nullptr, *h_guard_ = nullptr;
  uint8_t *d_input_ = nullptr, *d_output_base_ = nullptr, *d_output_ = nullptr;
};

std::unique_ptr<lp::Workload> make(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long records = options.take_int("records", kDefaultRecords);
  if (records < 1 || records > kMaxRecords)
    lp::fail(lp::kExitUsage,
             "records must be from 1 to %lld (2^24; the input is 128 bytes "
             "and the output 64 bytes per record), got %lld",
             kMaxRecords, records);
  const std::string cache = lp::asset_cache();
  return std::make_unique<VecaddProduct>(
      records, options.asset(kOperandsAsset, cache + "/" + kOperandsAsset),
      options.asset(kReferenceAsset, cache + "/" + kReferenceAsset));
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make); }
