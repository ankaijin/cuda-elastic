// loom-private/vecadd-product: batched CUDA extension of private Loom's
// cached-operand vecadd product interface. Upstream adds two cached 64-byte
// operands lane by lane (mod 256) into one 64-byte output slot per measured
// sample. This port runs R = 2^20 independent records (two 64-byte operands
// each, derived on the host from the 128-byte fixture) and writes one 64-byte
// slot per record. See contract.json and README.md.
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
#ifndef LP_VECADD_MUTANT
#define LP_VECADD_MUTANT 0
#endif

// Byte-lane sum modulo 256 of four packed bytes: add the low seven bits of
// each lane (no carry can leave a lane), then fix up each lane's top bit.
__device__ __forceinline__ uint32_t add_bytes(uint32_t a, uint32_t b) {
#if LP_VECADD_MUTANT == 1
  return __vaddus4(a, b);
#else
  return ((a & 0x7f7f7f7fu) + (b & 0x7f7f7f7fu)) ^ ((a ^ b) & 0x80808080u);
#endif
}

// One thread per 4 output bytes. Word w is lane group (w mod 16) of record
// (w / 16); the record's left operand is input words 32r .. 32r+15 and its
// right operand words 32r+16 .. 32r+31.
__global__ void add_records(const uint32_t *input, uint32_t *output,
                            uint64_t words) {
  const uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
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
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_input_, h_input_, records_ * 128,
                            cudaMemcpyHostToDevice, stream));
#ifdef LP_VECADD_TEST_SLOW_WARMUP
    if (application_invocations++ < 3) {
      spin_300ms<<<1, 1, 0, stream>>>();
      LP_CUDA_LAUNCH();
    }
#endif
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_output_, d_output_, records_ * 64,
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    const uint64_t words = records_ * 16;
    const uint64_t blocks = (words + kBlock - 1) / kBlock;
    add_records<<<(unsigned)blocks, kBlock, 0, stream>>>(
        reinterpret_cast<const uint32_t *>(d_input_),
        reinterpret_cast<uint32_t *>(d_output_), words);
    LP_CUDA_LAUNCH();
  }

  void poison_outputs(cudaStream_t stream) override {
    // Distinct poison per byte position; the canary guards on both sides get
    // their own pattern. Host poison is the same bytes the device receives.
    const size_t bytes = records_ * 64;
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
