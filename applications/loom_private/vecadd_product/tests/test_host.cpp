// Host tests for loom-private/vecadd-product: SHA-256, the reference-line
// parser, the oracle's arithmetic, and record 0 against the upstream sample.
// Needs the cached assets (applications/loom_private/tools/fetch_assets.py).
#include "../fixtures.h"
#include "../oracle.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fx = vecadd_fixtures;

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static std::string sha(const std::string &text) {
  return fx::sha256_hex((const uint8_t *)text.data(), text.size());
}

int main() {
  // SHA-256 known answers, including the 55/56/64-byte padding boundaries.
  CHECK(sha("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(sha(std::string(1000000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  CHECK(sha(std::string(55, 'x')) != sha(std::string(56, 'x')));
  CHECK(sha(std::string(64, 'x')) != sha(std::string(63, 'x')));

  const char *home = std::getenv("HOME");
  const char *env = std::getenv("LOOM_PRIVATE_ASSET_CACHE");
  const std::string cache = (env && *env) ? env
      : std::string(home ? home : ".") + "/.cache/cuda-elastic/loom-private-assets";
  std::vector<uint8_t> fixture, reference_text;
  const std::string p1 = fx::read_verified(
      cache + "/vecadd-product-operands", 128,
      "89712b453849a8d420cce0c7a5afff97c7c5f5a84509abef8fd47b67075815fb", fixture);
  const std::string p2 = fx::read_verified(
      cache + "/vecadd-product-upstream-expected-smoke", 154,
      "39a9a7849bacd718ad79ac23efb85bdc6dcd7a1e18fcf0a5302ccf04a914b39b",
      reference_text);
  if (!p1.empty() || !p2.empty()) {
    std::printf("FAIL assets: %s %s\n", p1.c_str(), p2.c_str());
    return 1;
  }
  // The verified reader rejects a wrong size and a wrong hash.
  std::vector<uint8_t> scratch;
  CHECK(!fx::read_verified(cache + "/vecadd-product-operands", 127,
                           "89712b453849a8d420cce0c7a5afff97c7c5f5a84509abef8fd47b67075815fb",
                           scratch).empty());
  CHECK(!fx::read_verified(cache + "/vecadd-product-operands", 128,
                           std::string(64, '0'), scratch).empty());
  CHECK(!fx::read_verified(cache + "/does-not-exist", 128, "", scratch).empty());

  // Record 0 equals upstream's one sample (byte-for-byte comparison with the
  // reference file; the upstream smoke configuration is not run).
  uint8_t upstream[64], slot[64];
  CHECK(fx::parse_reference_sample(reference_text, upstream).empty());
  vecadd_oracle::record_output(fixture.data(), 0, slot);
  CHECK(std::memcmp(slot, upstream, 64) == 0);
  // The same sample computed straight from the fixture's two operands.
  for (int k = 0; k < 64; ++k) CHECK(slot[k] == (uint8_t)(fixture[k] + fixture[64 + k]));

  // A malformed reference line is rejected.
  std::vector<uint8_t> bad = reference_text;
  bad[30] = 'g';
  CHECK(!fx::parse_reference_sample(bad, upstream).empty());
  bad = reference_text;
  bad.pop_back();
  CHECK(!fx::parse_reference_sample(bad, upstream).empty());

  // Oracle versus the closed form: byte k is fixture[k] + fixture[64+k] +
  // 2 (r mod 256), mod 256. Also: the output repeats with period 256 records,
  // and neighbouring slots differ in every lane (each lane moves by 2).
  long long wrapped = 0, unwrapped = 0;
  for (uint64_t r = 0; r < 1000; ++r) {
    uint8_t a[64], b[64], c[64];
    vecadd_oracle::record_output(fixture.data(), r, a);
    vecadd_oracle::record_output(fixture.data(), r + 1, b);
    vecadd_oracle::record_output(fixture.data(), r + 256, c);
    CHECK(std::memcmp(a, c, 64) == 0);
    for (int k = 0; k < 64; ++k) {
      CHECK(a[k] == (uint8_t)(fixture[k] + fixture[64 + k] + 2 * (r % 256)));
      CHECK(a[k] != b[k]);
      CHECK(b[k] == (uint8_t)(a[k] + 2));
    }
  }
  // Wrapping addition occurs: over one 256-record period some lanes' true sums
  // exceed 255 (so the byte wraps) and some do not.
  for (uint64_t r = 0; r < 256; ++r)
    for (int k = 0; k < 64; ++k) {
      const int left = (fixture[k] + (int)r) % 256, right = (fixture[64 + k] + (int)r) % 256;
      (left + right > 255 ? wrapped : unwrapped)++;
    }
  CHECK(wrapped > 0 && unwrapped > 0);
  std::printf("lanes over one 256-record period: %lld wrap, %lld do not\n",
              wrapped, unwrapped);

  // The threaded bulk oracle equals the per-record oracle for counts that do
  // not divide evenly among the workers.
  for (uint64_t n : {1ull, 2ull, 7ull, 255ull, 257ull, 10007ull}) {
    const std::vector<uint8_t> bulk = vecadd_oracle::all_outputs(fixture.data(), n);
    CHECK(bulk.size() == n * 64);
    for (uint64_t r = 0; r < n; ++r) {
      vecadd_oracle::record_output(fixture.data(), r, slot);
      CHECK(std::memcmp(bulk.data() + r * 64, slot, 64) == 0);
    }
  }

  if (failures) {
    std::printf("host tests FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("host tests passed\n");
  return 0;
}
