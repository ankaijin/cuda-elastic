// Host test of the experiment C reference (reference.cpp) against CUTLASS
// 4.8.0's own number types and sub-byte packing, and of the quantizers'
// defined edge cases (spec.json "rounding", "mxfp6", "nvfp4"). Exit 0 when
// every check passes.
#include "reference.h"

#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"
#include "cute/pointer.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace ref = lowprec_ref;

namespace {

int failures = 0;
void expect(bool ok, const char *what, double got = 0, double want = 0) {
  if (ok) return;
  if (++failures <= 20) std::printf("FAIL %s: got %.9g, expected %.9g\n", what, got, want);
}

template <class T> void decode_matches(ref::Format format, unsigned codes, const char *name) {
  for (unsigned code = 0; code < codes; ++code) {
    const double cutlass_value = (float)T::bitcast((typename T::Storage)code);
    if (std::isnan(cutlass_value)) continue; // NaN codes are never produced
    expect(ref::decode(format, code) == cutlass_value, name, ref::decode(format, code),
           cutlass_value);
  }
}

template <class T> void packing_matches(ref::Format format, unsigned (*unpack)(const uint8_t *, long long),
                                        int bits, const char *name) {
  std::mt19937 rng(17);
  const int count = 1024;
  std::vector<uint8_t> bytes(count * bits / 8 + 8, 0);
  std::vector<unsigned> codes(count);
  auto it = cute::recast_ptr<T>(bytes.data());
  for (int i = 0; i < count; ++i) {
    codes[i] = rng() % (1u << bits);
    it[i] = T::bitcast((typename T::Storage)codes[i]);
  }
  for (int i = 0; i < count; ++i) expect(unpack(bytes.data(), i) == codes[i], name, unpack(bytes.data(), i), codes[i]);
  (void)format;
}

} // namespace

int main() {
  decode_matches<cutlass::float_e2m1_t>(ref::Format::E2M1, 16, "E2M1 decode");
  decode_matches<cutlass::float_e3m2_t>(ref::Format::E3M2, 64, "E3M2 decode");
  decode_matches<cutlass::float_ue4m3_t>(ref::Format::UE4M3, 127, "UE4M3 decode");
  decode_matches<cutlass::float_ue8m0_t>(ref::Format::UE8M0, 255, "UE8M0 decode");
  packing_matches<cutlass::float_e2m1_t>(ref::Format::E2M1, ref::unpack_e2m1, 4, "E2M1 packing");
  packing_matches<cutlass::float_e3m2_t>(ref::Format::E3M2, ref::unpack_e3m2, 6, "E3M2 packing");

  // Every code encodes back to itself (except -0, which becomes +0).
  for (ref::Format f : {ref::Format::E2M1, ref::Format::E3M2, ref::Format::UE4M3}) {
    const unsigned codes = f == ref::Format::E2M1 ? 16 : f == ref::Format::E3M2 ? 64 : 127;
    const unsigned sign = f == ref::Format::E2M1 ? 8 : f == ref::Format::E3M2 ? 32 : 0;
    for (unsigned code = 0; code < codes; ++code) {
      const unsigned want = sign && (code & (sign - 1)) == 0 ? 0 : code;
      expect(ref::encode(f, ref::decode(f, code)) == want, "round trip", ref::encode(f, ref::decode(f, code)), want);
    }
  }
  // Ties to the even code, saturation, and +0.
  const struct { ref::Format f; double v; unsigned code; } cases[] = {
      {ref::Format::E2M1, 0.25, 0}, {ref::Format::E2M1, 0.75, 2}, {ref::Format::E2M1, 1.25, 2},
      {ref::Format::E2M1, 1.75, 4}, {ref::Format::E2M1, 2.5, 4},  {ref::Format::E2M1, 3.5, 6},
      {ref::Format::E2M1, 5.0, 6},  {ref::Format::E2M1, 7.0, 7},  {ref::Format::E2M1, 1e9, 7},
      {ref::Format::E2M1, -5.0, 14}, {ref::Format::E2M1, -0.1, 0}, {ref::Format::E2M1, 0.26, 1},
      {ref::Format::E3M2, 26.0, 0x1e}, {ref::Format::E3M2, 30.0, 0x1f}, {ref::Format::E3M2, 31.9, 0x1f},
      {ref::Format::E3M2, 0.03125, 0}, {ref::Format::E3M2, 0.09375, 2}, {ref::Format::E3M2, -0.01, 0},
      {ref::Format::E3M2, 0.28125, 4}, {ref::Format::UE4M3, 448.0, 0x7e}, {ref::Format::UE4M3, 500.0, 0x7e},
      {ref::Format::UE4M3, 0x1p-10, 0}, {ref::Format::UE4M3, 3 * 0x1p-10, 2}, {ref::Format::UE4M3, 1.0625, 0x38}};
  for (const auto &c : cases) expect(ref::encode(c.f, c.v) == c.code, "rounding case", ref::encode(c.f, c.v), c.code);
  expect(ref::to_bf16(1.0f + 0x1p-8f) == 0x3f80, "bf16 tie to even");
  expect(ref::to_bf16(1.0f + 0x1.8p-8f) == 0x3f81, "bf16 round up");

  // MXFP6: zero block, unit scale, saturation of a block maximum near 2^(E+1).
  {
    std::vector<float> row(64, 0.0f);
    row[32] = 16.0f;
    row[33] = 31.9f / 2;
    std::vector<uint8_t> codes(64), scales(2);
    ref::quantize_mxfp6(row.data(), 64, codes.data(), scales.data());
    expect(scales[0] == 0 && codes[0] == 0, "MXFP6 zero block");
    expect(scales[1] == 127, "MXFP6 scale of 16", scales[1], 127);
    expect(codes[32] == 0x1c, "MXFP6 16 at unit scale", codes[32], 0x1c);
    row[32] = 31.9f;
    ref::quantize_mxfp6(row.data(), 64, codes.data(), scales.data());
    expect(scales[1] == 127 && codes[32] == 0x1f, "MXFP6 saturation below 2^(E+1)", codes[32], 0x1f);
  }
  // NVFP4: encode scale, a block scale that rounds to zero, a unit block.
  {
    const float s = ref::nvfp4_encode_scale(2688.0f);
    expect(s == 1.0f, "NVFP4 encode scale", s, 1);
    expect(ref::nvfp4_encode_scale(0.0f) == 1.0f, "NVFP4 zero tensor");
    std::vector<float> row(32, 0.0f);
    row[0] = 1e-6f;
    for (int i = 16; i < 32; ++i) row[i] = (float)(i - 16) * 6.0f / 15.0f;
    std::vector<uint8_t> codes(32), scales(2);
    ref::quantize_nvfp4(row.data(), 32, s, codes.data(), scales.data());
    expect(scales[0] == 0 && codes[0] == 0, "NVFP4 tiny block", scales[0], 0);
    expect(scales[1] == 0x38, "NVFP4 unit block scale", scales[1], 0x38);
    expect(codes[31] == 7, "NVFP4 block maximum", codes[31], 7);
  }
  std::printf("%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
