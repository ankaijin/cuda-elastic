// Independent host reference for experiment C; see reference.h and spec.json.
#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace lowprec_ref {

namespace {

// Largest magnitude code and sign bit of each format (no sign bit: 0).
unsigned max_magnitude_code(Format format) {
  switch (format) {
  case Format::E2M1: return 7;
  case Format::E3M2: return 31;
  case Format::UE4M3: return 126; // 0x7f is NaN
  case Format::UE8M0: return 254; // 255 is NaN
  }
  std::abort();
}
unsigned sign_bit(Format format) {
  return format == Format::E2M1 ? 8 : format == Format::E3M2 ? 32 : 0;
}

double magnitude(Format format, unsigned code) {
  switch (format) {
  case Format::E2M1: {
    static const double values[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return values[code & 7];
  }
  case Format::E3M2: {
    const int e = (code >> 2) & 7, m = code & 3;
    return e == 0 ? std::ldexp(m / 4.0, -2) : std::ldexp(1.0 + m / 4.0, e - 3);
  }
  case Format::UE4M3: {
    const int e = (code >> 3) & 15, m = code & 7;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
  }
  case Format::UE8M0:
    return std::ldexp(1.0, (int)code - 127);
  }
  std::abort();
}

} // namespace

double decode(Format format, unsigned code) {
  const unsigned sign = sign_bit(format);
  const double value = magnitude(format, sign ? code & (sign - 1) : code);
  return sign && (code & sign) ? -value : value;
}

// Magnitudes of every non-negative code, in code order (increasing).
const std::vector<double> &magnitudes(Format format) {
  static const std::vector<std::vector<double>> tables = [] {
    std::vector<std::vector<double>> built(4);
    for (Format f : {Format::E2M1, Format::E3M2, Format::UE4M3, Format::UE8M0})
      for (unsigned code = 0; code <= max_magnitude_code(f); ++code)
        built[(int)f].push_back(magnitude(f, code));
    return built;
  }();
  return tables[(int)format];
}

unsigned encode(Format format, double v) {
  const double target = std::fabs(v);
  const std::vector<double> &table = magnitudes(format);
  // The last code at or below the target, compared with the next one.
  const unsigned top = (unsigned)table.size() - 1;
  const unsigned below =
      (unsigned)(std::upper_bound(table.begin(), table.end(), target) - table.begin()) - 1;
  unsigned best = below;
  if (below < top && table[below] < target) {
    const double low = target - table[below];
    const double high = table[below + 1] - target;
    if (high < low || (high == low && (below & 1))) best = below + 1;
  }
  if (best == 0) return 0;
  return v < 0 && sign_bit(format) ? best | sign_bit(format) : best;
}

uint16_t to_bf16(float v) {
  uint32_t bits;
  static_assert(sizeof bits == sizeof v, "float is 32 bits");
  std::memcpy(&bits, &v, sizeof bits);
  const uint16_t toward_zero = (uint16_t)(bits >> 16);
  if ((bits & 0xffffu) == 0) return toward_zero;
  const uint16_t away = (uint16_t)(toward_zero + 1); // next magnitude, same sign
  const double low = std::fabs(bf16_value(toward_zero) - (double)v);
  const double high = std::fabs(bf16_value(away) - (double)v);
  if (low < high) return toward_zero;
  if (high < low) return away;
  return (toward_zero & 1) ? away : toward_zero;
}

double bf16_value(uint16_t bits) {
  const uint32_t wide = (uint32_t)bits << 16;
  float v;
  std::memcpy(&v, &wide, sizeof v);
  return v;
}

void quantize_mxfp6(const float *row, long long width, uint8_t *codes, uint8_t *scales) {
  for (long long block = 0; block * 32 < width; ++block) {
    const float *x = row + block * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (amax == 0.0f) {
      scales[block] = 0;
      for (int i = 0; i < 32; ++i) codes[block * 32 + i] = 0;
      continue;
    }
    int exponent;
    std::frexp((double)amax, &exponent); // amax = m 2^exponent, 0.5 <= m < 1
    const int shared = std::min(127, std::max(-127, (exponent - 1) - 4));
    scales[block] = (uint8_t)(shared + 127);
    for (int i = 0; i < 32; ++i)
      codes[block * 32 + i] = (uint8_t)encode(Format::E3M2, std::ldexp((double)x[i], -shared));
  }
}

float nvfp4_encode_scale(float tensor_amax) {
  return tensor_amax > 0.0f ? 2688.0f / tensor_amax : 1.0f;
}
float nvfp4_decode_scale(float encode_scale) { return 1.0f / encode_scale; }

void quantize_nvfp4(const float *row, long long width, float encode_scale, uint8_t *codes,
                    uint8_t *scales) {
  for (long long block = 0; block * 16 < width; ++block) {
    const float *x = row + block * 16;
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[i]));
    float target = amax / 6.0f;
    target = target * encode_scale;
    const unsigned scale = encode(Format::UE4M3, target);
    scales[block] = (uint8_t)scale;
    const float f = (float)decode(Format::UE4M3, scale);
    for (int i = 0; i < 16; ++i) {
      if (scale == 0) {
        codes[block * 16 + i] = 0;
        continue;
      }
      float r = x[i] * encode_scale;
      r = r / f;
      codes[block * 16 + i] = (uint8_t)encode(Format::E2M1, r);
    }
  }
}

double dequantize_mxfp6(uint8_t code, uint8_t scale) {
  return decode(Format::E3M2, code) * decode(Format::UE8M0, scale);
}
double dequantize_nvfp4_unscaled(uint8_t code, uint8_t scale) {
  return decode(Format::E2M1, code) * decode(Format::UE4M3, scale);
}

unsigned unpack_e2m1(const uint8_t *row, long long i) {
  return (row[i / 2] >> (4 * (i % 2))) & 0xfu;
}
unsigned unpack_e3m2(const uint8_t *row, long long i) {
  const long long bit = 6 * i, byte = bit / 8;
  const int offset = (int)(bit % 8);
  unsigned window = row[byte];
  if (offset > 2) window |= (unsigned)row[byte + 1] << 8;
  return (window >> offset) & 0x3fu;
}

namespace {

struct SplitMix64 {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double uniform() { return (double)(next() >> 11) * 0x1.0p-53; }
};

std::vector<double> normals(int seed, int stream, size_t count) {
  SplitMix64 rng{(uint64_t)seed * 0x100000001b3ull + (uint64_t)stream};
  std::vector<double> out(count);
  const double two_pi = 6.283185307179586476925286766559;
  for (size_t i = 0; i < count; i += 2) {
    const double u1 = rng.uniform(), u2 = rng.uniform();
    const double radius = std::sqrt(-2.0 * std::log(1.0 - u1));
    out[i] = radius * std::cos(two_pi * u2);
    if (i + 1 < count) out[i + 1] = radius * std::sin(two_pi * u2);
  }
  return out;
}

} // namespace

Fixture make_fixture(int seed, long long width, long long vocab) {
  Fixture fixture;
  const size_t count = (size_t)(width * vocab);
  const std::vector<double> embedding = normals(seed, 1, count);
  const std::vector<double> classifier = normals(seed, 2, count);
  const float scale = (float)(1.0 / std::sqrt((double)width));
  fixture.embedding.resize(count);
  fixture.classifier.resize(count);
  for (size_t i = 0; i < count; ++i) {
    fixture.embedding[i] = (float)embedding[i];
    fixture.classifier[i] = (float)classifier[i] * scale;
  }
  return fixture;
}

void rmsnorm_double(const float *x, long long width, double *out) {
  double ss = 0.0;
  for (long long j = 0; j < width; ++j) ss += (double)x[j] * (double)x[j];
  const double scale = 1.0 / std::sqrt(ss / (double)width + (double)1e-5f);
  for (long long j = 0; j < width; ++j) out[j] = scale * (double)x[j];
}

void rmsnorm_float(const float *x, long long width, float *out) {
  float ss = 0.0f;
  for (long long j = 0; j < width; ++j) ss += x[j] * x[j];
  ss /= (float)width;
  ss += 1e-5f;
  ss = 1.0f / std::sqrt(ss);
  for (long long j = 0; j < width; ++j) out[j] = 1.0f * (ss * x[j]);
}

void softmax_double(const double *logits, long long count, double *out) {
  double maximum = -INFINITY, sum = 0.0;
  for (long long r = 0; r < count; ++r) maximum = std::max(maximum, logits[r]);
  for (long long r = 0; r < count; ++r) sum += out[r] = std::exp(logits[r] - maximum);
  for (long long r = 0; r < count; ++r) out[r] /= sum;
}

long long argmax_first(const double *values, long long count) {
  long long best = 0;
  for (long long r = 1; r < count; ++r)
    if (values[r] > values[best]) best = r;
  return best;
}

ErrorMetrics compare_logits(const std::vector<double> &reference,
                            const std::vector<double> &actual, long long rows,
                            long long vocab) {
  ErrorMetrics m;
  m.rows = rows;
  double error_squares = 0.0, reference_squares = 0.0, total_variation = 0.0;
  std::vector<double> p_reference(vocab), p_actual(vocab);
  for (long long b = 0; b < rows; ++b) {
    const double *ref = &reference[b * vocab], *act = &actual[b * vocab];
    for (long long r = 0; r < vocab; ++r) {
      const double error = act[r] - ref[r];
      m.max_abs_error = std::max(m.max_abs_error, std::fabs(error));
      error_squares += error * error;
      reference_squares += ref[r] * ref[r];
    }
    softmax_double(ref, vocab, p_reference.data());
    softmax_double(act, vocab, p_actual.data());
    double row_variation = 0.0;
    for (long long r = 0; r < vocab; ++r) {
      const double difference = std::fabs(p_actual[r] - p_reference[r]);
      m.max_abs_probability_error = std::max(m.max_abs_probability_error, difference);
      row_variation += difference;
    }
    total_variation += 0.5 * row_variation;
    m.top1_agree += argmax_first(ref, vocab) == argmax_first(act, vocab);
  }
  m.normalized_rms_error = reference_squares > 0 ? std::sqrt(error_squares / reference_squares) : 0;
  m.mean_total_variation = rows ? total_variation / rows : 0;
  return m;
}

} // namespace lowprec_ref
