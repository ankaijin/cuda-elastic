#include "fixtures.h"

#include <cstdio>
#include <cstring>

namespace vecadd_fixtures {
namespace {

const uint32_t kRound[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void compress(uint32_t state[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (uint32_t)block[4 * i] << 24 | (uint32_t)block[4 * i + 1] << 16 |
           (uint32_t)block[4 * i + 2] << 8 | (uint32_t)block[4 * i + 3];
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4],
           f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + ch + kRound[i] + w[i];
    const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  state[0] += a; state[1] += b; state[2] += c; state[3] += d;
  state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace

std::string sha256_hex(const uint8_t *data, size_t bytes) {
  uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t offset = 0;
  for (; offset + 64 <= bytes; offset += 64) compress(state, data + offset);
  uint8_t tail[128] = {0};
  const size_t rest = bytes - offset;
  if (rest) std::memcpy(tail, data + offset, rest);
  tail[rest] = 0x80;
  const size_t tail_bytes = rest < 56 ? 64 : 128;
  const uint64_t bits = (uint64_t)bytes * 8;
  for (int i = 0; i < 8; ++i)
    tail[tail_bytes - 1 - i] = (uint8_t)(bits >> (8 * i));
  for (size_t i = 0; i < tail_bytes; i += 64) compress(state, tail + i);
  char text[65];
  for (int i = 0; i < 8; ++i) std::snprintf(text + 8 * i, 9, "%08x", state[i]);
  return text;
}

std::string read_verified(const std::string &path, size_t bytes,
                          const std::string &sha256, std::vector<uint8_t> &out) {
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) return "cannot open " + path;
  out.assign(bytes + 1, 0);
  const size_t got = std::fread(out.data(), 1, bytes + 1, file);
  std::fclose(file);
  if (got != bytes)
    return path + " has " + std::to_string(got) + (got > bytes ? "+" : "") +
           " bytes, expected exactly " + std::to_string(bytes);
  out.resize(bytes);
  const std::string actual = sha256_hex(out.data(), out.size());
  if (actual != sha256)
    return path + " has sha256 " + actual + ", expected " + sha256;
  return "";
}

std::string parse_reference_sample(const std::vector<uint8_t> &text,
                                   uint8_t out[kLaneBytes]) {
  static const char kPrefix[] = "measured_sample=0 output=";
  const size_t prefix = sizeof kPrefix - 1;
  if (text.size() != prefix + 2 * kLaneBytes + 1 ||
      std::memcmp(text.data(), kPrefix, prefix) != 0 || text.back() != '\n')
    return "reference line is not 'measured_sample=0 output=<128 hex>\\n'";
  for (size_t i = 0; i < kLaneBytes; ++i) {
    unsigned value = 0;
    for (int nibble = 0; nibble < 2; ++nibble) {
      const char c = (char)text[prefix + 2 * i + nibble];
      unsigned digit;
      if (c >= '0' && c <= '9') digit = (unsigned)(c - '0');
      else if (c >= 'a' && c <= 'f') digit = (unsigned)(c - 'a' + 10);
      else return "reference line has a non-hex digit";
      value = value << 4 | digit;
    }
    out[i] = (uint8_t)value;
  }
  return "";
}

} // namespace vecadd_fixtures
