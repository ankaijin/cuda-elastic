#include "oracle.h"

#include <climits>
#include <string>

namespace vecadd_oracle {

namespace {
constexpr long long kMaxLength = 1ll << 27;
bool fits32(__int128 value) { return value >= INT_MIN && value <= INT_MAX; }
} // namespace

std::string validate(long long length, long long multiplier, long long bias) {
  if (length < 1) return "length must be at least 1, got " + std::to_string(length);
  if (length > kMaxLength)
    return "length must be at most " + std::to_string(kMaxLength) + " (2^27), got " +
           std::to_string(length);
  if (!fits32(multiplier))
    return "multiplier " + std::to_string(multiplier) + " does not fit in int32";
  if (!fits32(bias)) return "bias " + std::to_string(bias) + " does not fit in int32";
  // Every value is linear in the index, so the extremes are at i = 0 and
  // i = length - 1. Exact in 128-bit arithmetic.
  const __int128 last = length - 1;
  const __int128 rhs_last = last * multiplier + bias;
  const __int128 result_last = last * (multiplier + 1) + bias;
  if (!fits32(last)) return "an input index does not fit in int32";
  if (!fits32(bias) || !fits32(rhs_last))
    return "rhs values overflow int32 (rhs[length-1] = " +
           std::to_string((long long)rhs_last) + ")";
  if (!fits32(result_last))
    return "result values overflow int32 (result[length-1] = " +
           std::to_string((long long)result_last) + ")";
  return "";
}

Result compute(long long length, long long multiplier, long long bias) {
  Result out;
  out.result.resize((size_t)length);
  for (long long i = 0; i < length; ++i) {
    const int64_t lhs = i;
    const int64_t rhs = i * multiplier + bias;
    const int64_t sum = lhs + rhs;
    out.result[(size_t)i] = (int32_t)sum;
    out.checksum += sum;
  }
  return out;
}

} // namespace vecadd_oracle
