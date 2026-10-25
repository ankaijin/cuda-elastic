// Host tests for the vecadd-memory oracle and its dimension validation.
// None of the sizes here is an upstream smoke or validation configuration.
#include "../oracle.h"

#include <climits>
#include <cstdio>

static int failures = 0;

#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// Closed form of the checksum: sum of i * (m + 1) + b over i < n.
static long long closed_form(long long n, long long m, long long b) {
  return (m + 1) * (n * (n - 1) / 2) + b * n;
}

int main() {
  using namespace vecadd_oracle;

  // Large workload: the task's stated checksum, from the oracle, not a constant
  // inside it.
  {
    const Result r = compute(1048576, 2, 0);
    EXPECT(r.result.size() == 1048576);
    EXPECT(r.checksum == 1649265868800ll);
    EXPECT(r.result[0] == 0);
    EXPECT(r.result[1] == 3);
    EXPECT(r.result[1048575] == 3 * 1048575);
    std::printf("PASS large oracle checksum %lld\n", (long long)r.checksum);
  }

  // Oracle agrees with the closed form and with per-element formula.
  const long long cases[][3] = {{1, 2, 0},       {2, 1, 5},      {255, -3, 11},
                                {257, 7, -5},    {1000003, 2, 0}, {4097, 0, -100},
                                {5000, -1, 0},   {100, 21474836, 3}};
  for (const auto &c : cases) {
    EXPECT(validate(c[0], c[1], c[2]).empty());
    const Result r = compute(c[0], c[1], c[2]);
    EXPECT(r.checksum == closed_form(c[0], c[1], c[2]));
    bool exact = true;
    for (long long i = 0; i < c[0]; ++i)
      exact = exact && r.result[(size_t)i] == (int32_t)(i * (c[1] + 1) + c[2]);
    EXPECT(exact);
  }
  std::printf("PASS oracle matches closed form and per-element formula\n");

  // Validation boundaries.
  EXPECT(!validate(0, 2, 0).empty());
  EXPECT(!validate(-5, 2, 0).empty());
  EXPECT(validate(1, 2, 0).empty());
  EXPECT(validate(1ll << 27, 0, 0).empty());
  EXPECT(!validate((1ll << 27) + 1, 0, 0).empty());
  // length 2 -> last index 1: result[1] = m + 1 + b.
  EXPECT(validate(2, 1, INT_MAX - 2).empty());   // result = INT_MAX
  EXPECT(!validate(2, 1, INT_MAX - 1).empty());  // result = INT_MAX + 1
  EXPECT(validate(2, 0, INT_MAX - 1).empty());   // rhs = INT_MAX - 1, result = INT_MAX
  EXPECT(!validate(2, 0, INT_MAX).empty());      // result = INT_MAX + 1
  EXPECT(validate(2, 0, INT_MIN).empty());       // rhs = INT_MIN
  EXPECT(!validate(2, -2, INT_MIN).empty());     // rhs[1] = INT_MIN - 2
  EXPECT(!validate(3, 1073741823, 0).empty());   // result[2] = 2^31
  EXPECT(validate(3, 1073741822, 0).empty());    // result[2] = 2^31 - 2
  EXPECT(!validate(2, 1ll << 31, 0).empty());    // multiplier not int32
  EXPECT(!validate(2, 0, 1ll << 31).empty());    // bias not int32
  EXPECT(!validate(2, 0, -(1ll << 31) - 1).empty());
  EXPECT(!validate(1ll << 27, 100, 0).empty());  // overflows int32
  std::printf("PASS validation boundaries\n");

  if (failures) {
    std::printf("host tests FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("host tests passed\n");
  return 0;
}
