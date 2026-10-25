// Host helpers for loom-private/vecadd-product: SHA-256, exact-size file reads,
// and the parser for the upstream one-sample reference line. Plain C++ with no
// CUDA dependency so the host test can link it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vecadd_fixtures {

constexpr size_t kLaneBytes = 64;                // one operand and one output slot
constexpr size_t kRecordInputBytes = 2 * kLaneBytes; // left then right operand

// Lowercase hex SHA-256 of a byte range.
std::string sha256_hex(const uint8_t *data, size_t bytes);

// Reads a whole file that must be exactly `bytes` long and hash to `sha256`.
// Returns an empty string on success, otherwise a one-line problem description
// (missing, wrong size, or wrong hash).
std::string read_verified(const std::string &path, size_t bytes,
                          const std::string &sha256, std::vector<uint8_t> &out);

// Parses "measured_sample=0 output=<128 hex digits>\n" into 64 bytes. Returns
// an empty string on success, otherwise a problem description.
std::string parse_reference_sample(const std::vector<uint8_t> &text,
                                   uint8_t out[kLaneBytes]);

} // namespace vecadd_fixtures
