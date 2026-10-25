// Independent CPU oracle for loom-private/vecadd-product. It derives every
// record's operands from the 128-byte fixture with plain integer arithmetic
// and adds the two operands lane by lane. It shares no code with the CUDA path
// (vecadd_product.cu); it uses neither the packed-word arithmetic nor the
// record layout of the device buffers.
#pragma once

#include <cstdint>
#include <vector>

namespace vecadd_oracle {

// Output slot of one record: byte k = (a[k] + a[64 + k]) mod 256, where
// a[j] = (fixture[j] + (record mod 256)) mod 256. `fixture` has 128 bytes and
// `slot` receives 64.
void record_output(const uint8_t *fixture, uint64_t record, uint8_t *slot);

// Every record's output slot, back to back (records * 64 bytes), computed by
// a few host threads over disjoint record ranges.
std::vector<uint8_t> all_outputs(const uint8_t *fixture, uint64_t records);

} // namespace vecadd_oracle
