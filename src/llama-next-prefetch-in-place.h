#pragma once

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace llama_next_prefetch_in_place {

constexpr const char * mode_file_env = "LLAMA_EXPERT_HIP_MTP3_PREFETCH_IN_PLACE_MODE_FILE";
constexpr int slots = 2;
using pool = std::array<int32_t, slots>;

struct totals {
    size_t submitted = 0;
    size_t used = 0;
    size_t copied = 0;
    size_t copied_bytes = 0;
    size_t avoided_bytes = 0;
    size_t prefix_waits = 0;
    size_t reuse_busy = 0;
};

} // namespace llama_next_prefetch_in_place

LLAMA_API void llama_set_next_prefetch_in_place(bool enabled);
