#pragma once

#include <cstdint>

static __device__ __forceinline__ int q6_k_mmvq_signed_bytes(uint32_t codes) {
    // Setting each high bit prevents byte borrow during subtraction.
    return int(((codes | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
}
