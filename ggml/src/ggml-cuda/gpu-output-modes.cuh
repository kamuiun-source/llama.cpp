#pragma once

#include "../ggml-gpu-output-modes.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ggml_gpu_output_modes {

inline std::atomic<uint32_t> overrides{0};

inline void set_overrides(uint32_t modes) {
    overrides.store(modes, std::memory_order_relaxed);
}

inline uint32_t capture_key(uint32_t bits) {
    return overrides.load(std::memory_order_relaxed) & pack(bits, bits);
}

template<flag key>
inline bool enabled(int legacy = -1) {
    const auto modes = overrides.load(std::memory_order_relaxed);
    if (modes & (bit(key) << count)) {
        return (modes & bit(key)) != 0;
    }
    if (legacy >= 0) {
        return legacy != 0;
    }
    static const bool from_env = [] {
        const char * value = std::getenv(names[key].data());
        if constexpr (key == unary_mul_bcast) {
            return value && std::atoi(value) != 0;
        }
        return value && std::strcmp(value, "1") == 0;
    }();
    return from_env;
}

} // namespace ggml_gpu_output_modes
