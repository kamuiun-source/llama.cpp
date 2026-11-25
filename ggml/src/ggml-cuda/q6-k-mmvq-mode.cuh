#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ggml_cuda_q6_k_mmvq {

inline bool startup_signed() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_Q6_K_MMVQ_SIGNED");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool configured() {
    static const bool enabled = [] {
        const char * path = std::getenv("LLAMA_CUDA_Q6_K_MMVQ_MODE_FILE");
        return startup_signed() || (path && *path);
    }();
    return enabled;
}

inline std::atomic<bool> signed_mode{startup_signed()};
inline thread_local bool dispatched = false;

inline bool get_mode() {
    return signed_mode.load(std::memory_order_relaxed);
}

inline void set_mode(bool enabled) {
    signed_mode.store(enabled, std::memory_order_relaxed);
}

} // namespace ggml_cuda_q6_k_mmvq
#endif
