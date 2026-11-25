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

enum mode { off, signed_bytes, reuse };
inline std::atomic<int> active_mode{startup_signed() ? signed_bytes : off};
inline thread_local bool dispatched = false;

inline int get_mode() {
    return active_mode.load(std::memory_order_relaxed);
}

inline void set_mode(int requested) {
    if (requested >= off && requested <= reuse) {
        active_mode.store(requested, std::memory_order_relaxed);
    }
}

} // namespace ggml_cuda_q6_k_mmvq
#endif
