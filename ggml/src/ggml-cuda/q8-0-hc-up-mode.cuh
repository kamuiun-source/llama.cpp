#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <atomic>
#include <cstdlib>

namespace ggml_cuda_q8_0_hc_up {

inline bool configured() {
    static const bool enabled = [] {
        const char * path = std::getenv("LLAMA_CUDA_Q8_0_HC_UP_MODE_FILE");
        return path && *path;
    }();
    return enabled;
}

inline std::atomic<int> active_mode{0};
inline thread_local bool dispatched = false;

inline int get_mode() {
    return active_mode.load(std::memory_order_relaxed);
}

inline void set_mode(int requested) {
    if (requested == 0 || requested == 1) {
        active_mode.store(requested, std::memory_order_relaxed);
    }
}

} // namespace ggml_cuda_q8_0_hc_up
#endif
