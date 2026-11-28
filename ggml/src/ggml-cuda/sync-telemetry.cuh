#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <chrono>

extern thread_local uint64_t ggml_cuda_wait_calls[4];
extern thread_local uint64_t ggml_cuda_wait_ns[4];

static inline bool ggml_cuda_telemetry_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_DECODE_TELEMETRY");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}
#endif

template <typename F>
static inline cudaError_t ggml_cuda_time_wait(int index, F call) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_telemetry_enabled()) {
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        const cudaError_t result = call();
        ggml_cuda_wait_ns[index] += std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
        ++ggml_cuda_wait_calls[index];
        return result;
    }
#else
    GGML_UNUSED(index);
#endif
    return call();
}

static inline cudaError_t ggml_cuda_stream_synchronize(cudaStream_t stream) {
    return ggml_cuda_time_wait(0, [&] { return cudaStreamSynchronize(stream); });
}

static inline cudaError_t ggml_cuda_event_synchronize(cudaEvent_t event) {
    return ggml_cuda_time_wait(1, [&] { return cudaEventSynchronize(event); });
}

static inline cudaError_t ggml_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0) {
    return ggml_cuda_time_wait(2, [&] { return cudaStreamWaitEvent(stream, event, flags); });
}

static inline cudaError_t ggml_cuda_device_synchronize() {
    return ggml_cuda_time_wait(3, [] { return cudaDeviceSynchronize(); });
}
