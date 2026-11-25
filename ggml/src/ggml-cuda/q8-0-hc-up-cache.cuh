#pragma once

#include "q8-0-hc-up-mode.cuh"
#include "q8-0-hc-up-packed.cuh"
#include "sync-telemetry.cuh"

#include <mutex>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct hc_up_packed_cache {
    std::atomic<const void *> source{nullptr};
    std::mutex mutex;
    void * storage = nullptr;
    bool ready = false;

    ~hc_up_packed_cache() {
        if (storage) {
            ggml_cuda_q8_0_hc_up::pack_epoch.fetch_add(1, std::memory_order_relaxed);
            CUDA_CHECK(cudaFree(storage));
        }
    }

    void invalidate(const void * dst, size_t size) {
        const auto * original = source.load(std::memory_order_relaxed);
        if (!original || !size) {
            return;
        }
        const auto a = reinterpret_cast<uintptr_t>(original);
        const auto b = reinterpret_cast<uintptr_t>(dst);
        const bool overlaps = a <= b ? b - a < hc_up_packed_bytes : a - b < size;
        if (overlaps) {
            std::lock_guard<std::mutex> lock(mutex);
            if (ready) {
                ready = false;
                ggml_cuda_q8_0_hc_up::pack_epoch.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    hc_up_packed_weights prepare(cudaStream_t stream, const ggml_tensor * weight) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto * previous = source.load(std::memory_order_relaxed);
        if (previous && previous != weight->data) {
            return {};
        }
        if (!ready) {
            cudaStreamCaptureStatus capture;
            if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
                return {};
            }
            const int64_t start = ggml_time_us();
            if (!storage && cudaMalloc(&storage, hc_up_packed_bytes) != cudaSuccess) {
                (void) cudaGetLastError();
                storage = nullptr;
                return {};
            }
            source.store(weight->data, std::memory_order_relaxed);
            auto * scales = reinterpret_cast<half *>(static_cast<char *>(storage) + hc_up_packed_quant_bytes);
            cudaEvent_t begin = nullptr, end = nullptr;
            const bool timed = cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess;
            if (timed) {
                CUDA_CHECK(cudaEventRecord(begin, stream));
            } else {
                (void) cudaGetLastError();
            }
            constexpr int threads = 128;
            hc_up_pack_q8_0<<<(hc_up_packed_blocks + threads - 1) / threads, threads, 0, stream>>>(
                    static_cast<const block_q8_0 *>(weight->data), static_cast<int *>(storage), scales);
            CUDA_CHECK(cudaGetLastError());
            if (timed) {
                CUDA_CHECK(cudaEventRecord(end, stream));
            }
            CUDA_CHECK(ggml_cuda_stream_synchronize(stream));
            float gpu_ms = -1.0f;
            if (timed) {
                CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, begin, end));
            }
            if (begin) {
                CUDA_CHECK(cudaEventDestroy(begin));
            }
            if (end) {
                CUDA_CHECK(cudaEventDestroy(end));
            }
            ready = true;
            GGML_LOG_INFO("CUDA Ada Q8_0 HC up packed weight: name=%s bytes=%zu prep_wall_us=%lld pack_gpu_us=%.3f\n",
                    weight->name, hc_up_packed_bytes, static_cast<long long>(ggml_time_us() - start),
                    timed ? gpu_ms * 1000.0f : -1.0f);
        }
        return {static_cast<const int *>(storage),
                reinterpret_cast<const half *>(static_cast<const char *>(storage) + hc_up_packed_quant_bytes)};
    }
};
#endif
