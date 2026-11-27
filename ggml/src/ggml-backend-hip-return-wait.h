#pragma once

#include "ggml-backend-impl.h"
#include "ggml-backend-mtp.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstring>

using ggml_backend_cuda_return_stream_ordered_t = bool (*)(ggml_backend_t);

static std::atomic<bool> hip_return_dst_wait_skip{false};

void ggml_backend_sched_set_hip_return_dst_wait_skip(bool enabled) {
    hip_return_dst_wait_skip.store(enabled, std::memory_order_relaxed);
}

static ggml_backend_cuda_return_stream_ordered_t ggml_backend_hip_return_ordered_stream(
        ggml_backend_t * backends, int count) {
    ggml_backend_reg_t cuda = nullptr;
    bool hip = false;
    for (int i = 0; i < count; ++i) {
        const auto dev = ggml_backend_get_device(backends[i]);
        if (!dev) {
            return nullptr;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        const auto reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) {
            return nullptr;
        }
        const char * name = ggml_backend_reg_name(reg);
        if (std::strcmp(name, "CUDA") == 0 && !cuda) {
            cuda = reg;
        } else if (std::strcmp(name, "ROCm") == 0 && !hip) {
            hip = true;
        } else {
            return nullptr;
        }
    }
    return cuda && hip ? reinterpret_cast<ggml_backend_cuda_return_stream_ordered_t>(
            ggml_backend_reg_get_proc_address(cuda, "ggml_backend_cuda_return_stream_ordered")) : nullptr;
}

static bool ggml_backend_hip_return_can_queue_h2d(ggml_backend_t dst, ggml_backend_event_t & event) {
    if (!event && dst->iface.set_tensor_async) {
        event = ggml_backend_event_new(dst->device);
    }
    return event && event->device == dst->device && dst->iface.set_tensor_async;
}

struct ggml_backend_hip_return_wait_stats {
    size_t applied, fallback;
    size_t source_waits, destination_first_waits, destination_second_waits;
    size_t slab_event_waits, slab_stream_drains;
    int64_t source_us, destination_first_us, destination_second_us, slab_event_trace_us;

    void log(bool enabled) const {
        if (!source_waits) {
            return;
        }
        GGML_LOG_INFO("packed HIP MTP3 mapped return waits: mode=%s applied=%zu fallback=%zu"
                " mi50_source=%zu mi50_source_ms=%.3f rtx_dst_first=%zu rtx_dst_first_ms=%.3f"
                " rtx_dst_second=%zu rtx_dst_second_ms=%.3f slab_event=%zu slab_event_trace_ms=%.3f slab_stream_drained=%zu\n",
                enabled ? "on" : "off", applied, fallback, source_waits, source_us / 1000.0,
                destination_first_waits, destination_first_us / 1000.0,
                destination_second_waits, destination_second_us / 1000.0,
                slab_event_waits, slab_event_trace_us / 1000.0, slab_stream_drains);
    }
};
