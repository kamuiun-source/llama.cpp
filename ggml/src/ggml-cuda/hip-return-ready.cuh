#pragma once

#if defined(GGML_USE_HIP) && defined(__linux__)
#include "../ggml-backend-hip-return-ready.h"

#include <new>

struct ggml_hip_return_ready_event {
    hipEvent_t handle;
    // The scheduler can outlive the backend instance.
    int device;
};

static const ggml_backend_hip_return_ready_api ggml_backend_hip_return_ready_impl{
    [](ggml_backend_t backend) -> void * {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        auto * event = new (std::nothrow) ggml_hip_return_ready_event{nullptr, ctx->device};
        if (!event) {
            return nullptr;
        }
        ggml_cuda_set_device(event->device);
        if (hipEventCreateWithFlags(&event->handle, hipEventReleaseToSystem | hipEventDisableTiming) != hipSuccess) {
            delete event;
            return nullptr;
        }
        return event;
    },
    [](ggml_backend_t backend, void * payload) {
        auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        const auto * event = static_cast<ggml_hip_return_ready_event *>(payload);
        ggml_cuda_set_device(ctx->device);
        return ctx->curr_stream_no == 0 && hipEventRecord(event->handle, ctx->stream()) == hipSuccess;
    },
    [](void * payload) {
        const auto * event = static_cast<ggml_hip_return_ready_event *>(payload);
        ggml_cuda_set_device(event->device);
        return hipEventSynchronize(event->handle) == hipSuccess;
    },
    [](void * payload) {
        const auto * event = static_cast<ggml_hip_return_ready_event *>(payload);
        ggml_cuda_set_device(event->device);
        CUDA_CHECK(hipEventDestroy(event->handle));
        delete event;
    },
};
#endif
