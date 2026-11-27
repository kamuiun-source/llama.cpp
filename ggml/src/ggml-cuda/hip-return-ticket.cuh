#pragma once

#include "../ggml-backend-hip-return-ticket.h"

#if defined(USE_CUDA_GRAPH) && CUDART_VERSION >= 12090
static bool ggml_backend_cuda_hip_return_ticket(ggml_backend_t backend, ggml_cgraph * graph, uint64_t uid,
        const ggml_tensor * mirror, void * mirror_data, ggml_backend_hip_return_ticket_submit & submit, enum ggml_status * status) {
    if (!uid || !graph->n_nodes || !ggml_backend_cuda_return_stream_ordered(backend) || ggml_cuda_active_stream_telemetry) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ggml_cuda_info().devices[ctx->device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    namespace modes = ggml_gpu_output_modes;
    const auto key = ggml_cuda_graph_get_key(graph);
    const auto found = ctx->cuda_graphs.find(key);
    if (found == ctx->cuda_graphs.end()) {
        return false;
    }
    const auto & cached = *found->second;
    if (uid != cached.uid || !cached.is_enabled() || !cached.warmup_complete || cached.instances.size() != 1 ||
            cached.mapped_hc_output != mirror || cached.mapped_hc_data != mirror_data ||
            cached.gpu_output_modes != modes::capture_key(modes::common_bits | modes::ada_bits)) {
        return false;
    }
    using wait_t = CUresult (CUDAAPI *)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    static const auto wait = []() -> wait_t {
        void * proc = nullptr;
        cudaDriverEntryPointQueryResult query;
        const auto err = cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", &proc, 12090, cudaEnableDefault, &query);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        return query == cudaDriverEntryPointSuccess ? reinterpret_cast<wait_t>(proc) : nullptr;
    }();
    auto & ticket = submit.ticket;
    if (!wait || ticket.unavailable) {
        return false;
    }

    ggml_cuda_set_device(ctx->device);
    const auto stream = ctx->stream();
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    if (!ticket.host) {
        void * host = nullptr;
        void * device = nullptr;
        if (cudaHostAlloc(&host, 4096, cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
            if (host) {
                CUDA_CHECK(cudaFreeHost(host));
            }
            ticket.unavailable = true;
            return false;
        }
        ticket.host = new (host) uint32_t{0};
        ticket.device = reinterpret_cast<uintptr_t>(device);
        ticket.release = [](void * data) { CUDA_CHECK(cudaFreeHost(data)); };
    }
    static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr), "host ticket store");
    const auto instance = ctx->cuda_graph(key)->instances.front().handle;
    const uint32_t sequence = ++ticket.sequence;
    const auto result = wait(stream, ticket.device, sequence, CU_STREAM_WAIT_VALUE_EQ);
    if (result != CUDA_SUCCESS) {
        __atomic_store_n(ticket.host, sequence, __ATOMIC_RELEASE);
        ticket.unavailable = true;
        GGML_LOG_WARN("CUDA mapped return ticket wait unavailable: status=%d\n", int(result));
        return false;
    }

    // Only enqueue operations may run on CUDA while the stream waits for the host.
    submit.copy_start_us = ggml_time_us();
    const auto copy = cudaMemcpyAsync(submit.destination->data, submit.payload, ggml_nbytes(submit.destination),
            cudaMemcpyHostToDevice, stream);
    const auto record = copy == cudaSuccess ? cudaEventRecord(static_cast<cudaEvent_t>(submit.h2d_done->context), stream) : copy;
    submit.copy_end_us = ggml_time_us();
    submit.graph_start_us = submit.trace ? ggml_time_us() : 0;
    const auto launch = record == cudaSuccess ? cudaGraphLaunch(instance, stream) : record;
    submit.graph_end_us = submit.trace ? ggml_time_us() : 0;
    try {
        submit.ready(submit.ready_data);
    } catch (...) {
        __atomic_store_n(ticket.host, sequence, __ATOMIC_RELEASE);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        throw;
    }
    __atomic_store_n(ticket.host, sequence, __ATOMIC_RELEASE);
    if (launch != cudaSuccess) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GGML_LOG_ERROR("CUDA mapped return ticket enqueue failed: %s\n", cudaGetErrorString(launch));
        *status = GGML_STATUS_FAILED;
    } else {
        *status = GGML_STATUS_SUCCESS;
    }
    return true;
}
#endif
