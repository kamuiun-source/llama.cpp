#pragma once

#include "ggml-backend-mtp.h"

#include <cstdint>

struct ggml_backend_hip_return_ticket {
    uint32_t * host;
    uintptr_t device;
    uint32_t sequence;
    bool unavailable;
    void (*release)(void *);
};

struct ggml_backend_hip_return_ticket_submit {
    ggml_backend_hip_return_ticket & ticket;
    const ggml_tensor * destination;
    ggml_backend_event_t h2d_done;
    const void * payload;
    void (*ready)(void *);
    void * ready_data;
    bool trace;
    int64_t copy_start_us = 0;
    int64_t copy_end_us = 0;
    int64_t graph_start_us = 0;
    int64_t graph_end_us = 0;
};

using ggml_backend_cuda_hip_return_ticket_t = bool (*)(ggml_backend_t, ggml_cgraph *, uint64_t,
        const ggml_tensor *, void *, ggml_backend_hip_return_ticket_submit &, enum ggml_status *);
