#pragma once

#include "ggml-backend.h"

struct ggml_backend_hip_return_ready_api {
    void * (*create)(ggml_backend_t backend);
    bool (*record)(ggml_backend_t backend, void * event);
    bool (*wait)(void * event);
    void (*destroy)(void * event);
};

struct ggml_backend_hip_return_ready {
    ggml_backend_t owner;
    const ggml_backend_hip_return_ready_api * api;
    void * event;
    size_t recorded, event_waits, stream_waits, fallback;
};
