#pragma once

#include "ggml-cpp.h"

struct llama_mtp_rtx_chain {
    using publish_fn = bool (*)(ggml_backend_t, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
            ggml_tensor *, ggml_tensor *, int, int, const ggml_tensor *, void *, ggml_tensor *);
    using wait_fn = void (*)(uint64_t *);

    ggml_backend_t backend = nullptr;
    publish_fn publish = nullptr;
    wait_fn read_waits = nullptr;
    ggml_context_ptr storage;
    ggml_backend_buffer_ptr device;
    ggml_backend_buffer_ptr host;
    ggml_tensor * ids = nullptr;
    ggml_tensor * embedding = nullptr;
    void * embedding_table = nullptr;
    size_t embedding_rows = 0;
    ggml_tensor * hidden = nullptr;
    ggml_backend_sched_ptr scheduler;
    llm_graph_result_ptr result;
    llm_graph_result * active = nullptr;
    int step = -1;
    int64_t second_launch_us = 0;
    size_t applied = 0, failed = 0;
    int64_t second_before_submit_us = 0, second_submit_us = 0, finish_us = 0;
    uint64_t second_stream_wait_ns = 0;
};
