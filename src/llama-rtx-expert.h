#pragma once

#include "ggml-backend.h"

#include <memory>

class llama_rtx_expert {
public:
    llama_rtx_expert(ggml_backend_t cuda, int32_t slots);
    ~llama_rtx_expert();

    ggml_backend_t backend() const;
    bool enabled(int il) const;
    ggml_tensor * output(int il, ggml_tensor * map, ggml_tensor * cur,
            ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, ggml_tensor * gate_up, int64_t k, int64_t t);
    bool available(ggml_backend_t vk, ggml_backend_sched_t sched, int il, const ggml_cgraph * graph, const ggml_tensor * map);
    bool submit(int il, const ggml_cgraph * graph, const ggml_tensor * map, const int32_t * slots,
            const void * const * host_slices, size_t n_experts);
    void clear(int il, const ggml_cgraph * graph, const ggml_tensor * map);

private:
    struct impl;
    std::unique_ptr<impl> p;
};
