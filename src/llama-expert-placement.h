#pragma once

#include "ggml-cpp.h"
#include "llama-expert-profile.h"
#include "llama-mmap.h"
#include "llama.h"

#include <atomic>

struct llama_model_loader;

struct llama_expert_placement {
    struct weight {
        int layer;
        uint16_t shard;
        size_t offset;
        ggml_tensor * host;
        ggml_tensor * bank;
    };

    struct layer_map {
        std::array<int32_t, llama_expert_profile::n_experts> resident;
        std::array<int32_t, llama_expert_profile::n_experts> host;
    };

    llama_expert_placement(const char * path, ggml_backend_dev_t target, int32_t n_slots);
    ggml_tensor * add_weight(int layer, const ggml_tensor * source, uint16_t shard, size_t offset);
    bool load(llama_model_loader & loader, const llama_model_params & params);

    const ggml_backend_dev_t target;
    const int32_t n_slots;
    mutable std::atomic<bool> context_in_use{false};
    std::array<layer_map, llama_expert_profile::n_layers> maps;
    std::vector<weight> weights;
    ggml_context_ptr host_ctx;
    ggml_context_ptr bank_ctx;
    ggml_backend_buffer_ptr host_buf;
    ggml_backend_buffer_ptr bank_buf;
    llama_mlock host_lock;
};
