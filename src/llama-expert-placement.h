#pragma once

#include "ggml-cpp.h"
#include "llama-expert-profile.h"
#include "llama-mmap.h"
#include "llama.h"

#include <atomic>

struct llama_model_loader;

struct llama_expert_placement {
    enum class hip_overlap_mode { off, serial, async };

    struct weight {
        int layer;
        uint16_t shard;
        size_t offset;
        ggml_tensor * host;
        ggml_tensor * bank;
        ggml_tensor * cold;
    };

    struct layer_map {
        std::array<int32_t, llama_expert_profile::n_experts> resident;
        std::array<int32_t, llama_expert_profile::n_experts> host;
        int32_t hip_spare_hot_slot = llama_expert_profile::n_resident + 1;

        std::vector<int32_t> scratch_slots(int32_t n_slots) const;
    };

    llama_expert_placement(const char * path, ggml_backend_dev_t target, int32_t n_expert_used);
    ggml_tensor * add_weight(int layer, const ggml_tensor * source, uint16_t shard, size_t offset);
    bool load(llama_model_loader & loader, const llama_model_params & params);

    const ggml_backend_dev_t target;
    const int32_t max_ubatch;
    const hip_overlap_mode hip_overlap;
    const bool hip_mtp_verify_three;
    const int32_t n_slots;
    const hip_overlap_mode hip_prefill;
    const bool hip_overlap_promotion;
    const bool hip_protected_slots;
    const bool hip_slot_exchange;
    const bool hip_slot_exchange_usage;
    const bool overlap;
    const bool dynamic_resident;
    const int32_t prefill_tokens;
    int32_t prefill_waves(int64_t n_tokens, int64_t n_used) const;
    mutable std::atomic<bool> context_in_use{false};
    std::array<layer_map, llama_expert_profile::n_layers> maps;
    std::vector<weight> weights;
    ggml_context_ptr host_ctx;
    ggml_context_ptr bank_ctx;
    ggml_context_ptr cold_ctx;
    ggml_backend_buffer_ptr host_buf;
    ggml_backend_buffer_ptr bank_buf;
    ggml_backend_buffer_ptr cold_buf;
    llama_mlock host_lock;
};
