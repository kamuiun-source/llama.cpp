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
        size_t payload_bytes;
    };

    struct tuple_bank {
        ggml_tensor * host = nullptr;
        ggml_tensor * device = nullptr;
        size_t stride = 0;
        size_t payload_bytes = 0;
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
    const hip_overlap_mode hip_prefill;
    const bool hip_overlap_promotion;
    const bool overlap;
    const bool dynamic_resident;
    const bool hip_mtp_verify_four;
    const bool hip_mtp_verify_three;
    const bool hip_mtp_direct_ram;
    const bool hip_tuple_dma;
    const bool hip_single_miss_cpu;
    const int32_t n_slots;
    const int32_t prefill_tokens;
    int32_t prefill_waves(int64_t n_tokens, int64_t n_used) const;
    mutable std::atomic<bool> context_in_use{false};
    std::array<layer_map, llama_expert_profile::n_layers> maps;
    std::vector<weight> weights;
    std::array<tuple_bank, llama_expert_profile::n_layers> tuples;
    ggml_context_ptr host_ctx;
    ggml_context_ptr bank_ctx;
    ggml_context_ptr cold_ctx;
    ggml_context_ptr tuple_host_ctx;
    ggml_context_ptr tuple_bank_ctx;
    ggml_backend_buffer_ptr host_buf;
    ggml_backend_buffer_ptr bank_buf;
    ggml_backend_buffer_ptr cold_buf;
    llama_mlock host_lock;

private:
#ifdef __linux__
    struct tuple_buffer_locks {
        llama_memory_ranges ranges;
        size_t locked = 0;

        ~tuple_buffer_locks();
    };

    // Release OS locks before freeing the native host buffers.
    std::unique_ptr<tuple_buffer_locks> tuple_host_locks;

    std::unique_ptr<tuple_buffer_locks> lock_tuple_host() const;
#endif
    bool load_tuples(llama_model_loader & loader, const llama_model_params & params);
};
