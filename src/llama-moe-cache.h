#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llama_ubatch;
struct llama_expert_round_stats;
class llama_rtx_expert;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool mtp_verify_three_enabled = false);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;
    bool requires_slots(int32_t il) const;
    bool compact_prefill(int64_t n_tokens) const;
    ggml_tensor * build_compact_prefill(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * ids, int32_t il,
            ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, ggml_tensor * gate_up,
            ggml_tensor * up_scale, ggml_tensor * gate_scale, ggml_tensor * down_scale) const;
    void begin_prompt_trace(const llama_ubatch & ubatch, uint32_t batch_tokens, uint32_t batch_offset);
    void end_prompt_trace(bool success);
    void begin_generation_trace(const llama_ubatch & ubatch, uint32_t verify_width);
    void end_generation_trace(bool success);
    bool next_router_trace_active() const;
    void trace_next_router(int32_t il, const ggml_tensor * ids);
    bool next_router_prefetch_enabled() const;
    void begin_next_router_prefetch(const std::vector<ggml_tensor *> & predictions);
    void end_next_router_prefetch();
    void record_decode_width(uint32_t tokens);
    void finish_pending_promotion();
    llama_expert_round_stats round_stats() const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part, bool mtp_three = false) const;
    int32_t get_prefill_waves(int32_t il, int64_t n_tokens, int64_t n_used) const;
    ggml_tensor * get_prefill_map(int32_t il, int32_t wave) const;
    ggml_tensor * get_mtp_verify_map(int32_t il, int32_t wave) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w, bool cold = false) const;
    ggml_tensor * get_cold_slot_map(int32_t il, bool mtp_three = false) const;
    bool uses_hip_mtp_compact() const;
    ggml_tensor * get_hip_mtp_gate_up_full_map(int32_t il) const;
    uint64_t * get_hip_mtp_fused_hit_count() const;
    uint64_t * get_hip_mtp_shared_output_count() const;
    uint64_t * get_hip_mtp_q8_1_count() const;
    llama_rtx_expert * get_rtx_expert() const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool early_copy_enabled() const;
    bool router_host_ids_enabled() const;
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, bool profile,
            bool early = false, ggml_backend_sched_t sched = nullptr, bool fixed_mtp_pair = false, bool mtp_four = false);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
