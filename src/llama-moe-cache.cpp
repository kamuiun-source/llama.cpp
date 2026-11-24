#include "llama-moe-cache.h"

#include "llama-expert-placement.h"
#include "llama-ext.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-rtx-expert.h"

#include "ggml-cpp.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

static constexpr size_t packed_miss_bins = llama_expert_profile::n_experts - llama_expert_profile::n_resident + 1;
static thread_local std::array<uint64_t, packed_miss_bins> packed_miss_counts{};
static thread_local uint64_t packed_miss_bytes = 0;
static thread_local uint64_t packed_miss_bytes_3plus = 0;
static thread_local llama_expert_transfer_telemetry packed_transfer_totals;
static thread_local llama_expert_staging_telemetry packed_staging_totals;
static thread_local int64_t prompt_trace_request = -1;

template<typename T>
static void prompt_trace_array(std::ostream & out, const T & values) {
    out << '[';
    bool comma = false;
    for (const auto value : values) {
        if (comma) {
            out << ',';
        }
        out << value;
        comma = true;
    }
    out << ']';
}

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        impl * compact_owner = nullptr;
        int32_t compact_layer = -1;
        std::array<ggml_tensor *, 4> compact_weights{};
        std::array<ggml_tensor *, 3> compact_scales{};
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * cold_slot_map = nullptr;
        ggml_tensor * hip_mtp_hot_map = nullptr;
        ggml_tensor * hip_mtp_full_map = nullptr;
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        std::vector<int32_t> packed_scratch; // original expert ID per scratch slot, -1 if empty
        std::vector<ggml_tensor *> prefill_maps;
        std::array<ggml_tensor *, 2> mtp_verify_maps{};
        std::array<int32_t, llama_expert_profile::n_experts> prefill_wave;
        std::vector<int32_t> prefill_ids;
        std::vector<int32_t> packed_scratch_slots; // physical slots when the permanent map can change
        std::vector<uint64_t> packed_last_used;
        std::vector<int32_t> packed_ram_slots; // pinned tuple index per logical host slot, -1 for normal RAM
        uint64_t packed_lookups = 0;
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        std::array<const ggml_tensor *, 4> rtx_planes{}; // up, gate, down, gate_up
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
        ggml_tensor * cold = nullptr;
        size_t packed_offset = 0;
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    llama_expert_placement * placement;
    std::unique_ptr<const llama_expert_placement, void (*)(const llama_expert_placement *)> placement_owner{
        nullptr, [](const llama_expert_placement * p) { p->context_in_use.store(false); },
    };

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<int32_t> packed_cold_scratch;
    std::array<int32_t, llama_expert_profile::n_experts> packed_cold_slots{};
    int32_t pending_layer = -1;
    int32_t pending_mtp_promote = -1;
    int32_t pending_mtp_evict = -1;
    uint32_t pending_mtp_miss_mask = 0;
    int32_t pending_mtp_full_layer = -1;
    std::array<int32_t, llama_expert_profile::n_experts> pending_mtp_full_slots{};
    size_t pending_permanent_hits = 0;
    size_t pending_scratch_hits = 0;
    size_t pending_request_misses = 0;
    size_t pending_upload_bytes = 0;
    size_t packed_scratch_hits = 0;
    size_t packed_request_misses = 0;
    const bool mtp_verify_three_enabled;
    static constexpr int32_t mtp_pair_experts = 10;
    std::vector<std::array<uint64_t, mtp_pair_experts + 1>> mtp_pair_histograms;
    size_t mtp_verify_layers = 0;
    size_t mtp_verify_overflow_layers = 0;
    size_t mtp_overlap_fit_layers = 0;
    size_t mtp_overlap_attempts = 0;
    size_t mtp_overlap_submitted = 0;
    size_t mtp_verify_second_passes = 0;
    size_t mtp_verify_empty_no_copy = 0;
    int32_t mtp_verify_max_cold = 0;
    size_t prefill_layer_batches = 0;
    size_t prefill_waves_needed = 0;
    size_t prefill_waves_scheduled = 0;
    size_t prefill_unique_cold = 0;
    size_t prefill_max_tokens = 0;
    size_t prefill_waves_executed = 0;
    size_t prefill_waves_skipped = 0;
    bool prefill_active_waves = [] {
        const char * value = std::getenv("LLAMA_EXPERT_PREFILL_ACTIVE_WAVES");
        return value && std::strcmp(value, "1") == 0;
    }();
    std::unique_ptr<llama_rtx_expert> rtx_expert;
    size_t rtx_max_slots = 0;
    size_t rtx_eligible_steps = 0;
    size_t rtx_steps = 0;
    size_t rtx_unique_experts = 0;
    size_t rtx_ranks = 0;
    size_t rtx_uploaded_bytes = 0;
    bool packed_miss_histogram = false;
    using defer_slot_map_t = bool (*)(ggml_backend_t, ggml_tensor *, const int32_t *);
    defer_slot_map_t defer_slot_map = nullptr;
    using telemetry_cause_t = const char * (*)(const char *);
    telemetry_cause_t set_telemetry_cause = nullptr;
    using gate_up_diag_ids_t = void (*)(ggml_backend_t, const ggml_tensor *, const int32_t *, const int32_t *);
    gate_up_diag_ids_t gate_up_diag_ids = nullptr;

    struct telemetry_scope {
        telemetry_cause_t set;
        const char * previous;

        telemetry_scope(telemetry_cause_t set, const char * name) : set(set), previous(set ? set(name) : nullptr) {}
        ~telemetry_scope() {
            if (previous) {
                set(previous);
            }
        }
    };
    bool defer_slot_map_requested = false;
    using submit_expert_upload_t = bool (*)(ggml_backend_t);
    submit_expert_upload_t submit_expert_upload = nullptr;
    using stage_hip_expert_overlap_t = bool (*)(ggml_backend_t, const ggml_cgraph *, const ggml_tensor *,
            const ggml_tensor *, const void * const *, const int *, const int *, unsigned, size_t);
    stage_hip_expert_overlap_t stage_hip_expert_overlap = nullptr;
    size_t hip_expert_overlap_staged[4] = {};
    using overlap_expert_upload_t = bool (*)(ggml_backend_t, const ggml_cgraph *, ggml_tensor *, const int32_t *,
            const int32_t *, const uint32_t *, ggml_tensor * const *, const void *, const size_t *, const size_t *, const size_t *, size_t);
    overlap_expert_upload_t overlap_expert_upload = nullptr;
    using overlap_expert_promotion_t = bool (*)(ggml_backend_t, const ggml_cgraph *, ggml_tensor *, const int32_t *,
            const int32_t *, const uint32_t *, ggml_tensor * const *, const void *, const size_t *, const size_t *, const size_t *,
            size_t, int32_t, ggml_tensor * const *, const size_t *, size_t, void (*)(void *), void *);
    overlap_expert_promotion_t overlap_expert_promotion = nullptr;
    size_t packed_overlap_promotions = 0;
    size_t packed_overlap_writeback_bytes = 0;

    struct packed_promotion {
        struct plane {
            ggml_tensor * bank;
            uint8_t * host_data;
            size_t stride;
        };
        std::array<plane, 3> planes{};
        size_t n_planes = 0;
        size_t bytes = 0;
        int32_t il;
        int32_t scratch_index;
        int32_t evict;
        int32_t promote;
        int32_t host_slot;
        int32_t old_gpu_slot;
        int32_t new_gpu_slot;
        int32_t writeback_slot = -1;
    };
    packed_promotion pending_promotion{};
    const uint8_t * pending_writeback = nullptr;
    packed_promotion hip_pending_promotion{};
    size_t hip_writeback_offset = 0;
    bool overlap_upload_requested = false;
    size_t packed_overlap_uploads = 0;
    size_t packed_overlap_upload_bytes = 0;
    size_t packed_mtp_hot_routes = 0;
    size_t packed_mtp_miss_routes = 0;
    size_t packed_mtp_empty_groups = 0;
    size_t packed_mtp_gate_up_all_hit = 0;
    size_t packed_mtp_gate_up_all_miss = 0;
    size_t packed_mtp_gate_up_mixed = 0;
    uint64_t packed_mtp_gate_up_fused_hits = 0;
    uint64_t packed_mtp_gate_up_shared_dispatches = 0;
    uint64_t packed_mtp_gate_up_q8_1_dispatches = 0;
    bool defer_upload_requested = false;
    size_t packed_deferred_uploads = 0;
    size_t packed_synchronous_uploads = 0;
    size_t packed_deferred_upload_bytes = 0;
    size_t packed_deferred_maps = 0;
    size_t packed_synchronous_maps = 0;
    size_t router_host_hits = 0;
    size_t router_host_fallbacks = 0;
    size_t router_readback_elided_bytes = 0;
    int64_t packed_wait_us = 0;
    size_t packed_wait_count = 0;
    size_t hip_direct_upload_steps = 0;
    size_t hip_direct_upload_planes = 0;
    size_t hip_direct_upload_bytes = 0;
    uint64_t hip_direct_upload_wait_us = 0;
    std::vector<uint8_t> packed_writeback;
    size_t packed_promotions = 0;
    size_t packed_writeback_bytes = 0;
    int64_t packed_writeback_us = 0;
    bool profile_packed_ram_copy = false;
    bool profile_packed_callback = false;
    size_t packed_ram_memcpy_calls = 0;
    size_t packed_ram_memcpy_bytes = 0;
    int64_t packed_ram_memcpy_us = 0;
    uint8_t * packed_ram_base = nullptr;
    size_t packed_ram_stride = 0;
    size_t packed_ram_offset = 0;
    int32_t packed_ram_capacity = 0;
    int32_t packed_ram_spare = 0;
    int32_t packed_ram_next = 1;
    size_t packed_ram_rotations = 0;
    size_t packed_ram_rotated_bytes = 0;
    size_t packed_ram_restored_bytes = 0;
    int64_t profile_router_us = 0;
    size_t profile_router_reads = 0;
    int64_t profile_upload_us = 0;
    size_t profile_upload_steps = 0;
    size_t profile_upload_bytes = 0;
    ggml_backend_buffer_ptr packed_staging;
    ggml_backend_ptr hip_upload_backend;
    static constexpr int next_prefetch_slots = 8;
    ggml_backend_ptr next_prefetch_backend;
    ggml_context_ptr next_prefetch_ctx;
    ggml_context_ptr next_prefetch_views;
    ggml_backend_buffer_ptr next_prefetch_buffer;
    ggml_tensor * next_prefetch_bank = nullptr;
    const std::vector<ggml_tensor *> * next_prefetch_predictions = nullptr;
    const std::vector<ggml_tensor *> * next_prefetch_scores = nullptr;
    int next_prefetch_score_mode = 0;
    size_t next_prefetch_priority_layers = 0;
    size_t next_prefetch_priority_fallbacks = 0;
    size_t next_prefetch_top20_only = 0;
    size_t next_prefetch_top20_only_hits = 0;
    std::array<bool, next_prefetch_slots> next_prefetch_beyond_top10{};
    size_t next_prefetch_score_eligible = 0;
    size_t next_prefetch_score_rejected = 0;
    size_t next_prefetch_score_empty = 0;
    size_t next_prefetch_score_readback_fallback = 0;
    size_t next_prefetch_score_row_fallback = 0;
    std::array<int32_t, next_prefetch_slots> next_prefetch_ids{};
    std::array<ggml_backend_event_ptr, next_prefetch_slots> next_prefetch_events;
    std::array<int32_t, next_prefetch_slots> next_prefetch_source_slots{};
    size_t next_prefetch_sources = 0;
    size_t next_prefetch_completed = 0;
    int32_t next_prefetch_source_layer = -1;
    bool next_prefetch_tail_pending = false;
    bool next_prefetch_hit_wait = false;
    bool next_prefetch_after_submit = false;
    int32_t next_prefetch_submit_layer = -1;
    size_t next_prefetch_submit_scheduled = 0;
    size_t next_prefetch_submit_executed = 0;
    size_t next_prefetch_submit_fallback = 0;
    size_t next_prefetch_submit_canceled = 0;
    int64_t next_prefetch_submit_cpu_us = 0;
    size_t next_prefetch_dependencies = 0;
    size_t next_prefetch_dependency_slots = 0;
    size_t next_prefetch_source_waits = 0;
    size_t next_prefetch_tail_waits = 0;
    int64_t next_prefetch_consumer_wait_us = 0;
    int64_t next_prefetch_source_wait_us = 0;
    int64_t next_prefetch_tail_wait_us = 0;
    int32_t next_prefetch_layer = -1;
    size_t next_prefetch_count = 0;
    size_t next_prefetch_launches = 0;
    size_t next_prefetch_hits = 0;
    size_t next_prefetch_unused = 0;
    size_t next_prefetch_h2d_bytes = 0;
    size_t next_prefetch_d2d_bytes = 0;
    int64_t next_prefetch_wait_us = 0;
    bool next_profile_enabled = false;
    int64_t next_profile_start = 0;
    uint64_t next_profile_graphs = 0;
    uint64_t next_profile_failed = 0;
    uint64_t next_profile_nodes = 0;
    uint64_t next_profile_splits = 0;
    uint64_t next_profile_hc_pre = 0;
    uint64_t next_profile_hc_comb = 0;
    uint64_t next_profile_hc_post = 0;
    int64_t next_profile_graph_us = 0;
    llama_expert_staging_telemetry next_profile_callback;
    uint64_t next_profile_demand_count = 0;
    uint64_t next_profile_demand_bytes = 0;
    uint64_t next_profile_demand_waits = 0;
    uint64_t next_profile_readbacks = 0;
    int64_t next_profile_read_submit_us = 0;
    int64_t next_profile_read_wait_us = 0;
    int64_t next_profile_h2d_submit_us = 0;
    int64_t next_profile_d2d_submit_us = 0;
    uint64_t next_profile_prefetch_waits = 0;
    ggml_backend_buffer_ptr hip_mtp_slot_staging;
    bool hip_mtp_slot_async = false;
    bool hip_mtp4_slot_async = false;
    ggml_backend_buffer_ptr hip_mtp_promotion_staging;
    bool hip_mtp_promotion_async = false;
    bool hip_mtp4_promotion_async = false;
    packed_promotion hip_mtp_pending_move{};
    bool hip_mtp_pending_four = false;
    size_t hip_mtp4_async_maps = 0;
    size_t hip_mtp4_promotion_submits = 0;
    size_t hip_mtp4_promotion_commits = 0;
    size_t hip_mtp4_promotion_bytes = 0;
    int64_t hip_mtp4_promotion_wait_us = 0;
    size_t tuple_h2d_count = 0;
    size_t tuple_h2d_payload_bytes = 0;
    size_t tuple_h2d_bytes = 0;
    size_t tuple_prefill_h2d_count = 0;
    size_t tuple_d2h_count = 0;
    size_t tuple_d2h_payload_bytes = 0;
    size_t tuple_d2h_bytes = 0;
    size_t hip_mtp_async_maps = 0;
    size_t hip_mtp_async_map_bytes = 0;
    size_t hip_mtp_promotion_submits = 0;
    size_t hip_mtp_promotion_commits = 0;
    size_t hip_mtp_promotion_bytes = 0;
    int64_t hip_mtp_promotion_wait_us = 0;
    std::vector<moe_cache_lru::fill> fills; // packed fills use logical scratch indices
    std::vector<moe_cache_lru::fill> cpu;

    bool packed_early_copy = false;
    size_t packed_early_upload_steps = 0;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;
    FILE * prompt_trace = nullptr;
    uint64_t prompt_trace_segment = 0;
    uint64_t prompt_trace_batch = 0;
    uint64_t prompt_trace_rows = 0;
    int64_t prompt_trace_request_id = -1;
    std::vector<llama_pos> prompt_trace_positions;
    std::unique_ptr<FILE, decltype(&fclose)> generation_trace{nullptr, fclose};
    bool next_router_trace = false;
    std::vector<llama_pos> generation_trace_positions;
    uint64_t generation_trace_step = 0;
    uint64_t generation_trace_memcpy_us = 0;
    uint64_t generation_trace_upload_wait_us = 0;
    int64_t generation_trace_request = -1;
    uint32_t generation_trace_verify_width = 0;
    bool prompt_active = false;
    bool prompt_stats = false;
    uint64_t prompt_ubatches = 0;
    uint64_t prompt_tokens = 0;
    uint64_t prompt_max_tokens = 0;
    uint64_t decode_max_tokens = 0;
    uint64_t prompt_layer_steps = 0;
    uint64_t prompt_uploads = 0;
    uint64_t prompt_upload_bytes = 0;
    uint64_t compact_chunks = 0;
    ggml_backend_buffer_ptr compact_workspace;
    bool prefill_logical_ticks = false;
    uint64_t prefill_tick_steps = 0;
    uint64_t prefill_tick_tokens = 0;
    uint64_t prefill_tick_advanced = 0;
    uint64_t prefill_tick_min_delta = 0;
    uint64_t prefill_tick_max_delta = 0;
    uint64_t prefill_tick_age_sum = 0;
    uint64_t prefill_native_steps = 0;
    bool prefill_overlap_ten = false;
    using prefill_create_t = void * (*)(ggml_backend_t);
    using prefill_start_t = bool (*)(void *, ggml_tensor * const *, const void *, const size_t *, const size_t *, const size_t *, size_t, const int32_t *, size_t);
    using prefill_wait_t = bool (*)(void *);
    using prefill_pending_t = bool (*)(void *);
    using prefill_destroy_t = void (*)(void *);
    prefill_create_t prefill_create = nullptr;
    prefill_start_t prefill_start = nullptr;
    prefill_wait_t prefill_wait = nullptr;
    prefill_pending_t prefill_pending = nullptr;
    prefill_destroy_t prefill_destroy = nullptr;
    void * compact_upload_job = nullptr;
    bool compact_upload_attempted = false;
    bool compact_transfer_pending = false;
    bool compact_window_checked = false;
    std::array<std::vector<float>, 3> compact_overlap_scales;
    uint64_t prefill_overlap_submits = 0;
    uint64_t prefill_overlap_bytes = 0;
    uint64_t prefill_overlap_fallbacks = 0;
    uint64_t prefill_overlap_windows = 0;
    uint64_t prefill_overlap_pending_windows = 0;
    uint64_t prefill_overlap_fallback_layers = 0;

    static void write_expert_trace(FILE * file, const std::string & row) {
        if (fwrite(row.data(), 1, row.size(), file) != row.size() || fflush(file) != 0) {
            throw std::runtime_error("could not write packed expert trace");
        }
    }

    void write_prompt_trace(const std::string & row) {
        write_expert_trace(prompt_trace, row);
    }

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool mtp_verify_three_enabled) :
            n_expert_used(model.hparams.n_expert_used_max()), placement(model.expert_placement.get()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)),
            mtp_verify_three_enabled(mtp_verify_three_enabled) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        if (placement) {
            const auto device = std::find_if(devices.begin(), devices.end(), [&](const auto & d) {
                return ggml_backend_get_device(d.backend) == placement->target;
            });
            if (device == devices.end() || no_alloc || cpu_share != 0.0f) {
                throw std::runtime_error("packed expert execution requires the placement device and CPU share 0");
            }
            if (placement->context_in_use.exchange(true)) {
                throw std::runtime_error("packed expert scratch slots are already in use by another context");
            }
            placement_owner.reset(placement);
            const char * pair_overlap = std::getenv("LLAMA_MTP_PAIR_EXPERT_OVERLAP");
            if (!mtp_verify_three_enabled && n_expert_used == mtp_pair_experts && pair_overlap && std::strcmp(pair_overlap, "1") == 0) {
                mtp_pair_histograms.resize(layers.size());
            }
            LLAMA_LOG_INFO("packed expert router: reuse host IDs when captured by the scheduler\n");
            groups.emplace_back().id = static_cast<int32_t>(device - devices.begin());
            const int32_t n_prefill_waves = placement->prefill_waves(placement->prefill_tokens, n_expert_used);
            if (mtp_verify_three_enabled && (placement->overlap || placement->n_slots != 20 ||
                    placement->max_ubatch != 2 || n_expert_used != 10)) {
                throw std::runtime_error("3-position MTP requires packed Expert capacity 300+20 and top-10 routing");
            }
            const int32_t map_count = 1 + placement->overlap + 2 * placement->hip_mtp_overlap +
                    placement->hip_mtp_gate_up_overlap + n_prefill_waves + 2 * mtp_verify_three_enabled;
            const ggml_init_params params{placement->maps.size() * ggml_tensor_overhead() * map_count, nullptr, true};
            ctx_host.reset(ggml_init(params));
            if (!ctx_host) {
                throw std::runtime_error("could not allocate packed expert slot map context");
            }
            for (size_t il = 0; il < placement->maps.size(); ++il) {
                auto & l = layers.at(il);
                l.ig = 0;
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                ggml_format_name(l.slot_map, "moe_cache.packed_slots-%zu", il);
                layer_of.emplace(l.slot_map, std::make_pair(static_cast<int32_t>(il), false));
                for (int32_t wave = 0; wave < n_prefill_waves; ++wave) {
                    auto * map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                    ggml_format_name(map, "moe_cache.packed_slots-%zu-prefill-%d", il, wave);
                    l.prefill_maps.push_back(map);
                    layer_of.emplace(map, std::make_pair(static_cast<int32_t>(il), false));
                }
                if (mtp_verify_three_enabled) {
                    for (size_t wave = 0; wave < l.mtp_verify_maps.size(); ++wave) {
                        auto * map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                        ggml_format_name(map, "moe_cache.packed_slots-%zu-mtp3-%zu", il, wave);
                        l.mtp_verify_maps[wave] = map;
                        layer_of.emplace(map, std::make_pair(static_cast<int32_t>(il), false));
                    }
                }
                if (placement->hip_mtp_overlap) {
                    l.hip_mtp_hot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                    ggml_format_name(l.hip_mtp_hot_map, "moe_cache.packed_mtp_hot-%zu", il);
                    layer_of.emplace(l.hip_mtp_hot_map, std::make_pair(static_cast<int32_t>(il), false));
                }
                if (placement->hip_mtp_gate_up_overlap) {
                    l.hip_mtp_full_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                    ggml_format_name(l.hip_mtp_full_map, "moe_cache.packed_mtp_full-%zu", il);
                    layer_of.emplace(l.hip_mtp_full_map, std::make_pair(static_cast<int32_t>(il), false));
                }
                if (placement->overlap || placement->hip_mtp_overlap) {
                    l.cold_slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                    ggml_format_name(l.cold_slot_map, "moe_cache.packed_cold-%zu", il);
                    layer_of.emplace(l.cold_slot_map, std::make_pair(static_cast<int32_t>(il), true));
                }
                l.packed_scratch.assign(placement->n_slots, -1);
                if (placement->dynamic_resident) {
                    l.packed_scratch_slots = placement->maps[il].scratch_slots(placement->n_slots);
                }
                if (placement->dynamic_resident || placement->hip_overlap_promotion) {
                    l.packed_last_used.assign(llama_expert_profile::n_experts, 0);
                }
            }
            std::array<size_t, llama_expert_profile::n_layers> bytes_per_layer{};
            for (const auto & w : placement->weights) {
                layers.at(w.layer).experts.push_back(w.host);
                auto & bytes = bytes_per_layer.at(w.layer);
                bindings.emplace(w.host, binding{w.layer, 0, w.bank, w.cold, bytes});
                if (w.payload_bytes > SIZE_MAX - bytes) {
                    throw std::runtime_error("packed expert tuple is too large");
                }
                bytes += w.payload_bytes;
            }
            const size_t tuple_bytes = *std::max_element(bytes_per_layer.begin(), bytes_per_layer.end());
            const char * ram_copy = std::getenv("LLAMA_EXPERT_PROFILE_RAM_COPY");
            profile_packed_ram_copy = ram_copy && std::strcmp(ram_copy, "1") == 0;
            const char * callback = std::getenv("LLAMA_EXPERT_PROFILE_CALLBACK");
            profile_packed_callback = callback && std::strcmp(callback, "1") == 0;
            if (placement->dynamic_resident) {
                for (size_t il = 0; il < placement->maps.size(); ++il) {
                    const size_t n_planes = layers[il].experts.size();
                    if (n_planes != 2 && n_planes != 3) {
                        throw std::runtime_error("dynamic expert residency requires gate/up and down weights");
                    }
                }
                if (!placement->hip_tuple_dma) {
                    packed_writeback.resize(tuple_bytes);
                }
                LLAMA_LOG_INFO("packed expert dynamic residency: stale_lookups=%" PRIu64 " writeback_buffer_bytes=%zu\n",
                        packed_min_resident_age, packed_writeback.size());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), ggml_backend_cpu_buffer_type()));
            if (!buf_host) {
                throw std::runtime_error("could not allocate packed expert slot maps");
            }
            ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            const auto target_reg = ggml_backend_dev_backend_reg(placement->target);
            const char * target_name = target_reg ? ggml_backend_reg_name(target_reg) : "";
            const auto reg = std::strcmp(target_name, "Vulkan") == 0 ? target_reg : nullptr;
#ifdef __linux__
            const char * single_kernel_overlap = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_KERNEL_OVERLAP");
            if (placement->hip_tuple_dma && std::strcmp(target_name, "ROCm") == 0 &&
                    single_kernel_overlap && std::strcmp(single_kernel_overlap, "1") == 0) {
                stage_hip_expert_overlap = reinterpret_cast<stage_hip_expert_overlap_t>(
                        ggml_backend_reg_get_proc_address(target_reg, "ggml_backend_hip_stage_expert_overlap_v2"));
            }
#endif
            const char * histogram = std::getenv("LLAMA_EXPERT_MISS_HISTOGRAM");
            const char * telemetry = std::getenv("LLAMA_DECODE_TELEMETRY");
            packed_miss_histogram = telemetry && std::strcmp(telemetry, "1") == 0 &&
                    histogram && std::strcmp(histogram, "1") == 0;
            if (reg && telemetry && std::strcmp(telemetry, "1") == 0) {
                set_telemetry_cause = (telemetry_cause_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_set_telemetry_cause");
            }
            const char * gate_up_diag = std::getenv("GGML_VK_MTP_GATE_UP_TIMING");
            if (gate_up_diag && std::strcmp(gate_up_diag, "1") == 0) {
                gate_up_diag_ids = reg ? reinterpret_cast<gate_up_diag_ids_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_set_gate_up_diagnostic_ids")) : nullptr;
                LLAMA_LOG_INFO("mtp gate-up host IDs: callback=%s\n", gate_up_diag_ids ? "available" : "unavailable");
            }
            defer_slot_map_requested = !placement->overlap;
            if (defer_slot_map_requested) {
                defer_slot_map = reg ? (defer_slot_map_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_defer_slot_map") : nullptr;
            }
            const char * defer_upload = std::getenv("LLAMA_EXPERT_VK_DEFER_UPLOAD");
            defer_upload_requested = !placement->overlap && defer_upload && std::strcmp(defer_upload, "1") == 0;
            if (defer_upload_requested && defer_slot_map) {
                submit_expert_upload = (submit_expert_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_submit_expert_upload");
            }
            overlap_upload_requested = reg && !placement->overlap && placement->dynamic_resident && placement->max_ubatch == 2;
            if (placement->n_slots <= 0 || tuple_bytes > SIZE_MAX / placement->n_slots) {
                throw std::runtime_error("packed expert staging is too large");
            }
            const size_t staging_bytes = tuple_bytes * placement->n_slots;
            using alloc_host_upload_buffer_t = ggml_backend_buffer_t (*)(ggml_backend_dev_t, size_t);
            if (overlap_upload_requested && defer_slot_map && reg) {
                const auto alloc_overlap_staging = reinterpret_cast<alloc_host_upload_buffer_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_alloc_overlap_expert_staging"));
                const auto submit_overlap = reinterpret_cast<overlap_expert_upload_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_overlap_expert_upload"));
                if (alloc_overlap_staging && submit_overlap) {
                    overlap_expert_promotion = reinterpret_cast<overlap_expert_promotion_t>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_overlap_expert_promotion"));
                    if (overlap_expert_promotion) {
                        init_packed_ram_rotation(alloc_overlap_staging, staging_bytes, tuple_bytes);
                    }
                    if (!packed_staging) {
                        packed_staging.reset(alloc_overlap_staging(placement->target, staging_bytes));
                    }
                    if (packed_staging) {
                        overlap_expert_upload = submit_overlap;
                    } else {
                        overlap_expert_promotion = nullptr;
                    }
                }
            }
            const char * proc = placement->overlap ? "ggml_backend_vk_alloc_transfer_host_upload_buffer" :
                    "ggml_backend_vk_alloc_host_upload_buffer";
            const auto alloc_host_upload_buffer = reg ? (alloc_host_upload_buffer_t) ggml_backend_reg_get_proc_address(reg, proc) : nullptr;
            if (!packed_staging && alloc_host_upload_buffer) {
                packed_staging.reset(alloc_host_upload_buffer(placement->target, staging_bytes));
            }
            const char * hip_pinned = std::getenv("LLAMA_EXPERT_HIP_PINNED_UPLOAD");
            const bool hip_overlap = placement->hip_overlap != llama_expert_placement::hip_overlap_mode::off;
            const bool hip_prefill = placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off;
            const bool hip_upload_requested = hip_overlap || hip_prefill || placement->hip_mtp_overlap ||
                    (std::strcmp(target_name, "ROCm") == 0 && hip_pinned && std::strcmp(hip_pinned, "1") == 0);
            if (hip_upload_requested && !placement->hip_tuple_dma) {
                const auto buft = ggml_backend_dev_host_buffer_type(placement->target);
                const size_t writeback_bytes = placement->hip_overlap_promotion ? tuple_bytes : 0;
                if (writeback_bytes > SIZE_MAX - staging_bytes) {
                    throw std::runtime_error("HIP expert staging and writeback are too large");
                }
                ggml_backend_buffer_ptr buffer(buft ? ggml_backend_buft_alloc_buffer(buft, staging_bytes + writeback_bytes) : nullptr);
                // The HIP allocator can return a pageable CPU buffer on failure.
                if (buffer && ggml_backend_buffer_get_type(buffer.get()) == buft) {
                    packed_staging = std::move(buffer);
                    LLAMA_LOG_INFO("packed expert HIP pinned upload: %zu bytes of staging on %s\n",
                            ggml_backend_buffer_get_size(packed_staging.get()), ggml_backend_dev_name(placement->target));
                } else if (hip_overlap || placement->hip_mtp_overlap) {
                    throw std::runtime_error("HIP expert overlap requires pinned host staging");
                } else if (hip_prefill) {
                    throw std::runtime_error("HIP expert prefill requires pinned host staging");
                } else {
                    LLAMA_LOG_WARN("packed expert HIP pinned upload unavailable, using serial uploads\n");
                }
            }
            if (placement->hip_mtp_overlap) {
                LLAMA_LOG_INFO("packed expert HIP MTP3: single-bank upload overlap enabled\n");
            }
            if (packed_staging || placement->hip_mtp_direct_ram || placement->hip_tuple_dma) {
                fills.reserve(placement->n_slots);
            }
            if (packed_staging || placement->hip_tuple_dma) {
                if (hip_overlap || hip_prefill || placement->hip_mtp_overlap) {
                    hip_upload_backend.reset(ggml_backend_dev_init(placement->target, nullptr));
                    if (!hip_upload_backend) {
                        throw std::runtime_error("could not initialize the HIP expert upload backend");
                    }
                }
                if (hip_prefill) {
                    LLAMA_LOG_INFO("packed expert HIP compact prefill: cold uploads %s\n",
                            placement->hip_prefill == llama_expert_placement::hip_overlap_mode::async ? "overlapped" : "serialized");
                }
                if (placement->overlap || placement->hip_mtp_overlap) {
                    packed_cold_scratch.reserve(placement->n_slots);
                }
                if (placement->overlap) {
                    if (hip_overlap) {
                        if (placement->hip_overlap_promotion) {
                            hip_writeback_offset = staging_bytes;
                            ctx_views.reset(ggml_init({ 6 * ggml_tensor_overhead(), nullptr, true }));
                            if (!ctx_views) {
                                throw std::runtime_error("could not create HIP expert promotion views");
                            }
                        }
                        LLAMA_LOG_INFO("packed expert HIP split: cold uploads %s\n",
                                placement->hip_overlap == llama_expert_placement::hip_overlap_mode::async ? "overlapped" : "serialized");
                    } else {
                        const char * early_copy = std::getenv("LLAMA_EXPERT_OVERLAP_EARLY_COPY");
                        packed_early_copy = !early_copy || std::strcmp(early_copy, "0") != 0;
                        LLAMA_LOG_INFO("packed expert overlap: early cold uploads %s\n", packed_early_copy ? "enabled" : "disabled");
                    }
                }
                if (packed_staging) {
                    LLAMA_LOG_INFO("packed expert %s upload: %zu bytes of native host staging on %s (%s)\n",
                            placement->overlap ? "overlap" : "batch", ggml_backend_buffer_get_size(packed_staging.get()),
                            ggml_backend_dev_name(placement->target), ggml_backend_dev_description(placement->target));
                }
            } else if (placement->overlap) {
                throw std::runtime_error("expert overlap requires native Vulkan host staging on the target and an async transfer queue without serialized submissions");
            } else if (!hip_upload_requested) {
                LLAMA_LOG_WARN("packed expert batch upload unavailable on %s, using serial uploads\n",
                        ggml_backend_dev_name(placement->target));
            }
            if (placement->hip_mtp_verify_three && placement->dynamic_resident && !placement->overlap &&
                    !placement->hip_mtp_overlap && !placement->hip_mtp_direct_ram) {
                const auto buft = ggml_backend_dev_host_buffer_type(placement->target);
                const auto allocate_pinned = [&](const char * env, size_t bytes) -> ggml_backend_buffer_ptr {
                    const char * value = std::getenv(env);
                    if (!value || std::strcmp(value, "1") != 0) {
                        return {};
                    }
                    ggml_backend_buffer_ptr buffer(buft ? ggml_backend_buft_alloc_buffer(buft, bytes) : nullptr);
                    if (!buffer || ggml_backend_buffer_get_type(buffer.get()) != buft) {
                        LLAMA_LOG_WARN("%s: HIP pinned staging unavailable, using synchronous path\n", env);
                        return {};
                    }
                    return buffer;
                };
                const auto enabled = [](const char * env) {
                    const char * value = std::getenv(env);
                    return value && std::strcmp(value, "1") == 0;
                };
                const char * slot3 = "LLAMA_EXPERT_HIP_MTP3_ASYNC_SLOT_MAP";
                const char * slot4 = "LLAMA_EXPERT_HIP_MTP4_ASYNC_SLOT_MAP";
                hip_mtp_slot_async = enabled(slot3);
                hip_mtp4_slot_async = placement->hip_mtp_verify_four && enabled(slot4);
                if (hip_mtp_slot_async || hip_mtp4_slot_async) {
                    hip_mtp_slot_staging = allocate_pinned(hip_mtp_slot_async ? slot3 : slot4,
                            placement->maps.size() * sizeof(placement->maps[0].resident));
                }
                if (placement->hip_tuple_dma) {
                    hip_mtp_promotion_async = enabled("LLAMA_EXPERT_HIP_MTP3_ASYNC_PROMOTION");
                    hip_mtp4_promotion_async = placement->hip_mtp_verify_four && enabled("LLAMA_EXPERT_HIP_MTP4_ASYNC_PROMOTION");
                } else {
                    hip_mtp_promotion_staging = allocate_pinned("LLAMA_EXPERT_HIP_MTP3_ASYNC_PROMOTION", tuple_bytes);
                    hip_mtp_promotion_async = bool(hip_mtp_promotion_staging);
                }
                if ((hip_mtp_promotion_async || hip_mtp4_promotion_async) && !hip_upload_backend) {
                    hip_upload_backend.reset(ggml_backend_dev_init(placement->target, nullptr));
                    if (!hip_upload_backend) {
                        hip_mtp_promotion_staging.reset();
                        hip_mtp_promotion_async = false;
                        hip_mtp4_promotion_async = false;
                        LLAMA_LOG_WARN("HIP MTP promotion stream unavailable, using synchronous writeback\n");
                    }
                }
            }
            const char * next_profile = std::getenv("LLAMA_EXPERT_HIP_NEXT_ROUTER_PROFILE");
            next_profile_enabled = next_profile && std::strcmp(next_profile, "1") == 0 &&
                    model.arch == LLM_ARCH_QWEN4EXP && placement->hip_tuple_dma && placement->hip_mtp_verify_three &&
                    !placement->hip_mtp_overlap && !placement->overlap;
            const char * next_prefetch = std::getenv("LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH");
            if (next_prefetch && std::strcmp(next_prefetch, "1") == 0) {
                const char * single_kernel = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_KERNEL_OVERLAP");
                if (std::strcmp(target_name, "ROCm") != 0 || model.arch != LLM_ARCH_QWEN4EXP ||
                        model.hparams.n_expert_used() != 10 || model.hparams.n_expert_groups > 1 ||
                        !placement->hip_tuple_dma || !placement->hip_mtp_verify_three || placement->hip_mtp_verify_four ||
                        !placement->dynamic_resident || placement->overlap || placement->hip_mtp_overlap ||
                        placement->hip_prefill == llama_expert_placement::hip_overlap_mode::async ||
                        hip_mtp_promotion_async || (single_kernel && std::strcmp(single_kernel, "1") == 0)) {
                    throw std::runtime_error("HIP next Router prefetch requires Qwen4EXP packed MTP3 tuple DMA and serial promotion, with other Expert upload overlap off");
                }
                init_next_prefetch();
            }
            const char * rtx_slots_env = std::getenv("LLAMA_EXPERT_RTX_COLD_SLOTS");
            if (rtx_slots_env && std::strcmp(rtx_slots_env, "0") != 0) {
                if (placement->hip_mtp_overlap || placement->hip_mtp_direct_ram || placement->hip_tuple_dma) {
                    throw std::runtime_error("HIP MTP3 expert overlap, direct RAM and tuple DMA require RTX cold assist off");
                }
                char * end = nullptr;
                const long slots = std::strtol(rtx_slots_env, &end, 10);
                const auto cuda = std::find_if(devices.begin(), devices.end(), [](const auto & d) {
                    return std::strcmp(ggml_backend_dev_name(ggml_backend_get_device(d.backend)), "CUDA0") == 0;
                });
                if (end == rtx_slots_env || *end || slots < 1 || slots >= placement->n_slots ||
                        !reg || !packed_staging || placement->overlap || !placement->dynamic_resident ||
                        placement->max_ubatch != 2 || cuda == devices.end()) {
                    LLAMA_LOG_WARN("packed expert RTX offload unavailable: need CUDA0, dynamic packed batch-2 placement, native staging and a valid slot cap\n");
                } else {
                    for (size_t il = 0; il < placement->maps.size(); ++il) {
                        const auto & model_layer = model.layers[il];
                        layers[il].rtx_planes = {model_layer.ffn_up_exps, model_layer.ffn_gate_exps,
                                model_layer.ffn_down_exps, model_layer.ffn_gate_up_exps};
                    }
                    rtx_max_slots = slots;
                    rtx_expert = std::make_unique<llama_rtx_expert>(cuda->backend, static_cast<int32_t>(slots));
                    LLAMA_LOG_INFO("packed expert RTX offload: max_slots=%zu threshold=2\n", rtx_max_slots);
                }
            }
            if (const char * path = std::getenv("LLAMA_EXPERT_GENERATION_TRACE"); path && *path) {
                if (std::strcmp(target_name, "ROCm") != 0 || !placement->dynamic_resident ||
                        placement->overlap || placement->hip_mtp_overlap || rtx_expert) {
                    throw std::runtime_error("generation trace requires ROCm single-bank dynamic Expert placement with upload overlap and RTX assist off");
                }
                generation_trace.reset(fopen(path, "wb"));
                if (!generation_trace) {
                    throw std::runtime_error("could not open packed expert generation trace");
                }
                const char * next_router = std::getenv("LLAMA_EXPERT_NEXT_ROUTER_TRACE");
                next_router_trace = next_router && std::strcmp(next_router, "1") == 0;
                std::ostringstream header;
                header << "{\"type\":\"header\",\"version\":1,\"experts\":" << llama_expert_profile::n_experts
                       << ",\"resident_count\":" << llama_expert_profile::n_resident
                       << ",\"scratch_capacity\":" << placement->n_slots << "}\n";
                write_expert_trace(generation_trace.get(), header.str());
            }
            if (const char * path = std::getenv("LLAMA_EXPERT_PREFILL_TRACE"); path && *path) {
                if (placement->max_ubatch != 2 || placement->prefill_tokens || placement->overlap || rtx_expert) {
                    throw std::runtime_error("prompt trace requires packed batch-2 placement with live prefill and RTX cold assist off");
                }
                prompt_trace = fopen(path, "wb");
                if (!prompt_trace) {
                    throw std::runtime_error("could not open packed prompt trace");
                }
                std::ostringstream header;
                header << "{\"type\":\"header\",\"version\":1,\"baseline_width\":2,\"experts\":"
                    << llama_expert_profile::n_experts << ",\"resident_count\":" << llama_expert_profile::n_resident
                    << ",\"scratch_capacity\":" << placement->n_slots << ",\"layers\":[";
                for (size_t il = 0; il < placement->maps.size(); ++il) {
                    header << (il ? "," : "") << il;
                }
                header << "]}\n";
                try {
                    write_prompt_trace(header.str());
                } catch (...) {
                    fclose(prompt_trace);
                    prompt_trace = nullptr;
                    throw;
                }
                LLAMA_LOG_INFO("packed prompt trace: enabled, prompt-only batch-2 Router IDs\n");
            }
            const char * prompt_statistics = std::getenv("LLAMA_EXPERT_PREFILL_STATS");
            prompt_stats = placement->prefill_tokens == 512 ||
                (prompt_statistics && std::strcmp(prompt_statistics, "1") == 0);
            const char * logical_ticks = std::getenv("LLAMA_EXPERT_PREFILL_LOGICAL_TICKS");
            prefill_logical_ticks = placement->prefill_tokens == 512 &&
                logical_ticks && std::strcmp(logical_ticks, "1") == 0;
            const char * overlap_ten = std::getenv("LLAMA_EXPERT_PREFILL_OVERLAP_TEN");
            prefill_overlap_ten = hip_prefill || (placement->prefill_tokens == 512 &&
                (!overlap_ten || std::strcmp(overlap_ten, "0") != 0));
            if (prefill_overlap_ten && reg) {
                prefill_create = reinterpret_cast<prefill_create_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefill_upload_create"));
                prefill_start = reinterpret_cast<prefill_start_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefill_upload_start"));
                prefill_wait = reinterpret_cast<prefill_wait_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefill_upload_wait"));
                prefill_pending = reinterpret_cast<prefill_pending_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefill_upload_pending"));
                prefill_destroy = reinterpret_cast<prefill_destroy_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_prefill_upload_destroy"));
            }
            return;
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        g.lru.slot_map[il] = slots;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        // The context synchronizes its backends before destroying the cache.
        finish_next_prefetch();
        wait_hip_cold_upload();
        finish_hip_mtp_promotion();
        GGML_ASSERT(!pending_writeback);
        if (compact_upload_job) {
            prefill_destroy(compact_upload_job);
            compact_upload_job = nullptr;
        }
        restore_packed_ram_slots();
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
        if (prompt_trace) {
            fprintf(prompt_trace, "{\"type\":\"complete\",\"segments\":%" PRIu64 ",\"rows\":%" PRIu64 "}\n",
                    prompt_trace_segment, prompt_trace_rows);
            fclose(prompt_trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part, bool mtp_three) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (placement) {
            const int32_t max_tokens = placement->hip_mtp_verify_four ? 4 :
                    placement->hip_mtp_verify_three ? 3 : placement->max_ubatch;
            return n_tokens > 0 && n_tokens <= max_tokens && n_expert_used > 0 &&
                    n_expert_used <= placement->n_slots / n_tokens ? (mtp_three && l.hip_mtp_hot_map ? l.hip_mtp_hot_map : l.slot_map) : nullptr;
        }

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w, bool cold) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? (cold && !(placement && placement->hip_mtp_overlap) ? it->second.cold : it->second.cached) : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        if (placement) {
            return 0;
        }
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    static constexpr uint64_t packed_min_resident_age = 32;

    static int32_t packed_slot(const layer & l, int32_t scratch_index) {
        return l.packed_scratch_slots.empty() ? llama_expert_profile::n_resident + scratch_index :
                l.packed_scratch_slots[scratch_index];
    }

    static int32_t packed_promotion_victim(const layer & l, const llama_expert_placement::layer_map & map) {
        int32_t evict = -1;
        uint64_t oldest = l.packed_lookups;
        for (int32_t expert = 0; expert < llama_expert_profile::n_experts; ++expert) {
            if (map.resident[expert] >= 0 && l.packed_last_used[expert] < oldest) {
                evict = expert;
                oldest = l.packed_last_used[expert];
            }
        }
        return l.packed_lookups - oldest >= packed_min_resident_age ? evict : -1;
    }

    void init_packed_ram_rotation(ggml_backend_buffer_t (*alloc)(ggml_backend_dev_t, size_t), size_t upload_bytes, size_t tuple_bytes) {
        // Leave ordinary RAM slots pageable and reserve one pinned tuple for the next writeback.
        constexpr size_t pool_bytes = 64 * 1024 * 1024;
        constexpr size_t alignment = 32;
        if (!tuple_bytes || tuple_bytes > pool_bytes / 2 || upload_bytes > SIZE_MAX - pool_bytes - 2 * alignment) {
            return;
        }
        const size_t stride = (tuple_bytes + alignment - 1) / alignment * alignment;
        const size_t offset = (upload_bytes + alignment - 1) / alignment * alignment;
        const int32_t capacity = static_cast<int32_t>(pool_bytes / stride);
        const size_t bytes = offset + capacity * stride;
        ggml_backend_buffer_ptr buffer(alloc(placement->target, bytes));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) < bytes) {
            return;
        }

        for (size_t il = 0; il < placement->maps.size(); ++il) {
            auto & l = layers[il];
            l.packed_ram_slots.assign(llama_expert_profile::n_experts - llama_expert_profile::n_resident, -1);
        }
        packed_staging = std::move(buffer);
        packed_ram_offset = offset;
        packed_ram_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get())) + offset;
        packed_ram_stride = stride;
        packed_ram_capacity = capacity;
    }

    uint8_t * packed_host_data(const layer & l, const ggml_tensor * host, int32_t host_slot, size_t plane_offset) const {
        if (!l.packed_ram_slots.empty()) {
            const int32_t slot = l.packed_ram_slots[host_slot];
            if (slot >= 0) {
                return packed_ram_base + slot * packed_ram_stride + plane_offset;
            }
        }
        return static_cast<uint8_t *>(host->data) + host_slot * host->nb[2];
    }

    void restore_packed_ram_slots() noexcept {
        if (!packed_ram_base) {
            return;
        }
        // Placement can outlive this cache and its pinned buffer.
        for (const auto & l : layers) {
            for (size_t host_slot = 0; host_slot < l.packed_ram_slots.size(); ++host_slot) {
                const int32_t slot = l.packed_ram_slots[host_slot];
                if (slot < 0) {
                    continue;
                }
                const auto * data = packed_ram_base + slot * packed_ram_stride;
                for (const auto * host : l.experts) {
                    const size_t stride = host->nb[2];
                    std::memcpy(static_cast<uint8_t *>(host->data) + host_slot * stride, data, stride);
                    data += stride;
                    packed_ram_restored_bytes += stride;
                }
            }
        }
    }

    packed_promotion prepare_packed_promotion(const layer & l, int32_t il, int32_t scratch_index, int32_t evict) const {
        const auto & map = placement->maps[il];
        packed_promotion move{};
        move.il = il;
        move.scratch_index = scratch_index;
        move.evict = evict;
        move.promote = l.packed_scratch[scratch_index];
        move.host_slot = map.host[move.promote];
        move.old_gpu_slot = map.resident[evict];
        move.new_gpu_slot = packed_slot(l, scratch_index);
        GGML_ASSERT(move.host_slot >= 0 && map.resident[move.promote] == -1 && map.host[evict] == -1);
        if (placement->hip_tuple_dma) {
            const auto & tuple = placement->tuples[il];
            GGML_ASSERT(move.host_slot < llama_expert_profile::n_experts - llama_expert_profile::n_resident &&
                    move.old_gpu_slot >= 0 && size_t(move.old_gpu_slot) < ggml_nbytes(tuple.device) / tuple.stride);
            auto * data = static_cast<uint8_t *>(tuple.host->data) + move.host_slot * tuple.stride;
            move.n_planes = 1;
            move.bytes = tuple.stride;
            move.planes[0] = {tuple.device, data, tuple.stride};
            return move;
        }
        move.n_planes = l.experts.size();
        GGML_ASSERT(move.n_planes <= move.planes.size());
        for (size_t i = 0; i < move.n_planes; ++i) {
            ggml_tensor * host = l.experts[i];
            ggml_tensor * bank = bindings.at(host).cached;
            const size_t stride = host->nb[2];
            if (!stride || stride > packed_writeback.size() - move.bytes || !host->data || !bank || !bank->data ||
                    bank->nb[2] != stride || static_cast<size_t>(move.host_slot) >= ggml_nbytes(host) / stride ||
                    static_cast<size_t>(move.old_gpu_slot) >= ggml_nbytes(bank) / stride ||
                    static_cast<size_t>(move.new_gpu_slot) >= ggml_nbytes(bank) / stride) {
                throw std::runtime_error("invalid packed expert writeback range");
            }
            move.planes[i] = {bank, packed_host_data(l, host, move.host_slot, move.bytes), stride};
            move.bytes += stride;
        }
        return move;
    }

    void commit_packed_promotion(const packed_promotion & move, const uint8_t * victim_data) noexcept {
        auto & map = placement->maps[move.il];
        auto & l = layers[move.il];
        GGML_ASSERT(map.host[move.promote] == move.host_slot && map.resident[move.evict] == move.old_gpu_slot &&
                l.packed_scratch[move.scratch_index] == move.promote && packed_slot(l, move.scratch_index) == move.new_gpu_slot);
        // Publish the new RAM owner only after all victim planes are available.
        if (placement->hip_tuple_dma) {
            GGML_ASSERT(move.n_planes == 1 && victim_data == move.planes[0].host_data);
            ++tuple_d2h_count;
            tuple_d2h_payload_bytes += placement->tuples[move.il].payload_bytes;
            tuple_d2h_bytes += move.bytes;
        } else if (move.writeback_slot >= 0) {
            auto & slot = l.packed_ram_slots[move.host_slot];
            GGML_ASSERT(slot != move.writeback_slot);
            if (slot >= 0) {
                GGML_ASSERT(move.writeback_slot == packed_ram_spare);
                packed_ram_spare = slot;
            } else {
                GGML_ASSERT(move.writeback_slot == packed_ram_next && packed_ram_next < packed_ram_capacity);
                ++packed_ram_next;
            }
            slot = move.writeback_slot;
            ++packed_ram_rotations;
            packed_ram_rotated_bytes += move.bytes;
        } else {
            const int64_t start = profile_packed_ram_copy ? ggml_time_us() : 0;
            for (size_t i = 0; i < move.n_planes; ++i) {
                const auto & p = move.planes[i];
                std::memcpy(p.host_data, victim_data, p.stride);
                victim_data += p.stride;
            }
            if (profile_packed_ram_copy) {
                packed_ram_memcpy_us += ggml_time_us() - start;
                packed_ram_memcpy_calls += move.n_planes;
                packed_ram_memcpy_bytes += move.bytes;
            }
        }
        map.host[move.evict] = move.host_slot;
        map.host[move.promote] = -1;
        map.resident[move.evict] = -1;
        map.resident[move.promote] = move.new_gpu_slot;
        l.packed_scratch[move.scratch_index] = move.evict;
        l.packed_scratch_slots[move.scratch_index] = move.old_gpu_slot;
        ++packed_promotions;
        packed_writeback_bytes += move.bytes;
        if (packed_miss_histogram) {
            ++packed_transfer_totals.promotions;
            packed_transfer_totals.writeback_bytes += move.bytes;
        }
    }

    static void commit_overlap_promotion(void * data) noexcept {
        auto & self = *static_cast<impl *>(data);
        GGML_ASSERT(self.pending_writeback);
        self.commit_packed_promotion(self.pending_promotion, self.pending_writeback);
        ++self.packed_overlap_promotions;
        self.packed_overlap_writeback_bytes += self.pending_promotion.bytes;
        self.pending_writeback = nullptr;
    }

    void promote_packed(const layer & l, int32_t il, int32_t scratch_index, int32_t evict) {
        telemetry_scope scope(set_telemetry_cause, "expert_writeback");
        const auto move = prepare_packed_promotion(l, il, scratch_index, evict);
        wait_next_prefetch_source(il, move.host_slot);
        auto * writeback = placement->hip_tuple_dma ? move.planes[0].host_data : packed_writeback.data();
        const int64_t start = ggml_time_us();
        size_t offset = 0;
        for (size_t i = 0; i < move.n_planes; ++i) {
            const auto & p = move.planes[i];
            ggml_backend_tensor_get(p.bank, writeback + offset, move.old_gpu_slot * p.stride, p.stride);
            offset += p.stride;
        }
        commit_packed_promotion(move, writeback);
        packed_writeback_us += ggml_time_us() - start;
        if (!packed_staging && !placement->hip_tuple_dma) {
            LLAMA_LOG_DEBUG("packed expert serial promotion: layer=%d promoted=%d resident_slot=%d evicted=%d scratch_slot=%d host_slot=%d\n",
                    il, move.promote, move.new_gpu_slot, move.evict, move.old_gpu_slot, move.host_slot);
        }
    }

    void finish_hip_mtp_promotion() {
        if (!hip_mtp_pending_move.n_planes) {
            return;
        }
        const int64_t start = ggml_time_us();
        ggml_backend_synchronize(hip_upload_backend.get());
        (hip_mtp_pending_four ? hip_mtp4_promotion_wait_us : hip_mtp_promotion_wait_us) += ggml_time_us() - start;
        const auto * data = placement->hip_tuple_dma ? hip_mtp_pending_move.planes[0].host_data :
                static_cast<const uint8_t *>(ggml_backend_buffer_get_base(hip_mtp_promotion_staging.get()));
        commit_packed_promotion(hip_mtp_pending_move, data);
        packed_writeback_us += ggml_time_us() - start;
        ++(hip_mtp_pending_four ? hip_mtp4_promotion_commits : hip_mtp_promotion_commits);
        hip_mtp_pending_move.n_planes = 0;
    }

    void submit_hip_mtp_promotion(const layer & l, int32_t il, int32_t scratch_index, int32_t evict, bool mtp_four) {
        GGML_ASSERT(!hip_mtp_pending_move.n_planes);
        const auto move = prepare_packed_promotion(l, il, scratch_index, evict);
        auto * data = move.planes[0].host_data;
        if (!placement->hip_tuple_dma) {
            GGML_ASSERT(move.bytes <= ggml_backend_buffer_get_size(hip_mtp_promotion_staging.get()));
            data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(hip_mtp_promotion_staging.get()));
        }
        const int64_t start = ggml_time_us();
        try {
            for (size_t i = 0; i < move.n_planes; ++i) {
                const auto & p = move.planes[i];
                ggml_backend_tensor_get_async(hip_upload_backend.get(), p.bank, data, move.old_gpu_slot * p.stride, p.stride);
                data += p.stride;
            }
        } catch (...) {
            ggml_backend_synchronize(hip_upload_backend.get());
            throw;
        }
        hip_mtp_pending_move = move;
        hip_mtp_pending_four = mtp_four;
        ++(mtp_four ? hip_mtp4_promotion_submits : hip_mtp_promotion_submits);
        (mtp_four ? hip_mtp4_promotion_bytes : hip_mtp_promotion_bytes) += move.bytes;
        packed_writeback_us += ggml_time_us() - start;
    }

    void compact_chunk(ggml_backend_t backend, layer & l, ggml_tensor * dst,
            const std::vector<int32_t> & lanes, const std::vector<int32_t> & selected,
            const std::array<int32_t, llama_expert_profile::n_experts> & slots) {
        const int64_t count = lanes.size();
        const int64_t width = dst->ne[0];
        const int64_t ranks = dst->ne[1];
        constexpr int graph_size = 64;
        ggml_context_ptr context(ggml_init({graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true}));
        if (!context) {
            throw std::runtime_error("could not allocate compact prefill graph metadata");
        }
        auto * ctx = context.get();
        auto * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
        auto * mm_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, count);
        ggml_set_name(mm_ids, "ffn_moe_prefill_slots");
        auto * scatter = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
        auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, dst->ne[2]);
        source->data = dst->src[0]->data;
        source->buffer = dst->src[0]->buffer;
        auto * gathered = ggml_get_rows(ctx, source, rows);
        auto * input = ggml_reshape_3d(ctx, gathered, width, 1, count);
        std::array<std::vector<float>, 3> scales;
        std::array<ggml_tensor *, 3> scale_inputs{};
        for (size_t plane = 0; plane < scales.size(); ++plane) {
            auto * scale = l.compact_scales[plane];
            if (!scale) {
                continue;
            }
            std::vector<float> all;
            if (compact_upload_job || placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off) {
                all = compact_overlap_scales[plane];
            } else {
                all.resize(ggml_nelements(scale));
                ggml_backend_tensor_get(scale, all.data(), 0, all.size() * sizeof(float));
            }
            for (const auto lane : lanes) {
                scales[plane].push_back(all[selected[lane]]);
            }
            scale_inputs[plane] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, count);
        }
        auto project = [&](ggml_tensor * weight, ggml_tensor * x, int scale_index) {
            auto * result = ggml_mul_mat_id(ctx, weight, x, mm_ids);
            if (weight->type == GGML_TYPE_NVFP4) {
                ggml_prec_set_acc(result, GGML_PREC_BF16);
            }
            if (scale_inputs[scale_index]) {
                result = ggml_mul(ctx, result, scale_inputs[scale_index]);
            }
            return result;
        };
        ggml_tensor * gate;
        ggml_tensor * up;
        if (l.compact_weights[3]) {
            auto * both = project(l.compact_weights[3], input, 0);
            const int64_t ff = both->ne[0] / 2;
            gate = ggml_view_3d(ctx, both, ff, 1, count, both->nb[1], both->nb[2], 0);
            up = ggml_view_3d(ctx, both, ff, 1, count, both->nb[1], both->nb[2], ff * both->nb[0]);
        } else {
            up = project(l.compact_weights[0], input, 0);
            gate = project(l.compact_weights[1], input, 1);
        }
        auto * act = ggml_swiglu_split(ctx, gate, up);
        auto * down = project(l.compact_weights[2], act, 2);
        auto * destination = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, ranks * dst->ne[2]);
        destination->data = dst->data;
        destination->buffer = dst->buffer;
        auto * output = ggml_set_rows(ctx, destination, ggml_reshape_2d(ctx, down, width, count), scatter);
        auto * graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                throw std::runtime_error(format("compact prefill backend does not support %s", ggml_op_name(node->op)));
            }
        }
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        if (!compact_workspace || ggml_backend_buffer_get_size(compact_workspace.get()) < required) {
            compact_workspace.reset(ggml_backend_buft_alloc_buffer(buft, required));
            if (!compact_workspace) {
                throw std::runtime_error("could not allocate bounded compact prefill workspace");
            }
        }
        auto allocator = ggml_tallocr_new(compact_workspace.get());
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (tensor->view_src) {
                if (ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("could not initialize compact prefill view");
                }
            } else if (!tensor->data) {
                if (ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("could not allocate compact prefill tensor");
                }
            }
        }
        std::vector<int32_t> token_rows, physical;
        for (const auto lane : lanes) {
            token_rows.push_back(lane / ranks);
            physical.push_back(slots[selected[lane]]);
        }
        auto set_input = [&](ggml_tensor * tensor, const void * data, size_t bytes) {
            if (compact_transfer_pending) {
                ggml_backend_tensor_set_async(backend, tensor, data, 0, bytes);
            } else {
                ggml_backend_tensor_set(tensor, data, 0, bytes);
            }
        };
        set_input(rows, token_rows.data(), count * sizeof(int32_t));
        set_input(mm_ids, physical.data(), count * sizeof(int32_t));
        set_input(scatter, lanes.data(), count * sizeof(int32_t));
        for (size_t plane = 0; plane < scales.size(); ++plane) {
            if (scale_inputs[plane]) {
                set_input(scale_inputs[plane], scales[plane].data(), count * sizeof(float));
            }
        }
        const auto status = compact_transfer_pending ? ggml_backend_graph_compute_async(backend, graph) :
            ggml_backend_graph_compute(backend, graph);
        if (compact_transfer_pending) {
            if (!compact_window_checked) {
                ++prefill_overlap_windows;
                if (compact_upload_job) {
                    prefill_overlap_pending_windows += prefill_pending(compact_upload_job);
                }
                compact_window_checked = true;
            }
            ggml_backend_synchronize(backend);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("compact prefill execution failed");
        }
        ++compact_chunks;
    }

    void compact_wait_upload() {
        if (compact_transfer_pending) {
            if (placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off) {
                ggml_backend_synchronize(hip_upload_backend.get());
            } else if (!prefill_wait(compact_upload_job)) {
                throw std::runtime_error("could not join compact prefill upload");
            }
            compact_transfer_pending = false;
        }
    }

    std::vector<int32_t> compact_upload_ten(ggml_backend_t backend, layer & l, int32_t il,
            const std::vector<int32_t> & cold, size_t begin, int32_t bank, const std::vector<int32_t> & protected_slots,
            std::array<int32_t, llama_expert_profile::n_experts> & slots, size_t & uploaded) {
        GGML_ASSERT(!compact_transfer_pending);
        const auto & map = placement->maps[il];
        const bool hip_prefill = placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off;
        fills.clear();
        std::vector<int32_t> wave;
        for (size_t index = 0; index < 10 && begin + index < cold.size(); ++index) {
            const int32_t scratch = bank * 10 + index;
            const int32_t physical = packed_slot(l, scratch);
            if (std::find(protected_slots.begin(), protected_slots.end(), physical) != protected_slots.end()) {
                throw std::runtime_error("compact prefill upload would overwrite an active physical slot");
            }
            const auto expert = cold[begin + index];
            fills.push_back({expert, scratch});
            wave.push_back(expert);
            slots[expert] = physical;
        }
        for (const auto & fill : fills) {
            l.packed_scratch[fill.slot] = -1;
        }
        if (placement->hip_tuple_dma) {
            GGML_ASSERT(hip_prefill && hip_upload_backend);
            const bool async = placement->hip_prefill == llama_expert_placement::hip_overlap_mode::async;
            const size_t bytes = upload_packed(hip_upload_backend.get(), l, il, false, nullptr, 0, async);
            compact_transfer_pending = async;
            ++prefill_overlap_submits;
            prefill_overlap_bytes += bytes;
            tuple_prefill_h2d_count += fills.size();
            uploaded += bytes;
            return wave;
        }
        constexpr size_t max_regions = 30;
        std::array<ggml_tensor *, max_regions> weights{};
        std::array<size_t, max_regions> offsets{}, sizes{}, staging_offsets{};
        size_t regions = 0;
        size_t tuple_bytes = 0;
        for (const auto * host : l.experts) {
            tuple_bytes += host->nb[2];
        }
        auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get()));
        const size_t capacity = packed_ram_base ? packed_ram_offset : ggml_backend_buffer_get_size(packed_staging.get());
        size_t used = size_t(bank) * 10 * tuple_bytes;
        const size_t start = used;
        for (const auto & fill : fills) {
            const int32_t host_slot = map.host[fill.expert];
            for (auto * host : l.experts) {
                const auto & binding = bindings.at(host);
                const size_t stride = host->nb[2];
                if (host_slot < 0 || regions == max_regions || used > capacity || stride > capacity - used ||
                        (hip_prefill && (host_slot >= host->ne[2] || !host->data || !stride))) {
                    throw std::runtime_error("invalid compact prefill upload staging range");
                }
                weights[regions] = binding.cached;
                offsets[regions] = size_t(packed_slot(l, fill.slot)) * stride;
                sizes[regions] = stride;
                staging_offsets[regions++] = used;
                size_t no_retry = 0;
                copy_upload_staging(staging + used, packed_host_data(l, host, host_slot, binding.packed_offset), stride, no_retry);
                used += stride;
            }
        }
        const size_t bytes = used - start;
        bool submitted = false;
        try {
            if (hip_prefill) {
                for (size_t i = 0; i < regions; ++i) {
                    auto * weight = weights[i];
                    if (!weight || !weight->data || offsets[i] > ggml_nbytes(weight) ||
                            sizes[i] > ggml_nbytes(weight) - offsets[i]) {
                        throw std::runtime_error("invalid HIP compact prefill upload destination");
                    }
                }
                for (size_t i = 0; i < regions; ++i) {
                    ggml_backend_tensor_set_async(hip_upload_backend.get(), weights[i], staging + staging_offsets[i], offsets[i], sizes[i]);
                }
                if (placement->hip_prefill == llama_expert_placement::hip_overlap_mode::serial) {
                    ggml_backend_synchronize(hip_upload_backend.get());
                }
                submitted = true;
            } else {
                submitted = prefill_start(compact_upload_job, weights.data(), staging, offsets.data(), sizes.data(),
                        staging_offsets.data(), regions, protected_slots.data(), protected_slots.size());
            }
        } catch (...) {
            if (hip_prefill) {
                ggml_backend_synchronize(hip_upload_backend.get());
            } else {
                prefill_destroy(compact_upload_job);
                compact_upload_job = nullptr;
            }
            throw;
        }
        if (submitted) {
            compact_transfer_pending = !hip_prefill || placement->hip_prefill == llama_expert_placement::hip_overlap_mode::async;
            ++prefill_overlap_submits;
            prefill_overlap_bytes += bytes;
            uploaded += bytes;
        } else {
            // No transfer was submitted. The ordinary uploader can safely repack these planes.
            ++prefill_overlap_fallbacks;
            uploaded += upload_packed(backend, l, il, false, nullptr, bytes);
        }
        return wave;
    }

    void compact_execute(layer & l, ggml_tensor * dst) {
        finish_hip_mtp_promotion();
        const int32_t il = l.compact_layer;
        const auto backend = devices[groups[l.ig].id].backend;
        abandon_hip_mtp_upload();
        const bool hip_prefill = placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off;
        const int64_t ranks = dst->ne[1];
        std::vector<int32_t> selected(ranks * dst->ne[2]);
        ggml_backend_tensor_get(dst->src[1], selected.data(), 0, selected.size() * sizeof(int32_t));
        std::array<std::vector<int32_t>, llama_expert_profile::n_experts> by_expert;
        for (size_t lane = 0; lane < selected.size(); ++lane) {
            const int32_t expert = selected[lane];
            if (expert < 0 || expert >= llama_expert_profile::n_experts) {
                throw std::runtime_error("invalid compact prefill expert ID");
            }
            by_expert[expert].push_back(lane);
            if (by_expert[expert].size() > 512) {
                throw std::runtime_error("compact prefill requires unique top-k experts per token");
            }
        }
        auto slots = placement->maps[il].resident;
        for (size_t index = 0; index < l.packed_scratch.size(); ++index) {
            if (l.packed_scratch[index] >= 0) {
                slots[l.packed_scratch[index]] = packed_slot(l, index);
            }
        }
        std::vector<int32_t> hot, cold;
        hot.reserve(llama_expert_profile::n_experts);
        cold.reserve(llama_expert_profile::n_experts);
        size_t hot_lanes = 0;
        size_t scratch_lanes = 0;
        const uint64_t tick_base = l.packed_lookups;
        const uint64_t tick_delta = prefill_logical_ticks ? (dst->ne[2] + 1) / 2 : 1;
        l.packed_lookups += tick_delta;
        ++prefill_tick_steps;
        prefill_tick_tokens += dst->ne[2];
        prefill_tick_advanced += l.packed_lookups - tick_base;
        prefill_tick_min_delta = prefill_tick_min_delta ? std::min(prefill_tick_min_delta, tick_delta) : tick_delta;
        prefill_tick_max_delta = std::max(prefill_tick_max_delta, tick_delta);
        for (int32_t expert = 0; expert < llama_expert_profile::n_experts; ++expert) {
            if (by_expert[expert].empty()) {
                continue;
            }
            l.packed_last_used[expert] = prefill_logical_ticks ?
                    tick_base + 1 + (by_expert[expert].back() / ranks) / 2 : l.packed_lookups;
            if (prefill_logical_ticks) {
                prefill_tick_age_sum += l.packed_lookups - l.packed_last_used[expert];
            }
            (slots[expert] >= 0 ? hot : cold).push_back(expert);
            if (slots[expert] >= 0) {
                hot_lanes += by_expert[expert].size();
                scratch_lanes += placement->maps[il].resident[expert] < 0 ? by_expert[expert].size() : 0;
            }
        }
        auto execute = [&](const std::vector<int32_t> & experts) {
            std::vector<int32_t> lanes;
            for (const auto expert : experts) {
                if (lanes.size() + by_expert[expert].size() > 512) {
                    compact_chunk(backend, l, dst, lanes, selected, slots);
                    lanes.clear();
                }
                lanes.insert(lanes.end(), by_expert[expert].begin(), by_expert[expert].end());
            }
            if (!lanes.empty()) {
                compact_chunk(backend, l, dst, lanes, selected, slots);
            }
        };
        if (prefill_overlap_ten && !hip_prefill && !compact_upload_attempted && !cold.empty()) {
            compact_upload_attempted = true;
            if (placement->n_slots == 20 && packed_staging && prefill_create && prefill_start && prefill_wait && prefill_pending && prefill_destroy) {
                compact_upload_job = prefill_create(backend);
            }
        }
        if (compact_upload_job || hip_prefill) {
            for (size_t plane = 0; plane < l.compact_scales.size(); ++plane) {
                auto * scale = l.compact_scales[plane];
                auto & values = compact_overlap_scales[plane];
                values.clear();
                if (scale) {
                    values.resize(ggml_nelements(scale));
                    ggml_backend_tensor_get(scale, values.data(), 0, values.size() * sizeof(float));
                }
            }
        }
        struct upload_scope {
            impl & owner;
            ~upload_scope() {
                try {
                    owner.compact_wait_upload();
                } catch (const std::exception & err) {
                    LLAMA_LOG_ERROR("compact prefill upload cleanup: %s\n", err.what());
                    if (owner.compact_upload_job) {
                        owner.prefill_destroy(owner.compact_upload_job);
                        owner.compact_upload_job = nullptr;
                    }
                    owner.compact_transfer_pending = false;
                }
            }
        } upload_cleanup{*this};
        // All initial scratch hits must finish before either scratch bank is overwritten.
        execute(hot);
        size_t uploaded = 0;
        if ((compact_upload_job || hip_prefill) && !cold.empty()) {
            auto wave = compact_upload_ten(backend, l, il, cold, 0, 0, {}, slots, uploaded);
            compact_wait_upload();
            for (const auto & fill : fills) {
                l.packed_scratch[fill.slot] = fill.expert;
            }
            for (size_t begin = 0; begin < cold.size(); begin += 10) {
                std::vector<int32_t> next;
                if (begin + 10 < cold.size()) {
                    std::vector<int32_t> protected_slots;
                    for (const auto expert : wave) {
                        protected_slots.push_back(slots[expert]);
                    }
                    next = compact_upload_ten(backend, l, il, cold, begin + 10, ((begin / 10) + 1) % 2,
                            protected_slots, slots, uploaded);
                }
                compact_window_checked = false;
                execute(wave);
                if (!next.empty()) {
                    compact_wait_upload();
                    for (const auto & fill : fills) {
                        l.packed_scratch[fill.slot] = fill.expert;
                    }
                    wave = std::move(next);
                }
            }
        } else {
    for (size_t begin = 0; begin < cold.size(); begin += placement->n_slots) {
                prefill_overlap_fallback_layers += prefill_overlap_ten && begin == 0;
                fills.clear();
                std::vector<int32_t> wave;
                for (size_t index = 0; index < size_t(placement->n_slots) && begin + index < cold.size(); ++index) {
                    const auto expert = cold[begin + index];
                    l.packed_scratch[index] = -1;
                    fills.push_back({expert, int32_t(index)});
                    slots[expert] = packed_slot(l, index);
                    wave.push_back(expert);
                }
                try {
                    uploaded += upload_packed(backend, l, il, false);
                    for (const auto & fill : fills) {
                        l.packed_scratch[fill.slot] = fill.expert;
                    }
                } catch (...) {
                    for (const auto & fill : fills) {
                        l.packed_scratch[fill.slot] = -1;
                    }
                    throw;
                }
                execute(wave);
            }
        }
        stats_small.hits += hot_lanes;
        packed_scratch_hits += scratch_lanes;
        stats_small.misses += cold.size();
        packed_request_misses += selected.size() - hot_lanes;
        stats_small.bytes += uploaded;
        if (prompt_active) {
            ++prompt_layer_steps;
            prompt_uploads += cold.size();
            prompt_upload_bytes += uploaded;
        }
    }

    void copy_upload_staging(void * dst, const void * src, size_t size, size_t & retry_bytes, bool fine_profile = false) {
        const bool trace_generation = !generation_trace_positions.empty();
        if (!profile_packed_ram_copy && !trace_generation && !fine_profile) {
            std::memcpy(dst, src, size);
            return;
        }
        const int64_t start = ggml_time_us();
        std::memcpy(dst, src, size);
        const uint64_t us = ggml_time_us() - start;
        if (trace_generation) {
            generation_trace_memcpy_us += us;
        }
        if (fine_profile) {
            packed_staging_totals.callback_memcpy_us += us;
        }
        if (!profile_packed_ram_copy) {
            return;
        }
        auto & total = packed_staging_totals;
        ++total.calls;
        total.bytes += size;
        total.us += us;
        if (fills.size() >= 2) {
            total.fill_ge2_bytes += size;
            total.fill_ge2_us += us;
        }
        // Failed overlap plans pack a prefix of the same planes as the fallback.
        const size_t duplicated = std::min(size, retry_bytes);
        total.retry_bytes += duplicated;
        if (duplicated) {
            GGML_ASSERT(duplicated == size);
            total.retry_us += us;
            retry_bytes -= duplicated;
        }
    }

    uint8_t * hip_writeback_data() const {
        return static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get())) + hip_writeback_offset;
    }

    void submit_hip_promotion(const layer & l, int32_t il, int32_t scratch_index, int32_t evict) {
        const auto & map = placement->maps[il];
        GGML_ASSERT(scratch_index >= 0 && scratch_index < placement->n_slots);
        GGML_ASSERT(evict >= 0 && evict < llama_expert_profile::n_experts);
        packed_promotion move{};
        move.il = il;
        move.scratch_index = scratch_index;
        move.evict = evict;
        move.promote = l.packed_scratch[scratch_index];
        GGML_ASSERT(move.promote >= 0 && move.promote < llama_expert_profile::n_experts);
        move.host_slot = map.host[move.promote];
        move.old_gpu_slot = map.resident[evict];
        move.new_gpu_slot = map.hip_spare_hot_slot;
        move.n_planes = l.experts.size();
        if (move.n_planes < 2 || move.n_planes > move.planes.size() || move.host_slot < 0 ||
                move.old_gpu_slot < 0 || move.new_gpu_slot < 0 ||
                move.new_gpu_slot == llama_expert_profile::n_resident || move.new_gpu_slot == move.old_gpu_slot) {
            throw std::runtime_error("invalid HIP expert promotion mapping");
        }

        const size_t capacity = ggml_backend_buffer_get_size(packed_staging.get()) - hip_writeback_offset;
        std::array<ggml_tensor *, 3> src_views{}, dst_views{};
        ggml_reset(ctx_views.get());
        for (size_t i = 0; i < move.n_planes; ++i) {
            auto * host = l.experts[i];
            const auto & b = bindings.at(host);
            const size_t stride = host->nb[2];
            if (!stride || stride > capacity - move.bytes || !host->data || !b.cached || !b.cached->data ||
                    !b.cold || !b.cold->data || b.cached->nb[2] != stride || b.cold->nb[2] != stride ||
                    move.host_slot >= host->ne[2] || move.old_gpu_slot >= b.cached->ne[2] ||
                    move.new_gpu_slot >= b.cached->ne[2] || scratch_index >= b.cold->ne[2] - 1) {
                throw std::runtime_error("invalid HIP expert promotion range");
            }
            auto * src = ggml_view_3d(ctx_views.get(), b.cold, b.cold->ne[0], b.cold->ne[1], 1,
                    b.cold->nb[1], stride, scratch_index * stride);
            auto * dst = ggml_view_3d(ctx_views.get(), b.cached, b.cached->ne[0], b.cached->ne[1], 1,
                    b.cached->nb[1], stride, move.new_gpu_slot * stride);
            if (src->type != dst->type || !ggml_are_same_shape(src, dst) || !ggml_are_same_stride(src, dst) ||
                    !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || ggml_nbytes(src) != stride ||
                    ggml_backend_view_init(src) != GGML_STATUS_SUCCESS || ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("could not bind HIP expert promotion views");
            }
            src_views[i] = src;
            dst_views[i] = dst;
            move.planes[i] = {b.cached, packed_host_data(l, host, move.host_slot, move.bytes), stride};
            move.bytes += stride;
        }

        auto * backend = hip_upload_backend.get();
        auto * writeback = hip_writeback_data();
        for (size_t i = 0; i < move.n_planes; ++i) {
            const auto & p = move.planes[i];
            ggml_backend_tensor_get_async(backend, p.bank, writeback, move.old_gpu_slot * p.stride, p.stride);
            writeback += p.stride;
        }
        for (size_t i = 0; i < move.n_planes; ++i) {
            ggml_backend_tensor_copy_async(backend, backend, src_views[i], dst_views[i]);
        }
        hip_pending_promotion = move;
    }

    void commit_hip_promotion(layer & l) {
        const auto & move = hip_pending_promotion;
        if (move.n_planes == 0) {
            return;
        }
        auto & map = placement->maps[move.il];
        GGML_ASSERT(l.packed_scratch[move.scratch_index] == move.promote && map.host[move.promote] == move.host_slot &&
                map.resident[move.evict] == move.old_gpu_slot && map.hip_spare_hot_slot == move.new_gpu_slot);
        const auto * writeback = hip_writeback_data();
        for (size_t i = 0; i < move.n_planes; ++i) {
            const auto & p = move.planes[i];
            std::memcpy(p.host_data, writeback, p.stride);
            writeback += p.stride;
        }
        // The current cold map still refers to the old bytes. Reuse the slot only on a later lookup.
        l.packed_scratch[move.scratch_index] = -1;
        map.host[move.evict] = move.host_slot;
        map.host[move.promote] = -1;
        map.resident[move.evict] = -1;
        map.resident[move.promote] = move.new_gpu_slot;
        map.hip_spare_hot_slot = move.old_gpu_slot;
        ++packed_promotions;
        packed_writeback_bytes += move.bytes;
        LLAMA_LOG_DEBUG("packed expert HIP promotion: layer=%d promoted=%d resident_slot=%d evicted=%d spare_slot=%d host_slot=%d bytes=%zu\n",
                move.il, move.promote, move.new_gpu_slot, move.evict, move.old_gpu_slot, move.host_slot, move.bytes);
        hip_pending_promotion.n_planes = 0;
    }

    void abandon_hip_mtp_upload() {
        pending_mtp_full_layer = -1;
        if (placement && placement->hip_mtp_overlap && pending_layer >= 0) {
            wait_hip_cold_upload();
            pending_layer = -1;
        }
    }

    void wait_hip_cold_upload() {
        if (hip_upload_backend && (placement->hip_mtp_overlap || placement->hip_overlap == llama_expert_placement::hip_overlap_mode::async) &&
                pending_layer >= 0 && (pending_upload_bytes != 0 || hip_pending_promotion.n_planes != 0)) {
            const int64_t start = ggml_time_us();
            ggml_backend_synchronize(hip_upload_backend.get());
            const uint64_t us = ggml_time_us() - start;
            packed_wait_us += us;
            if (profile_packed_ram_copy && placement->hip_mtp_gate_up_overlap) {
                packed_staging_totals.upload_wait_us += us;
            }
        }
    }

    void init_next_prefetch() {
        size_t stride_max = 0;
        for (const auto & tuple : placement->tuples) {
            stride_max = std::max(stride_max, tuple.stride);
        }
        if (!stride_max || stride_max > INT64_MAX / next_prefetch_slots) {
            throw std::runtime_error("invalid HIP next Router prefetch tuple stride");
        }
        next_prefetch_ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        next_prefetch_views.reset(ggml_init({2 * next_prefetch_slots * ggml_tensor_overhead(), nullptr, true}));
        next_prefetch_backend.reset(ggml_backend_dev_init(placement->target, nullptr));
        if (!next_prefetch_ctx || !next_prefetch_views || !next_prefetch_backend) {
            throw std::runtime_error("could not initialize HIP next Router prefetch");
        }
        next_prefetch_bank = ggml_new_tensor_1d(next_prefetch_ctx.get(), GGML_TYPE_I8, stride_max * next_prefetch_slots);
        next_prefetch_buffer.reset(ggml_backend_alloc_ctx_tensors(next_prefetch_ctx.get(), next_prefetch_backend.get()));
        if (!next_prefetch_buffer) {
            throw std::runtime_error("could not allocate HIP next Router prefetch bank");
        }
        const char * score_filter = std::getenv("LLAMA_EXPERT_HIP_NEXT_ROUTER_SCORE_FILTER");
        if (score_filter && std::strcmp(score_filter, "1") == 0) {
            next_prefetch_score_mode = 1;
        } else if (score_filter && std::strcmp(score_filter, "2") == 0) {
            next_prefetch_score_mode = 2;
        }
        const char * hit_wait = std::getenv("LLAMA_EXPERT_HIP_NEXT_ROUTER_HIT_WAIT");
        if (hit_wait && std::strcmp(hit_wait, "1") == 0) {
            for (auto & event : next_prefetch_events) {
                event.reset(ggml_backend_event_new(placement->target));
                if (!event) {
                    break;
                }
            }
            next_prefetch_hit_wait = bool(next_prefetch_events.back());
            if (!next_prefetch_hit_wait) {
                LLAMA_LOG_WARN("packed expert HIP next Router prefetch: per-slot events unavailable, using full-stream wait\n");
            }
        }
        LLAMA_LOG_INFO("packed expert HIP next Router prefetch: slots=%d bank_bytes=%zu\n",
                next_prefetch_slots, ggml_backend_buffer_get_size(next_prefetch_buffer.get()));
        if (next_prefetch_score_mode) {
            LLAMA_LOG_INFO("packed expert HIP next Router prefetch: score_filter=%d\n", next_prefetch_score_mode);
        }
        if (next_prefetch_hit_wait) {
            LLAMA_LOG_INFO("packed expert HIP next Router prefetch: hit_wait=1\n");
        }
        const char * after_submit = std::getenv("LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH_AFTER_SUBMIT");
        if (after_submit && std::strcmp(after_submit, "1") == 0) {
            next_prefetch_after_submit = next_prefetch_hit_wait;
            if (!next_prefetch_after_submit) {
                LLAMA_LOG_WARN("packed expert HIP next Router prefetch: after-submit requires hit wait, using immediate launch\n");
            }
        }
    }

    void retire_next_prefetch(size_t hits) {
        next_prefetch_hits += hits;
        next_prefetch_unused += next_prefetch_count - hits;
        next_prefetch_count = 0;
        next_prefetch_layer = -1;
        ggml_reset(next_prefetch_views.get());
    }

    void finish_next_prefetch(size_t hits = 0) {
        if (next_prefetch_layer < 0 && !next_prefetch_tail_pending) {
            return;
        }
        const int64_t start = ggml_time_us();
        ggml_backend_synchronize(next_prefetch_backend.get());
        if (next_profile_start) {
            ++next_profile_prefetch_waits;
        }
        const int64_t elapsed = ggml_time_us() - start;
        next_prefetch_wait_us += elapsed;
        if (next_prefetch_hit_wait) {
            ++next_prefetch_tail_waits;
            next_prefetch_tail_wait_us += elapsed;
            next_prefetch_tail_pending = false;
        }
        retire_next_prefetch(hits);
    }

    void complete_next_prefetch_prefix(size_t count) {
        next_prefetch_completed = count;
        next_prefetch_tail_pending = count < next_prefetch_sources;
    }

    void wait_next_prefetch_source(int32_t il, int32_t host_slot) {
        if (!next_prefetch_tail_pending || il != next_prefetch_source_layer) {
            return;
        }
        for (size_t i = next_prefetch_completed; i < next_prefetch_sources; ++i) {
            if (next_prefetch_source_slots[i] == host_slot) {
                const int64_t start = ggml_time_us();
                ggml_backend_event_synchronize(next_prefetch_events[i].get());
                next_prefetch_source_wait_us += ggml_time_us() - start;
                ++next_prefetch_source_waits;
                complete_next_prefetch_prefix(i + 1);
                break;
            }
        }
    }

    static constexpr int next_prefetch_tokens = 3;
    static constexpr int next_prefetch_ranks = 10;
    static constexpr int next_prefetch_priority_ranks = 20;
    static constexpr int next_prefetch_experts = llama_expert_profile::n_experts;
    using next_prefetch_routes = std::array<int32_t, next_prefetch_tokens * next_prefetch_priority_ranks>;
    using next_prefetch_logits = std::array<float, next_prefetch_tokens * next_prefetch_experts>;
    using next_prefetch_mask = std::array<bool, next_prefetch_experts>;

    void filter_next_prefetch(const next_prefetch_routes & predicted, const next_prefetch_logits & logits,
            int32_t layer, next_prefetch_mask & accepted) {
        accepted.fill(false);
        next_prefetch_mask seen{};
        for (int token = 0; token < next_prefetch_tokens; ++token) {
            const auto * row = logits.data() + token * next_prefetch_experts;
            const auto * ids = predicted.data() + token * next_prefetch_ranks;
            bool valid = true;
            for (int rank = 0; rank < next_prefetch_ranks; ++rank) {
                GGML_ASSERT(ids[rank] >= 0 && ids[rank] < next_prefetch_experts);
                valid &= std::isfinite(row[ids[rank]]);
            }
            const float cutoff = row[ids[next_prefetch_ranks - 1]];
            const float range = row[ids[0]] - cutoff;
            valid &= std::isfinite(range) && range > 1e-6f;
            const float threshold = cutoff + 0.10f * range;
            next_prefetch_score_row_fallback += !valid;
            for (int rank = 0; rank < next_prefetch_ranks; ++rank) {
                const int expert = ids[rank];
                seen[expert] = true;
                accepted[expert] |= !valid || row[expert] >= threshold;
            }
        }

        const auto & map = placement->maps[layer];
        const auto & scratch = layers[layer].packed_scratch;
        size_t eligible = 0;
        for (int i = 0; i < next_prefetch_tokens * next_prefetch_ranks; ++i) {
            const int32_t expert = predicted[i];
            if (!seen[expert]) {
                continue;
            }
            seen[expert] = false;
            if (map.resident[expert] < 0 && map.host[expert] >= 0 &&
                    std::find(scratch.begin(), scratch.end(), expert) == scratch.end()) {
                eligible += accepted[expert];
                next_prefetch_score_rejected += !accepted[expert];
            }
        }
        next_prefetch_score_eligible += eligible;
        next_prefetch_score_empty += eligible == 0;
    }

    void submit_next_prefetch(int32_t layer, int32_t expert, int32_t host_slot, bool beyond_top10) {
        const auto & tuple = placement->tuples[layer];
        const auto * host = static_cast<const uint8_t *>(tuple.host->data);
        GGML_ASSERT(size_t(host_slot) < ggml_nbytes(tuple.host) / tuple.stride);
        next_prefetch_layer = layer;
        if (next_prefetch_hit_wait) {
            next_prefetch_source_layer = layer;
            next_prefetch_source_slots[next_prefetch_count] = host_slot;
            next_prefetch_sources = next_prefetch_count + 1;
            next_prefetch_completed = 0;
            next_prefetch_tail_pending = true;
        }
        const int64_t start = next_profile_start ? ggml_time_us() : 0;
        ggml_backend_tensor_set_async(next_prefetch_backend.get(), next_prefetch_bank, host + host_slot * tuple.stride,
                next_prefetch_count * tuple.stride, tuple.stride);
        if (next_prefetch_hit_wait) {
            ggml_backend_event_record(next_prefetch_events[next_prefetch_count].get(), next_prefetch_backend.get());
        }
        if (start) {
            next_profile_h2d_submit_us += ggml_time_us() - start;
        }
        next_prefetch_h2d_bytes += tuple.stride;
        next_prefetch_beyond_top10[next_prefetch_count] = beyond_top10;
        next_prefetch_top20_only += beyond_top10;
        next_prefetch_ids[next_prefetch_count++] = expert;
        if (next_prefetch_count == 1) {
            ++next_prefetch_launches;
        }
    }

    bool priority_next_prefetch(int32_t layer, const next_prefetch_routes & ids, const next_prefetch_logits & scores) {
        struct candidate {
            int32_t expert;
            float score;
            int order;
            bool top10;
        };
        std::array<candidate, next_prefetch_tokens * next_prefetch_priority_ranks> candidates;
        size_t count = 0;
        const auto & map = placement->maps[layer];
        const auto & scratch = layers[layer].packed_scratch;
        for (int rank = 0; rank < next_prefetch_priority_ranks; ++rank) {
            for (int token = 0; token < next_prefetch_tokens; ++token) {
                const int expert = ids[token * next_prefetch_priority_ranks + rank];
                GGML_ASSERT(expert >= 0 && expert < next_prefetch_experts);
                if (map.resident[expert] >= 0 || map.host[expert] < 0 ||
                        std::find(scratch.begin(), scratch.end(), expert) != scratch.end()) {
                    continue;
                }
                auto end = candidates.begin() + count;
                auto found = std::find_if(candidates.begin(), end, [expert](const candidate & c) { return c.expert == expert; });
                if (found == end) {
                    float score = 0.0f;
                    for (int position = 0; position < next_prefetch_tokens; ++position) {
                        const float prob = scores[position * next_prefetch_experts + expert];
                        if (!std::isfinite(prob) || prob < 0.0f || prob > 1.0f) {
                            return false;
                        }
                        score = std::max(score, prob);
                    }
                    candidates[count] = {expert, score, static_cast<int>(count), rank < next_prefetch_ranks};
                    ++count;
                } else {
                    found->top10 |= rank < next_prefetch_ranks;
                }
            }
        }
        const auto end = candidates.begin() + count;
        const auto top10 = std::count_if(candidates.begin(), end, [](const candidate & c) { return c.top10; });
        const size_t budget = std::min<size_t>(top10, next_prefetch_slots);
        std::sort(candidates.begin(), end, [](const candidate & a, const candidate & b) {
            return a.score == b.score ? a.order < b.order : a.score > b.score;
        });
        for (size_t i = 0; i < budget; ++i) {
            const auto & c = candidates[i];
            submit_next_prefetch(layer, c.expert, map.host[c.expert], !c.top10);
        }
        return true;
    }

    void launch_next_prefetch(ggml_backend_sched_t sched, int32_t il) {
        if (size_t(il) >= next_prefetch_predictions->size() || size_t(il + 1) >= placement->maps.size()) {
            return;
        }
        const int32_t target = il + 1;
        if (!next_prefetch_hit_wait || next_prefetch_layer >= 0 ||
                (next_prefetch_tail_pending && target <= next_prefetch_source_layer)) {
            finish_next_prefetch();
        }
        auto * prediction = (*next_prefetch_predictions)[il];
        auto * producer = ggml_backend_sched_get_tensor_backend(sched, prediction);
        GGML_ASSERT(producer && ggml_is_contiguous(prediction) && prediction->type == GGML_TYPE_I32);
        next_prefetch_routes predicted;
        next_prefetch_logits logits;
        bool read_scores = false;
        const int ranks = next_prefetch_score_mode == 2 ? next_prefetch_priority_ranks : next_prefetch_ranks;
        const size_t prediction_bytes = ranks * next_prefetch_tokens * sizeof(int32_t);
        GGML_ASSERT(prediction->ne[0] == ranks && prediction->ne[1] == next_prefetch_tokens &&
                ggml_nbytes(prediction) == prediction_bytes);
        const int64_t read_start = next_profile_start ? ggml_time_us() : 0;
        ggml_backend_tensor_get_async(producer, prediction, predicted.data(), 0, prediction_bytes);
        if (next_prefetch_scores) {
            auto * scores = (*next_prefetch_scores)[il];
            read_scores = scores && scores->type == GGML_TYPE_F32 && ggml_is_contiguous(scores) &&
                    scores->ne[0] == next_prefetch_experts && scores->ne[1] == next_prefetch_tokens &&
                    ggml_nbytes(scores) == sizeof(logits) && ggml_backend_sched_get_tensor_backend(sched, scores) == producer;
            if (read_scores) {
                ggml_backend_tensor_get_async(producer, scores, logits.data(), 0, sizeof(logits));
            } else {
                ++next_prefetch_score_readback_fallback;
            }
        }
        const int64_t wait_start = read_start ? ggml_time_us() : 0;
        ggml_backend_synchronize(producer);
        if (read_start) {
            ++next_profile_readbacks;
            next_profile_read_submit_us += wait_start - read_start;
            next_profile_read_wait_us += ggml_time_us() - wait_start;
        }

        if (next_prefetch_score_mode == 2) {
            if (read_scores && priority_next_prefetch(target, predicted, logits)) {
                ++next_prefetch_priority_layers;
                return;
            }
            ++next_prefetch_priority_fallbacks;
            read_scores = false;
        }
        next_prefetch_mask accepted;
        if (read_scores) {
            filter_next_prefetch(predicted, logits, target, accepted);
        }
        const auto & map = placement->maps[target];
        const auto & scratch = layers[target].packed_scratch;
        for (int rank = 0; rank < next_prefetch_ranks; ++rank) {
            for (int token = 0; token < next_prefetch_tokens; ++token) {
                const int32_t expert = predicted[token * ranks + rank];
                GGML_ASSERT(expert >= 0 && size_t(expert) < map.resident.size());
                if (read_scores && !accepted[expert]) {
                    continue;
                }
                const int32_t host_slot = map.host[expert];
                const auto end = next_prefetch_ids.begin() + next_prefetch_count;
                if (map.resident[expert] >= 0 || host_slot < 0 || std::find(scratch.begin(), scratch.end(), expert) != scratch.end() ||
                        std::find(next_prefetch_ids.begin(), end, expert) != end) {
                    continue;
                }
                submit_next_prefetch(target, expert, host_slot, false);
                if (next_prefetch_count == next_prefetch_slots) {
                    return;
                }
            }
        }
    }

    void schedule_next_prefetch(ggml_backend_sched_t sched, ggml_backend_t expert_backend, int32_t il) {
        if (next_prefetch_after_submit) {
            if (size_t(il) >= next_prefetch_predictions->size() || size_t(il + 1) >= placement->maps.size()) {
                return;
            }
            auto * prediction = (*next_prefetch_predictions)[il];
            const auto producer = ggml_backend_sched_get_tensor_backend(sched, prediction);
            // Readback must not synchronize the stream that runs the current Expert split.
            if (producer != expert_backend && next_prefetch_submit_layer < 0 &&
                    ggml_backend_sched_after_split_submit(sched, [](ggml_backend_sched_t sched, void * user_data) {
                        auto & self = *static_cast<impl *>(user_data);
                        const int32_t il = self.next_prefetch_submit_layer;
                        self.next_prefetch_submit_layer = -1;
                        ++self.next_prefetch_submit_executed;
                        const int64_t start = self.next_profile_start ? ggml_time_us() : 0;
                        self.launch_next_prefetch(sched, il);
                        if (start) {
                            self.next_prefetch_submit_cpu_us += ggml_time_us() - start;
                        }
                    }, this)) {
                next_prefetch_submit_layer = il;
                ++next_prefetch_submit_scheduled;
                return;
            }
            ++next_prefetch_submit_fallback;
        }
        launch_next_prefetch(sched, il);
    }

    int32_t next_prefetch_slot(int32_t il, int32_t expert) const {
        if (il != next_prefetch_layer) {
            return -1;
        }
        const auto end = next_prefetch_ids.begin() + next_prefetch_count;
        const auto found = std::find(next_prefetch_ids.begin(), end, expert);
        return found == end ? -1 : static_cast<int32_t>(found - next_prefetch_ids.begin());
    }

    void copy_next_prefetch(ggml_backend_t backend, const llama_expert_placement::tuple_bank & tuple, int32_t from, int32_t to) {
        const int64_t start = next_profile_start ? ggml_time_us() : 0;
        auto * ctx = next_prefetch_views.get();
        auto * src = ggml_view_1d(ctx, next_prefetch_bank, tuple.stride, from * tuple.stride);
        auto * dst = ggml_view_1d(ctx, tuple.device, tuple.stride, to * tuple.stride);
        if (ggml_backend_view_init(src) != GGML_STATUS_SUCCESS || ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("could not bind HIP next Router prefetch views");
        }
        ggml_backend_tensor_copy_async(backend, backend, src, dst);
        if (start) {
            next_profile_d2d_submit_us += ggml_time_us() - start;
        }
        next_prefetch_d2d_bytes += tuple.stride;
    }

    size_t stage_hip_single_kernel_upload(ggml_backend_t backend, const ggml_cgraph * graph, const ggml_tensor * lookup,
            const layer & l, int32_t il, int32_t promote_index, int32_t evict) {
        constexpr size_t max_uploads = 4;
        if (fills.empty() || fills.size() > max_uploads) {
            return 0;
        }
        const auto & tuple = placement->tuples[il];
        const auto & map = placement->maps[il];
        std::array<const void *, max_uploads> sources;
        std::array<int, max_uploads> slots, experts;
        const auto promotion = evict >= 0 ? prepare_packed_promotion(l, il, promote_index, evict) : packed_promotion{};
        for (size_t i = 0; i < fills.size(); ++i) {
            const auto & fill = fills[i];
            const int host_slot = map.host[fill.expert];
            const int slot = packed_slot(l, fill.slot);
            if (host_slot < 0 || size_t(host_slot) >= ggml_nbytes(tuple.host) / tuple.stride) {
                return 0;
            }
            // A concurrent promotion may only read and replace storage outside the DMA regions.
            if (evict >= 0 && (host_slot == promotion.host_slot || slot == promotion.old_gpu_slot ||
                    slot == promotion.new_gpu_slot)) {
                return 0;
            }
            sources[i] = static_cast<const uint8_t *>(tuple.host->data) + host_slot * tuple.stride;
            slots[i] = slot;
            experts[i] = fill.expert;
        }
        telemetry_scope scope(set_telemetry_cause, "expert_upload");
        if (!stage_hip_expert_overlap(backend, graph, lookup, tuple.device, sources.data(), slots.data(), experts.data(), fills.size(),
                tuple.stride)) {
            return 0;
        }
        const size_t bytes = fills.size() * tuple.stride;
        tuple_h2d_count += fills.size();
        tuple_h2d_payload_bytes += fills.size() * tuple.payload_bytes;
        tuple_h2d_bytes += bytes;
        ++hip_expert_overlap_staged[fills.size() - 1];
        return bytes;
    }

    size_t upload_packed(ggml_backend_t backend, const layer & l, int32_t il, bool cold,
                         bool * submitted = nullptr, size_t retry_bytes = 0, bool hip_async = false, bool hip_direct = false,
                         llama_expert_staging_telemetry * callback_totals = nullptr) {
        const bool tuple_dma = placement->hip_tuple_dma;
        GGML_ASSERT(!hip_async || ((cold || placement->hip_mtp_overlap || tuple_dma) && !submitted));
        GGML_ASSERT(!hip_direct || (!cold && !hip_async && !submitted && !retry_bytes));
        GGML_ASSERT(!tuple_dma || (!cold && !submitted && !retry_bytes && !hip_direct));
        telemetry_scope scope(set_telemetry_cause, "expert_upload");
        const auto & map = placement->maps[il];
        const bool direct = hip_direct || tuple_dma;
        auto * staging = direct ? nullptr : static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get()));
        const size_t staging_bytes = direct ? 0 : hip_writeback_offset ? hip_writeback_offset :
                packed_ram_base ? packed_ram_offset : ggml_backend_buffer_get_size(packed_staging.get());
        size_t bytes = 0;
        size_t prefetch_hits = 0;
        size_t prefetch_top20_hits = 0;
        std::array<moe_cache_lru::fill, next_prefetch_slots> needed;
        int32_t last_needed = -1;
        const auto submit = [&](ggml_tensor * tensor, const uint8_t * data, size_t offset, size_t size) {
            const int64_t start = callback_totals ? ggml_time_us() : 0;
            ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
            if (callback_totals) {
                callback_totals->submit_us += ggml_time_us() - start;
            }
            bytes += size;
        };
        try {
            for (const auto & fill : fills) {
                const int32_t slot = cold ? fill.slot : packed_slot(l, fill.slot);
                if (tuple_dma) {
                    const auto & tuple = placement->tuples[il];
                    const int32_t host_slot = map.host[fill.expert];
                    GGML_ASSERT(host_slot >= 0 && size_t(host_slot) < ggml_nbytes(tuple.host) / tuple.stride &&
                            slot >= 0 && size_t(slot) < ggml_nbytes(tuple.device) / tuple.stride);
                    const int32_t prefetched = next_prefetch_slot(il, fill.expert);
                    if (prefetched >= 0) {
                        if (next_prefetch_hit_wait) {
                            GGML_ASSERT(prefetch_hits < needed.size());
                            needed[prefetch_hits] = {prefetched, slot};
                            last_needed = std::max(last_needed, prefetched);
                        } else {
                            copy_next_prefetch(next_prefetch_backend.get(), tuple, prefetched, slot);
                        }
                        ++prefetch_hits;
                        prefetch_top20_hits += next_prefetch_beyond_top10[prefetched];
                        continue;
                    }
                    const auto * data = static_cast<const uint8_t *>(tuple.host->data) + host_slot * tuple.stride;
                    submit(tuple.device, data, slot * tuple.stride, tuple.stride);
                    if (next_profile_start) {
                        ++next_profile_demand_count;
                        next_profile_demand_bytes += tuple.stride;
                    }
                    ++tuple_h2d_count;
                    tuple_h2d_payload_bytes += tuple.payload_bytes;
                    tuple_h2d_bytes += tuple.stride;
                    continue;
                }
                for (ggml_tensor * host : l.experts) {
                    const auto & binding = bindings.at(host);
                    const size_t stride = host->nb[2];
                    const auto * data = packed_host_data(l, host, map.host[fill.expert], binding.packed_offset);
                    if (!hip_direct) {
                        GGML_ASSERT(stride <= staging_bytes - bytes);
                        copy_upload_staging(staging + bytes, data, stride, retry_bytes, callback_totals != nullptr);
                        data = staging + bytes;
                    }
                    submit(cold ? binding.cold : binding.cached, data, slot * stride, stride);
                }
            }
            if (submitted) {
                *submitted = submit_expert_upload(backend);
            }
            if (last_needed >= 0) {
                GGML_ASSERT(!hip_async && !submitted);
                // The H2D stream is FIFO. The last needed event covers all hits.
                ggml_backend_event_wait(backend, next_prefetch_events[last_needed].get());
                ++next_prefetch_dependencies;
                next_prefetch_dependency_slots += last_needed + 1;
                for (size_t i = 0; i < prefetch_hits; ++i) {
                    copy_next_prefetch(backend, placement->tuples[il], needed[i].expert, needed[i].slot);
                }
            }
        } catch (...) {
            // Queued copies can still read the staging buffer after a failed fill.
            if (bytes != 0 || last_needed >= 0) {
                ggml_backend_synchronize(backend);
            }
            if (il == next_prefetch_layer) {
                finish_next_prefetch();
            }
            throw;
        }
        if (il == next_prefetch_layer && !next_prefetch_hit_wait) {
            // The normal slots cannot be published until both upload streams have finished writing them.
            finish_next_prefetch(prefetch_hits);
        }
        if (!hip_async && (!submitted || !*submitted)) {
            const bool trace_wait = !generation_trace_positions.empty();
            const bool measure_wait = hip_direct || trace_wait || profile_packed_ram_copy || callback_totals || last_needed >= 0;
            const int64_t start = measure_wait ? ggml_time_us() : 0;
            ggml_backend_synchronize(backend);
            if (measure_wait) {
                const uint64_t us = ggml_time_us() - start;
                if (last_needed >= 0) {
                    next_prefetch_consumer_wait_us += us;
                }
                if (callback_totals) {
                    callback_totals->completion_us += us;
                }
                if (next_profile_start) {
                    ++next_profile_demand_waits;
                }
                if (profile_packed_ram_copy) {
                    packed_staging_totals.upload_wait_us += us;
                    packed_staging_totals.direct_bytes += direct ? bytes : 0;
                }
                if (trace_wait) {
                    generation_trace_upload_wait_us += us;
                }
                if (hip_direct) {
                    ++hip_direct_upload_steps;
                    hip_direct_upload_planes += fills.size() * l.experts.size();
                    hip_direct_upload_bytes += bytes;
                    hip_direct_upload_wait_us += us;
                }
            }
        }
        next_prefetch_top20_only_hits += prefetch_top20_hits;
        if (il == next_prefetch_layer && next_prefetch_hit_wait) {
            // The consumer sync above has drained the shared bank reads before its next use.
            complete_next_prefetch_prefix(last_needed + 1);
            retire_next_prefetch(prefetch_hits);
        }
        return bytes;
    }

    bool overlap_packed_upload(ggml_backend_t backend, const ggml_cgraph * graph, ggml_tensor * slot_map,
            const layer & l, int32_t il, int32_t n_used, const std::array<int32_t, llama_expert_profile::n_experts> & slots,
            int32_t promote_index, int32_t evict, size_t & bytes, size_t & copied_bytes) {
        copied_bytes = 0;
        constexpr size_t max_lanes = 32;
        constexpr size_t max_planes = 3;
        if (n_used <= 0 || ids.size() > max_lanes || ids.size() % n_used != 0 ||
                ids.size() / n_used > 3 || l.experts.size() > max_planes) {
            return false;
        }
        const auto & map = placement->maps[il];
        std::array<bool, llama_expert_profile::n_experts> filled{};
        for (const auto & fill : fills) {
            filled[fill.expert] = true;
        }
        std::array<int32_t, max_lanes> used_slots{};
        std::array<uint32_t, 3> hot_masks{};
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto expert = ids[i];
            used_slots[i] = slots[expert];
            if (!filled[expert] && used_slots[i] >= 0) {
                hot_masks[i / n_used] |= uint32_t{1} << (i % n_used);
            }
        }
        if (!(hot_masks[0] || hot_masks[1] || hot_masks[2])) {
            return false;
        }

        auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get()));
        const size_t staging_bytes = packed_ram_base ? packed_ram_offset : ggml_backend_buffer_get_size(packed_staging.get());
        constexpr size_t max_regions = max_lanes * max_planes;
        std::array<ggml_tensor *, max_regions> tensors;
        std::array<size_t, max_regions> offsets, sizes, staging_offsets;
        size_t n_regions = 0;
        size_t used_bytes = 0;
        for (const auto & fill : fills) {
            const int32_t slot = packed_slot(l, fill.slot);
            const int32_t host_slot = map.host[fill.expert];
            for (auto * host : l.experts) {
                const auto & binding = bindings.at(host);
                auto * bank = binding.cached;
                const size_t stride = host->nb[2];
                if (n_regions == max_regions || !stride || stride > staging_bytes - used_bytes ||
                        slot < 0 || static_cast<size_t>(slot) >= ggml_nbytes(bank) / stride ||
                        host_slot < 0 || static_cast<size_t>(host_slot) >= ggml_nbytes(host) / stride) {
                    return false;
                }
                tensors[n_regions] = bank;
                offsets[n_regions] = slot * stride;
                sizes[n_regions] = stride;
                staging_offsets[n_regions] = used_bytes;
                ++n_regions;
                size_t no_retry = 0;
                copy_upload_staging(staging + used_bytes, packed_host_data(l, host, host_slot, binding.packed_offset), stride, no_retry);
                used_bytes += stride;
                copied_bytes += stride;
            }
        }
        telemetry_scope scope(set_telemetry_cause, "expert_upload");
        if (evict >= 0) {
            GGML_ASSERT(!pending_writeback && overlap_expert_promotion);
            auto move = prepare_packed_promotion(l, il, promote_index, evict);
            if (move.bytes > staging_bytes - used_bytes ||
                    std::find(used_slots.begin(), used_slots.begin() + ids.size(), move.old_gpu_slot) != used_slots.begin() + ids.size()) {
                return false;
            }
            std::array<ggml_tensor *, max_planes> writeback_weights;
            std::array<size_t, max_planes> writeback_offsets;
            size_t offset = used_bytes;
            if (!l.packed_ram_slots.empty()) {
                const int32_t old_slot = l.packed_ram_slots[move.host_slot];
                if (old_slot >= 0) {
                    move.writeback_slot = packed_ram_spare;
                } else if (packed_ram_next < packed_ram_capacity) {
                    move.writeback_slot = packed_ram_next;
                }
                if (move.writeback_slot >= 0) {
                    offset = packed_ram_offset + move.writeback_slot * packed_ram_stride;
                }
            }
            pending_writeback = staging + offset;
            for (size_t i = 0; i < move.n_planes; ++i) {
                writeback_weights[i] = move.planes[i].bank;
                writeback_offsets[i] = offset;
                offset += move.planes[i].stride;
            }
            pending_promotion = move;
            try {
                if (!overlap_expert_promotion(backend, graph, slot_map, slots.data(), used_slots.data(), hot_masks.data(),
                        tensors.data(), staging, offsets.data(), sizes.data(), staging_offsets.data(), n_regions,
                        move.old_gpu_slot, writeback_weights.data(), writeback_offsets.data(), move.n_planes, commit_overlap_promotion, this)) {
                    pending_writeback = nullptr;
                    return false;
                }
            } catch (...) {
                pending_writeback = nullptr;
                throw;
            }
        } else if (!overlap_expert_upload(backend, graph, slot_map, slots.data(), used_slots.data(), hot_masks.data(),
                tensors.data(), staging, offsets.data(), sizes.data(), staging_offsets.data(), n_regions)) {
            return false;
        }
        bytes = used_bytes;
        return true;
    }

    size_t submit_rtx_misses(ggml_backend_t backend, ggml_backend_sched_t sched, int32_t il, const ggml_cgraph * graph, ggml_tensor * slot_map, const layer & l, int32_t n_used,
            const std::array<int32_t, llama_expert_profile::n_experts> & slots,
            std::array<int32_t, llama_expert_profile::n_experts> & rtx_slot_of, size_t & bytes) {
        if (n_used <= 0 || ids.empty() || ids.size() % n_used != 0 || ids.size() / n_used > 2 ||
                (l.experts.size() != 2 && l.experts.size() != 3)) {
            return 0;
        }
        std::array<int32_t, packed_miss_bins> candidates;
        size_t n_candidates = 0;
        for (const int32_t expert : ids) {
            if (slots[expert] >= 0 || rtx_slot_of[expert] >= 0) {
                continue;
            }
            rtx_slot_of[expert] = static_cast<int32_t>(n_candidates);
            candidates[n_candidates++] = expert;
            if (n_candidates > rtx_max_slots) {
                break;
            }
        }
        if (n_candidates < 2) {
            return 0;
        }
        // Leave at least one fresh expert for MI50, including when the cap was not reached.
        const size_t n_rtx = n_candidates - 1;
        rtx_slot_of[candidates[n_rtx]] = -1;
        ++rtx_eligible_steps;
        if (!rtx_expert->available(backend, sched, il, graph, slot_map)) {
            return 0;
        }

        const auto & map = placement->maps[il];
        const auto & planes = l.rtx_planes;
        const bool merged = planes[3] != nullptr;
        if (!planes[2] || (merged ? (planes[0] || planes[1]) : (!planes[0] || !planes[1])) ||
                l.experts.size() != (merged ? 2 : 3)) {
            return 0;
        }
        for (const auto * plane : l.experts) {
            if (!plane || std::count(planes.begin(), planes.end(), plane) != 1 ||
                    std::count(l.experts.begin(), l.experts.end(), plane) != 1 ||
                    !plane->data || !plane->buffer || !ggml_backend_buffer_is_host(plane->buffer) ||
                    !ggml_is_quantized(plane->type) || !ggml_is_contiguous(plane) || plane->ne[3] != 1) {
                return 0;
            }
        }
        std::array<const void *, 4 * packed_miss_bins> host_slices{};
        size_t rtx_bytes = 0;
        for (size_t r = 0; r < n_rtx; ++r) {
            const int32_t host_slot = map.host[candidates[r]];
            for (size_t p = 0; p < planes.size(); ++p) {
                const auto * host = planes[p];
                if (!host) {
                    continue;
                }
                const size_t stride = host->nb[2];
                if (host_slot < 0 || host_slot >= host->ne[2] || !stride ||
                        static_cast<size_t>(host_slot) >= ggml_nbytes(host) / stride || stride > SIZE_MAX - rtx_bytes) {
                    return 0;
                }
                host_slices[r * planes.size() + p] = packed_host_data(l, host, host_slot, bindings.at(host).packed_offset);
                rtx_bytes += stride;
            }
        }

        std::array<int32_t, packed_miss_bins> lanes;
        GGML_ASSERT(ids.size() <= lanes.size());
        size_t n_ranks = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            lanes[i] = rtx_slot_of[ids[i]];
            n_ranks += lanes[i] >= 0;
        }
        if (!rtx_expert->submit(il, graph, slot_map, lanes.data(), host_slices.data(), n_rtx)) {
            return 0;
        }
        ++rtx_steps;
        rtx_unique_experts += n_rtx;
        rtx_ranks += n_ranks;
        rtx_uploaded_bytes += rtx_bytes;
        bytes = rtx_bytes;
        return n_rtx;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, bool profile, bool early,
            ggml_backend_sched_t sched, bool fixed_mtp_pair, bool mtp_four) {
        const int64_t callback_start = (profile_packed_callback || next_profile_start) && !early ? ggml_time_us() : 0;
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        finish_hip_mtp_promotion();
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        if (early && (!packed_early_copy || !has_cpu || pending_layer != il || fills.empty() || pending_upload_bytes != 0)) {
            return false;
        }
        layer & l = layers[il];
        group & g = groups[l.ig];
        const auto wave_it = std::find(l.prefill_maps.begin(), l.prefill_maps.end(), src);
        const bool prefill = wave_it != l.prefill_maps.end();
        const bool mtp_verify = mtp_verify_three_enabled && !prefill &&
                (src == l.mtp_verify_maps[0] || src == l.mtp_verify_maps[1]);
        const bool use_waves = prefill || mtp_verify;
        static const bool overlap_three = [] {
            const char * value = std::getenv("LLAMA_MTP_VERIFY_THREE_OVERLAP");
            const char * fast = std::getenv("LLAMA_MTP_VERIFY_THREE_FAST");
            return value && std::strcmp(value, "1") == 0 && fast && std::strcmp(fast, "1") == 0;
        }();
        bool overlap_three_fit = false;
        const int32_t wave = mtp_verify ? int32_t(src == l.mtp_verify_maps[1]) :
                static_cast<int32_t>(wave_it - l.prefill_maps.begin());

        GGML_ASSERT(backend == devices[g.id].backend);
        // The scheduler's destination wait must commit the preceding promotion.
        GGML_ASSERT(!pending_writeback);

        if (src == l.hip_mtp_full_map) {
            GGML_ASSERT(pending_mtp_full_layer == il && pending_layer == -1);
            telemetry_scope scope(set_telemetry_cause, "slot_map");
            ggml_backend_tensor_set(dst, pending_mtp_full_slots.data(), 0, ggml_nbytes(dst));
            pending_mtp_full_layer = -1;
            return true;
        }

        auto find_lookup = [&]() -> ggml_tensor * {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                    return node;
                }
            }
            return nullptr;
        };

        if (placement && (placement->overlap || placement->hip_mtp_overlap) && has_cpu) {
            // The hot split does not read the slots being uploaded.
            GGML_ASSERT(pending_layer == il);
            if (hip_upload_backend) {
                wait_hip_cold_upload();
            } else if (!fills.empty() && pending_upload_bytes == 0) {
                // A failed upload can leave the old cold slot only partly intact.
                for (const auto & fill : fills) {
                    l.packed_scratch[fill.slot] = -1;
                }
                const int64_t start = ggml_time_us();
                pending_upload_bytes = upload_packed(backend, l, il, true);
                packed_wait_us += ggml_time_us() - start;
                ++packed_wait_count;
                packed_early_upload_steps += early;
            }
            if (early) {
                // Input copies can reuse memory. Publish the map in the normal callback.
                return true;
            }
            pending_layer = -1;
            l.packed_scratch.swap(packed_cold_scratch);
            {
                telemetry_scope scope(set_telemetry_cause, "slot_map");
                ggml_backend_tensor_set(dst, packed_cold_slots.data(), 0, ggml_nbytes(dst));
            }
            if (placement->hip_overlap_promotion) {
                const auto & map = placement->maps[il];
                for (const auto & fill : fills) {
                    LLAMA_LOG_DEBUG("packed expert HIP cold upload: layer=%d expert=%d scratch_slot=%d host_slot=%d\n",
                            il, fill.expert, fill.slot, map.host[fill.expert]);
                }
                commit_hip_promotion(l);
            }
            if (placement->hip_mtp_overlap) {
                if (placement->hip_mtp_compact) {
                    auto * lookup = find_lookup();
                    GGML_ASSERT(lookup);
                    lookup->op_params[0] = pending_mtp_miss_mask;
                    packed_mtp_hot_routes += pending_permanent_hits + pending_scratch_hits;
                    packed_mtp_miss_routes += pending_request_misses;
                    packed_mtp_empty_groups += pending_mtp_miss_mask == 0 || pending_mtp_miss_mask == ((uint32_t{1} << 30) - 1);
                }
                if (placement->hip_mtp_gate_up_overlap) {
                    const bool all_hit = pending_mtp_miss_mask == 0;
                    const bool all_miss = pending_mtp_miss_mask == ((uint32_t{1} << 30) - 1);
                    packed_mtp_gate_up_all_hit += all_hit;
                    packed_mtp_gate_up_all_miss += all_miss;
                    packed_mtp_gate_up_mixed += !all_hit && !all_miss;
                    pending_mtp_full_layer = il;
                }
                if (pending_mtp_evict >= 0) {
                    promote_packed(l, il, pending_mtp_promote, pending_mtp_evict);
                }
                packed_overlap_uploads += pending_upload_bytes != 0;
                packed_overlap_upload_bytes += pending_upload_bytes;
                ggml_backend_sched_trace_expert_misses(sched, il, 3, fills.size(), pending_upload_bytes);
                if (packed_miss_histogram) {
                    packed_transfer_totals.requests += pending_permanent_hits + pending_scratch_hits + pending_request_misses;
                    packed_transfer_totals.misses += pending_request_misses;
                    GGML_ASSERT(fills.size() < packed_miss_counts.size());
                    ++packed_miss_counts[fills.size()];
                    packed_miss_bytes += pending_upload_bytes;
                    if (fills.size() >= 3) {
                        packed_miss_bytes_3plus += pending_upload_bytes;
                    }
                }
            }
            stats_small.hits += pending_permanent_hits + pending_scratch_hits;
            packed_scratch_hits += pending_scratch_hits;
            packed_request_misses += pending_request_misses;
            stats_small.misses += fills.size();
            stats_small.bytes += pending_upload_bytes;
            LLAMA_LOG_DEBUG("packed experts: layer %d: %zu permanent hits, %zu scratch hits, %zu RAM uploads, %zu bytes\n",
                    il, pending_permanent_hits, pending_scratch_hits, fills.size(), pending_upload_bytes);
            return true;
        }

        abandon_hip_mtp_upload();
        const bool hip_mtp_hot = placement && src == l.hip_mtp_hot_map;
        if (hip_mtp_hot && (!sched || ggml_backend_sched_get_n_copies(sched) != 1)) {
            throw std::runtime_error("HIP MTP3 expert overlap requires a single-copy scheduler");
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        auto * lookup = find_lookup();
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(sel->type == GGML_TYPE_I32 && ggml_is_contiguous(sel));
        // A cross-device split receives a plain tensor copy of the reshaped IDs.
        GGML_ASSERT(placement || sel->op == GGML_OP_RESHAPE);
        const int32_t n_used = placement ? n_expert_used : (int32_t) sel->src[0]->ne[0];

        const size_t ids_bytes = ggml_nbytes(sel);
        const bool fine_profile = callback_start && placement && placement->hip_mtp_verify_three &&
                !placement->overlap && !placement->hip_mtp_overlap &&
                (packed_staging || placement->hip_tuple_dma) && src == l.slot_map &&
                ggml_nelements(sel) == 30;
        auto & callback_totals = next_profile_start ? next_profile_callback : packed_staging_totals;
        struct callback_measurement {
            int64_t start;
            llama_expert_staging_telemetry & totals;
            llama_expert_staging_telemetry previous;
            bool mirror;
            ~callback_measurement() {
                if (start) {
                    ++totals.callback_calls;
                    totals.callback_us += ggml_time_us() - start;
                }
                if (mirror) {
                    packed_staging_totals.callback_calls += totals.callback_calls - previous.callback_calls;
                    packed_staging_totals.callback_us += totals.callback_us - previous.callback_us;
                    packed_staging_totals.router_us += totals.router_us - previous.router_us;
                    packed_staging_totals.plan_us += totals.plan_us - previous.plan_us;
                    packed_staging_totals.submit_us += totals.submit_us - previous.submit_us;
                    packed_staging_totals.completion_us += totals.completion_us - previous.completion_us;
                    packed_staging_totals.promotion_us += totals.promotion_us - previous.promotion_us;
                    packed_staging_totals.map_us += totals.map_us - previous.map_us;
                }
            }
        };
        const bool mirror_profile = fine_profile && next_profile_start && profile_packed_callback;
        callback_measurement callback_measure{fine_profile ? callback_start : 0, callback_totals,
            mirror_profile ? callback_totals : llama_expert_staging_telemetry{}, mirror_profile};
        const int64_t router_begin = fine_profile ? ggml_time_us() : 0;
        ids.resize(ggml_nelements(sel));
        if (use_waves && wave > 0) {
            GGML_ASSERT(l.prefill_ids.size() == ids.size());
            ids = l.prefill_ids;
        } else if (placement && sched && ggml_backend_sched_get_host_copy(sched, sel, ids.data(), ids_bytes)) {
            // The scheduler waits for older work on this backend before exposing a host copy.
            ++router_host_hits;
            router_readback_elided_bytes += ids_bytes;
        } else {
            router_host_fallbacks += placement != nullptr;
            const int64_t router_start_us = profile ? ggml_time_us() : 0;
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ids_bytes);
            ggml_backend_synchronize(backend);
            if (profile) {
                profile_router_us += ggml_time_us() - router_start_us;
                profile_router_reads++;
            }
        }

        const int64_t plan_begin = fine_profile ? ggml_time_us() : 0;
        if (fine_profile) {
            callback_totals.router_us += plan_begin - router_begin;
        }
        if (placement) {
            const auto & map = placement->maps[il];
            const bool trace_generation = !generation_trace_positions.empty();
            const auto & trace_positions = trace_generation ? generation_trace_positions : prompt_trace_positions;
            std::unique_ptr<std::ostringstream> trace_row;
            size_t trace_uploads = 0;
            size_t trace_tuple_bytes = 0;
            if (!trace_positions.empty()) {
                if (ids.size() != trace_positions.size() * n_used) {
                    throw std::runtime_error("packed expert trace Router ID shape mismatch");
                }
                trace_row = std::make_unique<std::ostringstream>();
                auto & row = *trace_row;
                row << "{\"type\":\"layer\",\"request\":" << (trace_generation ? generation_trace_request : prompt_trace_request_id);
                if (trace_generation) {
                    generation_trace_memcpy_us = 0;
                    generation_trace_upload_wait_us = 0;
                    row << ",\"step\":" << generation_trace_step
                        << ",\"phase\":\"" << (generation_trace_verify_width ? "verify" : "decode") << "\""
                        << ",\"verify_width\":" << generation_trace_verify_width;
                } else {
                    row << ",\"segment\":" << prompt_trace_segment;
                }
                row << ",\"layer\":" << il << ",\"positions\":";
                prompt_trace_array(row, trace_positions);
                row << ",\"selected\":[";
                for (size_t token = 0; token < trace_positions.size(); ++token) {
                    row << (token ? ",[" : "[");
                    for (int rank = 0; rank < n_used; ++rank) {
                        row << (rank ? "," : "") << ids[token * n_used + rank];
                    }
                    row << ']';
                }
                std::vector<int32_t> resident, resident_slots, scratch_slots;
                for (size_t expert = 0; expert < map.resident.size(); ++expert) {
                    if (map.resident[expert] >= 0) {
                        resident.push_back(expert);
                        resident_slots.push_back(map.resident[expert]);
                    }
                }
                for (size_t slot = 0; slot < l.packed_scratch.size(); ++slot) {
                    scratch_slots.push_back(packed_slot(l, slot));
                }
                row << "],\"resident\":";
                prompt_trace_array(row, resident);
                row << ",\"resident_slots\":";
                prompt_trace_array(row, resident_slots);
                row << ",\"scratch\":";
                prompt_trace_array(row, l.packed_scratch);
                row << ",\"scratch_slots\":";
                prompt_trace_array(row, scratch_slots);
                row << ",\"strides\":[";
                for (size_t plane = 0; plane < l.experts.size(); ++plane) {
                    const auto * weight = l.experts[plane];
                    const size_t payload = ggml_row_size(weight->type, weight->ne[0]) * weight->ne[1];
                    row << (plane ? "," : "") << payload;
                    trace_tuple_bytes += payload;
                }
                if (placement->hip_tuple_dma) {
                    trace_tuple_bytes = placement->tuples[il].stride;
                }
                row << "],\"uploads\":[";
            }
            auto slots = map.resident;
            int32_t n_uploads = 0;
            size_t bytes = 0;
            size_t hits = 0;
            size_t scratch_hits = 0;
            const int64_t start = ggml_time_us();
            int64_t upload_start_us = 0;
            // Non-verification microbatches have at most two positions. Wide prefill uses its own path.
            const bool fixed_three = placement->hip_mtp_verify_three && src == l.slot_map && ids.size() == 30;
            const bool fixed_four = mtp_four && placement->hip_mtp_verify_four && src == l.slot_map && ids.size() == 40;
            const bool hip_direct = placement->hip_mtp_direct_ram && fixed_three;
            const bool hip_async_slot_map = ((fixed_three && hip_mtp_slot_async) || (fixed_four && hip_mtp4_slot_async)) &&
                    hip_mtp_slot_staging && sched &&
                    ggml_backend_sched_get_n_copies(sched) == 1 && !rtx_expert;
            const bool batch = packed_staging != nullptr || hip_direct || placement->hip_tuple_dma;
            if (batch) {
                fills.clear();
            }
            if (placement->overlap) {
                // An interrupted graph can leave an unconsumed plan.
                wait_hip_cold_upload();
                hip_pending_promotion.n_planes = 0;
                pending_layer = -1;
                pending_upload_bytes = 0;
                packed_cold_scratch = l.packed_scratch;
                packed_cold_slots.fill(placement->n_slots);
            }
            for (const int32_t expert : ids) {
                if (expert < 0 || static_cast<size_t>(expert) >= slots.size()) {
                    GGML_ABORT("invalid packed expert ID in layer %d", il);
                }
            }
            if (!mtp_pair_histograms.empty() && fixed_mtp_pair && src == l.slot_map && ids.size() == 2 * mtp_pair_experts) {
                // Top-k produces distinct IDs in each row. The packed lookup flattens the rows in token order.
                const auto second = ids.begin() + mtp_pair_experts;
                size_t common = 0;
                for (auto first = ids.begin(); first != second; ++first) {
                    common += std::find(second, ids.end(), *first) != ids.end();
                }
                ++mtp_pair_histograms[il][common];
            }
            if (use_waves) {
                if (wave == 0) {
                    // Each expert belongs to one wave, even if several tokens selected it.
                    l.prefill_ids = ids;
                    l.prefill_wave.fill(-1);
                    int32_t cold_count = 0;
                    if (mtp_verify) {
                        GGML_ASSERT(ids.size() == 3 * size_t(n_used));
                        // Consume selected scratch hits before replacing their slots.
                        for (const int32_t expert : l.packed_scratch) {
                            if (expert >= 0 && map.resident[expert] < 0 &&
                                    std::find(ids.begin(), ids.end(), expert) != ids.end()) {
                                l.prefill_wave[expert] = 0;
                                ++cold_count;
                            }
                        }
                    }
                    for (const int32_t expert : ids) {
                        if (l.prefill_wave[expert] < 0) {
                            l.prefill_wave[expert] = map.resident[expert] >= 0 ? 0 : cold_count++ / placement->n_slots;
                        }
                    }
                    if (mtp_verify) {
                        ++mtp_verify_layers;
                        mtp_verify_overflow_layers += cold_count > placement->n_slots;
                        overlap_three_fit = overlap_three && cold_count <= placement->n_slots;
                        mtp_overlap_fit_layers += overlap_three_fit;
                        mtp_verify_max_cold = std::max(mtp_verify_max_cold, cold_count);
                        if (sched && cold_count <= placement->n_slots) {
                            ggml_backend_sched_mark_mtp_three_empty_wave(sched, graph);
                        }
                    } else {
                        ++prefill_layer_batches;
                        prefill_max_tokens = std::max(prefill_max_tokens, ids.size() / n_used);
                        prefill_unique_cold += cold_count;
                        prefill_waves_needed += std::max(1, (cold_count + placement->n_slots - 1) / placement->n_slots);
                        prefill_waves_scheduled += placement->prefill_waves(ids.size() / n_used, n_used);
                    }
                }
                ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int32_t expert) {
                    return l.prefill_wave[expert] != wave;
                }), ids.end());
                if (wave > 0 && ids.empty() && (mtp_verify || prefill_active_waves)) {
                    constexpr const char accumulator_name[] = "ffn_moe_prefill_accumulator";
                    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                        auto * node = ggml_graph_node(graph, i);
                        if (std::strncmp(node->name, accumulator_name, sizeof(accumulator_name) - 1) == 0 &&
                                sched && ggml_backend_sched_carry_accumulator(sched, graph, node)) {
                            prefill_waves_skipped += prefill;
                            if (mtp_verify && node->data == node->src[0]->data) {
                                ++mtp_verify_empty_no_copy;
                            }
                            return true;
                        }
                    }
                    if (mtp_verify) {
                        throw std::runtime_error("could not skip the empty 3-position MTP Expert wave");
                    }
                }
                prefill_waves_executed += prefill;
                mtp_verify_second_passes += mtp_verify && wave > 0;
            }
            if ((placement->dynamic_resident || placement->hip_overlap_promotion) && (!use_waves || wave == 0)) {
                ++l.packed_lookups;
                if (placement->prefill_tokens == 512 && prompt_active) {
                    ++prefill_native_steps;
                }
                for (const int32_t expert : use_waves ? l.prefill_ids : ids) {
                    l.packed_last_used[expert] = l.packed_lookups;
                }
            }
            int32_t promote_index = -1;
            if (hip_mtp_hot) {
                packed_cold_scratch = l.packed_scratch;
            }
            auto & scratch = placement->overlap || hip_mtp_hot ? packed_cold_scratch : l.packed_scratch;
            // Mark all selected scratch residents before choosing any slot to replace.
            for (size_t i = 0; i < scratch.size(); ++i) {
                const int32_t expert = scratch[i];
                if (expert >= 0 && std::find(ids.begin(), ids.end(), expert) != ids.end()) {
                    slots[expert] = packed_slot(l, static_cast<int32_t>(i));
                    if (promote_index < 0) {
                        promote_index = static_cast<int32_t>(i);
                    }
                    ++scratch_hits;
                }
            }
            // Count requests before fills turn repeated misses into slot-map hits.
            size_t request_misses = 0;
            if (placement->max_ubatch > 1) {
                scratch_hits = 0;
                for (const int32_t expert : ids) {
                    if (map.resident[expert] < 0) {
                        if (slots[expert] >= 0) {
                            ++scratch_hits;
                        } else {
                            ++request_misses;
                        }
                    }
                }
            }
            std::array<int32_t, llama_expert_profile::n_experts> rtx_slots;
            size_t n_rtx = 0;
            size_t rtx_bytes = 0;
            if (mtp_verify && rtx_expert) {
                throw std::runtime_error("3-position MTP is incompatible with RTX Expert sharing");
            }
            if (rtx_expert) {
                if (!profile && sched && ggml_backend_sched_get_n_copies(sched) == 1) {
                    rtx_slots.fill(-1);
                    n_rtx = submit_rtx_misses(backend, sched, il, graph, dst, l, n_used, slots, rtx_slots, rtx_bytes);
                }
                if (n_rtx == 0) {
                    rtx_expert->clear(il, graph, dst);
                }
            }
            for (const int32_t expert : ids) {
                if (n_rtx && rtx_slots[expert] >= 0) {
                    continue;
                }
                if (slots[expert] >= 0) {
                    hits += map.resident[expert] >= 0;
                    continue;
                }
                if (n_uploads == placement->n_slots) {
                    GGML_ABORT("packed expert scratch slots exhausted in layer %d", il);
                }
                auto victim = std::find(scratch.begin(), scratch.end(), -1);
                if (victim == scratch.end()) {
                    victim = std::find_if(scratch.begin(), scratch.end(), [&](int32_t resident) {
                        return slots[resident] < 0;
                    });
                }
                if (victim == scratch.end()) {
                    GGML_ABORT("packed expert scratch slots exhausted in layer %d", il);
                }
                const int32_t scratch_index = static_cast<int32_t>(victim - scratch.begin());
                // A partial upload must not leave the slot mapped to the evicted expert.
                *victim = -1;
                ++n_uploads;
                if (profile && n_uploads == 1) {
                    upload_start_us = ggml_time_us();
                }
                const int32_t slot = packed_slot(l, scratch_index);
                if (trace_row) {
                    auto & row = *trace_row;
                    row << (trace_uploads++ ? "," : "") << "{\"expert\":" << expert << ",\"slot\":" << slot;
                    if (trace_generation) {
                        const bool prefetched = next_prefetch_slot(il, expert) >= 0;
                        row << ",\"bytes\":" << (prefetched ? 0 : trace_tuple_bytes);
                        if (prefetched) {
                            row << ",\"prefetched\":true";
                        }
                    }
                    row << '}';
                }
                if (batch) {
                    fills.push_back({expert, scratch_index});
                } else {
                    telemetry_scope scope(set_telemetry_cause, "expert_upload");
                    for (ggml_tensor * host : l.experts) {
                        ggml_tensor * bank = bindings.at(host).cached;
                        const size_t stride = host->nb[2];
                        const auto * data = packed_host_data(l, host, map.host[expert], bindings.at(host).packed_offset);
                        ggml_backend_tensor_set(bank, data, slot * stride, stride);
                        bytes += stride;
                    }
                    if (placement->dynamic_resident) {
                        LLAMA_LOG_DEBUG("packed expert serial upload: layer=%d expert=%d scratch_slot=%d host_slot=%d\n",
                                il, expert, slot, map.host[expert]);
                    }
                }
                *victim = expert;
                slots[expert] = slot;
            }
            if (hip_mtp_hot) {
                GGML_ASSERT(ids.size() == 30 && n_used == 10);
                const int32_t zero_slot = llama_expert_profile::n_resident + placement->n_slots;
                auto hot_slots = slots;
                packed_cold_slots.fill(zero_slot);
                for (const auto & fill : fills) {
                    packed_cold_slots[fill.expert] = slots[fill.expert];
                    hot_slots[fill.expert] = zero_slot;
                    l.packed_scratch[fill.slot] = -1;
                }
                if (placement->hip_mtp_compact) {
                    uint32_t hot_mask = 0;
                    for (size_t rank = 0; rank < ids.size(); ++rank) {
                        if (hot_slots[ids[rank]] != zero_slot) {
                            hot_mask |= uint32_t{1} << rank;
                        }
                    }
                    lookup->op_params[0] = hot_mask;
                    pending_mtp_miss_mask = ((uint32_t{1} << 30) - 1) ^ hot_mask;
                }
                if (placement->hip_mtp_gate_up_overlap) {
                    pending_mtp_full_slots = slots;
                }
                {
                    telemetry_scope scope(set_telemetry_cause, "slot_map");
                    ggml_backend_tensor_set(dst, hot_slots.data(), 0, ggml_nbytes(dst));
                }
                pending_permanent_hits = hits;
                pending_scratch_hits = scratch_hits;
                pending_request_misses = request_misses;
                pending_mtp_promote = promote_index;
                pending_mtp_evict = promote_index >= 0 ? packed_promotion_victim(l, map) : -1;
                pending_upload_bytes = 0;
                if (!fills.empty()) {
                    // Both the selected scratch hits and the zero slot stay outside the upload destinations.
                    const int64_t upload_start = ggml_time_us();
                    pending_upload_bytes = upload_packed(hip_upload_backend.get(), l, il, false, nullptr, 0, true);
                    packed_wait_us += ggml_time_us() - upload_start;
                    ++packed_wait_count;
                }
                pending_layer = il;
                return true;
            }
            if (placement->overlap) {
                // RAM experts use their cold slot and the zero slot in the resident bank.
                auto resident_slots = map.resident;
                for (const int32_t expert : ids) {
                    if (map.resident[expert] < 0) {
                        packed_cold_slots[expert] = slots[expert] - llama_expert_profile::n_resident;
                        resident_slots[expert] = llama_expert_profile::n_resident;
                    }
                }
                {
                    telemetry_scope scope(set_telemetry_cause, "slot_map");
                    ggml_backend_tensor_set(dst, resident_slots.data(), 0, ggml_nbytes(dst));
                }
                pending_layer = il;
                pending_permanent_hits = hits;
                pending_scratch_hits = scratch_hits;
                pending_request_misses = request_misses;
                const int32_t hip_evict = placement->hip_overlap_promotion && promote_index >= 0 ?
                        packed_promotion_victim(l, map) : -1;
                if (hip_upload_backend && (!fills.empty() || hip_evict >= 0)) {
                    // Older cold readers have completed before this callback. Do not retain slots from an abandoned plan.
                    for (const auto & fill : fills) {
                        l.packed_scratch[fill.slot] = -1;
                    }
                    const int64_t upload_start = ggml_time_us();
                    const bool async = placement->hip_overlap == llama_expert_placement::hip_overlap_mode::async;
                    try {
                        if (!fills.empty()) {
                            pending_upload_bytes = upload_packed(hip_upload_backend.get(), l, il, true, nullptr, 0, true);
                        }
                        if (hip_evict >= 0) {
                            submit_hip_promotion(l, il, promote_index, hip_evict);
                        }
                    } catch (...) {
                        ggml_backend_synchronize(hip_upload_backend.get());
                        throw;
                    }
                    if (!async) {
                        ggml_backend_synchronize(hip_upload_backend.get());
                    }
                    packed_wait_us += ggml_time_us() - upload_start;
                    ++packed_wait_count;
                    packed_early_upload_steps += async;
                }
                return true;
            }
            if (use_waves) {
                for (size_t expert = 0; expert < slots.size(); ++expert) {
                    if (l.prefill_wave[expert] != wave) {
                        slots[expert] = -1;
                    }
                }
            }
            const int32_t evict = placement->dynamic_resident && promote_index >= 0 ? packed_promotion_victim(l, map) : -1;
            if (fine_profile) {
                callback_totals.plan_us += ggml_time_us() - plan_begin;
            }
            const auto write_slot_map = [&] {
                telemetry_scope scope(set_telemetry_cause, "slot_map");
                const int64_t map_start = fine_profile ? ggml_time_us() : 0;
                const size_t bytes_map = ggml_nbytes(dst);
                if (hip_async_slot_map && dst->buffer &&
                        ggml_backend_buffer_get_type(dst->buffer) == ggml_backend_get_default_buffer_type(backend)) {
                    GGML_ASSERT(bytes_map == sizeof(slots));
                    auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(hip_mtp_slot_staging.get()));
                    staging += il * sizeof(slots);
                    // This callback's destination wait has drained any prior use of this layer's staging.
                    std::memcpy(staging, slots.data(), bytes_map);
                    ggml_backend_tensor_set_async(backend, dst, staging, 0, bytes_map);
                    if (fixed_four) {
                        ++hip_mtp4_async_maps;
                    } else {
                        ++hip_mtp_async_maps;
                        hip_mtp_async_map_bytes += bytes_map;
                    }
                } else {
                    ggml_backend_tensor_set(dst, slots.data(), 0, bytes_map);
                    packed_synchronous_maps += defer_slot_map_requested;
                }
                if (fine_profile) {
                    callback_totals.map_us += ggml_time_us() - map_start;
                }
            };
            bool upload_submitted = false;
            bool overlap_submitted = false;
            bool hip_single_kernel_submitted = false;
            bool slot_map_written = false;
            const bool can_overlap = !hip_direct && (!use_waves || overlap_three_fit) && overlap_expert_upload &&
                    !profile && sched && ggml_backend_sched_get_n_copies(sched) == 1;
            const bool can_overlap_promotion = can_overlap && overlap_expert_promotion && evict >= 0;
            if (batch && (!fills.empty() || can_overlap_promotion)) {
                // A single-copy scheduler synchronizes the backend before each host-weight callback.
                const bool can_defer = !hip_direct && submit_expert_upload && !profile && sched &&
                        ggml_backend_sched_get_n_copies(sched) == 1 && promote_index < 0;
                try {
                    size_t overlap_copied_bytes = 0;
                    if (can_overlap && (evict < 0 || can_overlap_promotion)) {
                        mtp_overlap_attempts += overlap_three_fit;
                        overlap_submitted = overlap_packed_upload(backend, graph, dst, l, il, n_used, slots, promote_index, evict, bytes, overlap_copied_bytes);
                        mtp_overlap_submitted += overlap_three_fit && overlap_submitted;
                    }
                    if (stage_hip_expert_overlap && fixed_three && !early && !profile && !use_waves && !rtx_expert &&
                            !fills.empty() && fills.size() <= std::size(hip_expert_overlap_staged) && sched && ggml_backend_sched_get_n_copies(sched) == 1 &&
                            std::strcmp(lookup->name, "ffn_moe_mtp_single_kernel_overlap") == 0) {
                        // The slot-map H2D must be submitted ahead of the large tuple copies.
                        write_slot_map();
                        slot_map_written = true;
                        bytes = stage_hip_single_kernel_upload(backend, graph, lookup, l, il, promote_index, evict);
                        hip_single_kernel_submitted = bytes != 0;
                    }
                    if (!hip_single_kernel_submitted && !overlap_submitted && !fills.empty()) {
                        bytes = upload_packed(backend, l, il, false, can_defer ? &upload_submitted : nullptr,
                                overlap_copied_bytes, false, hip_direct, fine_profile ? &callback_totals : nullptr);
                    }
                } catch (...) {
                    for (const auto & fill : fills) {
                        scratch[fill.slot] = -1;
                    }
                    throw;
                }
            }
            if (n_uploads > 0 && !hip_single_kernel_submitted && !upload_submitted && !overlap_submitted) {
                if (!batch) {
                    telemetry_scope scope(set_telemetry_cause, "expert_upload");
                    ggml_backend_synchronize(backend);
                }
                const int64_t end_us = ggml_time_us();
                packed_wait_us += end_us - start;
                ++packed_wait_count;
                if (profile) {
                    profile_upload_us += end_us - upload_start_us;
                    profile_upload_steps++;
                    profile_upload_bytes += bytes;
                }
            }
            if (il == next_prefetch_layer) {
                if (next_prefetch_hit_wait) {
                    retire_next_prefetch(0);
                } else {
                    finish_next_prefetch();
                }
            }
            if (evict >= 0 && !overlap_submitted) {
                const int64_t promotion_start = fine_profile ? ggml_time_us() : 0;
                if (((fixed_three && hip_mtp_promotion_async) || (fixed_four && hip_mtp4_promotion_async)) && !rtx_expert && sched &&
                        ggml_backend_sched_get_n_copies(sched) == 1) {
                    submit_hip_mtp_promotion(l, il, promote_index, evict, fixed_four);
                } else {
                    promote_packed(l, il, promote_index, evict);
                }
                if (fine_profile) {
                    callback_totals.promotion_us += ggml_time_us() - promotion_start;
                }
            }
            if (gate_up_diag_ids && fixed_mtp_pair && src == l.slot_map && ids.size() == 20) {
                int32_t selected_slots[20];
                for (uint32_t i = 0; i < ids.size(); ++i) {
                    selected_slots[i] = slots[ids[i]];
                }
                gate_up_diag_ids(backend, lookup, ids.data(), selected_slots);
            }
            // The scheduler synchronizes after each copy callback while profiling.
            if (overlap_submitted) {
                packed_overlap_uploads += bytes != 0;
                packed_overlap_upload_bytes += bytes;
                ++packed_deferred_maps;
            } else if (!slot_map_written && defer_slot_map && !profile && defer_slot_map(backend, dst, slots.data())) {
                ++packed_deferred_maps;
                if (upload_submitted) {
                    ++packed_deferred_uploads;
                    packed_deferred_upload_bytes += bytes;
                }
            } else {
                if (upload_submitted) {
                    telemetry_scope scope(set_telemetry_cause, "expert_upload");
                    ggml_backend_synchronize(backend);
                    packed_wait_us += ggml_time_us() - start;
                    ++packed_wait_count;
                    upload_submitted = false;
                }
                if (!slot_map_written) {
                    write_slot_map();
                }
            }
            packed_synchronous_uploads += defer_upload_requested && n_uploads > 0 && !hip_single_kernel_submitted && !upload_submitted && !overlap_submitted;
            stats_small.hits += hits + scratch_hits;
            packed_scratch_hits += scratch_hits;
            stats_small.misses += n_uploads;
            packed_request_misses += request_misses;
            stats_small.bytes += bytes;
            if (prompt_active) {
                ++prompt_layer_steps;
                prompt_uploads += n_uploads;
                prompt_upload_bytes += bytes;
            }
            ggml_backend_sched_trace_expert_misses(sched, il, n_used > 0 ? ids.size() / n_used : 0, n_uploads + n_rtx, bytes + rtx_bytes);
            if (packed_miss_histogram) {
                packed_transfer_totals.requests += hits + scratch_hits + request_misses;
                packed_transfer_totals.misses += request_misses;
                const size_t unique_misses = n_uploads + n_rtx;
                GGML_ASSERT(unique_misses < packed_miss_counts.size());
                ++packed_miss_counts[unique_misses];
                packed_miss_bytes += bytes + rtx_bytes;
                if (unique_misses >= 3) {
                    packed_miss_bytes_3plus += bytes + rtx_bytes;
                }
            }
            LLAMA_LOG_DEBUG("packed experts: layer %d: %zu permanent hits, %zu scratch hits, %d RAM uploads, %zu bytes\n",
                    il, hits, scratch_hits, n_uploads, bytes);
            if (trace_row) {
                *trace_row << "],\"bytes\":" << bytes;
                if (trace_generation) {
                    *trace_row << ",\"staging_memcpy_us\":" << generation_trace_memcpy_us
                               << ",\"upload_wait_us\":" << generation_trace_upload_wait_us
                               << ",\"direct_bytes\":" << (hip_direct || placement->hip_tuple_dma ? bytes : 0);
                }
                *trace_row << "}\n";
                if (trace_generation) {
                    write_expert_trace(generation_trace.get(), trace_row->str());
                } else {
                    write_prompt_trace(trace_row->str());
                    ++prompt_trace_rows;
                }
            }
            if (next_prefetch_predictions && fixed_three && !early && sched) {
                schedule_next_prefetch(sched, backend, il);
            }
            return true;
        }

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    void log_stats() const {
        if (next_profile_enabled) {
            const auto & c = next_profile_callback;
            LLAMA_LOG_INFO("packed expert HIP next Router profile: phase=verify3 prefetch=%d graphs=%" PRIu64
                    " failed=%" PRIu64 " graph_nodes=%" PRIu64 " splits=%" PRIu64 " scheduler_cpu_wall_us=%" PRId64
                    " hc_pre_nodes=%" PRIu64 " hc_comb_nodes=%" PRIu64 " hc_post_nodes=%" PRIu64 "\n",
                    next_prefetch_bank != nullptr, next_profile_graphs, next_profile_failed, next_profile_nodes,
                    next_profile_splits, next_profile_graph_us, next_profile_hc_pre, next_profile_hc_comb, next_profile_hc_post);
            LLAMA_LOG_INFO("packed expert HIP next Router profile: phase=verify3 layer_callbacks=%" PRIu64
                    " callback_cpu_wall_us=%" PRIu64 " router_read_cpu_wall_us=%" PRIu64 " plan_cpu_wall_us=%" PRIu64
                    " promotion_cpu_wall_us=%" PRIu64 " slot_map_cpu_wall_us=%" PRIu64 "\n",
                    c.callback_calls, c.callback_us, c.router_us, c.plan_us, c.promotion_us, c.map_us);
            LLAMA_LOG_INFO("packed expert HIP next Router profile: phase=verify3 demand_h2d_tuples=%" PRIu64
                    " demand_h2d_bytes=%" PRIu64 " demand_h2d_submit_cpu_us=%" PRIu64 " demand_stream_waits=%" PRIu64
                    " demand_stream_cpu_wait_us=%" PRIu64 "\n", next_profile_demand_count, next_profile_demand_bytes,
                    c.submit_us, next_profile_demand_waits, c.completion_us);
            LLAMA_LOG_INFO("packed expert HIP next Router profile: phase=verify3 prediction_reads=%" PRIu64
                    " prediction_read_submit_cpu_us=%" PRId64 " prediction_producer_cpu_wait_us=%" PRId64
                    " prefetch_h2d_submit_cpu_us=%" PRId64 " hit_d2d_submit_cpu_us=%" PRId64
                    " prefetch_stream_waits=%" PRIu64 " prefetch_stream_cpu_wait_us=%" PRId64 "\n",
                    next_profile_readbacks, next_profile_read_submit_us, next_profile_read_wait_us, next_profile_h2d_submit_us,
                    next_profile_d2d_submit_us, next_profile_prefetch_waits, next_prefetch_wait_us);
        }
        if (next_prefetch_bank) {
            LLAMA_LOG_INFO("packed expert HIP next Router prefetch: launches=%zu hits=%zu unused=%zu h2d_bytes=%zu d2d_bytes=%zu wait_us=%" PRId64 "\n",
                    next_prefetch_launches, next_prefetch_hits, next_prefetch_unused,
                    next_prefetch_h2d_bytes, next_prefetch_d2d_bytes, next_prefetch_wait_us);
            LLAMA_LOG_INFO("packed expert HIP next Router hit wait: enabled=%d dependencies=%zu dependency_prefix_slots=%zu"
                    " consumer_cpu_wait_us=%" PRId64 " source_waits=%zu source_cpu_wait_us=%" PRId64
                    " tail_waits=%zu tail_cpu_wait_us=%" PRId64 "\n",
                    next_prefetch_hit_wait, next_prefetch_dependencies, next_prefetch_dependency_slots,
                    next_prefetch_consumer_wait_us, next_prefetch_source_waits, next_prefetch_source_wait_us,
                    next_prefetch_tail_waits, next_prefetch_tail_wait_us);
            LLAMA_LOG_INFO("packed expert HIP next Router after-submit: enabled=%d scheduled=%zu executed=%zu fallback=%zu canceled=%zu"
                    " launch_cpu_wall_us=%" PRId64 "\n", next_prefetch_after_submit, next_prefetch_submit_scheduled,
                    next_prefetch_submit_executed, next_prefetch_submit_fallback, next_prefetch_submit_canceled, next_prefetch_submit_cpu_us);
            if (next_prefetch_score_mode == 1) {
                LLAMA_LOG_INFO("packed expert HIP next Router score filter: eligible=%zu rejected=%zu empty_layers=%zu"
                        " readback_fallbacks=%zu row_fallbacks=%zu\n", next_prefetch_score_eligible, next_prefetch_score_rejected,
                        next_prefetch_score_empty, next_prefetch_score_readback_fallback, next_prefetch_score_row_fallback);
            }
            if (next_prefetch_score_mode == 2) {
                LLAMA_LOG_INFO("packed expert HIP next Router score priority: layers=%zu fallbacks=%zu top20_only=%zu top20_only_hits=%zu\n",
                        next_prefetch_priority_layers, next_prefetch_priority_fallbacks,
                        next_prefetch_top20_only, next_prefetch_top20_only_hits);
            }
        }
        for (size_t il = 0; il < mtp_pair_histograms.size(); ++il) {
            if (!layers[il].slot_map) {
                continue;
            }
            const auto & histogram = mtp_pair_histograms[il];
            std::ostringstream bins;
            uint64_t samples = 0;
            for (size_t i = 0; i < histogram.size(); ++i) {
                bins << (i == 0 ? "" : ",") << histogram[i];
                samples += histogram[i];
            }
            LLAMA_LOG_INFO("MTP pair expert overlap: layer=%zu samples=%" PRIu64 " bins=%s\n",
                    il, samples, bins.str().c_str());
        }
        if (mtp_verify_three_enabled) {
            LLAMA_LOG_INFO("MTP 3-position expert: layers=%zu overflow_layers=%zu max_nonresident=%d\n",
                    mtp_verify_layers, mtp_verify_overflow_layers, mtp_verify_max_cold);
            LLAMA_LOG_INFO("MTP 3-position waves: second_passes=%zu empty_no_copy=%zu\n",
                    mtp_verify_second_passes, mtp_verify_empty_no_copy);
            LLAMA_LOG_INFO("MTP 3-position overlap: fit_layers=%zu attempts=%zu submitted=%zu\n",
                    mtp_overlap_fit_layers, mtp_overlap_attempts, mtp_overlap_submitted);
        }
        if (placement && prompt_stats) {
            LLAMA_LOG_INFO("packed expert prompt uploads: capacity=%d ubatches=%" PRIu64 " tokens=%" PRIu64
                    " max_tokens=%" PRIu64 " decode_max_tokens=%" PRIu64 " layer_steps=%" PRIu64 " uploads=%" PRIu64 " bytes=%" PRIu64
                    " compact_chunks=%" PRIu64 " workspace_bytes=%zu\n",
                    placement->prefill_tokens ? placement->prefill_tokens : placement->max_ubatch,
                    prompt_ubatches, prompt_tokens, prompt_max_tokens, decode_max_tokens, prompt_layer_steps, prompt_uploads, prompt_upload_bytes,
                    compact_chunks, compact_workspace ? ggml_backend_buffer_get_size(compact_workspace.get()) : 0);
            if (placement->hip_prefill != llama_expert_placement::hip_overlap_mode::off) {
                LLAMA_LOG_INFO("packed expert HIP prefill overlap10: mode=%s scratch_slots=%d bank_slots=10"
                        " transfer_submits=%" PRIu64 " transfer_bytes=%" PRIu64 " compute_windows=%" PRIu64 "\n",
                        placement->hip_prefill == llama_expert_placement::hip_overlap_mode::async ? "overlap" : "serial",
                        placement->n_slots, prefill_overlap_submits, prefill_overlap_bytes, prefill_overlap_windows);
            } else if (prefill_overlap_ten) {
                LLAMA_LOG_INFO("packed expert prefill overlap10: enabled=1 job=%d scratch_slots=20 bank_slots=10"
                        " transfer_submits=%" PRIu64 " transfer_bytes=%" PRIu64 " compute_windows=%" PRIu64
                        " pending_windows=%" PRIu64 " fallback_groups=%" PRIu64 " fallback_layers=%" PRIu64 "\n",
                        compact_upload_job != nullptr, prefill_overlap_submits, prefill_overlap_bytes, prefill_overlap_windows,
                        prefill_overlap_pending_windows, prefill_overlap_fallbacks, prefill_overlap_fallback_layers);
            }
            if (placement->prefill_tokens == 512) {
                LLAMA_LOG_INFO("packed expert prefill ticks: enabled=%d layer_steps=%" PRIu64 " tokens=%" PRIu64
                        " advanced_ticks=%" PRIu64 " min_delta=%" PRIu64 " max_delta=%" PRIu64
                        " last_use_age_sum=%" PRIu64 " native_steps=%" PRIu64 " native_ticks=%" PRIu64 "\n",
                        prefill_logical_ticks, prefill_tick_steps, prefill_tick_tokens, prefill_tick_advanced,
                        prefill_tick_min_delta, prefill_tick_max_delta, prefill_tick_age_sum, prefill_native_steps, prefill_native_steps);
            }
        }
        if (profile_router_reads > 0 || profile_upload_steps > 0) {
            LLAMA_LOG_INFO("decode_profile: expert_router_readback_ms=%.3f router_reads=%zu expert_upload_ms=%.3f upload_steps=%zu uploaded_MiB=%.3f\n",
                    profile_router_us / 1000.0, profile_router_reads, profile_upload_us / 1000.0,
                    profile_upload_steps, profile_upload_bytes / (1024.0 * 1024.0));
        }
        if (placement) {
            LLAMA_LOG_INFO("packed expert router: host_hits=%zu fallback_reads=%zu elided_readback_bytes=%zu\n",
                    router_host_hits, router_host_fallbacks, router_readback_elided_bytes);
        }
        if (placement) {
            LLAMA_LOG_INFO("packed expert transfer phase (%s): %.3f ms across %zu layer steps\n",
                    placement->overlap ? "overlap" : "sync", packed_wait_us / 1000.0, packed_wait_count);
            if (placement->hip_mtp_direct_ram) {
                LLAMA_LOG_INFO("packed expert HIP MTP3 direct RAM: steps=%zu planes=%zu bytes=%zu staging_bytes=0 completion_wait_us=%" PRIu64 "\n",
                        hip_direct_upload_steps, hip_direct_upload_planes, hip_direct_upload_bytes, hip_direct_upload_wait_us);
            }
            if (hip_mtp_slot_staging && hip_mtp_slot_async) {
                LLAMA_LOG_INFO("packed HIP MTP3 async slot map: copies=%zu bytes=%zu\n",
                        hip_mtp_async_maps, hip_mtp_async_map_bytes);
            }
            if (hip_mtp_slot_staging && hip_mtp4_slot_async) {
                LLAMA_LOG_INFO("packed HIP MTP4 async slot map: copies=%zu bytes=%zu\n", hip_mtp4_async_maps,
                        hip_mtp4_async_maps * sizeof(placement->maps[0].resident));
            }
            if (hip_mtp4_promotion_async) {
                LLAMA_LOG_INFO("packed HIP MTP4 async promotion: submits=%zu commits=%zu bytes=%zu wait_ms=%.3f pending=%d\n",
                        hip_mtp4_promotion_submits, hip_mtp4_promotion_commits, hip_mtp4_promotion_bytes,
                        hip_mtp4_promotion_wait_us / 1000.0, hip_mtp_pending_four && hip_mtp_pending_move.n_planes != 0);
            }
            if (hip_mtp_promotion_async) {
                LLAMA_LOG_INFO("packed HIP MTP3 async promotion: submits=%zu commits=%zu bytes=%zu wait_ms=%.3f pending=%d\n",
                        hip_mtp_promotion_submits, hip_mtp_promotion_commits, hip_mtp_promotion_bytes,
                        hip_mtp_promotion_wait_us / 1000.0, hip_mtp_pending_move.n_planes != 0);
            }
            if (stage_hip_expert_overlap) {
                LLAMA_LOG_INFO("packed expert HIP MTP3 single-kernel overlap staged: cold1=%zu cold2=%zu cold3=%zu cold4=%zu\n",
                        hip_expert_overlap_staged[0], hip_expert_overlap_staged[1],
                        hip_expert_overlap_staged[2], hip_expert_overlap_staged[3]);
            }
            if (placement->hip_tuple_dma) {
                LLAMA_LOG_INFO("packed HIP Expert tuple DMA: h2d=%zu h2d_payload_bytes=%zu h2d_bytes=%zu compact_h2d=%zu"
                        " d2h=%zu d2h_payload_bytes=%zu d2h_bytes=%zu\n",
                        tuple_h2d_count, tuple_h2d_payload_bytes, tuple_h2d_bytes, tuple_prefill_h2d_count,
                        tuple_d2h_count, tuple_d2h_payload_bytes, tuple_d2h_bytes);
            }
            if (placement->overlap) {
                LLAMA_LOG_INFO("packed expert overlap uploads: early_steps=%zu waited_steps=%zu\n",
                        packed_early_upload_steps, packed_wait_count - packed_early_upload_steps);
            }
            if (placement->prefill_tokens) {
                LLAMA_LOG_INFO("packed expert prefill: layer_batches=%zu max_tokens=%zu unique_cold=%zu waves_needed=%zu waves_scheduled=%zu\n",
                        prefill_layer_batches, prefill_max_tokens, prefill_unique_cold, prefill_waves_needed, prefill_waves_scheduled);
                LLAMA_LOG_INFO("packed expert prefill waves: active_only=%d executed=%zu skipped=%zu\n",
                        prefill_active_waves, prefill_waves_executed, prefill_waves_skipped);
            }
            LLAMA_LOG_INFO("packed expert slots: reuse=%s permanent_hits=%zu scratch_hits=%zu ram_uploads=%zu upload_mode=%s\n",
                    "on", stats_small.hits - packed_scratch_hits, packed_scratch_hits, stats_small.misses,
                    placement->overlap ? "overlap" : placement->hip_tuple_dma ? "tuple" : packed_staging ? "batch" : "serial");
            if (defer_upload_requested) {
                LLAMA_LOG_INFO("packed expert upload deferral: deferred=%zu synchronous=%zu bytes=%zu\n",
                        packed_deferred_uploads, packed_synchronous_uploads, packed_deferred_upload_bytes);
            }
            if (overlap_upload_requested) {
                LLAMA_LOG_INFO("packed expert Vulkan upload overlap: submitted=%zu bytes=%zu\n",
                        packed_overlap_uploads, packed_overlap_upload_bytes);
            }
            if (placement->hip_mtp_overlap) {
                LLAMA_LOG_INFO("packed expert HIP MTP3 upload overlap: submitted=%zu bytes=%zu\n",
                        packed_overlap_uploads, packed_overlap_upload_bytes);
            }
            if (placement->hip_mtp_compact) {
                LLAMA_LOG_INFO("packed expert HIP MTP3 active routes: hot=%zu miss=%zu empty_groups=%zu\n",
                        packed_mtp_hot_routes, packed_mtp_miss_routes, packed_mtp_empty_groups);
            }
            if (placement->hip_mtp_gate_up_overlap) {
                LLAMA_LOG_INFO("packed expert HIP MTP3 gate/up overlap: all_hit_layers=%zu all_miss_layers=%zu mixed_layers=%zu"
                        " swiglu_per_layer=1 down_per_layer=1\n",
                        packed_mtp_gate_up_all_hit, packed_mtp_gate_up_all_miss, packed_mtp_gate_up_mixed);
                if (placement->hip_mtp_gate_up_fuse_hits) {
                    LLAMA_LOG_INFO("packed expert HIP MTP3 gate/up overlap: fused_hit_dispatches=%" PRIu64 "\n",
                            packed_mtp_gate_up_fused_hits);
                }
                if (placement->hip_mtp_gate_up_shared_output) {
                    LLAMA_LOG_INFO("packed expert HIP MTP3 shared gate/up output: all_hit_layers=%zu"
                            " all_miss_layers=%zu mixed_layers=%zu projection_dispatches=%" PRIu64 "\n",
                            packed_mtp_gate_up_all_hit, packed_mtp_gate_up_all_miss, packed_mtp_gate_up_mixed,
                            packed_mtp_gate_up_shared_dispatches);
                }
                if (placement->hip_mtp_gate_up_reuse_q8_1) {
                    LLAMA_LOG_INFO("packed expert HIP MTP3 shared gate/up Q8_1: quantize_dispatches=%" PRIu64
                            " projection_consumers=%" PRIu64 "\n",
                            packed_mtp_gate_up_q8_1_dispatches, packed_mtp_gate_up_shared_dispatches);
                }
            }
            if (overlap_upload_requested) {
                LLAMA_LOG_INFO("packed expert Vulkan promotion overlap: committed=%zu writeback_bytes=%zu\n",
                        packed_overlap_promotions, packed_overlap_writeback_bytes);
            }
            if (defer_slot_map_requested) {
                LLAMA_LOG_INFO("packed expert slot map updates: deferred=%zu synchronous=%zu\n",
                        packed_deferred_maps, packed_synchronous_maps);
            }
            if (placement->max_ubatch > 1) {
                LLAMA_LOG_INFO("packed expert requests: total=%zu misses=%zu hit_rate=%.2f%%\n",
                        stats_small.hits + packed_request_misses, packed_request_misses,
                        stats_small.hits ? 100.0 * stats_small.hits / (stats_small.hits + packed_request_misses) : 0.0);
            }
            if (rtx_expert) {
                LLAMA_LOG_INFO("packed expert RTX offload: eligible_steps=%zu submitted_steps=%zu fallback_steps=%zu unique_experts=%zu ranks=%zu uploaded_bytes=%zu\n",
                        rtx_eligible_steps, rtx_steps, rtx_eligible_steps - rtx_steps, rtx_unique_experts, rtx_ranks, rtx_uploaded_bytes);
            }
            if (profile_packed_ram_copy) {
                LLAMA_LOG_INFO("packed expert promotion RAM memcpy: calls=%zu bytes=%zu cpu_ms=%.3f\n",
                        packed_ram_memcpy_calls, packed_ram_memcpy_bytes, packed_ram_memcpy_us / 1000.0);
            }
            if (packed_ram_base) {
                LLAMA_LOG_INFO("packed expert RAM rotation: rotations=%zu bytes=%zu pool_bytes=%zu live_slots=%d restored_bytes=%zu\n",
                        packed_ram_rotations, packed_ram_rotated_bytes, packed_ram_capacity * packed_ram_stride,
                        packed_ram_next - 1, packed_ram_restored_bytes);
            }
            if (placement->dynamic_resident) {
                LLAMA_LOG_INFO("packed expert dynamic residency: promotions=%zu evictions=%zu writeback_bytes=%zu writeback_ms=%.3f uploaded_bytes=%zu total_transfer_bytes=%zu total_transfer_ms=%.3f\n",
                        packed_promotions, packed_promotions, packed_writeback_bytes, packed_writeback_us / 1000.0,
                        stats_small.bytes, stats_small.bytes + packed_writeback_bytes, (packed_wait_us + packed_writeback_us) / 1000.0);
            }
            if (placement->hip_overlap_promotion) {
                LLAMA_LOG_INFO("packed expert HIP promotions: promotions=%zu evictions=%zu writeback_bytes=%zu\n",
                        packed_promotions, packed_promotions, packed_writeback_bytes);
            }
        }
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        if (placement && placement->max_ubatch > 1) {
            auto requests = stats_small;
            requests.misses = packed_request_misses;
            log("ubatch <= 8", requests);
        } else {
            log("ubatch <= 8", stats_small);
        }
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool mtp_verify_three_enabled) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share, mtp_verify_three_enabled)) {
}

llama_moe_cache::~llama_moe_cache() = default;

llama_expert_transfer_telemetry llama_get_expert_transfer_telemetry() {
    return packed_transfer_totals;
}

llama_expert_staging_telemetry llama_get_expert_staging_telemetry() {
    return packed_staging_totals;
}

int64_t llama_set_expert_prompt_trace_request(int64_t request) {
    const int64_t previous = prompt_trace_request;
    prompt_trace_request = request;
    return previous;
}

void llama_get_expert_miss_telemetry(uint64_t * bins, size_t n_bins, uint64_t & bytes, uint64_t & bytes_3plus) {
    const size_t count = std::min(n_bins, packed_miss_counts.size());
    std::copy_n(packed_miss_counts.begin(), count, bins);
    std::fill(bins + count, bins + n_bins, 0);
    bytes = packed_miss_bytes;
    bytes_3plus = packed_miss_bytes_3plus;
}

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

bool llama_moe_cache::requires_slots(int32_t il) const {
    return pimpl->placement && il >= 0 && static_cast<size_t>(il) < pimpl->placement->maps.size();
}

void llama_moe_cache::begin_prompt_trace(const llama_ubatch & ubatch, uint32_t batch_tokens, uint32_t batch_offset) {
    auto & self = *pimpl;
    if (self.prompt_stats) {
        self.prompt_active = true;
        ++self.prompt_ubatches;
        self.prompt_tokens += ubatch.n_tokens;
        self.prompt_max_tokens = std::max<uint64_t>(self.prompt_max_tokens, ubatch.n_tokens);
    }
    if (!self.prompt_trace || prompt_trace_request < 0) {
        return;
    }
    if (!ubatch.token || !ubatch.n_pos || ubatch.n_seqs_unq != 1 || ubatch.n_tokens > 2 || !self.prompt_trace_positions.empty()) {
        throw std::runtime_error("prompt trace requires a single text sequence and at most two tokens");
    }
    if (batch_offset == 0) {
        ++self.prompt_trace_batch;
    }
    ++self.prompt_trace_segment;
    self.prompt_trace_request_id = prompt_trace_request;
    // Text positions occupy the first section for IMROPE/MROPE as well.
    self.prompt_trace_positions.assign(ubatch.pos, ubatch.pos + ubatch.n_tokens);
    std::ostringstream row;
    row << "{\"type\":\"begin\",\"request\":" << self.prompt_trace_request_id << ",\"segment\":" << self.prompt_trace_segment
        << ",\"batch\":" << self.prompt_trace_batch << ",\"batch_tokens\":" << batch_tokens
        << ",\"batch_offset\":" << batch_offset << ",\"positions\":";
    prompt_trace_array(row, self.prompt_trace_positions);
    row << "}\n";
    self.write_prompt_trace(row.str());
}

void llama_moe_cache::end_prompt_trace(bool success) {
    auto & self = *pimpl;
    self.prompt_active = false;
    if (self.prompt_trace_positions.empty()) {
        return;
    }
    std::ostringstream row;
    row << "{\"type\":\"end\",\"request\":" << self.prompt_trace_request_id << ",\"segment\":" << self.prompt_trace_segment
        << ",\"success\":" << (success ? "true" : "false") << "}\n";
    self.prompt_trace_positions.clear();
    self.write_prompt_trace(row.str());
}

void llama_moe_cache::begin_generation_trace(const llama_ubatch & ubatch, uint32_t verify_width) {
    auto & self = *pimpl;
    if (!self.generation_trace) {
        return;
    }
    if (!ubatch.token || !ubatch.n_pos || ubatch.n_seqs_unq != 1 ||
            ubatch.n_tokens > 3 || !self.generation_trace_positions.empty()) {
        throw std::runtime_error("packed expert generation trace requires one text sequence and up to three tokens");
    }
    ++self.generation_trace_step;
    self.generation_trace_request = prompt_trace_request;
    self.generation_trace_verify_width = verify_width;
    self.generation_trace_positions.assign(ubatch.pos, ubatch.pos + ubatch.n_tokens);
}

void llama_moe_cache::end_generation_trace(bool success) {
    auto & self = *pimpl;
    if (self.generation_trace_positions.empty()) {
        return;
    }
    self.generation_trace_positions.clear();
    std::ostringstream row;
    row << "{\"type\":\"end\",\"request\":" << self.generation_trace_request << ",\"step\":" << self.generation_trace_step
        << ",\"success\":" << (success ? "true" : "false") << "}\n";
    self.write_expert_trace(self.generation_trace.get(), row.str());
}

bool llama_moe_cache::next_router_trace_active() const {
    const auto & self = *pimpl;
    return self.next_router_trace && self.generation_trace_verify_width == 3 &&
            self.generation_trace_positions.size() == 3;
}

bool llama_moe_cache::next_router_prefetch_enabled() const {
    return pimpl->next_prefetch_bank != nullptr;
}

int llama_moe_cache::next_router_score_mode() const {
    return pimpl->next_prefetch_score_mode;
}

void llama_moe_cache::begin_next_router_prefetch(const std::vector<ggml_tensor *> & predictions, const std::vector<ggml_tensor *> & scores) {
    auto & self = *pimpl;
    GGML_ASSERT(!self.next_prefetch_predictions && self.next_prefetch_layer < 0 && !self.next_prefetch_tail_pending);
    self.next_prefetch_predictions = &predictions;
    GGML_ASSERT(!self.next_prefetch_score_mode || scores.size() == predictions.size());
    self.next_prefetch_scores = self.next_prefetch_score_mode ? &scores : nullptr;
}

void llama_moe_cache::end_next_router_prefetch() {
    if (pimpl->next_prefetch_submit_layer >= 0) {
        ++pimpl->next_prefetch_submit_canceled;
        pimpl->next_prefetch_submit_layer = -1;
    }
    pimpl->finish_next_prefetch();
    pimpl->next_prefetch_predictions = nullptr;
    pimpl->next_prefetch_scores = nullptr;
}

bool llama_moe_cache::next_router_profile_enabled() const {
    return pimpl->next_profile_enabled;
}

void llama_moe_cache::begin_next_router_profile() {
    GGML_ASSERT(!pimpl->next_profile_start);
    pimpl->next_profile_start = ggml_time_us();
}

void llama_moe_cache::end_next_router_profile(int nodes, int splits, int hc_pre, int hc_comb, int hc_post, bool success) {
    auto & self = *pimpl;
    self.next_profile_graph_us += ggml_time_us() - self.next_profile_start;
    self.next_profile_start = 0;
    ++self.next_profile_graphs;
    self.next_profile_failed += !success;
    self.next_profile_nodes += nodes;
    self.next_profile_splits += splits;
    self.next_profile_hc_pre += hc_pre;
    self.next_profile_hc_comb += hc_comb;
    self.next_profile_hc_post += hc_post;
}

void llama_moe_cache::trace_next_router(int32_t il, const ggml_tensor * ids) {
    const auto & self = *pimpl;
    constexpr int n_tokens = 3;
    constexpr int n_used = 10;
    std::array<int32_t, n_tokens * 2 * n_used> predicted;
    const int ranks = ids->ne[0];
    const size_t bytes = ranks * n_tokens * sizeof(int32_t);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && ids->ne[1] == n_tokens &&
            (ranks == n_used || ranks == 2 * n_used) && ggml_nbytes(ids) == bytes);
    ggml_backend_tensor_get(ids, predicted.data(), 0, bytes);

    std::ostringstream row;
    row << "{\"type\":\"router_prediction\",\"request\":" << self.generation_trace_request
        << ",\"step\":" << self.generation_trace_step << ",\"layer\":" << il << ",\"positions\":";
    prompt_trace_array(row, self.generation_trace_positions);
    row << ",\"predicted\":[";
    for (int token = 0; token < n_tokens; ++token) {
        row << (token ? ",[" : "[");
        for (int rank = 0; rank < n_used; ++rank) {
            row << (rank ? "," : "") << predicted[token * ranks + rank];
        }
        row << ']';
    }
    row << "]}\n";
    self.write_expert_trace(self.generation_trace.get(), row.str());
}

void llama_moe_cache::record_decode_width(uint32_t tokens) {
    if (pimpl->prompt_stats) {
        pimpl->decode_max_tokens = std::max<uint64_t>(pimpl->decode_max_tokens, tokens);
    }
}

void llama_moe_cache::finish_pending_promotion() {
    pimpl->finish_hip_mtp_promotion();
}

llama_expert_round_stats llama_moe_cache::round_stats() const {
    const auto & self = *pimpl;
    return {
        self.stats_small.hits + self.packed_request_misses,
        self.packed_request_misses,
        self.stats_small.misses,
        self.stats_small.bytes,
        self.stats_small.hits - self.packed_scratch_hits,
        self.packed_scratch_hits,
        self.packed_promotions,
        self.packed_writeback_bytes,
        self.placement && self.placement->max_ubatch == 2 && !self.placement->overlap && !self.rtx_expert,
    };
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part, bool mtp_three) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part, mtp_three);
}

bool llama_moe_cache::compact_prefill(int64_t n_tokens) const {
    return pimpl->placement && pimpl->placement->prefill_tokens == 512 && n_tokens > 2 && n_tokens <= 512;
}

ggml_tensor * llama_moe_cache::build_compact_prefill(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * ids, int32_t il,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, ggml_tensor * gate_up,
        ggml_tensor * up_scale, ggml_tensor * gate_scale, ggml_tensor * down_scale) const {
    auto & self = *pimpl;
    auto & l = self.layers.at(il);
    if (!compact_prefill(cur->ne[1]) || cur->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !down || (!gate_up && (!up || !gate))) {
        throw std::runtime_error("unsupported compact prefill Expert layout");
    }
    l.compact_owner = &self;
    l.compact_layer = il;
    l.compact_weights = {up ? self.get_experts(up, false) : nullptr, gate ? self.get_experts(gate, false) : nullptr,
            self.get_experts(down, false), gate_up ? self.get_experts(gate_up, false) : nullptr};
    l.compact_scales = {up_scale, gate_scale, down_scale};
    if (!l.compact_weights[2] || (gate_up ? !l.compact_weights[3] : (!l.compact_weights[0] || !l.compact_weights[1]))) {
        throw std::runtime_error("compact prefill requires packed Expert weights");
    }
    for (auto * scale : l.compact_scales) {
        if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) ||
                ggml_nelements(scale) != llama_expert_profile::n_experts)) {
            throw std::runtime_error("compact prefill requires scalar F32 Expert scales");
        }
    }
    cur = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx, cur);
    ids = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx, ids);
    ggml_tensor * args[] = {cur, ids};
    auto * output = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], ids->ne[0], ids->ne[1], 1, args, 2,
            [](ggml_tensor * dst, int, int, void * userdata) {
                auto & layer = *static_cast<impl::layer *>(userdata);
                layer.compact_owner->compact_execute(layer, dst);
            }, 1, &l);
    ggml_format_name(output, "packed_expert_prefill512-%d", il);
    return output;
}

int32_t llama_moe_cache::get_prefill_waves(int32_t il, int64_t n_tokens, int64_t n_used) const {
    return requires_slots(il) ? pimpl->placement->prefill_waves(n_tokens, n_used) : 0;
}

ggml_tensor * llama_moe_cache::get_prefill_map(int32_t il, int32_t wave) const {
    return pimpl->layers.at(il).prefill_maps.at(wave);
}

ggml_tensor * llama_moe_cache::get_mtp_verify_map(int32_t il, int32_t wave) const {
    return pimpl->layers.at(il).mtp_verify_maps.at(wave);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w, bool cold) const {
    return pimpl->get_experts(w, cold);
}

llama_rtx_expert * llama_moe_cache::get_rtx_expert() const {
    return pimpl->rtx_expert.get();
}

ggml_tensor * llama_moe_cache::get_cold_slot_map(int32_t il, bool mtp_three) const {
    if (!requires_slots(il) || (pimpl->placement->hip_mtp_overlap && !mtp_three)) {
        return nullptr;
    }
    return pimpl->layers[il].cold_slot_map;
}

bool llama_moe_cache::uses_hip_mtp_compact() const {
    return pimpl->placement && pimpl->placement->hip_mtp_compact;
}

ggml_tensor * llama_moe_cache::get_hip_mtp_gate_up_full_map(int32_t il) const {
    return requires_slots(il) ? pimpl->layers[il].hip_mtp_full_map : nullptr;
}

uint64_t * llama_moe_cache::get_hip_mtp_fused_hit_count() const {
    return pimpl->placement && pimpl->placement->hip_mtp_gate_up_fuse_hits ? &pimpl->packed_mtp_gate_up_fused_hits : nullptr;
}

uint64_t * llama_moe_cache::get_hip_mtp_shared_output_count() const {
    return pimpl->placement && pimpl->placement->hip_mtp_gate_up_shared_output ?
            &pimpl->packed_mtp_gate_up_shared_dispatches : nullptr;
}

uint64_t * llama_moe_cache::get_hip_mtp_q8_1_count() const {
    return pimpl->placement && pimpl->placement->hip_mtp_gate_up_reuse_q8_1 ?
            &pimpl->packed_mtp_gate_up_q8_1_dispatches : nullptr;
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_map(il, n_tokens);
}

bool llama_moe_cache::early_copy_enabled() const {
    return pimpl->packed_early_copy;
}

bool llama_moe_cache::router_host_ids_enabled() const {
    return pimpl->placement != nullptr;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, bool profile, bool early,
        ggml_backend_sched_t sched, bool fixed_mtp_pair, bool mtp_four) {
    return pimpl->copy(backend, src, dst, graph, profile, early, sched, fixed_mtp_pair, mtp_four);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    if (pimpl->packed_staging) {
        const auto buffer = pimpl->packed_staging.get();
        res[ggml_backend_buffer_get_type(buffer)] += ggml_backend_buffer_get_size(buffer);
    }
    if (pimpl->next_prefetch_buffer) {
        const auto buffer = pimpl->next_prefetch_buffer.get();
        res[ggml_backend_buffer_get_type(buffer)] += ggml_backend_buffer_get_size(buffer);
    }
    return res;
}
