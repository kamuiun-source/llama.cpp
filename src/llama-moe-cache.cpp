#include "llama-moe-cache.h"

#include "llama-expert-placement.h"
#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-rtx-expert.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

static constexpr size_t packed_miss_bins = llama_expert_profile::n_experts - llama_expert_profile::n_resident + 1;
static thread_local std::array<uint64_t, packed_miss_bins> packed_miss_counts{};
static thread_local uint64_t packed_miss_bytes = 0;
static thread_local uint64_t packed_miss_bytes_3plus = 0;
static thread_local llama_expert_transfer_telemetry packed_transfer_totals;

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
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * cold_slot_map = nullptr;
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        std::vector<int32_t> packed_scratch; // original expert ID per scratch slot, -1 if empty
        std::vector<int32_t> packed_scratch_slots; // physical slots when the permanent map can change
        std::vector<uint64_t> packed_last_used;
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
    size_t pending_permanent_hits = 0;
    size_t pending_scratch_hits = 0;
    size_t pending_upload_bytes = 0;
    bool packed_scratch_reuse = false;
    size_t packed_scratch_hits = 0;
    size_t packed_request_misses = 0;
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
    using overlap_expert_upload_t = bool (*)(ggml_backend_t, const ggml_cgraph *, ggml_tensor *, const int32_t *,
            const int32_t *, const uint32_t *, ggml_tensor * const *, const void *, const size_t *, const size_t *, const size_t *, size_t);
    overlap_expert_upload_t overlap_expert_upload = nullptr;
    using overlap_expert_promotion_t = bool (*)(ggml_backend_t, const ggml_cgraph *, ggml_tensor *, const int32_t *,
            const int32_t *, const uint32_t *, ggml_tensor * const *, const void *, const size_t *, const size_t *, const size_t *,
            size_t, int32_t, ggml_tensor * const *, const size_t *, size_t, void (*)(void *), void *);
    overlap_expert_promotion_t overlap_expert_promotion = nullptr;
    bool overlap_promotion_requested = false;
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
    };
    packed_promotion pending_promotion{};
    const uint8_t * pending_writeback = nullptr;
    bool overlap_upload_requested = false;
    size_t packed_overlap_uploads = 0;
    size_t packed_overlap_upload_bytes = 0;
    bool defer_upload_requested = false;
    size_t packed_deferred_uploads = 0;
    size_t packed_synchronous_uploads = 0;
    size_t packed_deferred_upload_bytes = 0;
    size_t packed_deferred_maps = 0;
    size_t packed_synchronous_maps = 0;
    bool router_host_ids = false;
    size_t router_host_hits = 0;
    size_t router_host_fallbacks = 0;
    size_t router_readback_elided_bytes = 0;
    int64_t packed_wait_us = 0;
    size_t packed_wait_count = 0;
    std::vector<uint8_t> packed_writeback;
    size_t packed_promotions = 0;
    size_t packed_writeback_bytes = 0;
    int64_t packed_writeback_us = 0;
    int64_t profile_router_us = 0;
    size_t profile_router_reads = 0;
    int64_t profile_upload_us = 0;
    size_t profile_upload_steps = 0;
    size_t profile_upload_bytes = 0;
    ggml_backend_buffer_ptr packed_staging;
    std::vector<moe_cache_lru::fill> fills; // packed fills use logical scratch indices
    std::vector<moe_cache_lru::fill> cpu;

    bool packed_early_copy = false;
    size_t packed_early_upload_steps = 0;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
            n_expert_used(model.hparams.n_expert_used_max()), placement(model.expert_placement.get()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
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
            const char * host_ids = std::getenv("LLAMA_EXPERT_ROUTER_HOST_IDS");
            router_host_ids = host_ids && std::strcmp(host_ids, "1") == 0;
            if (router_host_ids) {
                LLAMA_LOG_INFO("packed expert router: reuse host IDs from CUDA to Vulkan copies\n");
            }
            const char * scratch_reuse = std::getenv("LLAMA_EXPERT_SCRATCH_REUSE");
            packed_scratch_reuse = !scratch_reuse || std::strcmp(scratch_reuse, "0") != 0;
            if (placement->dynamic_resident && !packed_scratch_reuse) {
                throw std::runtime_error("dynamic expert residency requires scratch reuse for every context");
            }
            groups.emplace_back().id = static_cast<int32_t>(device - devices.begin());
            const ggml_init_params params{placement->maps.size() * ggml_tensor_overhead() * (placement->overlap ? 2 : 1), nullptr, true};
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
                if (placement->overlap) {
                    l.cold_slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, model.hparams.n_expert);
                    ggml_format_name(l.cold_slot_map, "moe_cache.packed_cold-%zu", il);
                    layer_of.emplace(l.cold_slot_map, std::make_pair(static_cast<int32_t>(il), true));
                }
                if (packed_scratch_reuse) {
                    l.packed_scratch.assign(placement->n_slots, -1);
                }
                if (placement->dynamic_resident) {
                    l.packed_scratch_slots = placement->maps[il].scratch_slots(placement->n_slots);
                    l.packed_last_used.assign(llama_expert_profile::n_experts, 0);
                }
            }
            std::array<size_t, llama_expert_profile::n_layers> bytes_per_layer{};
            for (const auto & w : placement->weights) {
                layers.at(w.layer).experts.push_back(w.host);
                bindings.emplace(w.host, binding{w.layer, 0, w.bank, w.cold});
                auto & bytes = bytes_per_layer.at(w.layer);
                if (w.host->nb[2] > SIZE_MAX - bytes) {
                    throw std::runtime_error("packed expert tuple is too large");
                }
                bytes += w.host->nb[2];
            }
            const size_t tuple_bytes = *std::max_element(bytes_per_layer.begin(), bytes_per_layer.end());
            if (placement->dynamic_resident) {
                for (size_t il = 0; il < placement->maps.size(); ++il) {
                    const size_t n_planes = layers[il].experts.size();
                    if (n_planes != 2 && n_planes != 3) {
                        throw std::runtime_error("dynamic expert residency requires gate/up and down weights");
                    }
                }
                packed_writeback.resize(tuple_bytes);
                LLAMA_LOG_INFO("packed expert dynamic residency: stale_lookups=%" PRIu64 " writeback_buffer_bytes=%zu\n",
                        packed_min_resident_age, packed_writeback.size());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), ggml_backend_cpu_buffer_type()));
            if (!buf_host) {
                throw std::runtime_error("could not allocate packed expert slot maps");
            }
            ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            const auto reg = ggml_backend_dev_backend_reg(placement->target);
            const char * histogram = std::getenv("LLAMA_EXPERT_MISS_HISTOGRAM");
            const char * telemetry = std::getenv("LLAMA_DECODE_TELEMETRY");
            packed_miss_histogram = telemetry && std::strcmp(telemetry, "1") == 0 &&
                    histogram && std::strcmp(histogram, "1") == 0;
            if (reg && telemetry && std::strcmp(telemetry, "1") == 0) {
                set_telemetry_cause = (telemetry_cause_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_set_telemetry_cause");
            }
            const char * defer_maps = std::getenv("LLAMA_EXPERT_VK_DEFER_SLOT_MAP");
            defer_slot_map_requested = !placement->overlap && defer_maps && std::strcmp(defer_maps, "1") == 0;
            if (defer_slot_map_requested) {
                defer_slot_map = reg ? (defer_slot_map_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_defer_slot_map") : nullptr;
            }
            const char * defer_upload = std::getenv("LLAMA_EXPERT_VK_DEFER_UPLOAD");
            defer_upload_requested = !placement->overlap && defer_upload && std::strcmp(defer_upload, "1") == 0;
            if (defer_upload_requested && defer_slot_map) {
                submit_expert_upload = (submit_expert_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_submit_expert_upload");
            }
            const char * batch_upload = std::getenv("LLAMA_EXPERT_BATCH_UPLOAD");
            const bool batch_requested = !placement->overlap && batch_upload && std::strcmp(batch_upload, "1") == 0;
            const char * overlap_upload = std::getenv("LLAMA_EXPERT_VK_OVERLAP_UPLOAD");
            overlap_upload_requested = batch_requested && placement->dynamic_resident && placement->max_ubatch == 2 &&
                    overlap_upload && std::strcmp(overlap_upload, "1") == 0;
            const char * overlap_promotion = std::getenv("LLAMA_EXPERT_VK_OVERLAP_PROMOTION");
            overlap_promotion_requested = overlap_upload_requested && overlap_promotion && std::strcmp(overlap_promotion, "1") == 0;
            if (placement->n_slots <= 0 || tuple_bytes > SIZE_MAX / placement->n_slots) {
                throw std::runtime_error("packed expert staging is too large");
            }
            const size_t staging_bytes = tuple_bytes * placement->n_slots;
            if (placement->overlap || batch_requested) {
                using alloc_host_upload_buffer_t = ggml_backend_buffer_t (*)(ggml_backend_dev_t, size_t);
                if (overlap_upload_requested && defer_slot_map && reg) {
                    const auto alloc_overlap_staging = reinterpret_cast<alloc_host_upload_buffer_t>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_alloc_overlap_expert_staging"));
                    const auto submit_overlap = reinterpret_cast<overlap_expert_upload_t>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_overlap_expert_upload"));
                    if (alloc_overlap_staging && submit_overlap) {
                        packed_staging.reset(alloc_overlap_staging(placement->target, staging_bytes));
                        if (packed_staging) {
                            overlap_expert_upload = submit_overlap;
                            if (overlap_promotion_requested) {
                                overlap_expert_promotion = reinterpret_cast<overlap_expert_promotion_t>(
                                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_overlap_expert_promotion"));
                            }
                        }
                    }
                }
                const char * proc = placement->overlap ? "ggml_backend_vk_alloc_transfer_host_upload_buffer" :
                        "ggml_backend_vk_alloc_host_upload_buffer";
                const auto alloc_host_upload_buffer = reg ? (alloc_host_upload_buffer_t) ggml_backend_reg_get_proc_address(reg, proc) : nullptr;
                if (!packed_staging && alloc_host_upload_buffer) {
                    packed_staging.reset(alloc_host_upload_buffer(placement->target, staging_bytes));
                }
                if (packed_staging) {
                    fills.reserve(placement->n_slots);
                    if (placement->overlap) {
                        if (packed_scratch_reuse) {
                            packed_cold_scratch.reserve(placement->n_slots);
                        }
                        const char * early_copy = std::getenv("LLAMA_EXPERT_OVERLAP_EARLY_COPY");
                        packed_early_copy = !early_copy || std::strcmp(early_copy, "0") != 0;
                        LLAMA_LOG_INFO("packed expert overlap: early cold uploads %s\n", packed_early_copy ? "enabled" : "disabled");
                    }
                    LLAMA_LOG_INFO("packed expert %s upload: %zu bytes of native host staging on %s (%s)\n",
                            placement->overlap ? "overlap" : "batch", ggml_backend_buffer_get_size(packed_staging.get()),
                            ggml_backend_dev_name(placement->target), ggml_backend_dev_description(placement->target));
                } else if (placement->overlap) {
                    throw std::runtime_error("expert overlap requires native Vulkan host staging on the target and an async transfer queue without serialized submissions");
                } else {
                    LLAMA_LOG_WARN("packed expert batch upload unavailable on %s, using serial uploads\n",
                            ggml_backend_dev_name(placement->target));
                }
            }
            const char * rtx_slots_env = std::getenv("LLAMA_EXPERT_RTX_COLD_SLOTS");
            if (rtx_slots_env && std::strcmp(rtx_slots_env, "0") != 0) {
                char * end = nullptr;
                const long slots = std::strtol(rtx_slots_env, &end, 10);
                const auto cuda = std::find_if(devices.begin(), devices.end(), [](const auto & d) {
                    return std::strcmp(ggml_backend_dev_name(ggml_backend_get_device(d.backend)), "CUDA0") == 0;
                });
                if (end == rtx_slots_env || *end || slots < 1 || slots >= placement->n_slots ||
                        !packed_staging || placement->overlap || !placement->dynamic_resident ||
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
        GGML_ASSERT(!pending_writeback);
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (placement) {
            return n_tokens > 0 && n_tokens <= placement->max_ubatch && n_expert_used > 0 &&
                    n_expert_used <= placement->n_slots / n_tokens ? l.slot_map : nullptr;
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
        return it != bindings.end() ? (cold ? it->second.cold : it->second.cached) : nullptr;
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
            move.planes[i] = {bank, static_cast<uint8_t *>(host->data) + move.host_slot * stride, stride};
            move.bytes += stride;
        }
        return move;
    }

    void commit_packed_promotion(const packed_promotion & move, const uint8_t * victim_data) noexcept {
        auto & map = placement->maps[move.il];
        auto & l = layers[move.il];
        GGML_ASSERT(map.host[move.promote] == move.host_slot && map.resident[move.evict] == move.old_gpu_slot &&
                l.packed_scratch[move.scratch_index] == move.promote && packed_slot(l, move.scratch_index) == move.new_gpu_slot);
        // All victim planes are available before changing either original.
        for (size_t i = 0; i < move.n_planes; ++i) {
            const auto & p = move.planes[i];
            std::memcpy(p.host_data, victim_data, p.stride);
            victim_data += p.stride;
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
        const int64_t start = ggml_time_us();
        size_t offset = 0;
        for (size_t i = 0; i < move.n_planes; ++i) {
            const auto & p = move.planes[i];
            ggml_backend_tensor_get(p.bank, packed_writeback.data() + offset, move.old_gpu_slot * p.stride, p.stride);
            offset += p.stride;
        }
        commit_packed_promotion(move, packed_writeback.data());
        packed_writeback_us += ggml_time_us() - start;
    }

    size_t upload_packed(ggml_backend_t backend, const layer & l, int32_t il, bool cold, bool * submitted = nullptr) {
        telemetry_scope scope(set_telemetry_cause, "expert_upload");
        const auto & map = placement->maps[il];
        auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get()));
        const size_t staging_bytes = ggml_backend_buffer_get_size(packed_staging.get());
        size_t bytes = 0;
        for (const auto & fill : fills) {
            const int32_t slot = cold ? fill.slot : packed_slot(l, fill.slot);
            for (ggml_tensor * host : l.experts) {
                const auto & binding = bindings.at(host);
                const size_t stride = host->nb[2];
                const auto * data = static_cast<const uint8_t *>(host->data) + map.host[fill.expert] * stride;
                GGML_ASSERT(stride <= staging_bytes - bytes);
                std::memcpy(staging + bytes, data, stride);
                ggml_backend_tensor_set_async(backend, cold ? binding.cold : binding.cached, staging + bytes, slot * stride, stride);
                bytes += stride;
            }
        }
        if (submitted) {
            *submitted = submit_expert_upload(backend);
        }
        if (!submitted || !*submitted) {
            ggml_backend_synchronize(backend);
        }
        return bytes;
    }

    bool overlap_packed_upload(ggml_backend_t backend, const ggml_cgraph * graph, ggml_tensor * slot_map,
            const layer & l, int32_t il, int32_t n_used, const std::array<int32_t, llama_expert_profile::n_experts> & slots,
            int32_t promote_index, int32_t evict, size_t & bytes) {
        constexpr size_t max_lanes = 32;
        constexpr size_t max_planes = 3;
        if (n_used <= 0 || ids.size() > max_lanes || ids.size() % n_used != 0 ||
                ids.size() / n_used > 2 || l.experts.size() > max_planes) {
            return false;
        }
        const auto & map = placement->maps[il];
        std::array<bool, llama_expert_profile::n_experts> filled{};
        for (const auto & fill : fills) {
            filled[fill.expert] = true;
        }
        std::array<int32_t, max_lanes> used_slots{};
        std::array<uint32_t, 2> hot_masks{};
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto expert = ids[i];
            used_slots[i] = slots[expert];
            if (!filled[expert] && used_slots[i] >= 0) {
                hot_masks[i / n_used] |= uint32_t{1} << (i % n_used);
            }
        }
        if (!(hot_masks[0] || hot_masks[1])) {
            return false;
        }

        auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(packed_staging.get()));
        const size_t staging_bytes = ggml_backend_buffer_get_size(packed_staging.get());
        constexpr size_t max_regions = max_lanes * max_planes;
        std::array<ggml_tensor *, max_regions> tensors;
        std::array<size_t, max_regions> offsets, sizes, staging_offsets;
        size_t n_regions = 0;
        size_t used_bytes = 0;
        for (const auto & fill : fills) {
            const int32_t slot = packed_slot(l, fill.slot);
            const int32_t host_slot = map.host[fill.expert];
            for (auto * host : l.experts) {
                auto * bank = bindings.at(host).cached;
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
                std::memcpy(staging + used_bytes, static_cast<const uint8_t *>(host->data) + host_slot * stride, stride);
                used_bytes += stride;
            }
        }
        telemetry_scope scope(set_telemetry_cause, "expert_upload");
        if (evict >= 0) {
            GGML_ASSERT(!pending_writeback && overlap_expert_promotion);
            const auto move = prepare_packed_promotion(l, il, promote_index, evict);
            if (move.bytes > staging_bytes - used_bytes ||
                    std::find(used_slots.begin(), used_slots.begin() + ids.size(), move.old_gpu_slot) != used_slots.begin() + ids.size()) {
                return false;
            }
            std::array<ggml_tensor *, max_planes> writeback_weights;
            std::array<size_t, max_planes> writeback_offsets;
            size_t offset = used_bytes;
            for (size_t i = 0; i < move.n_planes; ++i) {
                writeback_weights[i] = move.planes[i].bank;
                writeback_offsets[i] = offset;
                offset += move.planes[i].stride;
            }
            pending_promotion = move;
            pending_writeback = staging + used_bytes;
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
                host_slices[r * planes.size() + p] = static_cast<const uint8_t *>(host->data) + static_cast<size_t>(host_slot) * stride;
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
            ggml_backend_sched_t sched) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        if (early && (!packed_early_copy || !has_cpu || pending_layer != il || fills.empty() || pending_upload_bytes != 0)) {
            return false;
        }
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);
        // The scheduler's destination wait must commit the preceding promotion.
        GGML_ASSERT(!pending_writeback);

        if (placement && placement->overlap && has_cpu) {
            // The resident split does not read the separate cold bank. The hot map readback waited for older users.
            GGML_ASSERT(pending_layer == il);
            if (!fills.empty() && pending_upload_bytes == 0) {
                if (packed_scratch_reuse) {
                    // A failed upload can leave the old cold slot only partly intact.
                    for (const auto & fill : fills) {
                        l.packed_scratch[fill.slot] = -1;
                    }
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
            stats_small.hits += pending_permanent_hits + pending_scratch_hits;
            packed_scratch_hits += pending_scratch_hits;
            stats_small.misses += fills.size();
            stats_small.bytes += pending_upload_bytes;
            LLAMA_LOG_DEBUG("packed experts: layer %d: %zu permanent hits, %zu scratch hits, %zu RAM uploads, %zu bytes\n",
                    il, pending_permanent_hits, pending_scratch_hits, fills.size(), pending_upload_bytes);
            return true;
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
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

        ids.resize(ggml_nelements(sel));
        const size_t ids_bytes = ggml_nbytes(sel);
        if (router_host_ids && sched && ggml_backend_sched_get_host_copy(sched, sel, ids.data(), ids_bytes)) {
            // The weight callback starts after the scheduler has finished older Vulkan work.
            ++router_host_hits;
            router_readback_elided_bytes += ids_bytes;
        } else {
            router_host_fallbacks += router_host_ids;
            const int64_t router_start_us = profile ? ggml_time_us() : 0;
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ids_bytes);
            ggml_backend_synchronize(backend);
            if (profile) {
                profile_router_us += ggml_time_us() - router_start_us;
                profile_router_reads++;
            }
        }

        if (placement) {
            const auto & map = placement->maps[il];
            auto slots = map.resident;
            int32_t n_uploads = 0;
            size_t bytes = 0;
            size_t hits = 0;
            size_t scratch_hits = 0;
            const int64_t start = ggml_time_us();
            int64_t upload_start_us = 0;
            const bool batch = packed_staging != nullptr;
            if (batch) {
                fills.clear();
            }
            if (placement->overlap) {
                // An interrupted graph can leave an unconsumed plan.
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
            if (placement->dynamic_resident) {
                ++l.packed_lookups;
                for (const int32_t expert : ids) {
                    l.packed_last_used[expert] = l.packed_lookups;
                }
            }
            int32_t promote_index = -1;
            auto & scratch = placement->overlap ? packed_cold_scratch : l.packed_scratch;
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
                int32_t scratch_index = n_uploads;
                auto victim = scratch.end();
                if (packed_scratch_reuse) {
                    victim = std::find(scratch.begin(), scratch.end(), -1);
                    if (victim == scratch.end()) {
                        victim = std::find_if(scratch.begin(), scratch.end(), [&](int32_t resident) {
                            return slots[resident] < 0;
                        });
                    }
                    if (victim == scratch.end()) {
                        GGML_ABORT("packed expert scratch slots exhausted in layer %d", il);
                    }
                    scratch_index = static_cast<int32_t>(victim - scratch.begin());
                    // A partial upload must not leave the slot mapped to the evicted expert.
                    *victim = -1;
                }
                ++n_uploads;
                if (profile && n_uploads == 1) {
                    upload_start_us = ggml_time_us();
                }
                const int32_t slot = packed_slot(l, scratch_index);
                if (batch) {
                    fills.push_back({expert, scratch_index});
                } else {
                    telemetry_scope scope(set_telemetry_cause, "expert_upload");
                    for (ggml_tensor * host : l.experts) {
                        ggml_tensor * bank = bindings.at(host).cached;
                        const size_t stride = host->nb[2];
                        const auto * data = static_cast<const uint8_t *>(host->data) + map.host[expert] * stride;
                        ggml_backend_tensor_set(bank, data, slot * stride, stride);
                        bytes += stride;
                    }
                }
                if (packed_scratch_reuse) {
                    *victim = expert;
                }
                slots[expert] = slot;
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
                return true;
            }
            const int32_t evict = placement->dynamic_resident && promote_index >= 0 ? packed_promotion_victim(l, map) : -1;
            bool upload_submitted = false;
            bool overlap_submitted = false;
            const bool can_overlap = overlap_expert_upload && !profile && sched && ggml_backend_sched_get_n_copies(sched) == 1;
            const bool can_overlap_promotion = can_overlap && overlap_expert_promotion && evict >= 0;
            if (batch && (!fills.empty() || can_overlap_promotion)) {
                // A single-copy scheduler synchronizes the backend before each host-weight callback.
                const bool can_defer = submit_expert_upload && !profile && sched &&
                        ggml_backend_sched_get_n_copies(sched) == 1 && promote_index < 0;
                try {
                    if (can_overlap && (evict < 0 || can_overlap_promotion)) {
                        overlap_submitted = overlap_packed_upload(backend, graph, dst, l, il, n_used, slots, promote_index, evict, bytes);
                    }
                    if (!overlap_submitted && !fills.empty()) {
                        bytes = upload_packed(backend, l, il, false, can_defer ? &upload_submitted : nullptr);
                    }
                } catch (...) {
                    if (packed_scratch_reuse) {
                        for (const auto & fill : fills) {
                            scratch[fill.slot] = -1;
                        }
                    }
                    throw;
                }
            }
            if (n_uploads > 0 && !upload_submitted && !overlap_submitted) {
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
            if (evict >= 0 && !overlap_submitted) {
                promote_packed(l, il, promote_index, evict);
            }
            // The scheduler synchronizes after each copy callback while profiling.
            if (overlap_submitted) {
                packed_overlap_uploads += bytes != 0;
                packed_overlap_upload_bytes += bytes;
                ++packed_deferred_maps;
            } else if (defer_slot_map && !profile && defer_slot_map(backend, dst, slots.data())) {
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
                telemetry_scope scope(set_telemetry_cause, "slot_map");
                ggml_backend_tensor_set(dst, slots.data(), 0, ggml_nbytes(dst));
                packed_synchronous_maps += defer_slot_map_requested;
            }
            packed_synchronous_uploads += defer_upload_requested && n_uploads > 0 && !upload_submitted && !overlap_submitted;
            stats_small.hits += hits + scratch_hits;
            packed_scratch_hits += scratch_hits;
            stats_small.misses += n_uploads;
            packed_request_misses += request_misses;
            stats_small.bytes += bytes;
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
        if (profile_router_reads > 0 || profile_upload_steps > 0) {
            LLAMA_LOG_INFO("decode_profile: expert_router_readback_ms=%.3f router_reads=%zu expert_upload_ms=%.3f upload_steps=%zu uploaded_MiB=%.3f\n",
                    profile_router_us / 1000.0, profile_router_reads, profile_upload_us / 1000.0,
                    profile_upload_steps, profile_upload_bytes / (1024.0 * 1024.0));
        }
        if (router_host_ids) {
            LLAMA_LOG_INFO("packed expert router: host_hits=%zu fallback_reads=%zu elided_readback_bytes=%zu\n",
                    router_host_hits, router_host_fallbacks, router_readback_elided_bytes);
        }
        if (placement) {
            LLAMA_LOG_INFO("packed expert transfer phase (%s): %.3f ms across %zu layer steps\n",
                    placement->overlap ? "overlap" : "sync", packed_wait_us / 1000.0, packed_wait_count);
            if (placement->overlap) {
                LLAMA_LOG_INFO("packed expert overlap uploads: early_steps=%zu waited_steps=%zu\n",
                        packed_early_upload_steps, packed_wait_count - packed_early_upload_steps);
            }
            LLAMA_LOG_INFO("packed expert slots: reuse=%s permanent_hits=%zu scratch_hits=%zu ram_uploads=%zu upload_mode=%s\n",
                    packed_scratch_reuse ? "on" : "off", stats_small.hits - packed_scratch_hits, packed_scratch_hits, stats_small.misses,
                    placement->overlap ? "overlap" : packed_staging ? "batch" : "serial");
            if (defer_upload_requested) {
                LLAMA_LOG_INFO("packed expert upload deferral: deferred=%zu synchronous=%zu bytes=%zu\n",
                        packed_deferred_uploads, packed_synchronous_uploads, packed_deferred_upload_bytes);
            }
            if (overlap_upload_requested) {
                LLAMA_LOG_INFO("packed expert Vulkan upload overlap: submitted=%zu bytes=%zu\n",
                        packed_overlap_uploads, packed_overlap_upload_bytes);
            }
            if (overlap_promotion_requested) {
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
            if (placement->dynamic_resident) {
                LLAMA_LOG_INFO("packed expert dynamic residency: promotions=%zu evictions=%zu writeback_bytes=%zu writeback_ms=%.3f uploaded_bytes=%zu total_transfer_bytes=%zu total_transfer_ms=%.3f\n",
                        packed_promotions, packed_promotions, packed_writeback_bytes, packed_writeback_us / 1000.0,
                        stats_small.bytes, stats_small.bytes + packed_writeback_bytes, (packed_wait_us + packed_writeback_us) / 1000.0);
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

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share)) {
}

llama_moe_cache::~llama_moe_cache() = default;

llama_expert_transfer_telemetry llama_get_expert_transfer_telemetry() {
    return packed_transfer_totals;
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

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w, bool cold) const {
    return pimpl->get_experts(w, cold);
}

llama_rtx_expert * llama_moe_cache::get_rtx_expert() const {
    return pimpl->rtx_expert.get();
}

ggml_tensor * llama_moe_cache::get_cold_slot_map(int32_t il) const {
    return requires_slots(il) ? pimpl->layers[il].cold_slot_map : nullptr;
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_map(il, n_tokens);
}

bool llama_moe_cache::early_copy_enabled() const {
    return pimpl->packed_early_copy;
}

bool llama_moe_cache::router_host_ids_enabled() const {
    return pimpl->router_host_ids;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, bool profile, bool early,
        ggml_backend_sched_t sched) {
    return pimpl->copy(backend, src, dst, graph, profile, early, sched);
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
    return res;
}
