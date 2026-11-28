#pragma once

#include "ggml-cpp.h"
#include "llama-expert-profile.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-ranked-rtx.h"

#include <cstring>
#include <fstream>
#include <sstream>

struct llama_ranked_rtx_experts {
    static constexpr int count = 10;
    static constexpr int32_t lookup_tag = 0x52545852;

    struct layer {
        std::array<int32_t, llama_expert_profile::n_experts> slots;
        ggml_tensor * map = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
    };

    std::array<layer, llama_expert_profile::n_layers> layers;
    ggml_backend_dev_t device;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;

    llama_ranked_rtx_experts(const char * path) {
        std::ifstream profile(path);
        std::string line;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            l.slots.fill(count);
            int layer_id;
            if (!std::getline(profile, line)) {
                throw std::runtime_error("ranked RTX Expert profile requires one row per layer");
            }
            std::istringstream row(line);
            if (!(row >> layer_id) || layer_id != int(il)) {
                throw std::runtime_error("invalid layer in ranked RTX Expert profile");
            }
            for (int slot = 0; slot < count; ++slot) {
                int expert;
                if (!(row >> expert) || expert < 0 || size_t(expert) >= l.slots.size() || l.slots[expert] != count) {
                    throw std::runtime_error("invalid or duplicate ID in ranked RTX Expert profile");
                }
                l.slots[expert] = slot;
            }
            row >> std::ws;
            if (!row.eof()) {
                throw std::runtime_error("invalid columns in ranked RTX Expert profile");
            }
        }
        profile >> std::ws;
        if (!profile.eof()) {
            throw std::runtime_error("extra rows in ranked RTX Expert profile");
        }

        device = ggml_backend_dev_by_name("CUDA0");
        if (!device) {
            throw std::runtime_error("ranked RTX Experts require CUDA0");
        }
        ctx.reset(ggml_init({4 * layers.size() * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::runtime_error("could not create ranked RTX Expert tensor context");
        }
        for (auto & l : layers) {
            l.map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, l.slots.size());
        }
    }

    bool owns(int layer_id, int expert) const {
        return layers[layer_id].slots[expert] != count;
    }

    ggml_tensor * add_plane(int il, const ggml_tensor * w) {
        auto & l = layers[il];
        auto & plane = std::strstr(w->name, ".ffn_gate_exps.weight") ? l.gate :
            std::strstr(w->name, ".ffn_up_exps.weight") ? l.up : l.down;
        if (plane) {
            throw std::runtime_error("duplicate ranked RTX Expert projection");
        }
        plane = ggml_new_tensor_3d(ctx.get(), w->type, w->ne[0], w->ne[1], count + 1);
        ggml_format_name(plane, "rtx_ranked.%s", w->name);
        return plane;
    }

    void allocate() {
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(device)));
        if (!buffer) {
            throw std::runtime_error("could not allocate ranked RTX Experts on CUDA0");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer.get(), 0);
        for (auto & l : layers) {
            if (!l.gate || !l.up || !l.down) {
                throw std::runtime_error("missing ranked RTX Expert projection");
            }
            auto slots = l.slots;
            for (auto & slot : slots) {
                if (slot == count) {
                    slot = ggml_ranked_rtx::inactive_id;
                }
            }
            ggml_backend_tensor_set(l.map, slots.data(), 0, sizeof(slots));
        }
    }

    ggml_tensor * output(ggml_context * graph, ggml_backend_sched_t sched, int il,
            ggml_tensor * ids, ggml_tensor * input, ggml_tensor * weights) const {
        ggml_backend_t cuda = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(backend) == device) {
                cuda = backend;
                break;
            }
        }
        if (!cuda) {
            throw std::runtime_error("ranked RTX Expert graph requires CUDA0 in the scheduler");
        }
        const auto on_cuda = [&](ggml_tensor * t) {
            if (!ggml_backend_supports_op(cuda, t)) {
                throw std::runtime_error("CUDA0 does not support ranked RTX Expert graph operation");
            }
            ggml_backend_sched_set_tensor_backend(sched, t, cuda);
            return t;
        };
        const auto & l = layers[il];
        auto * contiguous = ggml_is_contiguous(ids) ? ids : on_cuda(ggml_cont(graph, ids));
        auto * flat = ggml_reshape_1d(graph, contiguous, ggml_nelements(ids));
        auto * slots = on_cuda(ggml_get_rows(graph, l.map, flat));
        slots = ggml_reshape_2d(graph, slots, ids->ne[0], ids->ne[1]);
        slots->flags |= ggml_ranked_rtx::ids_flag;
        auto * gate = on_cuda(ggml_mul_mat_id(graph, l.gate, input, slots));
        auto * up = on_cuda(ggml_mul_mat_id(graph, l.up, input, slots));
        auto * act = on_cuda(ggml_swiglu_split(graph, gate, up));
        auto * down = on_cuda(ggml_mul_mat_id(graph, l.down, act, slots));
        auto * weighted = on_cuda(ggml_mul(graph, down, weights));
        ggml_tensor * sum = nullptr;
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            auto * lane = ggml_view_2d(graph, weighted, weighted->ne[0], ids->ne[1], weighted->nb[2], rank * weighted->nb[1]);
            sum = sum ? on_cuda(ggml_add(graph, sum, lane)) : lane;
        }
        return sum;
    }
};
