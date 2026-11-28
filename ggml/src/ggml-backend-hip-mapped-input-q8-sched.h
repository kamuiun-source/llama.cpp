#pragma once

static std::atomic<bool> hip_mapped_input_q8_enabled{false};

bool ggml_backend_sched_set_hip_mapped_input_q8(bool enabled) {
    return hip_mapped_input_q8_enabled.exchange(enabled, std::memory_order_relaxed) != enabled;
}

static bool ggml_backend_sched_mapped_input_q8_plan(ggml_backend_sched_t sched,
        ggml_backend_sched_split * split, const ggml_tensor * activation) {
    if (split->mapped_input_q8_plan_uid == split->graph.uid && split->mapped_input_q8_plan_input == activation) {
        return split->mapped_input_q8_plan_valid;
    }
    split->mapped_input_q8_plan_uid = split->graph.uid;
    split->mapped_input_q8_plan_input = activation;
    split->mapped_input_q8_plan_valid = false;
    if ((activation->flags & GGML_TENSOR_FLAG_OUTPUT) || activation->view_src) {
        return false;
    }
    int readers = 0;
    for (auto * part = sched->splits; part < sched->splits + sched->n_splits; ++part) {
        for (int j = 0; j < part->graph.n_nodes; ++j) {
            const auto * node = part->graph.nodes[j];
            if (ggml_hip_mapped_input_q8_base(node) == activation &&
                    (!ggml_is_view_op(node->op) || (node->flags & GGML_TENSOR_FLAG_OUTPUT))) {
                return false;
            }
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                const auto * src = node->src[k];
                if (ggml_hip_mapped_input_q8_base(src) != activation) {
                    continue;
                }
                const auto * weights = node->src[0];
                const auto * ids = node->src[2];
                if (part != split || k != 1 || node->op != GGML_OP_MUL_MAT_ID ||
                        !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || !ggml_hip_mapped_input_q8_layout(src) ||
                        src->data != activation->data || node->type != GGML_TYPE_F32 ||
                        node->ne[0] != 640 || node->ne[1] != 10 || node->ne[2] != 3 || node->ne[3] != 1 ||
                        !weights || weights->ne[0] != 2560 || weights->ne[1] != 640 ||
                        (weights->type != GGML_TYPE_IQ2_S && weights->type != GGML_TYPE_IQ3_XXS) ||
                        !ids || ids->type != GGML_TYPE_I32 || ids->ne[0] != 10 || ids->ne[1] != 3) {
                    return false;
                }
                ++readers;
            }
        }
    }
    split->mapped_input_q8_plan_valid = readers == 2;
    return split->mapped_input_q8_plan_valid;
}

static bool ggml_backend_sched_try_mapped_input_q8(ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        ggml_backend_t backend, ggml_tensor * destination, const void * mapped) {
    if (!hip_mapped_input_q8_enabled.load(std::memory_order_relaxed)) {
        return false;
    }
    if (!sched->hip_mapped_input_q8 || !sched->hip_mapped_input_q8_scope || split->has_custom ||
            sched->profile.enabled || sched->callback_eval || !ggml_backend_sched_mapped_input_q8_plan(sched, split, destination) ||
            !sched->hip_mapped_input_q8(backend, destination, mapped)) {
        ++sched->hip_mapped_input_q8_fallback;
        return false;
    }
    return true;
}

struct ggml_backend_sched_mapped_input_q8_scope {
    ggml_backend_hip_mapped_input_q8_scope_t set;
    ggml_backend_t backend;
    bool active;

    ggml_backend_sched_mapped_input_q8_scope(ggml_backend_sched_t sched, ggml_backend_t backend, const ggml_tensor * input)
        : set(sched->hip_mapped_input_q8_scope), backend(backend), active(input != nullptr) {
        if (active) {
            set(backend, input);
        }
    }

    void clear() {
        if (active) {
            set(backend, nullptr);
            active = false;
        }
    }

    ~ggml_backend_sched_mapped_input_q8_scope() { clear(); }
};
