#pragma once

#include "ggml-impl.h"

#ifdef __cplusplus
#include <algorithm>
#include <vector>

namespace ggml_ranked_rtx {
constexpr int branch_flag = 1 << 24;
constexpr int merge_flag = 1 << 25;
constexpr int ids_flag = 1 << 26;
constexpr int anchor_flag = 1 << 27;
constexpr int weighted_flag = 1 << 28;
constexpr uint64_t empty_uid_bit = UINT64_C(1) << 63;
constexpr uint64_t projected_uid_bit = UINT64_C(1) << 62;
constexpr int inactive_id = -1;
constexpr int count_param = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
constexpr int layer_param = count_param - 1;

struct early_projection {
    int begin;
    int end;
    ggml_tensor * weighted;
    ggml_tensor * base;
    ggml_cgraph graph;
    uint64_t suffix_uid[2][2];
    bool checked;
    bool safe;
};

inline early_projection find_early_projection(const ggml_cgraph & graph, const ggml_tensor * copy) {
    for (int i = 1; i < graph.n_nodes; ++i) {
        auto * weighted = graph.nodes[i];
        if (!(weighted->flags & weighted_flag) || weighted->op != GGML_OP_MUL) {
            continue;
        }
        auto * down = graph.nodes[i - 1];
        if (weighted->src[0] != down || down->op != GGML_OP_MUL_MAT_ID ||
                !(down->src[2]->flags & ids_flag) || down->ne[0] != 2560 || down->ne[1] != 10 || down->ne[2] != 3) {
            return {};
        }

        int begin = i - 1;
        while (begin > 0 && (graph.nodes[begin - 1]->flags & branch_flag) &&
                !(graph.nodes[begin - 1]->flags & anchor_flag)) {
            --begin;
        }
        if (begin < 2) {
            return {};
        }
        auto * anchor = graph.nodes[begin - 1];
        auto * base = graph.nodes[begin - 2];
        if (!(anchor->flags & anchor_flag) || anchor->op != GGML_OP_VIEW || anchor->view_src != base ||
                base->op != GGML_OP_ADD || base->src[0] != copy) {
            return {};
        }
        // A projection input is either earlier in this slice or was produced in an earlier split.
        for (int j = begin; j < i; ++j) {
            auto * node = graph.nodes[j];
            switch (node->op) {
                case GGML_OP_CONT:
                case GGML_OP_RESHAPE:
                case GGML_OP_GET_ROWS:
                case GGML_OP_MUL_MAT_ID:
                case GGML_OP_GLU:
                    break;
                default:
                    return {};
            }
            for (const auto * src : node->src) {
                if (src && std::find(graph.nodes, graph.nodes + graph.n_nodes, src) != graph.nodes + graph.n_nodes &&
                        std::find(graph.nodes + begin, graph.nodes + j, src) == graph.nodes + j) {
                    return {};
                }
            }
        }
        early_projection plan{};
        plan.begin = begin;
        plan.end = i;
        plan.weighted = weighted;
        plan.base = base;
        return plan;
    }
    return {};
}

inline ggml_cgraph omit_early_projection(const ggml_cgraph & source, const early_projection & plan,
        std::vector<ggml_tensor *> & nodes, uint64_t uid) {
    const auto begin = std::find(source.nodes, source.nodes + source.n_nodes, plan.graph.nodes[0]);
    const auto end = std::find(begin, source.nodes + source.n_nodes, plan.weighted);
    GGML_ASSERT(begin != source.nodes + source.n_nodes && end - begin == plan.graph.n_nodes);
    nodes.assign(source.nodes, begin);
    nodes.insert(nodes.end(), end, source.nodes + source.n_nodes);

    auto graph = source;
    graph.nodes = nodes.data();
    graph.n_nodes = nodes.size();
    graph.uid = uid;
    return graph;
}

inline bool can_skip(const ggml_cgraph & graph, ggml_tensor * merge) {
    if (!merge || !merge->data || merge->src[0]->op != GGML_OP_VIEW || merge->src[0]->data != merge->data ||
            merge->view_offs != 0 || merge->src[0]->buffer != merge->buffer || !ggml_are_same_layout(merge->src[0], merge)) {
        return false;
    }
    int tagged = 0;
    bool found = false;
    for (int i = 0; i < graph.n_nodes; ++i) {
        tagged += (graph.nodes[i]->flags & branch_flag) != 0;
        found |= graph.nodes[i] == merge;
    }
    return found && tagged == merge->op_params[count_param];

}

inline ggml_cgraph omit(const ggml_cgraph & source, std::vector<ggml_tensor *> & nodes,
        uint64_t uid) {
    nodes.clear();
    for (int i = 0; i < source.n_nodes; ++i) {
        auto * node = source.nodes[i];
        if (!(node->flags & branch_flag) || (node->flags & anchor_flag)) {
            nodes.push_back(node);
        }
    }
    auto graph = source;
    graph.nodes = nodes.data();
    graph.n_nodes = nodes.size();
    graph.uid = uid;
    return graph;
}
}
#endif
