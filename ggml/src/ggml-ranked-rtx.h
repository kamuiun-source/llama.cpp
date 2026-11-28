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
constexpr uint64_t empty_uid_bit = UINT64_C(1) << 63;
constexpr int inactive_id = -1;
constexpr int count_param = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
constexpr int layer_param = count_param - 1;

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
