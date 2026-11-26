#pragma once

#include "common.cuh"
#include "ggml-impl.h"

#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_next_router_top10_enabled();
void ggml_cuda_set_next_router_top10(bool enabled);
void ggml_cuda_launch_next_router_top10(const float * scores, int32_t * ids, cudaStream_t stream);

inline ggml_tensor * ggml_cuda_match_next_router_top10(const ggml_cgraph * graph, int i) {
    auto * order = graph->nodes[i];
    if (order->op != GGML_OP_ARGSORT || i + 2 >= graph->n_nodes || order->op_params[0] != GGML_SORT_ORDER_DESC) {
        return nullptr;
    }
    auto * view = graph->nodes[i + 1];
    auto * ids = graph->nodes[i + 2];
    const auto * scores = order->src[0];
    const auto * logits = scores && scores->op == GGML_OP_SOFT_MAX ? scores->src[0] : nullptr;
    // Score mode 1 exports logits; score mode 2 exports probabilities and selects 20 Experts.
    if (!logits || logits->op != GGML_OP_MUL_MAT || (logits->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (scores->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            std::strncmp(logits->name, "ffn_next_router_logits-", 23) != 0 ||
            scores->type != GGML_TYPE_F32 || scores->ne[0] != 512 || scores->ne[1] != 3 ||
            !ggml_is_matrix(scores) || !ggml_is_contiguous(scores) ||
            order->type != GGML_TYPE_I32 || !ggml_are_same_shape(scores, order) ||
            !ggml_is_contiguous(order) || view->op != GGML_OP_VIEW || view->src[0] != order || view->view_src != order ||
            view->view_offs != 0 || view->ne[0] != 10 || view->nb[0] != order->nb[0] || view->nb[1] != order->nb[1] ||
            view->ne[1] != 3 || !ggml_is_matrix(view) ||
            ids->op != GGML_OP_CONT || ids->src[0] != view || ids->type != GGML_TYPE_I32 ||
            !ggml_are_same_shape(view, ids) || !ggml_is_contiguous(ids) || !(ids->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            std::strncmp(ids->name, "ffn_next_router_topk-", 21) != 0) {
        return nullptr;
    }
    return ggml_can_fuse_subgraph(graph, i, { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_CONT }, { i + 2 }) ? ids : nullptr;
}

#endif
