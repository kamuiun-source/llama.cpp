#pragma once

#include "ggml-impl.h"

// A private MUL_MAT parameter. Precision and hints use indices 0-3.
static constexpr int ggml_mtp_q8_head_param = 15;
static constexpr int32_t ggml_mtp_q8_head_pair = 0x6d747038;

inline void ggml_mtp_q8_head_set_pair_rows(ggml_tensor * node) {
    GGML_ASSERT(node->op == GGML_OP_MUL_MAT);
    ggml_set_op_params_i32(node, ggml_mtp_q8_head_param, ggml_mtp_q8_head_pair);
}

inline bool ggml_mtp_q8_head_has_pair_rows(const ggml_tensor * node) {
    return node->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(node, ggml_mtp_q8_head_param) == ggml_mtp_q8_head_pair;
}
