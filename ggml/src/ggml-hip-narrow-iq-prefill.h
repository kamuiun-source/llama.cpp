#pragma once

#include "ggml.h"

namespace ggml_hip_narrow_iq_prefill {

inline constexpr char env[] = "GGML_HIP_GFX906_IQ_MMVQ_PRESERVE_FRACTION";
inline constexpr int32_t lookup_tag = 0x4e495150;

inline bool tagged_ids(const ggml_tensor * ids) {
    const auto * lookup = ids ? ids->view_src : nullptr;
    return ids && ids->type == GGML_TYPE_I32 && ids->ne[0] == 10 && ids->ne[1] >= 1 && ids->ne[1] <= 2 &&
        lookup && lookup->op == GGML_OP_GET_ROWS && lookup->type == GGML_TYPE_I32 &&
        lookup->op_params[0] == lookup_tag;
}

} // namespace ggml_hip_narrow_iq_prefill
