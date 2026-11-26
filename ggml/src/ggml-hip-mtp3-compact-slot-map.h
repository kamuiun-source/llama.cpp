#pragma once

#include "ggml-hip-iq3-strata-plan.h"

namespace ggml_hip_mtp3_compact_slot_map {

constexpr const char * mode_file_env = "LLAMA_EXPERT_HIP_MTP3_COMPACT_SLOT_MAP_MODE_FILE";
constexpr int32_t tag = 0x334d5043;
constexpr int routes = ggml_hip_iq3_strata_plan::routes;

inline bool is_lookup(const ggml_tensor * lookup) {
    return lookup && lookup->op == GGML_OP_GET_ROWS && lookup->op_params[0] == tag;
}

inline void pack(int32_t * dst, const int32_t * experts, const int32_t * slots) {
    for (int i = 0; i < routes; ++i) {
        dst[i] = slots[experts[i]];
    }
}

} // namespace ggml_hip_mtp3_compact_slot_map
