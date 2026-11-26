#pragma once

#include "ggml.h"

namespace ggml_hip_mtp3_skip_cpu_zero {

inline constexpr char mode_file_env[] = "GGML_HIP_MTP3_SKIP_CPU_ZERO_MODE_FILE";
inline constexpr char setter_proc[] = "ggml_backend_hip_set_mtp3_skip_cpu_zero";
inline constexpr char mode_proc[] = "ggml_backend_hip_mtp3_skip_cpu_zero_mode";
inline constexpr int slot_param = 1; // op_params[0] belongs to slot-map compaction.

inline constexpr int slot_tag = 0x4d330000;
inline constexpr int slot_mask = 0xffff;

inline int encode_slot(int slot) {
    return slot_tag | slot;
}

inline int lookup_slot(const ggml_tensor * lookup) {
    if (!lookup || lookup->op != GGML_OP_GET_ROWS || lookup->type != GGML_TYPE_I32 ||
            lookup->ne[0] != 1 || lookup->ne[1] != 30) {
        return -1;
    }
    const int param = lookup->op_params[slot_param];
    return (param & ~slot_mask) == slot_tag ? param & slot_mask : -1;
}

} // namespace ggml_hip_mtp3_skip_cpu_zero
