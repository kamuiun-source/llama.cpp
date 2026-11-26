#pragma once

#include "common.cuh"
#include "../ggml-hip-mtp3-compact-slot-map.h"

#ifdef GGML_USE_HIP
bool ggml_hip_mtp3_compact_slot_map_enabled();
void ggml_hip_set_mtp3_compact_slot_map(bool enabled);
void ggml_hip_copy_mtp3_compact_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * lookup);
#endif
