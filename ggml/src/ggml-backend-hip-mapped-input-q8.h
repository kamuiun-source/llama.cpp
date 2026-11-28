#pragma once

#include "ggml-backend-mtp.h"

#include <atomic>

using ggml_backend_hip_mapped_input_q8_t = bool (*)(ggml_backend_t, ggml_tensor *, const void *);
using ggml_backend_hip_mapped_input_q8_scope_t = void (*)(ggml_backend_t, const ggml_tensor *);

inline const ggml_tensor * ggml_hip_mapped_input_q8_base(const ggml_tensor * tensor) {
    while (tensor && tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor;
}

inline bool ggml_hip_mapped_input_q8_layout(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == 2560 &&
            tensor->ne[1] == 1 && tensor->ne[2] == 3 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}
