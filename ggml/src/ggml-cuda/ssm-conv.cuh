#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

bool ggml_cuda_is_ssm_conv_history(const ggml_tensor * op);
ggml_tensor * ggml_cuda_make_ssm_conv_history(ggml_context * ctx, ggml_tensor * history, ggml_tensor * x,
                                             ggml_tensor * weights, ggml_tensor * slots);
void ggml_cuda_op_ssm_conv_history(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
