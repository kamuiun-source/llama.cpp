#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__linux__) && CUDART_VERSION >= 12030
#    if __has_include(<cupti_activity.h>) && __has_include(<nvtx3/nvToolsExt.h>)
#        define GGML_CUDA_HAVE_MMVQ_TRACE
bool ggml_cuda_mmvq_trace_enabled(int device);
#    endif
#endif

#ifdef GGML_USE_HIP
bool ggml_hip_gfx906_set_lds_modes(int iq2_s, int iq3_xxs);
int ggml_hip_gfx906_lds_modes();
bool ggml_hip_gfx906_iq3_strata_enabled();
void ggml_hip_gfx906_set_iq3_strata(bool enabled);

#endif

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
