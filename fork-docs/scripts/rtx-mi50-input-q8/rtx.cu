#include "kernel-helpers.cuh"

#ifdef GGML_CUDA_USE_PDL
bool ggml_cuda_kernel_can_use_pdl(const void *) {
    return false; // sm_89 does not use PDL.
}
#endif

extern "C" void rtx_input_q8(const void * source, void * mapped_q8, void * stream) {
    quantize_fixed(source, mapped_q8, static_cast<cudaStream_t>(stream));
}
