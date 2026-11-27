#pragma once

namespace ggml_hip_narrow_iq_prefill {

// A 32-value Q8_1 block and the IQ grid bounds keep both numerators exact in FP32.
static __host__ __device__ constexpr float iq2_scaled_dot(int lo, int hi, int lo_scale, int hi_scale, float d) {
    return d * (float((2 * lo_scale + 1) * lo + (2 * hi_scale + 1) * hi) * 0.125f);
}

static __host__ __device__ constexpr float iq3_scaled_dot(int dot, int scale, float d) {
    return d * (float((2 * scale + 1) * dot) * 0.25f);
}

} // namespace ggml_hip_narrow_iq_prefill
