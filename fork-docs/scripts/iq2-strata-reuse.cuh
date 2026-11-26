#pragma once

#include "iq3-strata-reuse.cuh"
#include "../../ggml/src/ggml-cuda/mmvq-iq2-strata.cuh"

namespace iq2_strata_reuse {

__launch_bounds__(waves * wave, 1)
static __global__ void gate_up(
        const block_iq2_s * up, const block_iq2_s * gate, const block_q8_1 * activation,
        const expert_group * groups, float * out) {
    gate_up_group(up, gate, activation, groups[blockIdx.y], out, nrows * weight_blocks);
}

} // namespace iq2_strata_reuse
