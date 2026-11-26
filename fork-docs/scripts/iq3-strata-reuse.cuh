#pragma once

#include "../../ggml/src/ggml-cuda/mmvq-iq3-strata.cuh"

namespace iq3_strata_reuse {

__launch_bounds__(waves * wave, 1)
static __global__ void gate_up(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * activation,
        const expert_group * groups, float * out) {
    gate_up_group(up, gate, activation, groups[blockIdx.y], out, nrows * weight_blocks);
}

} // namespace iq3_strata_reuse
