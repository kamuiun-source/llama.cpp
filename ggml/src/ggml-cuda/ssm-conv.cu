#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

static void ggml_cuda_ssm_conv_history_cpu(ggml_tensor *, int, int, void *) {
    GGML_ABORT("SSM convolution with history requires the owning CUDA backend");
}

bool ggml_cuda_is_ssm_conv_history(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return false;
    }
    ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    return params.fun == ggml_cuda_ssm_conv_history_cpu;
}

ggml_tensor * ggml_cuda_make_ssm_conv_history(ggml_context * ctx, ggml_tensor * history, ggml_tensor * x,
                                             ggml_tensor * weights, ggml_tensor * slots) {
    const int64_t channels = x->ne[0];
    const int64_t tokens = x->ne[1];
    if (channels <= 0 || channels > INT_MAX / 3 || channels % 128 != 0 || (tokens != 1 && tokens != 2) ||
            x->type != GGML_TYPE_F32 || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) ||
            x->nb[1] % sizeof(float) != 0 || x->nb[1] < size_t(channels) * sizeof(float) ||
            history->type != GGML_TYPE_F32 || history->ne[0] != 3 || history->ne[1] != channels ||
            !ggml_is_matrix(history) || !ggml_is_contiguous(history) ||
            weights->type != GGML_TYPE_F32 || weights->ne[0] != 4 || weights->ne[1] != channels ||
            !ggml_is_matrix(weights) || !ggml_is_contiguous(weights) ||
            slots->type != GGML_TYPE_F32 || slots->ne[0] != 3 * channels || slots->ne[1] != 2 ||
            !ggml_is_matrix(slots) || slots->nb[0] != sizeof(float) || slots->nb[1] % sizeof(float) != 0 ||
            slots->nb[1] < 3 * size_t(channels) * sizeof(float)) {
        return nullptr;
    }

    ggml_tensor * args[] = {history, x, weights, slots};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, channels, tokens, 1, 1, args, 4,
            ggml_cuda_ssm_conv_history_cpu, 1, nullptr);
}

template <int tokens>
static __global__ void ssm_conv_history_f32(const float * history, const float * x, const float * weights, float * y,
                                            float * slots, size_t x_stride, size_t slot_stride, int channels) {
    ggml_cuda_pdl_lc();
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync();
    const float old0 = history[3 * channel];
    const float old1 = history[3 * channel + 1];
    const float old2 = history[3 * channel + 2];
    const float x0 = x[channel];
    const float w0 = weights[4 * size_t(channel)];
    const float w1 = weights[4 * size_t(channel) + 1];
    const float w2 = weights[4 * size_t(channel) + 2];
    const float w3 = weights[4 * size_t(channel) + 3];

    float sum = 0.0f;
    sum += old0 * w0;
    sum += old1 * w1;
    sum += old2 * w2;
    sum += x0 * w3;
    y[channel] = ggml_cuda_op_silu_single(sum + 0.0f);

    float * first = slots + 3 * channel;
    if constexpr (tokens == 2) {
        const float x1 = x[x_stride + channel];
        sum = 0.0f;
        sum += old1 * w0;
        sum += old2 * w1;
        sum += x0 * w2;
        sum += x1 * w3;
        y[channels + channel] = ggml_cuda_op_silu_single(sum + 0.0f);

        first[0] = old2;
        first[1] = x0;
        first[2] = x1;
        first += slot_stride;
    }
    first[0] = old1;
    first[1] = old2;
    first[2] = x0;
}

void ggml_cuda_op_ssm_conv_history(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * history = dst->src[0];
    const auto * x = dst->src[1];
    const auto * weights = dst->src[2];
    auto * slots = dst->src[3];
    auto * y = static_cast<float *>(dst->data);
    const int channels = dst->ne[0];
    constexpr int threads = 128;
    const auto params = ggml_cuda_kernel_launch_params(dim3(channels / threads), threads, 0, ctx.stream());
    const auto kernel = dst->ne[1] == 2 ? ssm_conv_history_f32<2> : ssm_conv_history_f32<1>;
    ggml_cuda_kernel_launch(kernel, params, static_cast<const float *>(history->data),
            static_cast<const float *>(x->data), static_cast<const float *>(weights->data), y,
            static_cast<float *>(slots->data), x->nb[1] / sizeof(float), slots->nb[1] / sizeof(float), channels);

    static const bool logged = [] {
        GGML_LOG_INFO("CUDA SSM convolution history applied\n");
        return true;
    }();
    GGML_UNUSED(logged);
}
