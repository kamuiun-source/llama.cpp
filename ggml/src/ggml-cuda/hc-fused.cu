/*
Adapted from tandem.cpp at 64a6b186f650bca7638b7c5baccb118587d1f715.
https://github.com/ewhacc/tandem.cpp/blob/64a6b186f650bca7638b7c5baccb118587d1f715/tandem.patch

MIT License

Copyright (c) 2026 ewhacc

tandem.patch modifies llama.cpp; the parts of it taken from llama.cpp are
Copyright (c) 2023-2026 The ggml authors, under the same MIT licence.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hc-fused.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "mmvq.cuh"
#include "norm.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <cmath>

static constexpr int HC_STREAMS = 4;
static constexpr int HC_DOWN_ROWS = 4;
static constexpr int HC_THREADS = 256;
static constexpr int HC_THREADS_PER_STREAM = HC_THREADS / HC_STREAMS;
static constexpr int HC_WARPS_PER_STREAM = HC_THREADS_PER_STREAM / WARP_SIZE;
static constexpr int HC_UP_OUTPUTS = HC_THREADS / WARP_SIZE;

static __device__ __forceinline__ float2 hc_bf16_pair(uint32_t bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

// Each block projects a few rows over all streams. Apply each stream's RMS factor after its partial dot.
static __global__ void hc_mix_down(
        const float4 * x, const float4 * norm, const uint4 * down,
        float4 * normalized, float * low, float * scaled,
        int embd, int rank, int rows, float eps, float down_scale) {
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int stream = tid / HC_THREADS_PER_STREAM;
    const int stream_tid = tid % HC_THREADS_PER_STREAM;
    const int first_row = blockIdx.x * HC_DOWN_ROWS;
    const int embd8 = embd / 8;
    const int width8 = HC_STREAMS * embd8;
    float squares = 0.0f;
    float dot[HC_DOWN_ROWS] = {};
    __shared__ float square_sum[HC_THREADS / WARP_SIZE];
    __shared__ float dot_sum[HC_THREADS / WARP_SIZE][HC_DOWN_ROWS];
    __shared__ float rms[HC_STREAMS];

    for (int k = stream_tid; k < embd8; k += HC_THREADS_PER_STREAM) {
        const int col8 = stream * embd8 + k;
        const float4 a = x[2 * col8];
        const float4 b = x[2 * col8 + 1];
        const float4 ga = norm[2 * col8];
        const float4 gb = norm[2 * col8 + 1];
        squares += a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w + b.x*b.x + b.y*b.y + b.z*b.z + b.w*b.w;
        const float4 aw = make_float4(a.x*ga.x, a.y*ga.y, a.z*ga.z, a.w*ga.w);
        const float4 bw = make_float4(b.x*gb.x, b.y*gb.y, b.z*gb.z, b.w*gb.w);

#pragma unroll
        for (int j = 0; j < HC_DOWN_ROWS; ++j) {
            const int row = first_row + j;
            if (row < rows) {
                const uint4 weight = down[int64_t(row) * width8 + col8];
                const float2 w0 = hc_bf16_pair(weight.x);
                const float2 w1 = hc_bf16_pair(weight.y);
                const float2 w2 = hc_bf16_pair(weight.z);
                const float2 w3 = hc_bf16_pair(weight.w);
                dot[j] += w0.x*aw.x + w0.y*aw.y + w1.x*aw.z + w1.y*aw.w
                        + w2.x*bw.x + w2.y*bw.y + w3.x*bw.z + w3.y*bw.w;
            }
        }
    }

    squares = warp_reduce_sum(squares);
    if (lane == 0) {
        square_sum[warp] = squares;
    }
#pragma unroll
    for (int j = 0; j < HC_DOWN_ROWS; ++j) {
        const float sum = warp_reduce_sum(dot[j]);
        if (lane == 0) {
            dot_sum[warp][j] = sum;
        }
    }
    __syncthreads();

    if (tid < HC_STREAMS) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < HC_WARPS_PER_STREAM; ++w) {
            sum += square_sum[tid * HC_WARPS_PER_STREAM + w];
        }
        rms[tid] = rsqrtf(sum / embd + eps);
    }
    __syncthreads();

    if (tid < HC_DOWN_ROWS && first_row + tid < rows) {
        float result = 0.0f;
#pragma unroll
        for (int s = 0; s < HC_STREAMS; ++s) {
            float partial = 0.0f;
#pragma unroll
            for (int w = 0; w < HC_WARPS_PER_STREAM; ++w) {
                partial += dot_sum[s * HC_WARPS_PER_STREAM + w][tid];
            }
            result += partial * rms[s];
        }
        result *= down_scale;
        const int row = first_row + tid;
        if (row < rank) {
            low[row] = result / (1.0f + expf(-result));
        }
        if (scaled) {
            scaled[row] = result;
        }
    }

    // The next kernel can write graph outputs even if the allocator reused x.
    const float r = rms[stream];
    for (int k = blockIdx.x * HC_THREADS_PER_STREAM + stream_tid;
         k < 2 * embd8; k += gridDim.x * HC_THREADS_PER_STREAM) {
        const int col4 = stream * (2 * embd8) + k;
        const float4 a = x[col4];
        const float4 g = norm[col4];
        normalized[col4] = make_float4((a.x*r)*g.x, (a.y*r)*g.y, (a.z*r)*g.z, (a.w*r)*g.w);
    }
}

// One warp per output, with one eight-lane group per stream's gate row.
static __global__ void hc_mix_up(
        const float * normalized, const float * low, const uint4 * up, const float * scaled,
        float * norm_dst, float * scaled_dst, float * dst, int embd, int rank, int rows, float pre_scale) {
    extern __shared__ float low_shared[];
    for (int j = threadIdx.x; j < rank; j += blockDim.x) {
        low_shared[j] = low[j];
    }
    if (scaled_dst && blockIdx.x == 0) {
        for (int j = threadIdx.x; j < rows; j += blockDim.x) {
            scaled_dst[j] = scaled[j];
        }
    }
    __syncthreads();

    constexpr int lanes_per_stream = WARP_SIZE / HC_STREAMS;
    const int e = blockIdx.x * HC_UP_OUTPUTS + threadIdx.x / WARP_SIZE;
    if (e >= embd) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int stream = lane / lanes_per_stream;
    const int channel = stream * embd + e;
    const uint4 * row = up + int64_t(channel) * (rank / 8);
    float gate = 0.0f;
    for (int k = lane % lanes_per_stream; k < rank / 8; k += lanes_per_stream) {
        const uint4 weight = row[k];
        const uint32_t pairs[] = { weight.x, weight.y, weight.z, weight.w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 w = hc_bf16_pair(pairs[j]);
            gate += w.x * low_shared[8*k + 2*j] + w.y * low_shared[8*k + 2*j + 1];
        }
    }
    gate = warp_reduce_sum<lanes_per_stream>(gate);
    float mixed = 0.0f;
    if (lane % lanes_per_stream == 0) {
        const float value = normalized[channel];
        norm_dst[channel] = value;
        mixed = value * (1.0f / (1.0f + expf(-gate)));
    }
    mixed += __shfl_xor_sync(0xffffffff, mixed, lanes_per_stream, WARP_SIZE);
    mixed += __shfl_xor_sync(0xffffffff, mixed, 2 * lanes_per_stream, WARP_SIZE);
    if (lane == 0) {
        dst[e] = pre_scale * mixed;
    }
}

static __global__ void hc_mix_silu_q8_1(const float * down, block_q8_1 * low, float scale) {
    const int k = blockIdx.x * QK8_1 + threadIdx.x;
    const float scaled = scale * down[k] + 0.0f;
    ggml_cuda_quantize_q8_1_block(ggml_cuda_op_silu_single(scaled), low + blockIdx.x, threadIdx.x);
}

// One warp computes one gate row. The four rows for the same channel share a block.
static __global__ void hc_mix_up_q8_0(
        const float * normalized, const block_q8_1 * low, const block_q8_0 * up,
        float * norm_dst, float * dst, int embd, int blocks_per_row, float pre_scale) {
    constexpr int lanes_per_q8_block = QI8_0 / VDR_Q8_0_Q8_1_MMVQ;
    const int e = blockIdx.x;
    const int stream = threadIdx.y;
    const int channel = stream * embd + e;
    const block_q8_0 * row = up + int64_t(channel) * blocks_per_row;
    __shared__ float normalized_stream[HC_STREAMS];
    __shared__ float sigmoid_stream[HC_STREAMS];

    const int lane = threadIdx.x;
    const int iqs = (lane % lanes_per_q8_block) * VDR_Q8_0_Q8_1_MMVQ;
    float gate = 0.0f;
    for (int k = lane / lanes_per_q8_block; k < blocks_per_row; k += WARP_SIZE / lanes_per_q8_block) {
        gate += vec_dot_q8_0_q8_1(row, low + k, k, iqs);
    }
    gate = warp_reduce_sum(gate);
    if (lane == 0) {
        const float value = normalized[channel];
        norm_dst[channel] = value;
        normalized_stream[stream] = value;
        sigmoid_stream[stream] = 1.0f / (1.0f + expf(-gate));
    }
    __syncthreads();

    if (lane == 0 && stream == 0) {
        float mixed = 0.0f;
#pragma unroll
        for (int s = 0; s < HC_STREAMS; ++s) {
            mixed += normalized_stream[s] * sigmoid_stream[s];
        }
        dst[e] = pre_scale * mixed;
    }
}

template<ggml_type type>
static void hc_mix_log_applied() {
    // This runs at launch or graph capture, so it does not count decode tokens.
    static const bool logged = [] {
        GGML_LOG_INFO("CUDA HC mix fusion applied (single-token %s projections)\n", ggml_type_name(type));
        return true;
    }();
    GGML_UNUSED(logged);
}

static bool hc_shape(const ggml_tensor * t, ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    return t && t->type == type && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == 1 &&
           ggml_is_contiguous(t);
}

static bool hc_reshape(const ggml_tensor * view, const ggml_tensor * src) {
    return view->op == GGML_OP_RESHAPE && view->src[0] == src && view->view_offs == 0 && view->data == src->data;
}

static bool hc_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

static bool hc_matmul_f32(const ggml_tensor * matmul) {
    const ggml_prec acc = ggml_prec_get_acc(matmul);
    const ggml_prec src = ggml_prec_get_src(matmul, 1);
    return (acc == GGML_PREC_UNDEFINED || acc == GGML_PREC_F32) &&
           (src == GGML_PREC_UNDEFINED || src == GGML_PREC_F32) &&
           ggml_get_op_params_i32(matmul, 1) == GGML_HINT_NONE;
}

static int hc_mix_fuse_at(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    constexpr ggml_op compute_ops[] = {
        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_SCALE,
        GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_DSV4_HC_PRE,
    };
    constexpr int n_compute = 7;
    constexpr int max_nodes = n_compute + 4;
    ggml_tensor * nodes[n_compute];
    int indices[n_compute];
    ggml_op ops[max_nodes];
    int count = 0;
    int n = 0;
    while (count < max_nodes && n < n_compute && count < cgraph->n_nodes - i) {
        ggml_tensor * node = cgraph->nodes[i + count];
        ops[count] = node->op;
        if (node->op == compute_ops[n]) {
            nodes[n] = node;
            indices[n++] = i + count;
        } else if (node->op != GGML_OP_RESHAPE && node->op != GGML_OP_VIEW) {
            return 0;
        }
        ++count;
    }
    if (n != n_compute) {
        return 0;
    }

    ggml_tensor * rms = nodes[0];
    ggml_tensor * mul = nodes[1];
    const ggml_tensor * down = nodes[2];
    ggml_tensor * scale = nodes[3];
    const ggml_tensor * silu = nodes[4];
    const ggml_tensor * up = nodes[5];
    ggml_tensor * pre = nodes[6];

    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * gamma = mul->src[1];
    const ggml_tensor * wd = down->src[0];
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * flat = down->src[1];
    const ggml_tensor * low = silu->src[0];
    const ggml_tensor * pre_x = pre->src[0];
    const ggml_tensor * pre_gate = pre->src[1];
    if (!x || !wd || !wu || !flat || !low || !pre_x || !pre_gate) {
        return 0;
    }
    const int64_t embd = x->ne[0];
    const int64_t rank = wu->ne[0];
    if (embd <= 0 || embd > 65536 || embd % 8 != 0 || rank <= 0 || rank > 2048 || rank % 8 != 0) {
        return 0;
    }
    const int64_t width = HC_STREAMS * embd;
    const int64_t rows = wd->ne[1];
    const bool combined = rows == rank + HC_STREAMS;
    if (!combined && rows != rank) {
        return 0;
    }
    const bool q8_0 = wd->type == GGML_TYPE_Q8_0;
    if (q8_0 && (combined || width % QK8_0 != 0 || rank % QK8_0 != 0 ||
            !ggml_cuda_should_use_mmvq(GGML_TYPE_Q8_0, ggml_cuda_info().devices[ctx.device].cc, 1))) {
        return 0;
    }
    const ggml_type projection_type = q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_BF16;

    const float eps = ggml_get_op_params_f32(rms, 0);
    const float down_scale = ggml_get_op_params_f32(scale, 0);
    const float pre_scale = ggml_get_op_params_f32(pre, 0);
    if (!(eps > 0.0f && std::isfinite(eps)) || down_scale != 0.25f ||
            ggml_get_op_params_f32(scale, 1) != 0.0f || pre_scale != 0.25f ||
            ggml_get_op_params_i32(pre, 1) != 1 || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return 0;
    }
    if (mul->src[0] != rms || !hc_reshape(flat, mul) || scale->src[0] != down ||
            up->src[1] != silu || !hc_reshape(pre_x, flat) || !hc_reshape(pre_gate, up) ||
            !hc_matmul_f32(down) || !hc_matmul_f32(up)) {
        return 0;
    }
    if (combined ? !(low->op == GGML_OP_VIEW && low->src[0] == scale && low->view_offs == 0 &&
                    low->data == scale->data && low->nb[1] == scale->nb[1]) : low != scale) {
        return 0;
    }
    if (!hc_shape(x, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(rms, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(gamma, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(mul, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(flat, GGML_TYPE_F32, width) ||
            !hc_shape(wd, projection_type, width, rows) ||
            !hc_shape(down, GGML_TYPE_F32, rows) ||
            !hc_shape(scale, GGML_TYPE_F32, rows) ||
            !hc_shape(low, GGML_TYPE_F32, rank) ||
            !hc_shape(silu, GGML_TYPE_F32, rank) ||
            !hc_shape(wu, projection_type, rank, width) ||
            !hc_shape(up, GGML_TYPE_F32, width) ||
            !hc_shape(pre_x, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(pre_gate, GGML_TYPE_F32, embd, HC_STREAMS) ||
            !hc_shape(pre, GGML_TYPE_F32, embd)) {
        return 0;
    }

    // Include only the reshape and view nodes used by this chain.
    const ggml_tensor * views[] = { flat, pre_x, pre_gate, combined ? low : nullptr };
    int n_views = 0;
    int flat_index = -1;
    for (int j = 0; j < count; ++j) {
        if (ops[j] == GGML_OP_RESHAPE || ops[j] == GGML_OP_VIEW) {
            const ggml_tensor * view = cgraph->nodes[i + j];
            if (std::find(std::begin(views), std::end(views), view) == std::end(views)) {
                return 0;
            }
            if (view == flat) {
                flat_index = i + j;
            }
            ++n_views;
        }
    }
    if (flat_index < 0 || n_views != (combined ? 4 : 3)) {
        return 0;
    }

    // The separate inject projection may still read the normalized input through its flat view.
    const int outputs[] = { indices[1], indices[6], flat_index, indices[3] };
    if (!ggml_can_fuse_subgraph(cgraph, i, count, ops, outputs, combined ? 4 : 3)) {
        return 0;
    }

    const auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor * tensors[] = { x, gamma, wd, wu, mul, pre, scale };
    for (const ggml_tensor * t : tensors) {
        if (!t->buffer || ggml_backend_buffer_get_type(t->buffer) != buft ||
                !t->data || reinterpret_cast<uintptr_t>(t->data) % 16 != 0) {
            return 0;
        }
    }
    if (hc_overlap(mul, pre) || (combined && (hc_overlap(mul, scale) || hc_overlap(pre, scale))) ||
            hc_overlap(mul, wu) || hc_overlap(pre, wu) || (combined && hc_overlap(scale, wu))) {
        return 0;
    }

    if (q8_0) {
        if (ggml_backend_buffer_get_usage(wd->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                ggml_backend_buffer_get_usage(wu->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            return 0;
        }

        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), width + rank);
        ggml_cuda_pool_alloc<block_q8_1> low_quant(ctx.pool(), rank / QK8_1);
        ggml_tensor norm_tmp = *mul;
        norm_tmp.data = scratch.get();
        ggml_cuda_op_rms_norm_fused(ctx, rms, &norm_tmp);

        ggml_tensor flat_tmp = *flat;
        flat_tmp.data = norm_tmp.data;
        ggml_tensor down_tmp = *down;
        down_tmp.data = scratch.get() + width;
        ggml_cuda_mul_mat_vec_q(ctx, wd, &flat_tmp, nullptr, &down_tmp);

        const cudaStream_t stream = ctx.stream();
        hc_mix_silu_q8_1<<<rank / QK8_1, QK8_1, 0, stream>>>(
                static_cast<const float *>(down_tmp.data), low_quant.get(), down_scale);
        hc_mix_up_q8_0<<<embd, dim3(WARP_SIZE, HC_STREAMS), 0, stream>>>(
                scratch.get(), low_quant.get(), static_cast<const block_q8_0 *>(wu->data),
                static_cast<float *>(mul->data), static_cast<float *>(pre->data), embd, rank / QK8_0, pre_scale);
        CUDA_CHECK(cudaGetLastError());
        hc_mix_log_applied<GGML_TYPE_Q8_0>();
        return count - 1;
    }

    // This changes the reduction order and moves RMS scaling after each stream's down dot.
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), width + rank + (combined ? rows : 0));
    float * normalized = scratch.get();
    float * low_buf = normalized + width;
    float * scaled = combined ? low_buf + rank : nullptr;
    const cudaStream_t stream = ctx.stream();
    hc_mix_down<<<(rows + HC_DOWN_ROWS - 1) / HC_DOWN_ROWS, HC_THREADS, 0, stream>>>(
            static_cast<const float4 *>(x->data), static_cast<const float4 *>(gamma->data),
            static_cast<const uint4 *>(wd->data), reinterpret_cast<float4 *>(normalized), low_buf, scaled,
            embd, rank, rows, eps, down_scale);
    hc_mix_up<<<(embd + HC_UP_OUTPUTS - 1) / HC_UP_OUTPUTS, HC_THREADS, rank * sizeof(float), stream>>>(
            normalized, low_buf, static_cast<const uint4 *>(wu->data), scaled,
            static_cast<float *>(mul->data), combined ? static_cast<float *>(scale->data) : nullptr,
            static_cast<float *>(pre->data), embd, rank, rows, pre_scale);
    CUDA_CHECK(cudaGetLastError());
    hc_mix_log_applied<GGML_TYPE_BF16>();
    return count - 1;
}

#endif

int ggml_cuda_hc_mix_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
    GGML_UNUSED(i);
    return 0;
#else
    if (ggml_cuda_info().devices[ctx.device].cc < GGML_CUDA_CC_AMPERE) {
        return 0;
    }
    return hc_mix_fuse_at(ctx, cgraph, i);
#endif
}
