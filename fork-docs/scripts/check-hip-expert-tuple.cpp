#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int n_experts = 16;
constexpr int n_tokens = 3;
constexpr int n_used = 10;
constexpr int n_embd = 2560;
constexpr int n_ff = 640;

constexpr int32_t routes[n_tokens][n_used] = {
    { 5,  1, 15,  2,  9,  7, 12,  0,  8,  4 },
    {13,  0,  6, 11,  2, 15,  3, 10,  9, 14 },
    { 8, 14,  4,  6,  1, 10,  7, 13, 11, 12 },
};

struct plane {
    const char * name;
    ggml_type type;
    int cols;
    int rows;

    size_t row_bytes() const { return ggml_row_size(type, cols); }
    size_t expert_bytes() const { return rows * row_bytes(); }
};

constexpr std::array<plane, 3> planes{{
    {"gate", GGML_TYPE_IQ3_XXS, n_embd, n_ff},
    {"up",   GGML_TYPE_IQ3_XXS, n_embd, n_ff},
    {"down", GGML_TYPE_IQ4_NL,  n_ff, n_embd},
}};

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

size_t round_up(size_t bytes, size_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

struct tuple_layout {
    std::array<size_t, planes.size()> offsets{};
    std::array<size_t, planes.size()> tails{};
    size_t stride;

    explicit tuple_layout(ggml_backend_buffer_type_t buft) {
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        ggml_context_ptr ctx(ggml_init({planes.size() * ggml_tensor_overhead(), nullptr, true}));
        check(bool(ctx), "cannot allocate tuple layout metadata");
        size_t end = 0;
        size_t stride_alignment = alignment;
        for (size_t p = 0; p < planes.size(); ++p) {
            const auto & dims = planes[p];
            auto * tensor = ggml_new_tensor_3d(ctx.get(), dims.type, dims.cols, dims.rows, n_experts);
            tails[p] = ggml_backend_buft_get_alloc_size(buft, tensor) - ggml_nbytes(tensor);
            offsets[p] = round_up(end, alignment);
            end = offsets[p] + dims.expert_bytes() + tails[p];
            stride_alignment = std::lcm(stride_alignment, ggml_type_size(dims.type));
        }
        stride = round_up(end, stride_alignment);
    }
};

struct expert_graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * raw = nullptr;
    std::array<ggml_tensor *, planes.size()> weights{};
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * activated;
    ggml_tensor * output;
    ggml_cgraph * graph;

    expert_graph(ggml_backend_t gpu, const tuple_layout * layout)
        : ctx(ggml_init({ 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true })) {
        check(bool(ctx), "cannot allocate graph metadata");
        if (layout) {
            raw = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, n_experts * layout->stride);
            ggml_set_name(raw, "expert_tuple_bank");
        }
        for (size_t p = 0; p < planes.size(); ++p) {
            const auto & dims = planes[p];
            auto * weight = ggml_new_tensor_3d(ctx.get(), dims.type, dims.cols, dims.rows, n_experts);
            ggml_set_name(weight, dims.name);
            if (layout) {
                // The raw bank owns the bytes. Each typed view walks the same Expert stride.
                weight->view_src = raw;
                weight->view_offs = layout->offsets[p];
                weight->nb[2] = layout->stride;
                weight->nb[3] = n_experts * layout->stride;
                check(weight->view_offs + ggml_nbytes(weight) <= ggml_nbytes(raw), "tuple plane exceeds bank");
            }
            weights[p] = weight;
        }
        input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, 1, n_tokens);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
        auto * gate = ggml_mul_mat_id(ctx.get(), weights[0], input, ids);
        auto * up = ggml_mul_mat_id(ctx.get(), weights[1], input, ids);
        activated = ggml_swiglu_split(ctx.get(), gate, up);
        output = ggml_mul_mat_id(ctx.get(), weights[2], activated, ids);
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu));
        check(buffer && !ggml_backend_buffer_is_host(buffer.get()), "cannot allocate ROCm device graph");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * op : {gate, up, activated, output}) {
            check(ggml_backend_supports_op(gpu, op), "ROCm does not support graph operation");
        }
    }
};

float value(std::mt19937 & rng) {
    return (float(rng() >> 16) / 32768.0f - 1.0f) * 0.1f;
}

std::vector<uint8_t> quantized_rows(const plane & dims, std::mt19937 & rng) {
    constexpr int pool_rows = 128;
    std::vector<float> values(pool_rows * dims.cols);
    for (auto & x : values) {
        x = value(rng);
    }
    const std::vector<float> imatrix(dims.cols, 1.0f);
    std::vector<uint8_t> rows(pool_rows * dims.row_bytes());
    check(ggml_quantize_chunk(dims.type, values.data(), rows.data(), 0, pool_rows, dims.cols, imatrix.data()) == rows.size(),
            "quantized row pool has wrong byte size");
    return rows;
}

void compare(const char * name, const ggml_tensor * dense, const ggml_tensor * tuple) {
    check(ggml_nbytes(dense) == ggml_nbytes(tuple), "result sizes differ");
    std::vector<float> expected(ggml_nelements(dense)), actual(expected.size());
    ggml_backend_tensor_get(dense, expected.data(), 0, ggml_nbytes(dense));
    ggml_backend_tensor_get(tuple, actual.data(), 0, ggml_nbytes(tuple));
    bool exact = true;
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite graph output");
        exact &= std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0;
        max_abs = std::fmax(max_abs, std::fabs(expected[i] - actual[i]));
    }
    std::printf("%s values=%zu bitwise_equal=%d max_abs=%g\n", name, actual.size(), exact, double(max_abs));
    check(exact, "tuple graph output differs from dense ROCm graph");
}

} // namespace

int main() try {
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_name("ROCm0");
    check(device && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "ROCm") == 0,
            "ROCm0 HIP backend required");
    ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
    check(bool(gpu), "cannot initialize ROCm0");

    const tuple_layout layout(ggml_backend_dev_buffer_type(device));
    expert_graph dense(gpu.get(), nullptr);
    expert_graph tuple(gpu.get(), &layout);
    const auto host_buft = ggml_backend_dev_host_buffer_type(device);
    ggml_backend_buffer_ptr pinned(host_buft ? ggml_backend_buft_alloc_buffer(host_buft, 2 * layout.stride) : nullptr);
    check(pinned && ggml_backend_buffer_get_type(pinned.get()) == host_buft, "genuine ROCm pinned host memory required");
    auto * original = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
    auto * readback = original + layout.stride;

    std::mt19937 rng(20261122);
    const std::array<std::vector<uint8_t>, planes.size()> pools{
        quantized_rows(planes[0], rng), quantized_rows(planes[1], rng), quantized_rows(planes[2], rng),
    };
    size_t dense_upload_bytes = 0;
    for (int expert = 0; expert < n_experts; ++expert) {
        std::memset(original, 0, layout.stride);
        for (size_t p = 0; p < planes.size(); ++p) {
            const auto & dims = planes[p];
            const size_t row_bytes = dims.row_bytes();
            auto * weight = original + layout.offsets[p];
            for (int row = 0; row < dims.rows; ++row) {
                const size_t index = rng() % (pools[p].size() / row_bytes);
                std::memcpy(weight + row * row_bytes, pools[p].data() + index * row_bytes, row_bytes);
            }
            ggml_backend_tensor_set(dense.weights[p], weight, expert * dims.expert_bytes(), dims.expert_bytes());
            dense_upload_bytes += dims.expert_bytes();
        }
        const size_t offset = expert * layout.stride;
        ggml_backend_tensor_set_async(gpu.get(), tuple.raw, original, offset, layout.stride);
        ggml_backend_tensor_get_async(gpu.get(), tuple.raw, readback, offset, layout.stride);
        ggml_backend_synchronize(gpu.get());
        check(std::memcmp(original, readback, layout.stride) == 0, "round-trip tuple bytes differ");
    }

    std::array<float, n_tokens * n_embd> activations;
    for (auto & x : activations) {
        x = value(rng);
    }
    for (auto * fixture : {&dense, &tuple}) {
        ggml_backend_tensor_set(fixture->input, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(fixture->ids, routes, 0, sizeof(routes));
        check(ggml_backend_graph_compute(gpu.get(), fixture->graph) == GGML_STATUS_SUCCESS, "ROCm graph failed");
    }

    std::printf("ROCm0 experts=%d width=%d top_k=%d stride=%zu offsets=%zu,%zu,%zu zero_tails=%zu,%zu,%zu\n",
            n_experts, n_tokens, n_used, layout.stride, layout.offsets[0], layout.offsets[1], layout.offsets[2],
            layout.tails[0], layout.tails[1], layout.tails[2]);
    std::printf("dense h2d=%zu bytes=%zu tuple h2d=%d d2h=%d bytes_per_direction=%zu roundtrip_exact=1\n",
            n_experts * planes.size(), dense_upload_bytes, n_experts, n_experts, n_experts * layout.stride);
    compare("gate/up SwiGLU", dense.activated, tuple.activated);
    compare("down", dense.output, tuple.output);
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "HIP Expert tuple fixture: %s\n", e.what());
    return 1;
}
