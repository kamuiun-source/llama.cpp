#include "vecdotq.cuh"
#include "unary.cuh"
#include "quantize.cuh"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

constexpr int k = 2560;
constexpr int m = 640;
constexpr int positions = 3;
constexpr int top_k = 10;
constexpr int experts = 16;
constexpr int max_cold_experts = 4;
constexpr int wave_size = 64;
constexpr int rows_per_block = 4;
constexpr int grid_size = m / rows_per_block;
constexpr int blocks_per_row = k / QK_K;
constexpr int input_stride = k / QK8_1;
constexpr size_t output_count = positions * top_k * m;
constexpr size_t plane_bytes = m * blocks_per_row * sizeof(block_iq3_xxs);
constexpr size_t down_bytes = k * (m / QK4_NL) * sizeof(block_iq4_nl);
constexpr size_t alignment = 128;

constexpr size_t round_up(size_t bytes, size_t multiple) {
    return (bytes + multiple - 1) / multiple * multiple;
}

constexpr size_t up_offset = round_up(plane_bytes, alignment);
constexpr size_t down_offset = round_up(up_offset + plane_bytes, alignment);
// Match the tuple stride alignment for both IQ3_XXS and IQ4_NL views.
constexpr size_t tuple_alignment = std::lcm(alignment, std::lcm(sizeof(block_iq3_xxs), sizeof(block_iq4_nl)));
constexpr size_t tuple_bytes = round_up(down_offset + down_bytes, tuple_alignment);

constexpr int expert_ids[positions][top_k] = {
    {  5,  1, 15,  2,  9,  7, 12,  0,  8,  4 },
    { 13,  0,  6, 11,  2, 15,  3, 10,  9, 14 },
    {  8, 14,  4,  6,  1, 10,  7, 13, 11, 12 },
};

struct route {
    int ranks[positions];
};

struct hot_sample {
    unsigned first_ticket;
    unsigned last_ticket;
    uint64_t first_tick;
    uint64_t last_tick;
};

__device__ __forceinline__ unsigned load_ticket(const unsigned * ticket) {
    return __hip_atomic_load(const_cast<unsigned *>(ticket), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void prepare_q8_1(const float * activations, block_q8_1 * quantized) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    ggml_cuda_quantize_q8_1_block(activations[index], quantized + index / QK8_1, index % QK8_1);
}

__global__ void expert_swiglu(const uint8_t * bank, const block_q8_1 * input, const route * routes,
                             const unsigned * ticket, unsigned base, int hot_experts, uint64_t timeout_ticks,
                             float * output, unsigned * timeouts, hot_sample * samples) {
    const int lane = threadIdx.x % wave_size;
    const int row = blockIdx.x * rows_per_block + threadIdx.x / wave_size;
    constexpr int lanes_per_block = QI3_XXS / VDR_IQ3_XXS_Q8_1_MMVQ;
    const int iqs = (lane % lanes_per_block) * VDR_IQ3_XXS_Q8_1_MMVQ;
    if (samples && threadIdx.x == 0) {
        samples[blockIdx.x].first_ticket = load_ticket(ticket) - base;
        samples[blockIdx.x].first_tick = wall_clock64();
    }

    for (int expert = 0; expert < experts; ++expert) {
        if (expert == hot_experts) {
            // Every wave in this block finishes its hot rows before any cold wait.
            __syncthreads();
            if (samples && threadIdx.x == 0) {
                samples[blockIdx.x].last_tick = wall_clock64();
                samples[blockIdx.x].last_ticket = load_ticket(ticket) - base;
            }
        }
        if (expert >= hot_experts) {
            const unsigned expected = base + (expert - hot_experts + 1);
            const uint64_t begin = wall_clock64();
            // Each lane acquires before it can read the copied tuple.
            while (!__all(load_ticket(ticket) >= expected)) {
                if (__any(wall_clock64() - begin >= timeout_ticks)) {
                    if (lane == 0) {
                        atomicAdd(timeouts, 1u);
                    }
                    return;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }

        const uint8_t * tuple = bank + expert * tuple_bytes;
        const auto * gate = reinterpret_cast<const block_iq3_xxs *>(tuple) + row * blocks_per_row;
        const auto * up = reinterpret_cast<const block_iq3_xxs *>(tuple + up_offset) + row * blocks_per_row;
        for (int position = 0; position < positions; ++position) {
            const int rank = routes[expert].ranks[position];
            if (rank < 0) {
                continue;
            }
            float g = 0.f;
            float u = 0.f;
            for (int block = lane / lanes_per_block; block < blocks_per_row; block += wave_size / lanes_per_block) {
                const auto * x = input + position * input_stride + block * (QK_K / QK8_1);
                g += vec_dot_iq3_xxs_q8_1(gate, x, block, iqs);
                u += vec_dot_iq3_xxs_q8_1(up, x, block, iqs);
            }
            g = warp_reduce_sum<wave_size>(g);
            u = warp_reduce_sum<wave_size>(u);
            if (lane == 0) {
                output[(position * top_k + rank) * m + row] = ggml_cuda_op_silu_single(g) * u;
            }
        }
    }
}

void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void check_hip(hipError_t status, const char * call) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(status));
    }
}

#define HIP(call) check_hip((call), #call)

template<typename T>
void allocate(T * & ptr, size_t count) {
    HIP(hipMalloc(&ptr, count * sizeof(T)));
}

struct production_graph {
    ggml_backend_ptr backend;
    ggml_context_ptr weights_ctx;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr weights;
    ggml_backend_buffer_ptr tensors;
    ggml_tensor * bank = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit production_graph(const char * library) {
        const auto reg = ggml_backend_load(library);
        check(reg && std::strcmp(ggml_backend_reg_name(reg), "ROCm") == 0, "cannot load requested ROCm backend");
        const auto device = ggml_backend_dev_by_name("ROCm0");
        check(device && ggml_backend_dev_backend_reg(device) == reg, "requested backend has no ROCm0 device");
        backend.reset(ggml_backend_dev_init(device, nullptr));
        check(bool(backend), "cannot initialize ROCm0 backend");

        weights_ctx.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
        ctx.reset(ggml_init({7 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        check(weights_ctx && ctx, "cannot allocate production graph metadata");
        bank = ggml_new_tensor_1d(weights_ctx.get(), GGML_TYPE_I8, experts * tuple_bytes);
        ggml_set_name(bank, "expert_tuple_bank");
        auto make_plane = [&](size_t offset, const char * name) {
            auto * plane = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_IQ3_XXS, k, m, experts);
            plane->view_src = bank;
            plane->view_offs = offset;
            plane->nb[2] = tuple_bytes;
            plane->nb[3] = experts * tuple_bytes;
            check(offset + ggml_nbytes(plane) <= ggml_nbytes(bank), "weight plane exceeds tuple bank");
            ggml_set_name(plane, name);
            return plane;
        };
        auto * gate_weight = make_plane(0, "expert_bank.blk.0.ffn_gate_exps.weight");
        auto * up_weight = make_plane(up_offset, "expert_bank.blk.0.ffn_up_exps.weight");
        input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, 1, positions);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, top_k, positions);
        auto * gate = ggml_mul_mat_id(ctx.get(), gate_weight, input, ids);
        auto * up = ggml_mul_mat_id(ctx.get(), up_weight, input, ids);
        output = ggml_swiglu_split(ctx.get(), gate, up);
        ggml_set_name(gate, "ffn_moe_gate-0");
        ggml_set_name(up, "ffn_moe_up-0");
        ggml_set_name(output, "ffn_moe_swiglu-0");
        graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        for (const auto * op : {gate, up, output}) {
            check(ggml_backend_supports_op(backend.get(), op), "ROCm0 does not support the production fixture");
        }

        weights.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), backend.get()));
        tensors.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        check(weights && tensors && !ggml_backend_buffer_is_host(weights.get()), "cannot allocate ROCm0 tensors");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(ids, expert_ids, 0, sizeof(expert_ids));
    }

    void run() {
        const auto status = ggml_backend_graph_compute_async(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
        check(status == GGML_STATUS_SUCCESS, "production graph compute failed");
    }

    void poison_output() const {
        ggml_backend_tensor_memset(output, 0xff, 0, ggml_nbytes(output));
    }

    std::vector<float> read_output() const {
        std::vector<float> values(output_count);
        ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
};

using bench_clock = std::chrono::steady_clock;

struct timing {
    double total_us;
    double dma_to_done_us;
    double graph_us = 0;
};

double elapsed_us(bench_clock::time_point begin, bench_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct fixture {
    hipStream_t copy_stream = nullptr;
    hipStream_t compute_stream = nullptr;
    std::array<hipEvent_t, max_cold_experts> copied{};
    int cold_experts = 0;
    int hot_experts = 0;
    uint8_t * payload = nullptr;
    uint8_t * bank = nullptr;
    production_graph * production = nullptr;
    block_q8_1 * input = nullptr;
    route * routes = nullptr;
    float * output = nullptr;
    unsigned * ticket = nullptr;
    unsigned * device_ticket = nullptr;
    unsigned * timeouts = nullptr;
    hot_sample * samples = nullptr;
    unsigned base = 0;
    uint64_t timeout_ticks = 0;

    // Drain both streams before freeing any buffers, including on a failed submission.
    ~fixture() {
        if (copy_stream) {
            static_cast<void>(hipStreamSynchronize(copy_stream));
        }
        if (compute_stream) {
            static_cast<void>(hipStreamSynchronize(compute_stream));
        }
        for (auto event : copied) {
            if (event) {
                static_cast<void>(hipEventDestroy(event));
            }
        }
        if (copy_stream) {
            static_cast<void>(hipStreamDestroy(copy_stream));
        }
        if (compute_stream) {
            static_cast<void>(hipStreamDestroy(compute_stream));
        }
        if (payload) {
            static_cast<void>(hipHostFree(payload));
        }
        if (ticket) {
            static_cast<void>(hipHostFree(ticket));
        }
        if (bank && !production) {
            static_cast<void>(hipFree(bank));
        }
        for (void * ptr : {static_cast<void *>(input), static_cast<void *>(routes),
                           static_cast<void *>(output), static_cast<void *>(timeouts), static_cast<void *>(samples)}) {
            if (ptr) {
                static_cast<void>(hipFree(ptr));
            }
        }
    }

    void prepare(production_graph * comparator, int cold_count) {
        production = comparator;
        cold_experts = cold_count;
        hot_experts = experts - cold_experts;
        hipDeviceProp_t properties{};
        HIP(hipGetDeviceProperties(&properties, 0));
        check(std::strncmp(properties.gcnArchName, "gfx906", 6) == 0 && properties.warpSize == wave_size,
              "gfx906 device 0 is required");
        check(properties.canMapHostMemory && properties.asyncEngineCount > 0,
              "device must support mapped host memory and an asynchronous copy engine");
        int wall_clock_khz = 0;
        HIP(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, 0));
        check(wall_clock_khz > 0, "invalid device wall clock rate");
        timeout_ticks = 1000ull * wall_clock_khz;
        HIP(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
        HIP(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));
        for (int i = 0; i < cold_experts; ++i) {
            HIP(hipEventCreateWithFlags(&copied[i], hipEventDisableTiming | hipEventReleaseToSystem));
        }
        HIP(hipHostMalloc(&payload, 2 * experts * tuple_bytes, hipHostMallocDefault));
        HIP(hipHostMalloc(&ticket, sizeof(*ticket), hipHostMallocMapped | hipHostMallocCoherent));
        new (ticket) unsigned(0);
        static_assert(__atomic_always_lock_free(sizeof(unsigned), nullptr), "ticket stores must be lock free");
        HIP(hipHostGetDevicePointer(reinterpret_cast<void **>(&device_ticket), ticket, 0));
        if (production) {
            bank = static_cast<uint8_t *>(production->bank->data);
        } else {
            allocate(bank, experts * tuple_bytes);
        }
        allocate(input, positions * input_stride);
        allocate(routes, experts);
        allocate(output, output_count);
        allocate(timeouts, 1);
        allocate(samples, grid_size);

        route host_routes[experts];
        for (auto & route : host_routes) {
            std::fill_n(route.ranks, positions, -1);
        }
        for (int position = 0; position < positions; ++position) {
            for (int rank = 0; rank < top_k; ++rank) {
                host_routes[expert_ids[position][rank]].ranks[position] = rank;
            }
        }
        HIP(hipMemcpy(routes, host_routes, sizeof(host_routes), hipMemcpyHostToDevice));

        std::mt19937 rng(1234);
        const auto next_value = [&]() { return static_cast<float>(rng() >> 16) / 32768.0f - 1.0f; };
        std::vector<float> activations(positions * k);
        std::generate(activations.begin(), activations.end(), next_value);
        if (production) {
            ggml_backend_tensor_set(production->input, activations.data(), 0, activations.size() * sizeof(float));
        }
        float * device_activations = production ? static_cast<float *>(production->input->data) : nullptr;
        if (!production) {
            allocate(device_activations, activations.size());
        }
        // Use the production Q8_1 quantizer on both paths to isolate the gate/up calculation.
        const auto status = [&]() {
            if (!production) {
                const auto uploaded = hipMemcpy(device_activations, activations.data(), activations.size() * sizeof(float),
                                                hipMemcpyHostToDevice);
                if (uploaded != hipSuccess) {
                    return uploaded;
                }
            }
            constexpr int threads = CUDA_QUANTIZE_BLOCK_SIZE;
            static_assert((positions * k) % threads == 0);
            prepare_q8_1<<<positions * k / threads, threads, 0, compute_stream>>>(device_activations, input);
            const auto launched = hipGetLastError();
            return launched == hipSuccess ? hipStreamSynchronize(compute_stream) : launched;
        }();
        if (!production) {
            const auto freed = hipFree(device_activations);
            HIP(status);
            HIP(freed);
        }
        HIP(status);

        constexpr int pool_rows = 128;
        constexpr size_t row_bytes = plane_bytes / m;
        std::vector<float> pool_input(pool_rows * k);
        const std::vector<float> imatrix(k, 1.f);
        std::vector<uint8_t> pool(pool_rows * row_bytes);
        std::memset(payload, 0, 2 * experts * tuple_bytes);
        ggml_quantize_init(GGML_TYPE_IQ3_XXS);
        for (size_t offset : {size_t{0}, up_offset}) {
            std::generate(pool_input.begin(), pool_input.end(), next_value);
            check(ggml_quantize_chunk(GGML_TYPE_IQ3_XXS, pool_input.data(), pool.data(), 0, pool_rows, k, imatrix.data()) ==
                  pool.size(), "unexpected quantized row size");
            for (int variant = 0; variant < 2; ++variant) {
                // Generate the same payload for each physical slot at every miss count.
                for (int expert = variant == 0 ? 0 : experts - max_cold_experts; expert < experts; ++expert) {
                    auto * plane = payload + (variant * experts + expert) * tuple_bytes + offset;
                    for (int row = 0; row < m; ++row) {
                        std::memcpy(plane + row * row_bytes, pool.data() + (rng() % pool_rows) * row_bytes, row_bytes);
                    }
                }
            }
        }
        ggml_quantize_free();
        HIP(hipMemcpy(bank, payload, hot_experts * tuple_bytes, hipMemcpyHostToDevice));
    }

    void launch(bool sample_hot) {
        expert_swiglu<<<grid_size, rows_per_block * wave_size, 0, compute_stream>>>(
                bank, input, routes, device_ticket, base, hot_experts, timeout_ticks, output, timeouts, sample_hot ? samples : nullptr);
        HIP(hipGetLastError());
    }

    void upload(int variant, bool tickets) {
        for (int i = 0; i < cold_experts; ++i) {
            const int expert = hot_experts + i;
            HIP(hipMemcpyAsync(bank + expert * tuple_bytes, payload + (variant * experts + expert) * tuple_bytes,
                               tuple_bytes, hipMemcpyHostToDevice, copy_stream));
            if (tickets) {
                HIP(hipEventRecord(copied[i], copy_stream));
            }
        }
    }

    timing run_production(int variant) {
        const auto begin = bench_clock::now();
        upload(variant, false);
        HIP(hipStreamSynchronize(copy_stream));
        const auto graph_begin = bench_clock::now();
        production->run();
        const auto end = bench_clock::now();
        return {elapsed_us(begin, end), elapsed_us(begin, end), elapsed_us(graph_begin, end)};
    }

    timing run(bool overlap, int variant, bool sample_hot = false) {
        const auto begin = bench_clock::now();
        if (overlap) {
            launch(sample_hot);
        }
        const auto dma_begin = bench_clock::now();
        upload(variant, true);
        const auto deadline = begin + std::chrono::seconds(1);
        for (int i = 0; i < cold_experts; ++i) {
            hipError_t status;
            while ((status = hipEventQuery(copied[i])) == hipErrorNotReady) {
                check(bench_clock::now() < deadline, "H2D completion timed out");
            }
            HIP(status);
            // The system-release event completes after SDMA. Publish only completed tuples.
            __atomic_store_n(ticket, base + i + 1u, __ATOMIC_RELEASE);
        }
        if (!overlap) {
            launch(sample_hot);
        }
        HIP(hipStreamSynchronize(compute_stream));
        const auto end = bench_clock::now();
        base += cold_experts;
        return {elapsed_us(begin, end), elapsed_us(dma_begin, end)};
    }

    void verify_timeouts() const {
        unsigned count = 0;
        HIP(hipMemcpy(&count, timeouts, sizeof(count), hipMemcpyDeviceToHost));
        check(count == 0, "one or more waves timed out waiting for a cold tuple");
    }

    void poison_output() const {
        HIP(hipMemsetAsync(output, 0xff, output_count * sizeof(float), compute_stream));
        HIP(hipStreamSynchronize(compute_stream));
    }

    std::vector<float> read_output() const {
        std::vector<float> values(output_count);
        HIP(hipMemcpy(values.data(), output, values.size() * sizeof(float), hipMemcpyDeviceToHost));
        return values;
    }
};

void compare_output(const std::vector<float> & expected, const std::vector<float> & actual, int variant) {
    double squared_error = 0;
    double squared_reference = 0;
    double max_error = 0;
    double max_reference = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]), "nonfinite output in production reference");
        const double value = expected[i];
        const double error = std::abs(actual[i] - value);
        squared_error += error * error;
        squared_reference += value * value;
        max_error = std::max(max_error, error);
        max_reference = std::max(max_reference, std::abs(value));
    }
    const bool exact = std::memcmp(expected.data(), actual.data(), output_count * sizeof(float)) == 0;
    const double normalized_rmse = std::sqrt(squared_error / squared_reference);
    std::printf("production variant=%d prototype_bitwise_equal=%d max_abs_error=%.9g normalized_rmse=%.9g\n",
                variant, exact, max_error, normalized_rmse);
    check(normalized_rmse <= 1e-5 && max_error <= 1e-4 * max_reference,
          "production SwiGLU output exceeds the prototype cross-check tolerance");
}

} // namespace

int main(int argc, char ** argv) try {
    const std::string mode = argc > 1 ? argv[1] : "both";
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
    const bool compare = mode == "compare";
    const int cold_arg = compare ? 4 : 3;
    const int cold_experts = argc > cold_arg ? std::stoi(argv[cold_arg]) : max_cold_experts;
    const int hot_experts = experts - cold_experts;
    check((compare ? argc == 4 || argc == 5 : argc <= 4 && (mode == "both" || mode == "serial" || mode == "overlap")) &&
          iterations > 0 && iterations <= 1000000 && cold_experts >= 1 && cold_experts <= max_cold_experts,
          "usage: check-hip-expert-single-kernel-overlap [both|serial|overlap] [iterations=20] [cold_experts=4]\n"
          "       check-hip-expert-single-kernel-overlap compare iterations /path/to/libggml-hip.so [cold_experts=4]");

    HIP(hipSetDeviceFlags(hipDeviceMapHost));
    std::unique_ptr<production_graph> production;
    if (compare) {
        production = std::make_unique<production_graph>(argv[3]);
    }
    fixture test;
    test.prepare(production.get(), cold_experts);
    HIP(hipMemset(test.timeouts, 0, sizeof(*test.timeouts)));
    HIP(hipDeviceSynchronize());
    std::array<std::vector<float>, 2> expected;
    for (int variant = 0; variant < 2; ++variant) {
        test.poison_output();
        test.run(false, variant);
        expected[variant] = test.read_output();
        check(std::all_of(expected[variant].begin(), expected[variant].end(),
                          [](float value) { return std::isfinite(value); }), "nonfinite output in serialized reference");
    }
    // Check that a missed upload cannot pass merely because the old tuple remained in the slot.
    for (int position = 0; position < positions; ++position) {
        for (int rank = 0; rank < top_k; ++rank) {
            if (expert_ids[position][rank] >= hot_experts) {
                const size_t offset = (position * top_k + rank) * m;
                check(std::memcmp(expected[0].data() + offset, expected[1].data() + offset, m * sizeof(float)) != 0,
                      "cold route did not change with the tuple payload");
            }
        }
    }
    test.verify_timeouts();
    std::printf("device=gfx906 K=%d M=%d positions=%d top_k=%d hot=%d cold=%d H2D_bytes=%zu tuple_bytes=%zu\n",
                k, m, positions, top_k, hot_experts, cold_experts, cold_experts * tuple_bytes, tuple_bytes);

    if (compare) {
        std::array<std::vector<float>, 2> production_expected;
        for (int i = 0; i < 8; ++i) {
            const int variant = i % 2;
            production->poison_output();
            test.run_production(variant);
            production_expected[variant] = production->read_output();
            compare_output(expected[variant], production_expected[variant], variant);
        }
        double dma_to_done_us = 0;
        double graph_us = 0;
        for (int i = 0; i < iterations; ++i) {
            const int variant = i % 2;
            const auto times = test.run_production(variant);
            dma_to_done_us += times.dma_to_done_us;
            graph_us += times.graph_us;
            const auto actual = production->read_output();
            check(std::memcmp(production_expected[variant].data(), actual.data(), output_count * sizeof(float)) == 0,
                  "production SwiGLU output differs from its poisoned warmup reference");
        }
        std::printf("production_serial iterations=%d dma_to_done_mean_us=%.3f graph_with_q8_quantization_mean_us=%.3f output=exact_to_production_reference\n",
                    iterations, dma_to_done_us / iterations, graph_us / iterations);
    }

    for (bool overlap : {false, true}) {
        const char * arm = overlap ? "overlap" : "serial";
        if (!compare && mode != "both" && mode != arm) {
            continue;
        }
        double total_us = 0;
        double dma_to_done_us = 0;
        for (int i = 0; i < iterations; ++i) {
            const int variant = i % 2;
            const auto times = test.run(overlap, variant);
            total_us += times.total_us;
            dma_to_done_us += times.dma_to_done_us;
            test.verify_timeouts();
            const auto actual = test.read_output();
            check(std::memcmp(expected[variant].data(), actual.data(), output_count * sizeof(float)) == 0,
                  "SwiGLU output differs from serialized reference");
        }
        std::printf("prototype_%s iterations=%d total_mean_us=%.3f dma_to_done_mean_us=%.3f output=exact\n",
                    arm, iterations, total_us / iterations, dma_to_done_us / iterations);
    }

    if (mode != "serial") {
        test.poison_output();
        test.run(true, iterations % 2, true);
        test.verify_timeouts();
        const auto actual = test.read_output();
        check(std::memcmp(expected[iterations % 2].data(), actual.data(), output_count * sizeof(float)) == 0,
              "diagnostic dispatch output differs from serialized reference");
        std::array<hot_sample, grid_size> observed{};
        HIP(hipMemcpy(observed.data(), test.samples, sizeof(observed), hipMemcpyDeviceToHost));
        unsigned started_before_all_ready = 0;
        unsigned before_all_ready = 0;
        uint64_t first_tick = UINT64_MAX;
        uint64_t last_tick = 0;
        for (const auto & sample : observed) {
            started_before_all_ready += sample.first_ticket < cold_experts;
            before_all_ready += sample.last_ticket < cold_experts;
            first_tick = std::min(first_tick, sample.first_tick);
            last_tick = std::max(last_tick, sample.last_tick);
        }
        std::printf("diagnostic hot_first_tick=%llu hot_last_tick=%llu blocks_starting_hot_before_all_tickets=%u/%d blocks_finishing_hot_before_all_tickets=%u/%d\n",
                    static_cast<unsigned long long>(first_tick), static_cast<unsigned long long>(last_tick),
                    started_before_all_ready, grid_size, before_all_ready, grid_size);
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "expert single-kernel overlap: %s\n", error.what());
    return 1;
}
