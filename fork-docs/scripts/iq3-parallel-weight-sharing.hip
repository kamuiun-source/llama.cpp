#include "../../ggml/src/ggml-cuda/mmvq.cu"
#include "iq3-parallel-weight-sharing.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>

void check(hipError_t result) {
    if (result != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(result));
        std::exit(1);
    }
}

template<typename T>
T * to_device(const std::vector<T> & host) {
    T * result = nullptr;
    check(hipMalloc(&result, host.size()*sizeof(T)));
    check(hipMemcpy(result, host.data(), host.size()*sizeof(T), hipMemcpyHostToDevice));
    return result;
}

template<typename Launch>
double time_batch(Launch launch, hipEvent_t start, hipEvent_t stop, int count) {
    check(hipEventRecord(start, nullptr));
    for (int i = 0; i < count; ++i) {
        launch();
        check(hipGetLastError());
    }
    check(hipEventRecord(stop, nullptr));
    check(hipEventSynchronize(stop));
    float elapsed_ms = 0;
    check(hipEventElapsedTime(&elapsed_ms, start, stop));
    return double(elapsed_ms) * 1000.0 / count;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.0;
}

int main() {
    constexpr int ncols = 2560, nrows = 640, experts = 30, tokens = 3, ranks = 10;
    constexpr int weight_row_stride = ncols / QK_K;
    constexpr int weight_expert_stride = nrows * weight_row_stride;
    constexpr int input_token_stride = ncols / QK8_1;
    constexpr int output_token_stride = ranks * nrows;
    constexpr int output_elements = tokens * output_token_stride;
    std::mt19937 rng(264032);
    std::vector<block_iq3_xxs> up(experts * weight_expert_stride);
    auto gate = up;
    for (auto * weights : {&up, &gate}) {
        for (auto & block : *weights) {
            block.d = __float2half(0.001f + float(rng() % 100) / 10000);
            for (auto & byte : block.qs) {
                byte = uint8_t(rng());
            }
        }
    }
    std::vector<block_q8_1> input(tokens * input_token_stride);
    for (auto & block : input) {
        block.ds = __floats2half2_rn(0.001f + float(rng() % 100) / 10000, 0.0f);
        for (auto & val : block.qs) {
            val = int8_t(int(rng() % 255) - 127);
        }
    }
    auto * up_gpu = to_device(up);
    auto * gate_gpu = to_device(gate);
    auto * input_gpu = to_device(input);
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate_gpu;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    std::vector<float> off(output_elements), on(output_elements);
    auto * out_gpu = to_device(off);
    const dim3 grid(nrows / 4, ranks);
    const dim3 block(64, tokens);
    constexpr int warmup_batches = 8, measured_pairs = 60, launches_per_batch = 64;
    hipEvent_t event_start, event_stop;
    check(hipEventCreate(&event_start));
    check(hipEventCreate(&event_stop));
    std::printf("timing_config warmup_batches=%d measured_pairs=%d launches_per_batch=%d\n",
                warmup_batches, measured_pairs, launches_per_batch);
    for (int multiplicity = 1; multiplicity <= 3; ++multiplicity) {
        std::vector<int32_t> ids_host(ranks * tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int rank = 0; rank < ranks; ++rank) {
                const int expert = (rank + 3*t) % ranks;
                ids_host[t*ranks + rank] = (t < multiplicity ? 0 : t*ranks) + expert;
            }
        }
        std::vector<int32_t> identity_slots(ggml_hip_iq3_route_plan::experts);
        std::iota(identity_slots.begin(), identity_slots.end(), 0);
        const auto route_plan_array = ggml_hip_iq3_route_plan::build(ids_host.data(), identity_slots.data(), -1);
        const auto groups = iq3_parallel_weights::group_routes(route_plan_array);
        auto * ids_gpu = to_device(ids_host);
        auto * groups_gpu = to_device(groups);
        const size_t followers = std::count(route_plan_array.begin(), route_plan_array.end(), ggml_hip_iq3_route_plan::follower);
        const size_t duplicate_payload_bytes = followers * nrows * weight_row_stride * sizeof(block_iq3_xxs) * 2;
        const dim3 parallel_grid(nrows / 4, groups.size());
        std::printf("parallel_plan overlap_positions=%d cta_groups=%zu threads=%u duplicate_payload_bytes=%zu\n",
                    multiplicity, groups.size(), block.x * block.y, duplicate_payload_bytes);
        const auto launch_off = [&] {
            mul_mat_vec_q_moe<GGML_TYPE_IQ3_XXS, 2, true, true, 3, 2, false, false, false>
                <<<grid, block>>>(up_gpu, input_gpu, ids_gpu, fusion, out_gpu, ncols, make_uint3(1, 1, 1), nrows,
                    weight_row_stride, input_token_stride, output_token_stride, weight_expert_stride, 0, nrows, tokens, ranks);
        };
        const auto launch_on = [&] {
            iq3_parallel_weights::gate_up<<<parallel_grid, block>>>(up_gpu, gate_gpu, input_gpu, ids_gpu, groups_gpu, out_gpu);
        };
        check(hipMemset(out_gpu, 0xff, sizeof(float)*output_elements));
        launch_off();
        check(hipGetLastError());
        check(hipMemcpy(off.data(), out_gpu, sizeof(float)*output_elements, hipMemcpyDeviceToHost));
        check(hipMemset(out_gpu, 0xff, sizeof(float)*output_elements));
        launch_on();
        check(hipGetLastError());
        check(hipMemcpy(on.data(), out_gpu, sizeof(float)*output_elements, hipMemcpyDeviceToHost));
        size_t different = 0;
        float max_abs = 0;
        for (int i = 0; i < output_elements; ++i) {
            if (!std::isfinite(off[i]) || !std::isfinite(on[i])) {
                std::fprintf(stderr, "missing or non-finite result at %d\n", i);
                return 2;
            }
            different += std::memcmp(&off[i], &on[i], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::fabs(off[i] - on[i]));
        }
        std::printf("positions_per_shared_expert=%d outputs=%d bitwise_differences=%zu max_abs=%g\n",
                    multiplicity, output_elements, different, max_abs);
        if (different) {
            return 3;
        }
        std::vector<double> off_us, on_us;
        off_us.reserve(measured_pairs);
        on_us.reserve(measured_pairs);
        for (int i = 0; i < warmup_batches; ++i) {
            if (i % 2 == 0) {
                time_batch(launch_off, event_start, event_stop, launches_per_batch);
                time_batch(launch_on, event_start, event_stop, launches_per_batch);
            } else {
                time_batch(launch_on, event_start, event_stop, launches_per_batch);
                time_batch(launch_off, event_start, event_stop, launches_per_batch);
            }
        }
        for (int i = 0; i < measured_pairs; ++i) {
            double off_sample, on_sample;
            if (i % 2 == 0) {
                off_sample = time_batch(launch_off, event_start, event_stop, launches_per_batch);
                on_sample = time_batch(launch_on, event_start, event_stop, launches_per_batch);
            } else {
                on_sample = time_batch(launch_on, event_start, event_stop, launches_per_batch);
                off_sample = time_batch(launch_off, event_start, event_stop, launches_per_batch);
            }
            off_us.push_back(off_sample);
            on_us.push_back(on_sample);
            std::printf("timing_sample overlap_positions=%d trial=%d order=%s off_us=%.6f on_us=%.6f\n",
                        multiplicity, i, i % 2 == 0 ? "off-on" : "on-off", off_sample, on_sample);
        }
        const auto bounds = [](const std::vector<double> & values) {
            return std::minmax_element(values.begin(), values.end());
        };
        const auto off_bounds = bounds(off_us), on_bounds = bounds(on_us);
        std::printf("timing_summary overlap_positions=%d samples=%zu launches_per_sample=%d off_median_us=%.6f off_min_us=%.6f off_max_us=%.6f on_median_us=%.6f on_min_us=%.6f on_max_us=%.6f paired_median_delta_us=%.6f\n",
                    multiplicity, off_us.size(), launches_per_batch, median(off_us), *off_bounds.first, *off_bounds.second,
                    median(on_us), *on_bounds.first, *on_bounds.second, median(on_us) - median(off_us));
        check(hipFree(groups_gpu));
        check(hipFree(ids_gpu));
    }
    check(hipFree(out_gpu));
    check(hipFree(input_gpu));
    check(hipFree(gate_gpu));
    check(hipFree(up_gpu));
    check(hipEventDestroy(event_start));
    check(hipEventDestroy(event_stop));
}
