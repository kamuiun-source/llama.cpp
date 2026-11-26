#include "../../ggml/src/ggml-cuda/mmvq.cu"
#include "iq3-strata-reuse.cuh"
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
    using namespace iq3_strata_reuse;
    constexpr int max_experts = tokens * ranks;
    constexpr int output_elements = tokens * ranks * nrows;
    constexpr int expert_stride = nrows * weight_blocks;
    constexpr int warmup = 8, trials = 60, launches_per_trial = 64;

    // Route equality and rank order from fixed-three request 0, step 1, IQ3 layers 25 and 17.
    // Expert labels are replaced by compact synthetic IDs in first-occurrence order.
    const std::array<int32_t, max_experts> replay[] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 3, 11, 12, 13, 14, 1, 15, 16, 5, 0, 9, 17, 1, 8, 18, 19, 2, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 0, 2, 10, 11, 12, 7, 13, 14, 15, 16, 17, 18, 9, 19, 20, 5, 21, 22, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5},
    };
    constexpr const char * cases[] = {"mixed", "lower_overlap", "fully_shared"};

    std::mt19937 rng(264032);
    std::vector<block_iq3_xxs> up(max_experts * expert_stride);
    auto gate = up;
    for (auto * weights : {&up, &gate}) {
        for (auto & block : *weights) {
            block.d = __float2half(0.001f + float(rng() % 100) / 10000);
            for (auto & byte : block.qs) {
                byte = uint8_t(rng());
            }
        }
    }
    std::vector<block_q8_1> input(tokens * activation_blocks);
    for (auto & block : input) {
        block.ds = __floats2half2_rn(0.001f + float(rng() % 100) / 10000, 0.0f);
        for (auto & value : block.qs) {
            value = int8_t(int(rng() % 255) - 127);
        }
    }
    auto * up_gpu = to_device(up);
    auto * gate_gpu = to_device(gate);
    auto * input_gpu = to_device(input);
    std::vector<float> stock_out(output_elements), reuse_out(output_elements);
    auto * out_gpu = to_device(stock_out);
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate_gpu;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    hipEvent_t start, stop;
    check(hipEventCreate(&start));
    check(hipEventCreate(&stop));

    const auto stock = mul_mat_vec_q_moe<GGML_TYPE_IQ3_XXS, 2, true, true, 3, 2, false, false>;
    const auto print_resources = [](const char * name, auto kernel, int threads) {
        hipFuncAttributes attrs{};
        check(hipFuncGetAttributes(&attrs, reinterpret_cast<const void *>(kernel)));
        int blocks = 0;
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
        std::printf("resources kernel=%s threads=%d vgpr=%d lds=%zu max_blocks_per_cu=%d\n", name, threads, attrs.numRegs,
                    attrs.sharedSizeBytes, blocks);
    };
    print_resources("stock", stock, wave * tokens);
    print_resources("strata_reuse", gate_up, wave * waves);

    for (int c = 0; c < int(sizeof(cases) / sizeof(cases[0])); ++c) {
        std::array<expert_group, max_experts> grouped{};
        for (int route = 0; route < max_experts; ++route) {
            const int expert = replay[c][route];
            auto & group = grouped[expert];
            if (group.count == tokens || (group.count && group.routes[group.count - 1] / ranks == route / ranks)) {
                std::fprintf(stderr, "invalid replay route for case %s\n", cases[c]);
                return 1;
            }
            group.expert = expert;
            group.routes[group.count++] = route;
        }
        std::vector<expert_group> groups;
        int counts[tokens + 1] = {};
        for (const auto & group : grouped) {
            if (group.count) {
                groups.push_back(group);
                ++counts[group.count];
            }
        }
        const dim3 stock_grid(nrows / 4, ranks);
        const dim3 stock_block(wave, tokens);
        const dim3 reuse_grid(nrows / rows_per_tile, groups.size());
        std::printf("route_plan case=%s experts=%zu singles=%d doubles=%d triples=%d stock_ctas=%u reuse_ctas=%u\n",
                    cases[c], groups.size(), counts[1], counts[2], counts[3],
                    stock_grid.x * stock_grid.y, reuse_grid.x * reuse_grid.y);
        auto * ids_gpu = to_device(std::vector<int32_t>(replay[c].begin(), replay[c].end()));
        auto * groups_gpu = to_device(groups);
        const auto launch_stock = [&] {
            stock<<<stock_grid, stock_block>>>(up_gpu, input_gpu, ids_gpu, fusion, out_gpu, ncols, make_uint3(1, 1, 1),
                    nrows, weight_blocks, activation_blocks, ranks * nrows, expert_stride, 0, nrows, tokens, ranks);
        };
        const auto launch_reuse = [&] {
            gate_up<<<reuse_grid, waves * wave>>>(up_gpu, gate_gpu, input_gpu, groups_gpu, out_gpu);
        };
        const auto read = [&](auto launch, std::vector<float> & output) {
            check(hipMemset(out_gpu, 0xff, output_elements * sizeof(float)));
            launch();
            check(hipGetLastError());
            check(hipMemcpy(output.data(), out_gpu, output_elements * sizeof(float), hipMemcpyDeviceToHost));
        };
        read(launch_stock, stock_out);
        read(launch_reuse, reuse_out);
        int different = 0, non_finite = 0;
        float max_abs = 0;
        for (int i = 0; i < output_elements; ++i) {
            different += std::memcmp(&stock_out[i], &reuse_out[i], sizeof(float)) != 0;
            non_finite += !std::isfinite(stock_out[i]) || !std::isfinite(reuse_out[i]);
            max_abs = std::max(max_abs, std::fabs(stock_out[i] - reuse_out[i]));
        }
        std::printf("correctness case=%s outputs=%d bitwise_differences=%d non_finite=%d max_abs=%g\n",
                    cases[c], output_elements, different, non_finite, max_abs);
        if (different || non_finite) {
            return 2;
        }

        std::vector<double> stock_us, reuse_us, delta;
        for (int trial = -warmup; trial < trials; ++trial) {
            double s, r;
            if (trial % 2 == 0) {
                s = time_batch(launch_stock, start, stop, launches_per_trial);
                r = time_batch(launch_reuse, start, stop, launches_per_trial);
            } else {
                r = time_batch(launch_reuse, start, stop, launches_per_trial);
                s = time_batch(launch_stock, start, stop, launches_per_trial);
            }
            if (trial >= 0) {
                stock_us.push_back(s);
                reuse_us.push_back(r);
                delta.push_back(r - s);
                std::printf("timing_sample case=%s trial=%d order=%s stock_us=%.6f reuse_us=%.6f\n",
                            cases[c], trial, trial % 2 == 0 ? "stock-reuse" : "reuse-stock", s, r);
            }
        }
        const auto bounds = std::minmax_element(delta.begin(), delta.end());
        std::printf("timing_summary case=%s trials=%d launches_per_trial=%d stock_median_us=%.6f reuse_median_us=%.6f "
                    "paired_median_us=%.6f paired_min_us=%.6f paired_max_us=%.6f\n",
                    cases[c], trials, launches_per_trial, median(stock_us), median(reuse_us), median(delta), *bounds.first, *bounds.second);
        check(hipFree(groups_gpu));
        check(hipFree(ids_gpu));
    }
    check(hipEventDestroy(stop));
    check(hipEventDestroy(start));
    check(hipFree(out_gpu));
    check(hipFree(input_gpu));
    check(hipFree(gate_gpu));
    check(hipFree(up_gpu));
}
