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

constexpr int ablation_arms = 5;

template<typename Kernel>
int active_blocks(Kernel kernel, int threads, size_t dynamic_lds) {
    int blocks = 0;
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, dynamic_lds));
    return blocks;
}

template<typename Baseline, typename Grouped>
size_t report_ablation_resources(Baseline baseline, Grouped grouped, int threads) {
    hipFuncAttributes stock{}, prototype{};
    check(hipFuncGetAttributes(&stock, reinterpret_cast<const void *>(baseline)));
    check(hipFuncGetAttributes(&prototype, reinterpret_cast<const void *>(grouped)));
    const int stock_blocks = active_blocks(baseline, threads, 0);
    const int grouped_blocks = active_blocks(grouped, threads, 0);
    const size_t same_lds = prototype.sharedSizeBytes > stock.sharedSizeBytes ?
            prototype.sharedSizeBytes - stock.sharedSizeBytes : 0;
    size_t padding = same_lds;
    int padded_blocks = active_blocks(baseline, threads, padding);
    if (padded_blocks != grouped_blocks) {
        int device = 0;
        hipDeviceProp_t properties{};
        check(hipGetDevice(&device));
        check(hipGetDeviceProperties(&properties, device));
        const size_t max_padding = properties.sharedMemPerBlock - stock.sharedSizeBytes;
        size_t best_distance = max_padding + 1;
        for (size_t candidate = 256; candidate <= max_padding; candidate += 256) {
            const size_t distance = candidate > same_lds ? candidate - same_lds : same_lds - candidate;
            if (distance < best_distance && active_blocks(baseline, threads, candidate) == grouped_blocks) {
                padding = candidate;
                padded_blocks = grouped_blocks;
                best_distance = distance;
            }
        }
    }

    for (int arm = 0; arm < ablation_arms; ++arm) {
        const auto & attrs = arm < 2 ? stock : prototype;
        const size_t dynamic = arm == 1 ? padding : 0;
        const int blocks = arm == 0 ? stock_blocks : arm == 1 ? padded_blocks : grouped_blocks;
        std::printf("ablation_resources arm=%c lds_static=%zu lds_dynamic=%zu lds_total=%zu vgpr_per_thread=%d max_active_blocks_per_cu=%d\n",
                    'A' + arm, attrs.sharedSizeBytes, dynamic, attrs.sharedSizeBytes + dynamic, attrs.numRegs, blocks);
    }
    std::printf("ablation_occupancy padded_matches_grouped=%d\n", padded_blocks == grouped_blocks);
    return padding;
}

template<typename Launch>
void time_ablation(Launch launch, hipEvent_t start, hipEvent_t stop, int warmup, int trials, int launches, const char * copy_contrast) {
    // Five rotations and their reverses balance arm positions and adjacent pairs over ten trials.
    constexpr int order[ablation_arms] = {0, 1, 4, 2, 3};
    std::array<std::vector<double>, ablation_arms> samples;
    std::vector<double> b_minus_a, c_minus_b, d_minus_c, e_minus_c, d_minus_e, e_minus_d;
    for (int trial = -warmup; trial < trials; ++trial) {
        const int sequence = (trial + warmup) % (2 * ablation_arms);
        std::array<int, ablation_arms> arms{};
        std::array<double, ablation_arms> timings{};
        for (int position = 0; position < ablation_arms; ++position) {
            const int index = sequence < ablation_arms ? position : ablation_arms - 1 - position;
            const int arm = (order[index] + sequence) % ablation_arms;
            arms[position] = arm;
            timings[arm] = time_batch([&] { launch(arm); }, start, stop, launches);
        }
        if (trial < 0) {
            continue;
        }
        for (int arm = 0; arm < ablation_arms; ++arm) {
            samples[arm].push_back(timings[arm]);
        }
        b_minus_a.push_back(timings[1] - timings[0]);
        c_minus_b.push_back(timings[2] - timings[1]);
        d_minus_c.push_back(timings[3] - timings[2]);
        e_minus_c.push_back(timings[4] - timings[2]);
        d_minus_e.push_back(timings[3] - timings[4]);
        e_minus_d.push_back(timings[4] - timings[3]);
        std::printf("ablation_sample trial=%d order=%c%c%c%c%c A_us=%.6f B_us=%.6f C_us=%.6f D_us=%.6f E_us=%.6f\n",
                    trial, 'A' + arms[0], 'A' + arms[1], 'A' + arms[2], 'A' + arms[3], 'A' + arms[4],
                    timings[0], timings[1], timings[2], timings[3], timings[4]);
    }
    for (int arm = 0; arm < ablation_arms; ++arm) {
        const auto bounds = std::minmax_element(samples[arm].begin(), samples[arm].end());
        std::printf("ablation_timing arm=%c samples=%d launches_per_sample=%d median_us=%.6f min_us=%.6f max_us=%.6f\n",
                    'A' + arm, trials, launches, median(samples[arm]), *bounds.first, *bounds.second);
    }
    if (copy_contrast) {
        const auto stage = std::minmax_element(c_minus_b.begin(), c_minus_b.end());
        const auto consume = std::minmax_element(e_minus_d.begin(), e_minus_d.end());
        std::printf("ablation_paired_delta contrast=%s_copy_only median_us=%.6f min_us=%.6f max_us=%.6f\n",
                    copy_contrast, median(c_minus_b), *stage.first, *stage.second);
        std::printf("ablation_paired_delta contrast=%s_full median_us=%.6f min_us=%.6f max_us=%.6f\n",
                    copy_contrast, median(e_minus_d), *consume.first, *consume.second);
        return;
    }
    std::printf("ablation_paired_delta B_minus_A_us=%.6f C_minus_B_us=%.6f D_minus_C_us=%.6f\n",
                median(b_minus_a), median(c_minus_b), median(d_minus_c));
    const auto stage = std::minmax_element(e_minus_c.begin(), e_minus_c.end());
    const auto consume = std::minmax_element(d_minus_e.begin(), d_minus_e.end());
    std::printf("ablation_paired_delta contrast=E_minus_C median_us=%.6f min_us=%.6f max_us=%.6f\n",
                median(e_minus_c), *stage.first, *stage.second);
    std::printf("ablation_paired_delta contrast=D_minus_E median_us=%.6f min_us=%.6f max_us=%.6f\n",
                median(d_minus_e), *consume.first, *consume.second);
}

template<typename Launch>
void time_row_overlap(Launch launch, hipEvent_t start, hipEvent_t stop, int warmup, int trials, int launches) {
    constexpr int order[4][4] = {{0, 1, 3, 2}, {1, 2, 0, 3}, {2, 3, 1, 0}, {3, 0, 2, 1}};
    std::array<std::vector<double>, 4> samples;
    std::vector<double> split_minus_full, overlap_minus_split, overlap_minus_full;
    for (int trial = -warmup; trial < trials; ++trial) {
        const auto & arms = order[(trial + warmup) % 4];
        std::array<double, 4> timings{};
        for (int arm : arms) {
            timings[arm] = time_batch([&] { launch(arm); }, start, stop, launches);
        }
        if (trial < 0) {
            continue;
        }
        for (int arm = 0; arm < 4; ++arm) {
            samples[arm].push_back(timings[arm]);
        }
        split_minus_full.push_back(timings[2] - timings[1]);
        overlap_minus_split.push_back(timings[3] - timings[2]);
        overlap_minus_full.push_back(timings[3] - timings[1]);
        std::printf("ablation_sample trial=%d order=%c%c%c%c A_us=%.6f B_us=%.6f C_us=%.6f D_us=%.6f\n",
                    trial, 'A' + arms[0], 'A' + arms[1], 'A' + arms[2], 'A' + arms[3],
                    timings[0], timings[1], timings[2], timings[3]);
    }
    for (int arm = 0; arm < 4; ++arm) {
        const auto bounds = std::minmax_element(samples[arm].begin(), samples[arm].end());
        std::printf("ablation_timing arm=%c samples=%d launches_per_sample=%d median_us=%.6f min_us=%.6f max_us=%.6f\n",
                    'A' + arm, trials, launches, median(samples[arm]), *bounds.first, *bounds.second);
    }
    const auto report = [](const char * name, const std::vector<double> & timings) {
        const auto bounds = std::minmax_element(timings.begin(), timings.end());
        std::printf("ablation_paired_delta contrast=%s median_us=%.6f min_us=%.6f max_us=%.6f\n",
                    name, median(timings), *bounds.first, *bounds.second);
    };
    report("synchronous_minus_full", split_minus_full);
    report("overlap_minus_synchronous", overlap_minus_split);
    report("overlap_minus_full", overlap_minus_full);
}

int main(int argc, char ** argv) {
    const bool batch_ablation = argc == 2 && std::strcmp(argv[1], "--ablate-load-batching") == 0;
    const bool vector_ablation = argc == 2 && std::strcmp(argv[1], "--ablate-vector-copy") == 0;
    const bool row_overlap = argc == 2 && std::strcmp(argv[1], "--ablate-row-overlap") == 0;
    const bool copy_ablation = batch_ablation || vector_ablation || row_overlap;
    const bool ablation = copy_ablation || (argc == 2 && std::strcmp(argv[1], "--ablate-lds") == 0);
    if (argc != 1 && !ablation) {
        std::fprintf(stderr, "usage: %s [--ablate-lds|--ablate-load-batching|--ablate-vector-copy|--ablate-row-overlap]\n", argv[0]);
        return 1;
    }
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
    const int first_multiplicity = ablation ? 2 : 1;
    const int last_multiplicity = ablation ? 2 : 3;
    const auto baseline = mul_mat_vec_q_moe<GGML_TYPE_IQ3_XXS, 2, true, true, 3, 2, false, false, false>;
    const auto split = iq3_parallel_weights::gate_up<true, false, false, true>;
    const auto grouped = vector_ablation || row_overlap ? iq3_parallel_weights::gate_up<true, false, true> :
            batch_ablation ? iq3_parallel_weights::gate_up<true, true> : iq3_parallel_weights::gate_up<true>;
    const char * copy_contrast = vector_ablation ? "vector_minus_batched" : batch_ablation ? "batched_minus_serial" : nullptr;
    for (int multiplicity = first_multiplicity; multiplicity <= last_multiplicity; ++multiplicity) {
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
        const auto launch_stock = [&](size_t padding) {
            baseline<<<grid, block, padding>>>(up_gpu, input_gpu, ids_gpu, fusion, out_gpu, ncols,
                    make_uint3(1, 1, 1), nrows, weight_row_stride, input_token_stride, output_token_stride,
                    weight_expert_stride, 0, nrows, tokens, ranks, nullptr);
        };
        const auto launch_off = [&] { launch_stock(0); };
        const auto launch_on = [&] {
            iq3_parallel_weights::gate_up<><<<parallel_grid, block>>>(up_gpu, gate_gpu, input_gpu, ids_gpu, groups_gpu, out_gpu);
        };
        const auto copy_output = [&](auto launch, std::vector<float> & output) {
            check(hipMemset(out_gpu, 0xff, sizeof(float)*output_elements));
            launch();
            check(hipGetLastError());
            check(hipMemcpy(output.data(), out_gpu, sizeof(float)*output_elements, hipMemcpyDeviceToHost));
        };
        copy_output(launch_off, off);
        if (ablation) {
            // A and B execute the same instructions. B requests additional, untouched dynamic LDS.
            // C, D, and E use one runtime-switched kernel and reserve identical resources.
            size_t padding = 0;
            const int arm_count = row_overlap ? 4 : ablation_arms;
            if (row_overlap) {
                constexpr const char * labels[] = {"direct", "vector_full", "split_synchronous", "split_overlap"};
                for (int arm = 0; arm < arm_count; ++arm) {
                    const auto kernel = arm < 2 ? grouped : split;
                    hipFuncAttributes attrs{};
                    check(hipFuncGetAttributes(&attrs, reinterpret_cast<const void *>(kernel)));
                    const int blocks = active_blocks(kernel, block.x * block.y, 0);
                    std::printf("ablation_resources arm=%c mode=%s lds_static=%zu lds_dynamic=0 vgpr_per_thread=%d max_active_blocks_per_cu=%d\n",
                                'A' + arm, labels[arm], attrs.sharedSizeBytes, attrs.numRegs, blocks);
                }
            } else if (copy_ablation) {
                hipFuncAttributes attrs{};
                check(hipFuncGetAttributes(&attrs, reinterpret_cast<const void *>(grouped)));
                const int blocks = active_blocks(grouped, block.x * block.y, 0);
                constexpr const char * batch_labels[] = {"direct", "serial_copy_only", "batched_copy_only", "serial_full", "batched_full"};
                constexpr const char * vector_labels[] = {"direct", "batched_copy_only", "vector_copy_only", "batched_full", "vector_full"};
                const auto * labels = vector_ablation ? vector_labels : batch_labels;
                for (int arm = 0; arm < ablation_arms; ++arm) {
                    std::printf("ablation_resources arm=%c mode=%s lds_static=%zu lds_dynamic=0 vgpr_per_thread=%d max_active_blocks_per_cu=%d\n",
                                'A' + arm, labels[arm], attrs.sharedSizeBytes, attrs.numRegs, blocks);
                }
            } else {
                padding = report_ablation_resources(baseline, grouped, block.x * block.y);
            }
            const auto launch = [&](int arm) {
                using namespace iq3_parallel_weights;
                if (row_overlap) {
                    const auto kernel = arm < 2 ? grouped : split;
                    constexpr weight_mode modes[] = {weights_direct, weights_share_vector, weights_wait_early, weights_direct};
                    kernel<<<parallel_grid, block>>>(up_gpu, gate_gpu, input_gpu, ids_gpu, groups_gpu, out_gpu, modes[arm]);
                } else if (copy_ablation) {
                    constexpr weight_mode batch_modes[] = {
                        weights_direct, weights_stage, weights_stage_batched, weights_share, weights_share_batched,
                    };
                    constexpr weight_mode vector_modes[] = {
                        weights_direct, weights_stage_batched, weights_stage_vector, weights_share_batched, weights_share_vector,
                    };
                    const weight_mode mode = vector_ablation ? vector_modes[arm] : batch_modes[arm];
                    grouped<<<parallel_grid, block>>>(up_gpu, gate_gpu, input_gpu, ids_gpu, groups_gpu, out_gpu, mode);
                } else if (arm < 2) {
                    launch_stock(arm == 1 ? padding : 0);
                } else {
                    constexpr weight_mode modes[] = {weights_direct, weights_share, weights_stage};
                    grouped<<<parallel_grid, block>>>(up_gpu, gate_gpu, input_gpu, ids_gpu, groups_gpu, out_gpu, modes[arm - 2]);
                }
            };
            for (int arm = 0; arm < arm_count; ++arm) {
                if (arm != 0 || copy_ablation) {
                    copy_output([&] { launch(arm); }, on);
                }
                const auto & output = arm == 0 && !copy_ablation ? off : on;
                int different = 0, non_finite = 0;
                for (int i = 0; i < output_elements; ++i) {
                    non_finite += !std::isfinite(output[i]);
                    different += std::memcmp(&off[i], &output[i], sizeof(float)) != 0;
                }
                std::printf("ablation_correctness arm=%c outputs=%d bitwise_differences=%d non_finite=%d\n",
                            'A' + arm, output_elements, different, non_finite);
                if (different || non_finite) {
                    return 3;
                }
            }
            if (row_overlap) {
                time_row_overlap(launch, event_start, event_stop, warmup_batches, measured_pairs, launches_per_batch);
            } else {
                time_ablation(launch, event_start, event_stop, warmup_batches, measured_pairs, launches_per_batch, copy_contrast);
            }
            check(hipFree(groups_gpu));
            check(hipFree(ids_gpu));
            continue;
        }
        copy_output(launch_on, on);
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
