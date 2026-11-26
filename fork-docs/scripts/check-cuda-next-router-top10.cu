#include "../../ggml/src/ggml-cuda/argsort.cu"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <random>

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

template<typename Launch>
static double time_batch(Launch launch, cudaEvent_t start, cudaEvent_t stop) {
    constexpr int samples = 256;
    check(cudaEventRecord(start));
    for (int j = 0; j < samples; ++j) {
        launch();
    }
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    return ms * 1000.0 / samples;
}

int main() {
    constexpr int rows = 3, cols = 512, top = 10;
    std::array<float, rows * cols> input{};
    std::array<int32_t, rows * cols> stock{};
    std::array<int32_t, rows * top> compact{};
    float * input_d = nullptr;
    int32_t * stock_d = nullptr, * compact_d = nullptr;
    check(cudaMalloc(&input_d, sizeof(input)));
    check(cudaMalloc(&stock_d, sizeof(stock)));
    check(cudaMalloc(&compact_d, sizeof(compact)));
    const auto launch_stock = [&] {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<rows, cols, cols * sizeof(int32_t)>>>(input_d, stock_d, cols, cols);
        check(cudaGetLastError());
    };
    const auto launch_compact = [&] {
        next_router_top10_kernel<<<rows, next_router_threads>>>(input_d, compact_d);
        check(cudaGetLastError());
    };
    std::mt19937 rng(1960);
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int test = 0; test < 3; ++test) {
        for (int row = 0; row < rows; ++row) {
            auto * begin = input.data() + row * cols;
            std::iota(begin, begin + cols, 1.0f);
            std::shuffle(begin, begin + cols, rng);
        }
        if (test == 1) {
            // Exercise reversed tie order and a tie spanning the selection cutoff.
            std::replace(input.begin(), input.begin() + cols, 511.0f, 512.0f);
            std::replace(input.begin() + cols, input.begin() + 2 * cols, 502.0f, 503.0f);
            std::fill(input.begin() + 2 * cols, input.end(), 0.0f);
        } else if (test == 2) {
            input[12] = std::numeric_limits<float>::quiet_NaN();
            input[cols + 2] = std::numeric_limits<float>::infinity();
            input[2 * cols + 1] = -std::numeric_limits<float>::infinity();
        }
        check(cudaMemcpy(input_d, input.data(), sizeof(input), cudaMemcpyHostToDevice));
        launch_stock();
        launch_compact();
        check(cudaMemcpy(stock.data(), stock_d, sizeof(stock), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(compact.data(), compact_d, sizeof(compact), cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            if (!std::equal(compact.begin() + row * top, compact.begin() + (row + 1) * top, stock.begin() + row * cols)) {
                std::fprintf(stderr, "rank mismatch case=%d row=%d\n", test, row);
                return 2;
            }
        }
        std::printf("next_router_top10 case=%d matched_ids=%d\n", test, rows * top);
        if (test == 0) {
            // Stock timings exclude CONT.
            constexpr int trials = 30;
            std::array<double, trials> stock_us{}, compact_us{};
            for (int trial = -4; trial < trials; ++trial) {
                double a, b;
                if (trial % 2 == 0) {
                    a = time_batch(launch_stock, start, stop);
                    b = time_batch(launch_compact, start, stop);
                } else {
                    b = time_batch(launch_compact, start, stop);
                    a = time_batch(launch_stock, start, stop);
                }
                if (trial >= 0) {
                    stock_us[trial] = a;
                    compact_us[trial] = b;
                }
            }
            std::sort(stock_us.begin(), stock_us.end());
            std::sort(compact_us.begin(), compact_us.end());
            std::printf("next_router_top10 stock_sort_us=%.6f compact_us=%.6f\n",
                        (stock_us[14] + stock_us[15]) / 2, (compact_us[14] + compact_us[15]) / 2);
        }
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(compact_d));
    check(cudaFree(stock_d));
    check(cudaFree(input_d));
    return 0;
}
