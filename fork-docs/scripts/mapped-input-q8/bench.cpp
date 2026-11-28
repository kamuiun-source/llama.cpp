#include "quantize.cuh"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

[[noreturn]] void ggml_cuda_error(const char * statement, const char * function, const char * file, int line, const char * message) {
    std::fprintf(stderr, "%s:%d: %s: %s: %s\n", file, line, function, statement, message);
    std::exit(2);
}

[[noreturn]] void ggml_abort(const char * file, int line, const char * format, ...) {
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::exit(2);
}

namespace {

constexpr int width = 2560;
constexpr int rows = 3;
constexpr int floats = width * rows;
constexpr size_t f32_bytes = floats * sizeof(float);
constexpr size_t q8_bytes = floats / QK8_1 * sizeof(block_q8_1);
static_assert(q8_bytes == 8640 && width == GGML_PAD(width, MATRIX_ROW_PADDING), "fixed MTP3 activation");

using f32x4 = float __attribute__((ext_vector_type(4)));

__global__ void blit(const float4 * source, float4 * dest) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = source[i];
}

__global__ void quantize_vector(const f32x4 * source, block_q8_1 * dest) {
    constexpr int lanes_per_quant = QK8_1 / 4;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.z * (width / 4) + col;
    const f32x4 value = source[i];
    const float local_max = fmaxf(fmaxf(fabsf(value[0]), fabsf(value[1])), fmaxf(fabsf(value[2]), fabsf(value[3])));
    const float amax = warp_reduce_max<lanes_per_quant>(local_max);
    const float d = amax / 127.0f;
    int sum = 0;
    const int lane = col % lanes_per_quant;
    auto & block = dest[i / lanes_per_quant];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int8_t q = amax == 0.0f ? 0 : roundf(value[k] / d);
        block.qs[lane * 4 + k] = q;
        sum += q;
    }
    sum = warp_reduce_sum<lanes_per_quant>(sum);
    if (lane == 0) {
        block.ds = make_half2(d, d * sum);
    }
}

unsigned number(const char * text, unsigned low, unsigned high) {
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!*text || *end || errno || value < low || value > high) {
        std::fprintf(stderr, "invalid argument: %s\n", text);
        std::exit(1);
    }
    return value;
}

void pattern(float * data, unsigned test) {
    for (int i = 0; i < floats; ++i) {
        if (test == 2) {
            const int lane = i % QK8_1;
            data[i] = lane == 0 ? 127.0f : (lane % 2 ? 1.0f : -1.0f) * (lane + 0.5f);
        } else {
            const unsigned bits = (unsigned(i) * 0x9e3779b9u) ^ (unsigned(i) * 0x85ebca6bu >> 7);
            data[i] = (int(bits % 2047) - 1023) * (i / width + 1) / 512.0f;
            if (test == 1 && (i / QK8_1) % 7 == 0) {
                data[i] = 0.0f;
            }
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 4) {
        std::fprintf(stderr, "usage: %s [device=0] [repetitions=128] [samples=30]\n", argv[0]);
        return 1;
    }
    const int device = argc > 1 ? number(argv[1], 0, 255) : 0;
    const unsigned repetitions = argc > 2 ? number(argv[2], 1, 4096) : 128;
    const unsigned samples = argc > 3 ? number(argv[3], 3, 1000) : 30;
    CUDA_CHECK(hipSetDevice(device));
    hipDeviceProp_t props{};
    CUDA_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strncmp(props.gcnArchName, "gfx906", 6) != 0 || !props.canMapHostMemory) {
        std::printf("skip: device=%d arch=%s mapped_host=%d\n", device, props.gcnArchName, props.canMapHostMemory);
        return 77;
    }

    const long page = sysconf(_SC_PAGESIZE);
    void * host = nullptr;
    const size_t registered_bytes = page > 0 ? (f32_bytes + page - 1) / page * page : 0;
    if (!registered_bytes || posix_memalign(&host, page, registered_bytes) != 0) {
        std::fprintf(stderr, "could not allocate page-aligned host payload\n");
        return 2;
    }
    std::memset(host, 0, registered_bytes);
    CUDA_CHECK(hipHostRegister(host, registered_bytes, hipHostRegisterPortable | hipHostRegisterMapped));
    void * mapped = nullptr;
    CUDA_CHECK(hipHostGetDevicePointer(&mapped, host, 0));
    float * staging = nullptr;
    block_q8_1 * q8 = nullptr;
    CUDA_CHECK(hipMalloc(&staging, f32_bytes));
    CUDA_CHECK(hipMalloc(&q8, q8_bytes));
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    const auto scalar = [&](const float * source) {
        quantize_row_q8_1_cuda(source, q8, width, width, width, width * rows, width, 1, rows, 1, stream);
    };
    constexpr unsigned modes = 3;
    const char * names[modes] = {"blit_scalar_q8", "mapped_scalar_q8", "mapped_vector_q8"};
    const auto launch = [&](unsigned mode) {
        if (mode == 0) {
            constexpr unsigned threads = 128;
            blit<<<floats / (4 * threads), threads, 0, stream>>>(
                    static_cast<const float4 *>(mapped), reinterpret_cast<float4 *>(staging));
            CUDA_CHECK(hipGetLastError());
            scalar(staging);
        } else if (mode == 1) {
            scalar(static_cast<const float *>(mapped));
        } else {
            constexpr unsigned threads = CUDA_QUANTIZE_BLOCK_SIZE / 4;
            quantize_vector<<<dim3(width / (4 * threads), 1, rows), threads, 0, stream>>>(
                    static_cast<const f32x4 *>(mapped), q8);
            CUDA_CHECK(hipGetLastError());
        }
    };

    std::array<unsigned char, q8_bytes> reference, actual;
    for (unsigned test = 0; test < 3; ++test) {
        pattern(static_cast<float *>(host), test);
        for (unsigned mode = 0; mode < modes; ++mode) {
            CUDA_CHECK(hipMemsetAsync(q8, 0xa5 + mode, q8_bytes, stream));
            launch(mode);
            auto & output = mode == 0 ? reference : actual;
            CUDA_CHECK(hipMemcpyAsync(output.data(), q8, q8_bytes, hipMemcpyDeviceToHost, stream));
            CUDA_CHECK(hipStreamSynchronize(stream));
            if (mode != 0 && reference != actual) {
                const auto mismatch = std::mismatch(reference.begin(), reference.end(), actual.begin());
                std::fprintf(stderr, "bitwise=FAIL pattern=%u mode=%s byte=%td expected=0x%02x got=0x%02x\n",
                        test, names[mode], mismatch.first - reference.begin(), *mismatch.first, *mismatch.second);
                return 3;
            }
        }
        std::printf("bitwise=PASS pattern=%u bytes=%zu\n", test, q8_bytes);
    }

    pattern(static_cast<float *>(host), 0);
    std::array<hipGraph_t, modes> graphs;
    std::array<hipGraphExec_t, modes> executable;
    for (unsigned mode = 0; mode < modes; ++mode) {
        CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        for (unsigned i = 0; i < repetitions; ++i) {
            launch(mode);
        }
        CUDA_CHECK(hipStreamEndCapture(stream, &graphs[mode]));
        CUDA_CHECK(hipGraphInstantiate(&executable[mode], graphs[mode], nullptr, nullptr, 0));
    }
    constexpr unsigned warmup = 5;
    for (unsigned i = 0; i < warmup; ++i) {
        for (unsigned mode = 0; mode < modes; ++mode) {
            CUDA_CHECK(hipGraphLaunch(executable[(mode + i) % modes], stream));
            CUDA_CHECK(hipStreamSynchronize(stream));
        }
    }

    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
    std::array<std::vector<double>, modes> results;
    for (unsigned sample = 0; sample < samples; ++sample) {
        for (unsigned ord = 0; ord < modes; ++ord) {
            const unsigned mode = (ord + sample) % modes;
            CUDA_CHECK(hipEventRecord(start, stream));
            CUDA_CHECK(hipGraphLaunch(executable[mode], stream));
            CUDA_CHECK(hipEventRecord(stop, stream));
            CUDA_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
            results[mode].push_back(ms * 1000.0 / repetitions);
        }
    }
    std::printf("device=%s arch=%s shape=%dx%d method=hip_graph_events repetitions=%u samples=%u warmup=%u\n",
            props.name, props.gcnArchName, rows, width, repetitions, samples, warmup);
    for (unsigned mode = 0; mode < modes; ++mode) {
        auto & times = results[mode];
        std::sort(times.begin(), times.end());
        std::printf("mode=%s p50_us=%.4f p90_us=%.4f min_us=%.4f\n", names[mode],
                times[(samples - 1) * 50 / 100], times[(samples - 1) * 90 / 100], times.front());
        CUDA_CHECK(hipGraphExecDestroy(executable[mode]));
        CUDA_CHECK(hipGraphDestroy(graphs[mode]));
    }
    CUDA_CHECK(hipEventDestroy(stop));
    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipFree(q8));
    CUDA_CHECK(hipFree(staging));
    CUDA_CHECK(hipHostUnregister(host));
    std::free(host);
}
