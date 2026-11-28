#include "common.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

namespace {
using clock_type = std::chrono::steady_clock;
using us = std::chrono::duration<double, std::micro>;

void check(CUresult result, const char * operation) {
    if (result != CUDA_SUCCESS) {
        const char * name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "%s: %d %s\n", operation, int(result), name ? name : "unknown");
        std::exit(2);
    }
}

unsigned number(const char * text, unsigned min, unsigned max) {
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!*text || *end || errno || value < min || value > max) {
        std::fprintf(stderr, "invalid argument: %s\n", text);
        std::exit(1);
    }
    return value;
}

void pattern(std::array<float, elements> & data, unsigned test) {
    for (int i = 0; i < elements; ++i) {
        if (test == 2) {
            const int lane = i % 32;
            data[i] = lane == 0 ? 127.0f : (lane % 2 ? 1.0f : -1.0f) * (lane + 0.5f);
        } else {
            const unsigned bits = (unsigned(i) * 0x9e3779b9u) ^ (unsigned(i) * 0x85ebca6bu >> 7);
            data[i] = (int(bits % 2047) - 1023) * (i / width + 1) / 512.0f;
            if (test == 1 && (i / 32) % 7 == 0) {
                data[i] = 0.0f;
            }
        }
    }
}

void report(const char * arm, const char * interval, const char * correctness, std::vector<double> & times) {
    std::sort(times.begin(), times.end());
    auto at = [&](unsigned p) { return times[(times.size() - 1) * p / 100]; };
    std::printf("mode=%s interval=%s correctness=%s n=%zu p50_us=%.3f p90_us=%.3f p99_us=%.3f\n",
            arm, interval, correctness, times.size(), at(50), at(90), at(99));
}
}

int main(int argc, char ** argv) {
    if (argc > 5) {
        std::fprintf(stderr, "usage: %s [hip=0] [cuda=0] [samples=3000] [warmup=100]\n", argv[0]);
        return 1;
    }
    const unsigned amd = argc > 1 ? number(argv[1], 0, 255) : 0;
    const unsigned nvidia = argc > 2 ? number(argv[2], 0, 255) : 0;
    const unsigned samples = argc > 3 ? number(argv[3], 1, 100000) : 3000;
    const unsigned warmup = argc > 4 ? number(argv[4], 0, 10000) : 100;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        std::fprintf(stderr, "could not read system page size\n");
        return 2;
    }
    const size_t q8_offset = (f32_bytes + page - 1) / page * page;
    const size_t total = q8_offset + (q8_bytes + page - 1) / page * page;
    void * host = nullptr;
    if (posix_memalign(&host, page, total) != 0) {
        std::fprintf(stderr, "could not allocate shared payload\n");
        return 2;
    }
    std::memset(host, 0, total);
    mi50_input_setup(amd, host, total, q8_offset);
    check(cuInit(0), "cuInit");
    CUdevice device;
    CUcontext context;
    CUstream stream;
    CUdeviceptr f32_device, q8_device, mapped;
    check(cuDeviceGet(&device, nvidia), "cuDeviceGet");
    char name[128];
    check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(cuMemAlloc(&f32_device, f32_bytes), "cuMemAlloc(F32)");
    check(cuMemAlloc(&q8_device, q8_bytes), "cuMemAlloc(Q8_1)");
    check(cuMemHostRegister(host, total, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP), "cuMemHostRegister");
    check(cuMemHostGetDevicePointer(&mapped, host, 0), "cuMemHostGetDevicePointer");
    auto * host_q8 = static_cast<char *>(host) + q8_offset;
    auto * mapped_q8 = reinterpret_cast<void *>(uintptr_t(mapped + q8_offset));
    std::printf("cuda_device=%u name=%s f32_bytes=%zu q8_bytes=%zu q8_offset=%zu samples=%u warmup=%u\n",
            nvidia, name, f32_bytes, q8_bytes, q8_offset, samples, warmup);

    constexpr unsigned arms = 4;
    const char * names[arms] = {"hip_blit_scalar_q8", "hip_mapped_scalar_q8", "rtx_mapped_q8", "rtx_device_q8_d2h"};
    const auto prepare = [&](unsigned test) {
        std::array<float, elements> input;
        pattern(input, test);
        check(cuMemcpyHtoD(f32_device, input.data(), f32_bytes), "cuMemcpyHtoD(setup)");
        return input;
    };
    auto run = [&](unsigned arm) {
        const auto start = clock_type::now();
        check(cuMemcpyDtoHAsync(host, f32_device, f32_bytes, stream), "cuMemcpyDtoHAsync(F32 mirror)");
        const auto * rtx_input = reinterpret_cast<const void *>(uintptr_t(f32_device));
        if (arm == 2) {
            rtx_input_q8(rtx_input, mapped_q8, stream);
        } else if (arm == 3) {
            rtx_input_q8(rtx_input, reinterpret_cast<void *>(uintptr_t(q8_device)), stream);
            check(cuMemcpyDtoHAsync(host_q8, q8_device, q8_bytes, stream), "cuMemcpyDtoHAsync(Q8_1)");
        }
        check(cuStreamSynchronize(stream), "cuStreamSynchronize(F32/Q8 publication)");
        const auto cuda_done = clock_type::now();
        mi50_input_submit(arm);
        mi50_input_sync();
        const auto hip_done = clock_type::now();
        return std::array<double, 3>{us(hip_done - start).count(), us(cuda_done - start).count(), us(hip_done - cuda_done).count()};
    };
    std::array<bool, arms> valid;
    valid.fill(true);
    std::array<unsigned char, q8_bytes> ref, got;
    for (unsigned test = 0; test < 3; ++test) {
        const auto input = prepare(test);
        for (unsigned arm = 0; arm < arms; ++arm) {
            std::memset(host, 0xa5 + arm, total);
            run(arm);
            mi50_input_read(arm == 0 ? ref.data() : got.data());
            const bool f32_ok = std::memcmp(host, input.data(), f32_bytes) == 0;
            const bool q8_ok = arm == 0 || ref == got;
            if (!q8_ok) {
                const auto mismatch = std::mismatch(ref.begin(), ref.end(), got.begin());
                std::printf("first_q8_diff mode=%s pattern=%u byte=%td expected=0x%02x got=0x%02x\n",
                        names[arm], test, mismatch.first - ref.begin(), *mismatch.first, *mismatch.second);
            }
            valid[arm] &= f32_ok && q8_ok;
            std::printf("bitwise mode=%s pattern=%u f32=%s q8=%s f32_bytes=%zu q8_bytes=%zu\n",
                    names[arm], test, f32_ok ? "PASS" : "FAIL", q8_ok ? "PASS" : "FAIL", f32_bytes, q8_bytes);
        }
    }

    prepare(0);
    std::array<std::array<std::vector<double>, 3>, arms> times;
    for (unsigned i = 0; i < samples + warmup; ++i) {
        for (unsigned j = 0; j < arms; ++j) {
            const unsigned arm = (j + i) % arms;
            const auto elapsed = run(arm);
            if (i >= warmup) {
                for (unsigned span = 0; span < elapsed.size(); ++span) {
                    times[arm][span].push_back(elapsed[span]);
                }
            }
        }
    }
    const char * intervals[] = {"rtx_ready_to_mi50_ready", "rtx_publish", "mi50_prepare"};
    for (unsigned arm = 0; arm < arms; ++arm) {
        for (unsigned span = 0; span < times[arm].size(); ++span) {
            report(names[arm], intervals[span], valid[0] && valid[arm] ? "pass" : "fail", times[arm][span]);
        }
    }

    check(cuMemHostUnregister(host), "cuMemHostUnregister");
    check(cuMemFree(q8_device), "cuMemFree(Q8_1)");
    check(cuMemFree(f32_device), "cuMemFree(F32)");
    check(cuStreamDestroy(stream), "cuStreamDestroy");
    check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    mi50_input_cleanup(host);
    std::free(host);
    return std::all_of(valid.begin(), valid.end(), [](bool value) { return value; }) ? 0 : 3;
}
