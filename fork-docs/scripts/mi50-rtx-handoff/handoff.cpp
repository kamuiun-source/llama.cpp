#include <cuda.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>
#include <vector>

extern "C" void setup_hip(int device, void * shared, size_t shared_bytes, size_t bytes);
extern "C" void map_hip_ticket(void * host_ticket);
extern "C" void fill_hip(const void * expected, size_t bytes);
extern "C" void copy_from_hip(void * shared, size_t bytes);
extern "C" void signal_hip(uint32_t value);
extern "C" void sync_hip();
extern "C" void cleanup_hip(void * shared);

static void check(CUresult code, const char * stage) {
    if (code != CUDA_SUCCESS) {
        const char * name = nullptr;
        const char * description = nullptr;
        cuGetErrorName(code, &name);
        cuGetErrorString(code, &description);
        std::fprintf(stderr, "failure stage=%s cuda_status=%d %s %s\n", stage, int(code),
                name ? name : "unknown", description ? description : "");
        std::exit(2);
    }
}

static unsigned number(const char * str, unsigned minimum, unsigned maximum) {
    char * end = nullptr;
    errno = 0;
    const unsigned long result = std::strtoul(str, &end, 10);
    if (!*str || *end || errno || result < minimum || result > maximum) {
        std::fprintf(stderr, "invalid integer %s\n", str);
        std::exit(1);
    }
    return result;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const char * mode = argc > 1 ? argv[1] : "both";
    const bool baseline = std::strcmp(mode, "both") == 0 || std::strcmp(mode, "cpu") == 0;
    const bool doorbell = std::strcmp(mode, "both") == 0 || std::strcmp(mode, "gpu") == 0 || std::strcmp(mode, "gpu-flush") == 0;
    const bool flush = std::strcmp(mode, "gpu-flush") == 0;
    if ((!baseline && !doorbell) || argc > 7) {
        std::fprintf(stderr, "usage: %s [both|cpu|gpu|gpu-flush] [bytes=30720] [samples=3000] [warmup=100] [hip=0] [cuda=0]\n", argv[0]);
        return 1;
    }
    const size_t bytes = argc > 2 ? number(argv[2], 1, 16*1024*1024) : 30720;
    const unsigned samples = argc > 3 ? number(argv[3], 1, 100000) : 3000;
    const unsigned warmup = argc > 4 ? number(argv[4], 0, 10000) : 100;
    const int amd = argc > 5 ? number(argv[5], 0, 255) : 0;
    const int nvidia = argc > 6 ? number(argv[6], 0, 255) : 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        std::fprintf(stderr, "failure stage=sysconf(page_size)\n");
        return 2;
    }
    const size_t page = page_size;
    const size_t total = page + ((bytes + page - 1) / page) * page;
    void * raw = nullptr;
    if (int err = posix_memalign(&raw, page, total)) {
        std::fprintf(stderr, "failure stage=posix_memalign code=%d\n", err);
        return 2;
    }
    std::memset(raw, 0, total);
    auto * shared = static_cast<uint8_t *>(raw) + page;

    setup_hip(amd, raw, total, bytes);
    check(cuInit(0), "cuInit");
    CUdevice device;
    CUcontext context;
    CUstream stream;
    CUdeviceptr destination;
    check(cuDeviceGet(&device, nvidia), "cuDeviceGet");
    check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    char name[128] = {}, pci[32] = {};
    int version = 0, memops = 0, flush_supported = 0;
    check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    check(cuDeviceGetPCIBusId(pci, sizeof(pci), device), "cuDeviceGetPCIBusId");
    check(cuDriverGetVersion(&version), "cuDriverGetVersion");
    const CUresult memops_status = cuDeviceGetAttribute(&memops, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, device);
    const CUresult flush_status = cuDeviceGetAttribute(&flush_supported, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device);
    std::printf("# cuda device=%d name=%s pci=%s driver=%d memops_v1=%d memops_v1_status=%d flush=%d flush_status=%d\n",
            nvidia, name, pci, version, memops, int(memops_status), flush_supported, int(flush_status));
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(cuMemAlloc(&destination, bytes), "cuMemAlloc");
    check(cuMemHostRegister(raw, total, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP), "cuMemHostRegister");
    std::printf("# dual_registered=%zu page=%zu ticket_offset=0 payload_offset=%zu payload_bytes=%zu samples=%u warmup=%u\n",
            total, page, page, bytes, samples, warmup);
    std::printf("# latency=host_steady_clock_before_amd_d2h_enqueue_to_after_rtx_stream_sync verification=outside_timer\n");

    uint32_t ticket = 0;
    CUdeviceptr cuda_ticket = 0;
    std::vector<uint8_t> expected(bytes), received(bytes);
    auto run = [&](const char * label, bool use_doorbell) {
        std::printf("# stage=%s begin\n", label);
        std::vector<int64_t> timings;
        timings.reserve(samples);
        for (unsigned i = 0; i < warmup + samples; ++i) {
            ++ticket;
            uint32_t value = ticket * 0x9e3779b9u;
            for (auto & byte : expected) {
                value ^= value << 13;
                value ^= value >> 17;
                value ^= value << 5;
                byte = value >> 24;
            }
            fill_hip(expected.data(), bytes);
            using clock = std::chrono::steady_clock;
            if (i == 0) {
                std::fprintf(stderr, "stage=%s first_handoff enqueue_amd_d2h\n", label);
            }
            const auto start = clock::now();
            copy_from_hip(shared, bytes);
            if (use_doorbell) {
                signal_hip(ticket);
                const unsigned flags = static_cast<unsigned>(CU_STREAM_WAIT_VALUE_EQ) | (flush ? static_cast<unsigned>(CU_STREAM_WAIT_VALUE_FLUSH) : 0u);
                check(cuStreamWaitValue32(stream, cuda_ticket, ticket, flags), "cuStreamWaitValue32");
            } else {
                sync_hip();
            }
            check(cuMemcpyHtoDAsync(destination, shared, bytes, stream), "cuMemcpyHtoDAsync(RTX_payload)");
            if (i == 0) {
                std::fprintf(stderr, "stage=%s first_handoff queued_wait_and_payload joining_rtx\n", label);
            }
            check(cuStreamSynchronize(stream), "cuStreamSynchronize(RTX_payload)");
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
            check(cuMemcpyDtoH(received.data(), destination, bytes), "cuMemcpyDtoH(RTX_verify)");
            const auto difference = std::mismatch(expected.begin(), expected.end(), received.begin());
            if (difference.first != expected.end()) {
                std::fprintf(stderr, "failure stage=verify mode=%s iteration=%u offset=%td got=%u expected=%u\n",
                        label, i, difference.first - expected.begin(), unsigned(*difference.second), unsigned(*difference.first));
                std::exit(3);
            }
            if (i >= warmup) {
                timings.push_back(elapsed);
            }
        }
        std::printf("mode,iteration,latency_ns\n");
        for (size_t i = 0; i < timings.size(); ++i) {
            std::printf("%s,%zu,%lld\n", label, i, static_cast<long long>(timings[i]));
        }
        std::sort(timings.begin(), timings.end());
        const auto percentile = [&](size_t percent) { return timings[(percent * timings.size() + 99) / 100 - 1]; };
        std::printf("# summary mode=%s p50_ns=%lld p99_ns=%lld max_ns=%lld verified_payloads=%u\n", label,
                static_cast<long long>(percentile(50)), static_cast<long long>(percentile(99)),
                static_cast<long long>(timings.back()), samples + warmup);
    };
    if (baseline) {
        run("cpu", false);
    }
    if (doorbell) {
        if (flush && !flush_supported) {
            std::fprintf(stderr, "failure stage=check_flush_support\n");
            return 4;
        }
        map_hip_ticket(raw);
        check(cuMemHostGetDevicePointer(&cuda_ticket, raw, 0), "cuMemHostGetDevicePointer(ticket)");
        std::printf("# cuda mapped_ticket=0x%llx wait=eq flush=%d\n", static_cast<unsigned long long>(cuda_ticket), flush);
        run(flush ? "gpu-flush" : "gpu", true);
        sync_hip();
    }
    check(cuMemFree(destination), "cuMemFree");
    check(cuStreamDestroy(stream), "cuStreamDestroy");
    check(cuMemHostUnregister(raw), "cuMemHostUnregister");
    check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    cleanup_hip(raw);
    std::free(raw);
}
