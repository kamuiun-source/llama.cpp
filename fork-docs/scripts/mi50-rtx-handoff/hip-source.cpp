#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
hipStream_t stream;
void * source;
void * ticket;
float * add_lhs;
float * add_rhs;
float * mapped_payload;
void * expert_host;
void * expert_device;
constexpr unsigned threads = 256;

__global__ void add_fp32(float * dst, const float * lhs, const float * rhs, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        dst[i] = lhs[i] + rhs[i];
    }
}

__global__ void blit_fp32(float * dst, const float * src, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        dst[i] = src[i];
    }
}

void check(hipError_t code, const char * stage) {
    if (code != hipSuccess) {
        std::fprintf(stderr, "failure stage=%s hip_status=%d %s\n", stage, int(code), hipGetErrorString(code));
        std::exit(2);
    }
}
}

extern "C" void setup_hip(int device, void * shared, size_t shared_bytes, size_t bytes) {
    check(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t prop{};
    int runtime = 0, driver = 0;
    check(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
    check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    check(hipDriverGetVersion(&driver), "hipDriverGetVersion");
    std::printf("# hip device=%d name=%s arch=%s pci=%x:%x:%x runtime=%d driver=%d can_map_host=%d\n",
            device, prop.name, prop.gcnArchName, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, runtime, driver, prop.canMapHostMemory);
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    check(hipMalloc(&source, bytes), "hipMalloc");
    check(hipHostRegister(shared, shared_bytes, hipHostRegisterPortable | hipHostRegisterMapped), "hipHostRegister");
}

extern "C" void map_hip_ticket(void * host_ticket) {
    check(hipHostGetDevicePointer(&ticket, host_ticket, 0), "hipHostGetDevicePointer(ticket)");
    std::printf("# hip mapped_ticket=%p\n", ticket);
}

extern "C" void fill_hip(const void * expected, size_t bytes) {
    check(hipMemcpy(source, expected, bytes, hipMemcpyHostToDevice), "hipMemcpy(H2D_setup)");
}

extern "C" void copy_from_hip(void * shared, size_t bytes) {
    check(hipMemcpyAsync(shared, source, bytes, hipMemcpyDeviceToHost, stream), "hipMemcpyAsync(D2H_payload)");
}

extern "C" void setup_hip_add(void * raw, size_t payload_offset, const float * rhs, size_t bytes, bool map, size_t expert_bytes) {
    check(hipMalloc(&add_lhs, bytes), "hipMalloc(add_lhs)");
    check(hipMalloc(&add_rhs, bytes), "hipMalloc(add_rhs)");
    check(hipMemcpy(add_rhs, rhs, bytes, hipMemcpyHostToDevice), "hipMemcpy(H2D_add_rhs)");
    if (map) {
        void * mapped = nullptr;
        check(hipHostGetDevicePointer(&mapped, raw, 0), "hipHostGetDevicePointer(payload)");
        mapped_payload = reinterpret_cast<float *>(static_cast<char *>(mapped) + payload_offset);
    }
    if (expert_bytes) {
        check(hipHostMalloc(&expert_host, expert_bytes), "hipHostMalloc(expert)");
        std::memset(expert_host, 0x5a, expert_bytes);
        check(hipMalloc(&expert_device, expert_bytes), "hipMalloc(expert)");
    }
}

extern "C" void fill_hip_add(const float * lhs, size_t bytes) {
    check(hipMemcpy(add_lhs, lhs, bytes, hipMemcpyHostToDevice), "hipMemcpy(H2D_add_lhs)");
}

extern "C" void copy_hip_expert(size_t bytes) {
    check(hipMemcpyAsync(expert_device, expert_host, bytes, hipMemcpyHostToDevice, stream), "hipMemcpyAsync(H2D_expert)");
}

extern "C" void launch_hip_add(size_t count, bool direct) {
    auto * dst = direct ? mapped_payload : static_cast<float *>(source);
    hipLaunchKernelGGL(add_fp32, dim3((count + threads - 1) / threads), dim3(threads), 0, stream, dst, add_lhs, add_rhs, count);
    check(hipGetLastError(), "launch(add_fp32)");
}

extern "C" void blit_from_hip(size_t count) {
    hipLaunchKernelGGL(blit_fp32, dim3((count + threads - 1) / threads), dim3(threads), 0, stream,
            mapped_payload, static_cast<const float *>(source), count);
    check(hipGetLastError(), "launch(blit_fp32)");
}

extern "C" void signal_hip(uint32_t value) {
    check(hipStreamWriteValue32(stream, ticket, value, 0), "hipStreamWriteValue32");
}

extern "C" void sync_hip() {
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
}

extern "C" void cleanup_hip(void * shared) {
    if (expert_host) {
        check(hipFree(expert_device), "hipFree(expert)");
        check(hipHostFree(expert_host), "hipHostFree(expert)");
    }
    if (add_lhs) {
        check(hipFree(add_lhs), "hipFree(add_lhs)");
        check(hipFree(add_rhs), "hipFree(add_rhs)");
    }
    check(hipHostUnregister(shared), "hipHostUnregister");
    check(hipFree(source), "hipFree");
    check(hipStreamDestroy(stream), "hipStreamDestroy");
}
