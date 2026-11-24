#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
hipStream_t stream;
void * source;
void * ticket;

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

extern "C" void signal_hip(uint32_t value) {
    check(hipStreamWriteValue32(stream, ticket, value, 0), "hipStreamWriteValue32");
}

extern "C" void sync_hip() {
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
}

extern "C" void cleanup_hip(void * shared) {
    check(hipHostUnregister(shared), "hipHostUnregister");
    check(hipFree(source), "hipFree");
    check(hipStreamDestroy(stream), "hipStreamDestroy");
}
