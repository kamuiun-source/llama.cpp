#include "kernel-helpers.cuh"
#include <cstring>

namespace {
cudaStream_t stream;
float * staging;
void * result;
const void * mapped_f32;
const void * mapped_q8;
constexpr unsigned threads = 128;

__global__ void blit_f32(const float4 * source, float4 * dest) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = source[i];
}

__global__ void blit_q8(const uint4 * source, uint4 * dest) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q8_bytes / sizeof(uint4)) {
        dest[i] = source[i];
    }
}
}

extern "C" void mi50_input_setup(int device, void * host, size_t bytes, size_t q8_offset) {
    CUDA_CHECK(hipSetDevice(device));
    hipDeviceProp_t props{};
    CUDA_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strncmp(props.gcnArchName, "gfx906", 6) != 0 || !props.canMapHostMemory) {
        std::printf("skip: device=%d arch=%s mapped_host=%d\n", device, props.gcnArchName, props.canMapHostMemory);
        std::exit(77);
    }
    std::printf("hip_device=%d name=%s arch=%s\n", device, props.name, props.gcnArchName);
    CUDA_CHECK(hipHostRegister(host, bytes, hipHostRegisterPortable | hipHostRegisterMapped));
    void * mapped = nullptr;
    CUDA_CHECK(hipHostGetDevicePointer(&mapped, host, 0));
    mapped_f32 = mapped;
    mapped_q8 = static_cast<const char *>(mapped) + q8_offset;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUDA_CHECK(hipMalloc(&staging, f32_bytes));
    CUDA_CHECK(hipMalloc(&result, q8_bytes));
}

extern "C" void mi50_input_submit(unsigned arm) {
    if (arm == 0) {
        blit_f32<<<elements / (4 * threads), threads, 0, stream>>>(
                static_cast<const float4 *>(mapped_f32), reinterpret_cast<float4 *>(staging));
        CUDA_CHECK(hipGetLastError());
        quantize_fixed(staging, result, stream);
    } else if (arm == 1) {
        quantize_fixed(mapped_f32, result, stream);
    } else {
        constexpr unsigned vectors = q8_bytes / sizeof(uint4);
        static_assert(q8_bytes % sizeof(uint4) == 0, "full Q8_1 vector blocks");
        blit_q8<<<(vectors + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<const uint4 *>(mapped_q8), static_cast<uint4 *>(result));
        CUDA_CHECK(hipGetLastError());
    }
}

extern "C" void mi50_input_sync() {
    CUDA_CHECK(hipStreamSynchronize(stream));
}

extern "C" void mi50_input_read(void * output) {
    CUDA_CHECK(hipMemcpy(output, result, q8_bytes, hipMemcpyDeviceToHost));
}

extern "C" void mi50_input_cleanup(void * host) {
    CUDA_CHECK(hipFree(result));
    CUDA_CHECK(hipFree(staging));
    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipHostUnregister(host));
}
