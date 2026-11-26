#include "mmvq.cuh"
#include "hc-fused.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#ifdef GGML_USE_HIP
#include "mmvq-iq2-strata.cuh"
#include "../ggml-hip-mtp3-skip-cpu-zero.h"
#include "mtp3-compact-slot-map.cuh"
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
#include <cupti_activity.h>
#include <dlfcn.h>
#include <nvtx3/nvToolsExt.h>

bool ggml_cuda_mmvq_trace_enabled(int device) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MMVQ_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_ADA_LOVELACE;
}

namespace {
struct mmvq_trace_tensors {
    const ggml_tensor * weight;
    const ggml_tensor * src;
    const ggml_tensor * dst;
};

thread_local const mmvq_trace_tensors * mmvq_current_trace = nullptr;

struct mmvq_trace_scope {
    mmvq_trace_tensors tensors;
    const mmvq_trace_tensors * previous;

    mmvq_trace_scope(int device, bool dense, const ggml_tensor * weight, const ggml_tensor * src, const ggml_tensor * dst)
        : tensors{weight, src, dst}, previous(mmvq_current_trace) {
        mmvq_current_trace = dense && ggml_cuda_mmvq_trace_enabled(device) ? &tensors : nullptr;
    }

    ~mmvq_trace_scope() { mmvq_current_trace = previous; }
};

static uint64_t mmvq_graph_node_id(cudaGraphNode_t node) {
    static const auto get_id = [] {
        auto * symbol = dlsym(RTLD_DEFAULT, "cuptiGetGraphNodeId");
        if (!symbol) {
            char soname[48];
            std::snprintf(soname, sizeof(soname), "libcupti.so.%d", CUDART_VERSION / 1000);
            // Keep the CUPTI library loaded for the cached function pointer.
            auto * lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
            if (lib) {
                symbol = dlsym(lib, "cuptiGetGraphNodeId");
            }
        }
        if (!symbol) {
            GGML_LOG_WARN("CUDA MMVQ trace: cuptiGetGraphNodeId unavailable\n");
        }
        return reinterpret_cast<decltype(&cuptiGetGraphNodeId)>(symbol);
    }();
    uint64_t id = 0;
    return get_id && get_id(node, &id) == CUPTI_SUCCESS ? id : 0;
}

struct mmvq_launch_trace {
    const mmvq_trace_tensors * tensors = mmvq_current_trace;
    cudaStream_t stream;
    char label[768];

    mmvq_launch_trace(const ggml_cuda_kernel_launch_params & params, int cols, bool fusion,
                     bool small_k, bool halve_iters, bool hc_reuse = false, int row_groups = 1,
                     const char * hc_down_stage = nullptr) : stream(params.stream) {
        if (!tensors) {
            return;
        }
        const auto * weight = tensors->weight;
        const auto * src = tensors->src;
        const auto * dst = tensors->dst;
        constexpr auto shape = [](const ggml_tensor * tensor, int i) { return static_cast<long long>(tensor->ne[i]); };
        std::snprintf(label, sizeof(label),
            "mmvq_launch dst=\"%s\" weight=\"%s\" quant=%s weight_ne=%lldx%lldx%lldx%lld src_ne=%lldx%lldx%lldx%lld dst_ne=%lldx%lldx%lldx%lld"
            " grid=%ux%ux%u block=%ux%ux%u smem=%zu cols=%d fusion=%d small_k=%d halve_iters=%d hc_reuse=%d row_groups=%d%s%s",
            dst->name, weight->name, ggml_type_name(weight->type),
            shape(weight, 0), shape(weight, 1), shape(weight, 2), shape(weight, 3),
            shape(src, 0), shape(src, 1), shape(src, 2), shape(src, 3),
            shape(dst, 0), shape(dst, 1), shape(dst, 2), shape(dst, 3),
            params.block_nums.x, params.block_nums.y, params.block_nums.z,
            params.block_dims.x, params.block_dims.y, params.block_dims.z, params.shmem,
            cols, fusion, small_k, halve_iters, hc_reuse, row_groups,
            hc_down_stage ? " hc_down_stage=" : "", hc_down_stage ? hc_down_stage : "");
        nvtxRangePushA(label);
    }

    ~mmvq_launch_trace() {
        if (!tensors) {
            return;
        }
        nvtxRangePop();
        cudaStreamCaptureStatus capture;
        const cudaGraphNode_t * deps = nullptr;
        size_t count = 0;
        // The sole frontier node after this launch is the kernel just added to this stream.
        if (cudaStreamGetCaptureInfo_v3(stream, &capture, nullptr, nullptr, &deps, nullptr, &count) != cudaSuccess ||
                capture != cudaStreamCaptureStatusActive || count != 1) {
            return;
        }
        cudaGraphNodeType type;
        const auto node = deps[0];
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) {
            return;
        }
        const uint64_t id = mmvq_graph_node_id(node);
        if (id) {
            char mark[832];
            std::snprintf(mark, sizeof(mark), "mmvq_graph_node graphNodeId=%llu %s",
                    static_cast<unsigned long long>(id), label + sizeof("mmvq_launch ") - 1);
            nvtxMarkA(mark);
        }
    }
};
} // namespace
#endif

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

template <bool packed_signs = false>
static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1<packed_signs>;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1<packed_signs>;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (small_k && type == GGML_TYPE_Q8_0 && ncols_dst == 2 &&
            (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_TURING)) {
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false,
          bool hc_reuse = false, int row_groups = 1, int split_k = 1>
__launch_bounds__(row_groups*calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int group = row_groups == 1 ? 0 : threadIdx.z;
    const     int row0 = rows_per_cuda_block*(row_groups*blockIdx.x + group);
    const     int blocks_per_row_x = ncols_x / qk;
    const     int k_first = split_k == 1 ? 0 : blockIdx.z * blocks_per_row_x / split_k;
    const     int k_end = split_k == 1 ? blocks_per_row_x : (blockIdx.z + 1) * blocks_per_row_x / split_k;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = split_k == 1 ? blockIdx.z : 0;
    if constexpr (split_k != 1) {
        static_assert(hc_reuse && type == GGML_TYPE_Q8_0 && ncols_dst == 3 && row_groups == 1);
        dst += blockIdx.z * stride_sample_dst;
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = k_first + tid / (qi/vdr); kbx < k_end; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < k_end) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

        if constexpr (hc_reuse) {
            static_assert(type == GGML_TYPE_Q8_0 && (ncols_dst == 2 || ncols_dst == 3) && rows_per_cuda_block == 2 && !has_fusion);
            constexpr int n = VDR_Q8_0_Q8_1_MMVQ;
            int u[ncols_dst][n];
            float dy[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const block_q8_1 & activation = y[j*stride_col_y + kby];
                dy[j] = __low2half(activation.ds);
#pragma unroll
                for (int k = 0; k < n; ++k) {
                    u[j][k] = get_int_b4(activation.qs, kqs + k);
                }
            }
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const block_q8_0 & weight = ((const block_q8_0 *) vx)[kbx_offset + i*stride_row_x + kbx];
                const float dx = weight.d;
                int v[n];
#pragma unroll
                for (int k = 0; k < n; ++k) {
                    v[k] = get_int_b2(weight.qs, kqs + k);
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += vec_dot_q8_0_q8_1_impl<float, n>(v, u[j], dx, dy[j]);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[row_groups][nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[group][threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[group][l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static constexpr int hc_down_split_k_parts = 4;

static __global__ void hc_down_split_k_reduce(const float * partial, float * dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = partial[i];
#pragma unroll
        for (int part = 1; part < hc_down_split_k_parts; ++part) {
            sum += partial[part * n + i];
        }
        dst[i] = sum;
    }
}
#endif

#ifdef GGML_USE_HIP
static __device__ __forceinline__ void iq3_xxs_dot_shared_tokens(
        const block_iq3_xxs & weight, const block_q8_1 * input, int input_stride,
        int kqs, unsigned tokens, float (&acc)[3]) {
    const int2 packed = make_int2(get_int_b2(weight.qs, kqs), get_int_b2(weight.qs, kqs + 1));
    const auto * q3 = reinterpret_cast<const uint8_t *>(&packed);
    const uint32_t aux = get_int_b2(weight.qs, QK_K/16 + kqs/2);
    int sum[3] = {};
    input += kqs/2;

#pragma unroll
    for (int l = 0; l < 8; l += 2) {
        const uint32_t signs = unpack_ksigns(aux >> (7*l/2));
        const int sign_l = __vcmpne4(signs & 0x08040201, 0);
        const int sign_h = __vcmpne4(signs & 0x80402010, 0);
        const int grid_l = __vsub4(iq3xxs_grid[q3[l]] ^ sign_l, sign_l);
        const int grid_h = __vsub4(iq3xxs_grid[q3[l + 1]] ^ sign_h, sign_h);
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            if (tokens & (1u << t)) {
                const auto & activation = input[t * input_stride];
                sum[t] = ggml_cuda_dp4a(grid_l, get_int_b4(activation.qs, l), sum[t]);
                sum[t] = ggml_cuda_dp4a(grid_h, get_int_b4(activation.qs, l + 1), sum[t]);
            }
        }
    }
    const int scale = aux >> 28;
    const float weight_d = __half2float(weight.d);
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        if (tokens & (1u << t)) {
            const int scaled = (scale*sum[t] + sum[t]/2)/2;
            const float d = weight_d * __low2float(input[t * input_stride].ds);
            acc[t] += d * scaled;
        }
    }
}

__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_iq3_xxs_moe_share_weights(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * input,
        const int32_t * ids, float * dst, int ids_stride, int expert_stride, int row_stride,
        int input_stride, int dst_rank_stride, int dst_token_stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int lane_group = QI3_XXS / VDR_IQ3_XXS_Q8_1_MMVQ;
    const int token = threadIdx.y;
    const int rank = blockIdx.y;
    const int expert = ids[token * ids_stride + rank];
    int ranks[3];
    unsigned tokens = 0;
    bool duplicate = false;
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const auto matches = __ballot(threadIdx.x < 10 && ids[t * ids_stride + threadIdx.x] == expert);
        ranks[t] = __ffsll(matches) - 1;
        tokens |= matches ? 1u << t : 0;
        duplicate |= (matches & (matches - 1)) != 0;
    }
    if (duplicate) {
        // Multiple routes in the same token cannot share one rank destination.
        tokens = 1u << token;
        ranks[token] = rank;
    } else if (tokens & ((1u << token) - 1)) {
        return;
    }

    const int row0 = 2 * blockIdx.x;
    const int weight_base = expert * expert_stride + row0 * row_stride;
    const int kqs = VDR_IQ3_XXS_Q8_1_MMVQ * (threadIdx.x % lane_group);
    float acc_up[2][3] = {};
    float acc_gate[2][3] = {};
    for (int k = threadIdx.x / lane_group; k < 10; k += warp_size / lane_group) {
        const auto * activation = input + k * (QK_K / QK8_1);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int block = weight_base + r * row_stride + k;
            iq3_xxs_dot_shared_tokens(up[block], activation, input_stride, kqs, tokens, acc_up[r]);
            iq3_xxs_dot_shared_tokens(gate[block], activation, input_stride, kqs, tokens, acc_gate[r]);
        }
    }
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        if (tokens & (1u << t)) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float u = warp_reduce_sum<warp_size>(acc_up[r][t]);
                const float g = warp_reduce_sum<warp_size>(acc_gate[r][t]);
                if (threadIdx.x == r) {
                    dst[t * dst_token_stride + ranks[t] * dst_rank_stride + row0 + r] = u * ggml_cuda_op_silu_single(g);
                }
            }
        }
    }
}
#endif

static constexpr int iq_lds_ncols = 2560;
static constexpr int iq_lds_activation_blocks = iq_lds_ncols / QK8_1;

#ifdef GGML_USE_HIP
static __device__ __forceinline__ uint32_t iq4_nl_down_bank_selector(uint32_t shifted) {
#if defined(__gfx906__)
    uint32_t result;
    asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(result) :
        "v"(shifted), "s"(uint32_t{0x04040404}), "v"(uint32_t{0x03020100}));
    return result;
#else
    return 0x03020100u | (shifted & 0x04040404u);
#endif
}

static __device__ __forceinline__ int2 iq4_nl_down_decode(uint32_t q) {
    const auto * table = reinterpret_cast<const uint32_t *>(kvalues_iq4nl);
    const uint32_t even = q & 0x07070707u;
    const uint32_t odd = (q >> 4) & 0x07070707u;
    const uint32_t even_low = ggml_hip_perm(table[1], table[0], even);
    const uint32_t even_high = ggml_hip_perm(table[3], table[2], even);
    const uint32_t odd_low = ggml_hip_perm(table[1], table[0], odd);
    const uint32_t odd_high = ggml_hip_perm(table[3], table[2], odd);
    // Move each nibble's high bit directly to the byte selector's bank bit.
    const uint32_t even_select = iq4_nl_down_bank_selector(q >> 1);
    const uint32_t odd_select = iq4_nl_down_bank_selector(q >> 5);
    return make_int2(ggml_hip_perm(even_high, even_low, even_select), ggml_hip_perm(odd_high, odd_low, odd_select));
}

static __device__ __forceinline__ float vec_dot_iq4_nl_down_compact(
        const void * weights, const block_q8_1 * activation, const int & block, const int & iqs) {
    const auto & weight = static_cast<const block_iq4_nl *>(weights)[block];
    const auto * q8 = reinterpret_cast<const int *>(activation->qs) + iqs;
    int sum = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int2 decoded = iq4_nl_down_decode(get_int_b2(weight.qs, iqs + l));
        sum = ggml_cuda_dp4a(decoded.x, q8[l], sum);
        sum = ggml_cuda_dp4a(decoded.y, q8[l + 4], sum);
    }
    const float d = __half2float(weight.d) * __low2float(activation->ds);
    return d * sum;
}
#endif

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false, bool packed_signs = false,
          int iq_lds_tokens = 0, int iq_row_groups = 1, bool iq2_paired_dot = false, bool iq4_compact_decode = false,
          bool skip_cpu_zero = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const int zero_slot) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    static_assert(!iq4_compact_decode || (type == GGML_TYPE_IQ4_NL && c_rows_per_block == 2 && !has_fusion &&
        !packed_signs && iq_lds_tokens == 0 && iq_row_groups == 1 && !iq2_paired_dot));
#ifdef GGML_USE_HIP
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = iq4_compact_decode ? vec_dot_iq4_nl_down_compact : get_vec_dot_q_cuda<packed_signs>(type);
#else
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda<packed_signs>(type);
#endif

    static_assert(iq_row_groups == 1 || (iq_row_groups == 2 && (iq_lds_tokens == 3 ||
        (iq_lds_tokens == 0 && type == GGML_TYPE_IQ2_S && c_rows_per_block == 2 && has_fusion && packed_signs))));

    static_assert(!iq2_paired_dot || (type == GGML_TYPE_IQ2_S && c_rows_per_block == 2 && has_fusion &&
        packed_signs && iq_lds_tokens == 0 && iq_row_groups == 1));

    using grid_entry = std::conditional_t<type == GGML_TYPE_IQ2_S, uint64_t, uint32_t>;
    const grid_entry * staged_grid = nullptr;
    const block_q8_1 * staged_y = nullptr;
    if constexpr (iq_lds_tokens != 0) {
        static_assert((type == GGML_TYPE_IQ2_S && iq_lds_tokens == 3) ||
                      (type == GGML_TYPE_IQ3_XXS && (iq_lds_tokens == 2 || iq_lds_tokens == 3)));
        static_assert(c_rows_per_block == 2 && has_fusion && packed_signs);
        static_assert(sizeof(block_q8_1) % sizeof(uint32_t) == 0 && alignof(block_q8_1) <= alignof(uint32_t));
        constexpr int grid_size = type == GGML_TYPE_IQ2_S ?
            sizeof(iq2s_grid) / sizeof(iq2s_grid[0]) : sizeof(iq3xxs_grid) / sizeof(iq3xxs_grid[0]);
        constexpr int activation_words = iq_lds_activation_blocks * sizeof(block_q8_1) / sizeof(uint32_t);
        __shared__ grid_entry grid[grid_size];
        __shared__ uint32_t activation[iq_lds_tokens][activation_words];

        for (int i = threadIdx.y * warp_size + threadIdx.x; i < grid_size; i += iq_lds_tokens * warp_size) {
            if constexpr (type == GGML_TYPE_IQ2_S) {
                grid[i] = iq2s_grid[i];
            } else {
                grid[i] = iq3xxs_grid[i];
            }
        }
        const auto * src = reinterpret_cast<const uint32_t *>(static_cast<const block_q8_1 *>(vy) + threadIdx.y * stride_col_y);
        auto * token_activation = activation[threadIdx.y];
        for (int i = threadIdx.x; i < activation_words; i += warp_size) {
            token_activation[i] = src[i];
        }
        __syncthreads();
        staged_grid = grid;
        staged_y = reinterpret_cast<const block_q8_1 *>(token_activation);
    }

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx = threadIdx.y;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
#ifdef GGML_USE_HIP
    if constexpr (skip_cpu_zero) {
        static_assert(iq4_compact_decode ||
                (type == GGML_TYPE_Q2_0 && !has_fusion) ||
                (type == GGML_TYPE_IQ4_XS && has_fusion));
        static_assert(iq_lds_tokens == 0 && iq_row_groups == 1);
        if (channel_x == uint32_t(zero_slot)) {
            if (threadIdx.x < c_rows_per_block) {
                float * route_dst = dst + channel_dst * stride_channel_dst + token_idx * stride_col_dst;
                const uint32_t row = c_rows_per_block * blockIdx.x + threadIdx.x;
                ggml_hip_mtp3_skip_cpu_zero::store_wave(route_dst, row, nrows_x);
            }
            return;
        }
    } else
#endif
    {
        GGML_UNUSED(zero_slot);
    }
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;

#pragma unroll 1
    for (int group = 0; group < iq_row_groups; ++group) {
        const int row0 = c_rows_per_block*(iq_row_groups*blockIdx.x + group);
        const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

        // partial sum for each thread
        float tmp[c_rows_per_block] = {0.0f};
        float tmp_gate[c_rows_per_block] = {0.0f};

        if constexpr (iq_lds_tokens != 0) {
            const int kbx = threadIdx.x / (qi/vdr);
            const int kqs = vdr * (threadIdx.x % (qi/vdr));
            const int kbx_next = kbx + blocks_per_iter;
            const bool has_next = kbx_next < blocks_per_row_x;
            using block_iq = std::conditional_t<type == GGML_TYPE_IQ2_S, block_iq2_s, block_iq3_xxs>;
            const auto * up = static_cast<const block_iq *>(vx) + kbx_offset;
            const auto * gate = static_cast<const block_iq *>(vgate) + kbx_offset;
            const auto * activation = staged_y + kqs/2;

#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                const int row = i * stride_row_x;
                const auto up_current = load_iq_mmvq_weights(up[row + kbx], kqs);
                const auto gate_current = load_iq_mmvq_weights(gate[row + kbx], kqs);
                decltype(load_iq_mmvq_weights(up[row + kbx], kqs)) up_next, gate_next;
                if (has_next) {
                    up_next = load_iq_mmvq_weights(up[row + kbx_next], kqs);
                    gate_next = load_iq_mmvq_weights(gate[row + kbx_next], kqs);
                }

                const auto & current = activation[kbx * (qk/QK8_1)];
                tmp[i] += vec_dot_iq_q8_1_loaded<packed_signs>(up_current, current, staged_grid);
                tmp_gate[i] += vec_dot_iq_q8_1_loaded<packed_signs>(gate_current, current, staged_grid);
                if (has_next) {
                    const auto & next = activation[kbx_next * (qk/QK8_1)];
                    tmp[i] += vec_dot_iq_q8_1_loaded<packed_signs>(up_next, next, staged_grid);
                    tmp_gate[i] += vec_dot_iq_q8_1_loaded<packed_signs>(gate_next, next, staged_grid);
                }
            }
        } else {
            for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                const int kby = kbx * (qk/QK8_1);
                const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    const int weight = kbx_offset + i*stride_row_x + kbx;
                    if constexpr (iq2_paired_dot) {
                        const auto up = load_iq_mmvq_weights(static_cast<const block_iq2_s *>(vx)[weight], kqs);
                        const auto gate = load_iq_mmvq_weights(static_cast<const block_iq2_s *>(vgate)[weight], kqs);
                        vec_dot_iq2_s_q8_1_pair(up, gate, y[kby + kqs/2], tmp[i], tmp_gate[i]);
                    } else {
                        tmp[i] += vec_dot_q_cuda(vx, &y[kby], weight, kqs);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], weight, kqs);
                            }
                        }
                    }
                }
            }
        }

        ggml_cuda_pdl_lc();

        // Warp-level reduction only - no shared memory needed
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
                }
            }
        }

        // Write results
        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            float result = tmp[threadIdx.x];
            if constexpr (has_fusion) {
                const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (x_scale) {
                        result *= x_scale[channel_x];
                    }
                }
                if (x_bias) {
                    result += x_bias[bias_idx];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[threadIdx.x];
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        if (gate_scale) {
                            gate_value *= gate_scale[channel_x];
                        }
                    }
                    if (gate_bias) {
                        gate_value += gate_bias[bias_idx];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
        }

        if constexpr (!has_fusion) {
            GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
        } else if constexpr (type != GGML_TYPE_NVFP4) {
            GGML_UNUSED_VARS(x_scale, gate_scale);
        }
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
            const mmvq_launch_trace trace(launch_params, c_ncols_dst, true, small_k, halve_iters);
#endif
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
    const mmvq_launch_trace trace(launch_params, c_ncols_dst, false, small_k, halve_iters);
#endif
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

#ifdef GGML_USE_HIP
// Pack both modes into one atomic so a launch cannot see a partially applied pair.
static constexpr int gfx906_lds_mode_base = 5;
static std::atomic<int> gfx906_lds_modes{-1};

bool ggml_hip_gfx906_set_lds_modes(int iq2_s, int iq3_xxs) {
    const auto valid_mode = [](int mode) { return mode == 0 || mode == 2 || mode == 4; };
    if (!valid_mode(iq2_s) || !valid_mode(iq3_xxs)) {
        return false;
    }
    gfx906_lds_modes.store(iq2_s * gfx906_lds_mode_base + iq3_xxs, std::memory_order_relaxed);
    return true;
}

int ggml_hip_gfx906_lds_modes() {
    return gfx906_lds_modes.load(std::memory_order_relaxed);
}

static bool ggml_hip_gfx906_iq_sign_decode_configured() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GFX906_IQ_SIGN_DECODE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool ggml_hip_gfx906_iq_sign_decode() {
    return ggml_hip_gfx906_iq_sign_decode_configured() && ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VEGA20;
}

static bool ggml_hip_gfx906_iq2_paired_dot() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GFX906_IQ2_S_PAIRED_DOT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool ggml_hip_gfx906_iq2_lds_prefetch() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GFX906_IQ2_S_LDS_PREFETCH");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool ggml_hip_gfx906_iq2_no_lds_row_groups() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GFX906_IQ2_S_NO_LDS_ROW_GROUPS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

bool ggml_hip_gfx906_iq2_strata_enabled() {
    const int modes = ggml_hip_gfx906_lds_modes();
    return iq2_strata_mode.load(std::memory_order_relaxed) && ggml_hip_gfx906_iq_sign_decode_configured() &&
        ggml_hip_gfx906_iq2_paired_dot() &&
        (modes >= 0 ? modes / gfx906_lds_mode_base == 0 :
            !ggml_hip_gfx906_iq2_lds_prefetch() && !ggml_hip_gfx906_iq2_no_lds_row_groups());
}
#endif

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream, const int zero_slot) {

#ifdef GGML_USE_HIP
    int live_lds_mode = -1;
    if constexpr (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS) {
        const int modes = ggml_hip_gfx906_lds_modes();
        if (modes >= 0) {
            live_lds_mode = type == GGML_TYPE_IQ2_S ? modes / gfx906_lds_mode_base : modes % gfx906_lds_mode_base;
        }
    }
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        static const int swiglu_rows = [] {
            const char * value = std::getenv("GGML_HIP_GFX906_MOE_SWIGLU_ROWS");
            if (value && std::strcmp(value, "1") == 0) {
                return 1;
            }
            return value && std::strcmp(value, "4") == 0 ? 4 : 2;
        }();
        static const bool mtp4_swiglu_single_row = [] {
            const char * value = std::getenv("GGML_HIP_GFX906_MOE_SWIGLU_MTP4_ROWS");
            return value && std::strcmp(value, "1") == 0;
        }();
        const int swiglu_rows_selected = ncols_dst == 4 ? (mtp4_swiglu_single_row ? 1 : 2) : swiglu_rows;
        if (swiglu_rows_selected != 2 && ncols_x == 2560 && nrows_x == 640 && nchannels_dst == 10 && nchannels_y.z == 1 &&
                ncols_dst >= 2 && ncols_dst <= 4 && fusion.gate && fusion.glu_op == GGML_GLU_OP_SWIGLU &&
                !fusion.x_bias && !fusion.gate_bias && !fusion.x_scale && !fusion.gate_scale &&
                !fusion.shared_up && !fusion.shared_gate && !fusion.shared_dst &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VEGA20 &&
                (ncols_dst == 4 || live_lds_mode < 0 || !ggml_hip_gfx906_iq_sign_decode())) {
            const auto launch = [&](auto rows) {
                constexpr int rows_per_block = decltype(rows)::value;
                const dim3 block_nums(nrows_x / rows_per_block, nchannels_dst);
                const dim3 block_dims(warp_size, ncols_dst);
                const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, -1);
            };
            if (swiglu_rows_selected == 1) {
                launch(std::integral_constant<int, 1>{});
            } else {
                launch(std::integral_constant<int, 4>{});
            }
            if (ncols_dst == 4) {
                static const bool logged = [] {
                    GGML_LOG_INFO("HIP gfx906 IQ3_XXS routed SwiGLU applied: MTP positions=4 rows_per_block=1\n");
                    return true;
                }();
                GGML_UNUSED(logged);
            }
            return;
        }
    }
#endif

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

#ifdef GGML_USE_HIP
    if constexpr (type == GGML_TYPE_IQ4_NL) {
        static const bool compact_decode = [] {
            const char * value = std::getenv("GGML_HIP_GFX906_IQ4_NL_DOWN_COMPACT_DECODE");
            return value && std::strcmp(value, "1") == 0;
        }();
        if (compact_decode && !has_fusion && !fusion.shared_up && ncols_x == 640 && nrows_x == 2560 &&
                ncols_dst == 3 && nchannels_dst == 10 && nchannels_y.z == 10 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VEGA20) {
            if (zero_slot >= 0) {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false, false, 0, 1, false, true, true>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, zero_slot);
                static const bool skip_logged = (GGML_LOG_INFO("HIP gfx906 CPU zero-slot skip applied: IQ4_NL down\n"), true);
                GGML_UNUSED(skip_logged);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false, false, 0, 1, false, true>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, -1);
            }
            static const bool logged = [] {
                GGML_LOG_INFO("HIP gfx906 IQ4_NL routed down compact decode applied: positions=3 rows_per_block=2\n");
                return true;
            }();
            GGML_UNUSED(logged);
            return;
        }
    }
    if constexpr (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS) {
        if ((ncols_dst == 2 || ncols_dst == 3) && ncols_x == iq_lds_ncols && nrows_x == 640 && nchannels_dst == 10 &&
                nchannels_y.z == 1 &&
                fusion.gate && fusion.glu_op == GGML_GLU_OP_SWIGLU && !fusion.x_bias && !fusion.gate_bias &&
                !fusion.x_scale && !fusion.gate_scale && !fusion.shared_up && !fusion.shared_gate && !fusion.shared_dst &&
                ggml_hip_gfx906_iq_sign_decode()) {
            const auto launch = [&](auto lds, auto groups) {
                constexpr int lds_tokens = decltype(lds)::value;
                constexpr int row_groups = decltype(groups)::value;
                const dim3 block_nums_iq(row_groups == 1 ? block_nums.x : nrows_x / (rows_per_block*row_groups), block_nums.y);
                const ggml_cuda_kernel_launch_params launch_params_iq(block_nums_iq, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true, true, lds_tokens, row_groups>, launch_params_iq,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, -1);
                if constexpr (lds_tokens != 0) {
                    static const bool logged_lds = [] {
                        constexpr const char * type_name = type == GGML_TYPE_IQ2_S ? "IQ2_S" : "IQ3_XXS";
                        if constexpr (row_groups == 2) {
                            GGML_LOG_INFO("HIP gfx906 %s routed SwiGLU LDS prefetch applied: positions=%d row_groups=%d\n",
                                    type_name, lds_tokens, row_groups);
                        } else {
                            GGML_LOG_INFO("HIP gfx906 %s routed SwiGLU LDS prefetch applied: positions=%d\n", type_name, lds_tokens);
                        }
                        return true;
                    }();
                    GGML_UNUSED(logged_lds);
                } else if constexpr (row_groups == 2) {
                    static const bool logged_groups = [] {
                        GGML_LOG_INFO("HIP gfx906 IQ2_S routed SwiGLU without LDS prefetch applied: positions=3 row_groups=2\n");
                        return true;
                    }();
                    GGML_UNUSED(logged_groups);
                }
            };
            bool use_no_lds_row_groups = false;
            if constexpr (type == GGML_TYPE_IQ2_S) {
                use_no_lds_row_groups = live_lds_mode < 0 && ggml_hip_gfx906_iq2_no_lds_row_groups() && ncols_dst == 3 &&
                    nrows_x % (2*rows_per_block) == 0;
            }
            static const bool lds_prefetch = [] {
                if constexpr (type == GGML_TYPE_IQ2_S) {
                    return ggml_hip_gfx906_iq2_lds_prefetch();
                }
                const char * value = std::getenv("GGML_HIP_GFX906_IQ3_XXS_LDS_PREFETCH");
                return value && std::strcmp(value, "1") == 0;
            }();
            const bool use_lds = (live_lds_mode < 0 ? lds_prefetch : live_lds_mode != 0) &&
                stride_col_y == iq_lds_activation_blocks &&
                (type == GGML_TYPE_IQ3_XXS || ncols_dst == 3);
            static std::once_flag logged[2];
            std::call_once(logged[ncols_dst - 2], [ncols_dst] {
                GGML_LOG_INFO("HIP gfx906 packed IQ sign decoding applied: type=%s routed SwiGLU positions=%u\n",
                        ggml_type_name(type), ncols_dst);
            });
            if constexpr (type == GGML_TYPE_IQ2_S) {
                if (ggml_hip_gfx906_iq2_paired_dot() && ncols_dst == 3 && !use_lds && !use_no_lds_row_groups) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true, true, 0, 1, true>, launch_params,
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                        stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                        ncols_dst, ids_stride, -1);
                    static const bool logged = [] {
                        GGML_LOG_INFO("HIP gfx906 IQ2_S routed SwiGLU paired dot applied: positions=3 rows_per_block=2\n");
                        return true;
                    }();
                    GGML_UNUSED(logged);
                    return;
                }
            }
            if (use_no_lds_row_groups) {
                if constexpr (type == GGML_TYPE_IQ2_S) {
                    launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{});
                }
            } else if (!use_lds) {
                launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
            } else if (ncols_dst == 3) {
                static const bool lds_row_groups = [] {
                    const char * value = std::getenv(type == GGML_TYPE_IQ2_S ?
                        "GGML_HIP_GFX906_IQ2_S_LDS_ROW_GROUPS" : "GGML_HIP_GFX906_IQ3_XXS_LDS_ROW_GROUPS");
                    return value && std::strcmp(value, "1") == 0;
                }();
                if ((live_lds_mode < 0 ? lds_row_groups : live_lds_mode == 4) && nrows_x % (2*rows_per_block) == 0) {
                    launch(std::integral_constant<int, 3>{}, std::integral_constant<int, 2>{});
                } else {
                    launch(std::integral_constant<int, 3>{}, std::integral_constant<int, 1>{});
                }
            } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
                launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
            }
            return;
        }
    }
    if (zero_slot >= 0 && !fusion.shared_up && !fusion.shared_gate && !fusion.shared_dst &&
            ncols_dst == 3 && nchannels_dst == 10) {
        if constexpr (type == GGML_TYPE_Q2_0) {
            if (!has_fusion && ncols_x == 640 && nrows_x == 2560 && nchannels_y.z == 10) {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false, false, 0, 1, false, false, true>,
                    launch_params, vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y,
                    stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride, zero_slot);
                static const bool skip_logged = (GGML_LOG_INFO("HIP gfx906 CPU zero-slot skip applied: Q2_0 down\n"), true);
                GGML_UNUSED(skip_logged);
                return;
            }
        }
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            if (fusion.gate && fusion.glu_op == GGML_GLU_OP_SWIGLU && !fusion.x_bias && !fusion.gate_bias &&
                    !fusion.x_scale && !fusion.gate_scale && ncols_x == 2560 && nrows_x == 640 && nchannels_y.z == 1) {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true, false, 0, 1, false, false, true>,
                    launch_params, vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y,
                    stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, ids_stride, zero_slot);
                static const bool skip_logged = (GGML_LOG_INFO("HIP gfx906 CPU zero-slot skip applied: IQ4_XS gate/up\n"), true);
                GGML_UNUSED(skip_logged);
                return;
            }
        }
    }
#endif

    GGML_UNUSED(zero_slot);
    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, -1);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, -1);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const int zero_slot = -1) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream, zero_slot);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const int zero_slot = -1) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, zero_slot);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, zero_slot);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, zero_slot);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
    const mmvq_trace_scope trace_scope(ctx.device, ids == nullptr, src0, src1, dst);
#endif
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static const bool hc_down_split_k = [] {
        const char * value = std::getenv("GGML_CUDA_HC_DOWN_SPLIT_K");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool fixed_mtp_three = [] {
        const char * value = std::getenv("LLAMA_MTP_VERIFY_THREE");
        return value && std::strcmp(value, "1") == 0;
    }();
    const auto & device = ggml_cuda_info().devices[ctx.device];
    const bool hc_down_width3 =
            (hc_down_split_k
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
             || mmvq_current_trace
#endif
            ) && fixed_mtp_three && !ids && !fusion && src0->type == GGML_TYPE_Q8_0 && device.cc == GGML_CUDA_CC_ADA_LOVELACE &&
            ne00 == 10240 && ne01 == 320 && ne11 == 3 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
            ggml_is_contiguous(dst) && ggml_cuda_hc_projection_for_weight(src0->name) == GGML_CUDA_HC_DOWN;
#endif
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        const auto quantize = [&] {
            quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        };
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
        if (mmvq_current_trace && hc_down_width3) {
            const ggml_cuda_kernel_launch_params params(
                    dim3((ne10_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, ne11, ne12 * ne13),
                    dim3(CUDA_QUANTIZE_BLOCK_SIZE), 0, stream);
            const mmvq_launch_trace trace(params, ne11, false, false, false, false, 1, "quantize");
            quantize();
        } else
#endif
        {
            quantize();
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    int zero_slot = -1;
#ifdef GGML_USE_HIP
    namespace cpu_zero = ggml_hip_mtp3_skip_cpu_zero;
    if (cpu_zero::enabled() && ids && ids->ne[0] == 10 && ids->ne[1] == 3 &&
            ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VEGA20) {
        const int tagged_slot = cpu_zero::lookup_slot(ids->view_src);
        if (tagged_slot >= 0 && tagged_slot < ne02) {
            zero_slot = tagged_slot;
        }
    }

    const bool iq2_strata = src0->type == GGML_TYPE_IQ2_S && ggml_hip_gfx906_iq2_strata_enabled();
    const bool iq3_strata = src0->type == GGML_TYPE_IQ3_XXS && ggml_hip_gfx906_iq3_strata_enabled();
    if ((iq2_strata || iq3_strata) && ne00 == 2560 && ne01 == 640 &&
            ne03 == 1 && ne13 == 1 && ne3 == 1 && ncols_dst == 3 && nchannels_y == 1 && nchannels_dst == 10 &&
            s01 == iq3_strata_reuse::weight_blocks && s02 >= ne01 * s01 && s02 <= INT32_MAX && src0->nb[2] % ts_src0 == 0 &&
            stride_col_y == iq3_strata_reuse::activation_blocks && ggml_is_contiguous(dst) &&
            fusion_local.gate && fusion_local.glu_op == GGML_GLU_OP_SWIGLU &&
            !fusion_local.x_bias && !fusion_local.gate_bias && !fusion_local.x_scale && !fusion_local.gate_scale &&
            !fusion_local.shared_up && !fusion_local.shared_gate && !fusion_local.shared_dst &&
            ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VEGA20) {
        namespace plan = ggml_hip_iq3_strata_plan;
        namespace reuse = iq3_strata_reuse;
        const auto * map = plan::map_from_ids(ids, iq2_strata ? plan::iq2 : plan::iq3);
        if (map && map->data) {
            const bool compact = ggml_hip_mtp3_compact_slot_map_enabled() &&
                    ggml_hip_mtp3_compact_slot_map::is_lookup(ids->view_src);
            const auto * groups = static_cast<const int32_t *>(map->data) + (compact ? plan::routes : plan::experts);
            const ggml_cuda_kernel_launch_params params(dim3(reuse::nrows / reuse::rows_per_tile, plan::routes),
                    dim3(reuse::waves * reuse::wave), 0, stream);
            const auto * q8 = reinterpret_cast<const block_q8_1 *>(src1_q8_1.get());
            if (iq2_strata) {
                if (zero_slot >= 0) {
                    ggml_cuda_kernel_launch(iq2_strata_reuse::gate_up_packed<true>, params,
                            static_cast<const block_iq2_s *>(src0->data), static_cast<const block_iq2_s *>(fusion_local.gate),
                            q8, groups, dst_d, int(s02), zero_slot);
                    static const bool skip_logged = (GGML_LOG_INFO("HIP gfx906 CPU zero-slot skip applied: IQ2_S Strata gate/up\n"), true);
                    GGML_UNUSED(skip_logged);
                } else {
                    ggml_cuda_kernel_launch(iq2_strata_reuse::gate_up_packed<false>, params,
                            static_cast<const block_iq2_s *>(src0->data), static_cast<const block_iq2_s *>(fusion_local.gate),
                            q8, groups, dst_d, int(s02), -1);
                }
                static const bool logged = (GGML_LOG_INFO("HIP gfx906 IQ2_S Strata gate/up applied: MTP positions=3 top_k=10\n"), true);
                GGML_UNUSED(logged);
            } else {
                if (zero_slot >= 0) {
                    ggml_cuda_kernel_launch(reuse::gate_up_packed<true>, params,
                            static_cast<const block_iq3_xxs *>(src0->data), static_cast<const block_iq3_xxs *>(fusion_local.gate),
                            q8, groups, dst_d, int(s02), zero_slot);
                    static const bool skip_logged = (GGML_LOG_INFO("HIP gfx906 CPU zero-slot skip applied: IQ3_XXS Strata gate/up\n"), true);
                    GGML_UNUSED(skip_logged);
                } else {
                    ggml_cuda_kernel_launch(reuse::gate_up_packed<false>, params,
                            static_cast<const block_iq3_xxs *>(src0->data), static_cast<const block_iq3_xxs *>(fusion_local.gate),
                            q8, groups, dst_d, int(s02), -1);
                }
                static const bool logged = (GGML_LOG_INFO("HIP gfx906 IQ3_XXS Strata gate/up applied: MTP positions=3 top_k=10\n"), true);
                GGML_UNUSED(logged);
            }
            return;
        }
    }
    const auto * lookup = ids ? ids->view_src : nullptr;
    if (lookup && lookup->op == GGML_OP_GET_ROWS && std::strcmp(lookup->name, "ffn_moe_mtp_share_iq3_weights") == 0 &&
            src0->type == GGML_TYPE_IQ3_XXS && ne00 == 2560 && ne01 == 640 && ncols_dst == 3 &&
            nchannels_y == 1 && nchannels_dst == 10 && ggml_is_contiguous(dst) &&
            ids->ne[0] == 10 && ids->ne[1] == 3 && fusion_local.gate && fusion_local.glu_op == GGML_GLU_OP_SWIGLU &&
            !fusion_local.x_bias && !fusion_local.gate_bias && !fusion_local.x_scale && !fusion_local.gate_scale &&
            !fusion_local.shared_up && !fusion_local.shared_gate && !fusion_local.shared_dst &&
            ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VEGA20) {
        const auto & device = ggml_cuda_info().devices[ctx.device];
        const ggml_cuda_kernel_launch_params params(dim3(320, 10), dim3(device.warp_size, 3), 0, stream);
        ggml_cuda_kernel_launch(mul_mat_vec_iq3_xxs_moe_share_weights, params,
                static_cast<const block_iq3_xxs *>(src0->data), static_cast<const block_iq3_xxs *>(fusion_local.gate),
                reinterpret_cast<const block_q8_1 *>(src1_q8_1.get()), ids_d, dst_d, int(ids_stride), int(s02), int(s01),
                int(stride_col_y), int(stride_channel_dst), int(stride_col_dst));
        static const bool logged = [] {
            GGML_LOG_INFO("HIP gfx906 IQ3_XXS shared routed weights: MTP positions=3 top_k=10\n");
            return true;
        }();
        GGML_UNUSED(logged);
        return;
    }
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (hc_down_split_k && hc_down_width3) {
        const int n = ne01 * ne11;
        ggml_cuda_pool_alloc<float> partial(ctx.pool(), n * hc_down_split_k_parts);
        auto dims = calc_launch_params<GGML_TYPE_Q8_0>(3, ne01, 1, 1, device.warp_size, get_device_table_id(device.cc));
        dims.first.z = hc_down_split_k_parts;
        {
            const ggml_cuda_kernel_launch_params params(dims.first, dims.second, 0, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
            const mmvq_launch_trace trace(params, 3, false, false, false, true, 1, "partial");
#endif
            ggml_cuda_kernel_launch(mul_mat_vec_q<GGML_TYPE_Q8_0, 3, false, false, false, true, 1, hc_down_split_k_parts>, params,
                src0->data, src1_q8_1.get(), nullptr, fusion_local, partial.get(), uint32_t(ne00), make_uint3(0, 0, 0),
                uint32_t(s01), uint32_t(s11), uint32_t(s1), init_fastdiv_values(1), uint32_t(s02), uint32_t(s12), uint32_t(s2),
                init_fastdiv_values(1), uint32_t(s03), uint32_t(s13), uint32_t(n), uint32_t(0));
        }
        {
            constexpr int threads = 128;
            const ggml_cuda_kernel_launch_params params(dim3((n + threads - 1) / threads), dim3(threads), 0, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
            const mmvq_launch_trace trace(params, 3, false, false, false, true, 1, "reduce");
#endif
            ggml_cuda_kernel_launch(hc_down_split_k_reduce, params, partial.get(), dst_d, n);
        }
        static const bool logged = (GGML_LOG_INFO("CUDA HC down split-K applied: Q8_0 K=10240 rows=320 tokens=3 splits=%d\n",
                hc_down_split_k_parts), true);
        GGML_UNUSED(logged);
        return;
    }

    static const bool hc_up_pair = [] {
        const char * value = std::getenv("GGML_CUDA_HC_UP_PAIR");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool hc_down_reuse = [] {
        const char * value = std::getenv("GGML_CUDA_HC_DOWN_REUSE");
        return value && std::strcmp(value, "1") == 0;
    }();
    static const bool hc_up_reuse = [] {
        const char * value = std::getenv("GGML_CUDA_HC_UP_REUSE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if ((hc_down_reuse || hc_up_reuse) && !ids && !fusion && src0->type == GGML_TYPE_Q8_0 &&
            device.cc == GGML_CUDA_CC_ADA_LOVELACE && (ne11 == 2 || (ne11 == 3 && ggml_cuda_hc_mtp_three_enabled())) &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ((hc_down_reuse && ne00 == 10240 && ne01 == 320) || (hc_up_reuse && ne00 == 320 && ne01 == 10240)) &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        const auto projection = ggml_cuda_hc_projection_for_weight(src0->name);
        const bool down = hc_down_reuse && ne00 == 10240 && projection == GGML_CUDA_HC_DOWN;
        const bool up = hc_up_reuse && ne00 == 320 && projection == GGML_CUDA_HC_UP;
        if (down || up) {
            const auto launch = [&](auto cols, auto groups) {
                constexpr int ncols = decltype(cols)::value;
                constexpr int row_groups = decltype(groups)::value;
                auto dims = calc_launch_params<GGML_TYPE_Q8_0>(ncols, ne01, 1, 1, device.warp_size, get_device_table_id(device.cc));
                dims.first.x /= row_groups;
                dims.second.z = row_groups;
                const ggml_cuda_kernel_launch_params params(dims.first, dims.second, 0, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
                const mmvq_launch_trace trace(params, ncols, false, false, false, true, row_groups,
                        down && ncols == 3 ? "dot" : nullptr);
#endif
                ggml_cuda_kernel_launch(mul_mat_vec_q<GGML_TYPE_Q8_0, ncols, false, false, false, true, row_groups>, params,
                    src0->data, src1_q8_1.get(), nullptr, fusion_local, dst_d, uint32_t(ne00), make_uint3(0, 0, 0),
                    uint32_t(s01), uint32_t(s11), uint32_t(s1), init_fastdiv_values(1), uint32_t(s02), uint32_t(s12), uint32_t(s2),
                    init_fastdiv_values(1), uint32_t(s03), uint32_t(s13), uint32_t(s3), uint32_t(0));
            };
            const auto launch_projection = [&](auto cols) {
                constexpr int ncols = decltype(cols)::value;
                if (down) {
                    launch(cols, std::integral_constant<int, 1>{});
                    static const bool logged = (GGML_LOG_INFO("%s\n",
                        ncols == 2 ? "CUDA HC down reuse applied" : "CUDA HC down reuse applied: tokens=3"), true);
                    GGML_UNUSED(logged);
                } else {
                    // Pack two row pairs without changing the reduction for either pair.
                    launch(cols, std::integral_constant<int, 2>{});
                    static const bool logged = (GGML_LOG_INFO("%s\n",
                        ncols == 2 ? "CUDA HC up reuse applied" : "CUDA HC up reuse applied: tokens=3"), true);
                    GGML_UNUSED(logged);
                }
            };
            if (ne11 == 2) {
                launch_projection(std::integral_constant<int, 2>{});
            } else {
                launch_projection(std::integral_constant<int, 3>{});
            }
            return;
        }
    }
    if (hc_up_pair && !ids && !fusion && src0->type == GGML_TYPE_Q8_0 &&
            device.cc == GGML_CUDA_CC_ADA_LOVELACE &&
            ne00 == 320 && ne01 == 10240 && ne11 == 2 && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
            (std::strstr(src0->name, ".hc_attn_up.weight") || std::strstr(src0->name, ".hc_ffn_up.weight"))) {
        // Short reductions need fewer warps; keep the Q8_1 input and dot product unchanged.
        const auto dims = calc_launch_params<GGML_TYPE_Q8_0>(2, ne01, 1, 1, device.warp_size, get_device_table_id(device.cc), true);
        mul_mat_vec_q_switch_fusion<GGML_TYPE_Q8_0, 2, true>(
            src0->data, src1_q8_1.get(), nullptr, fusion_local, dst_d, ne00, make_uint3(0, 0, 0), s01, s11, s1,
            init_fastdiv_values(1), s02, s12, s2, init_fastdiv_values(1), s03, s13, s3,
            dims.first, dims.second, 0, 0, stream);
        static const bool logged = [] {
            GGML_LOG_INFO("CUDA HC up pair applied: Q8_0 K=320 rows=10240 tokens=2 warps=1\n");
            return true;
        }();
        GGML_UNUSED(logged);
        return;
    }
#endif

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, zero_slot);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
    const mmvq_trace_scope trace_scope(id, true, src0, src1, dst);
#endif

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
