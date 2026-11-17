# Packed Expert upload overlap with dynamic residency

This opt-in experiment keeps the existing persistent Expert bank, dynamic residency policy, scratch reuse, and MTP two-token verification. Enable it with `LLAMA_EXPERT_VK_OVERLAP_UPLOAD=1`. The old `LLAMA_EXPERT_OVERLAP` and global `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` remain unset.

The first eligible routed matrix-vector projection runs the already available selected positions before waiting on the dedicated transfer queue. Cold weights use one upload per unique missed Expert. Compact per-token position masks preserve original output positions; the cold positions run after a timeline semaphore wait. Remaining Expert operations are unchanged. This overlaps transfer with the first hot projection only, not the entire hot feed-forward computation.

Only packed Expert bank and native pinned staging allocations use concurrent sharing between the compute and transfer queue families. Ordinary Vulkan buffers keep their existing allocation mode. Hot reads and cold writes must cover disjoint Expert slots. Dynamic promotion layers and unsupported shapes or paths use synchronous upload.

The comparison fixes MTP target microbatch 2, dynamic residency, batch uploads, scratch reuse, router host-ID reuse, device-input batching, deferred slot maps, CUDA Spin and final graph fence reuse. Compact Expert submits, deferred same-queue uploads, CUDA component profiling and old forced-synchronization profiling remain off.

## Comparison

After building, run `fork-docs/scripts/compare-vulkan-expert-overlap.ps1 -Mode speed`, then `-Mode telemetry`. Each runs baseline and overlap cases with the same prompt and 256-token limit. Logs and recorded settings stay below `G:/AI/llama-hetero-moe-local/logs` and must not be committed.

Check output equality, Expert request counts, hit rate, promotions, uploaded bytes and writeback bytes before interpreting throughput. The transfer callback timer excludes asynchronous upload completion and cannot be compared as total transfer time. GPU compute-segment spans and host waits overlap; do not sum them.

`packed expert Vulkan upload overlap` counts accepted uploads. Native backend counters distinguish split first projections, hot/cold positions and paired submission batches. A positive transfer-pending counter shows transfer had not completed after the compute submission, not a measured duration of simultaneous GPU execution. A zero count does not prove absence of overlap. Compute timestamps alone do not measure the independent transfer queue's full execution interval.

## Validation

MSVC syntax checks and the packed cache helper fixture passed. The Vulkan translation unit compiled with project flags; the MMV, IQ3_XXS and MMVQ shaders regenerated successfully. A small MI50 F32 fixture verified asymmetric two-token masks, original output positions and staging reuse. Separate graph-mismatch and cancelled-graph cases joined outstanding transfers safely. These tests do not establish full-model IQ3_XXS correctness or a throughput improvement; the user comparison remains required.
