# RTX cold Expert assist: sizing and first measurement

Status: experimental opt-in implementation under review. Use LLAMA_EXPERT_RTX_COLD_SLOTS=1 for the first comparison; an unset value preserves the adopted graph. Do not enable full-hot overlap in the same comparison.

Keep the accepted MI50 upload and promotion overlap enabled. The baseline log `vulkan-promotion-overlap-telemetry-20261117-123704` has 143990 requests, 7477 missed request lanes and 6875 unique uploads over 7248 packed callbacks, including startup/prefill. There are 3703 callbacks with uploads, averaging 1.8566 unique uploads each. These totals cannot recover the number of callbacks with three or more unique misses. MTP can select the same Expert in both tokens; count that weight transfer only once.

The maximum Expert tuple in the current GGUF is 2662400 bytes (2.539 MiB), summed across its gate/up/down planes. A reusable pool of 4/8/16/20 slots requires approximately 10.16/20.31/40.63/50.78 MiB for quantized weights only. Activation, output scatter, CUDA workspaces, padding, shape/type variants and an optional second bank cost extra. This is a design estimate, not a measured RTX allocation. The pool should be shared across sequential layers, not allocated independently for all 48 layers.

Context, MTP and established CUDA allocations take priority. Configure a total Expert-assist VRAM cap and an independent maximum number of selected cold Experts per invocation. Budget must account for execution buffers and a reserve after model/context warmup; do not reduce the requested context automatically. On insufficient space, reduce assist capacity or retain the MI50 path. More available slots do not establish a throughput improvement.

The prospective execution point is after required CUDA-to-Vulkan input synchronization and before Vulkan Expert compute. Use a pinned upload source and a separate CUDA execution path, then join before the original router weighting/reduction. Preserve original token/Expert positions and verify numerical behavior of the CUDA quantized kernels. Avoid launching RTX work before an ordinary scheduler synchronization that would immediately wait for it. If an Expert is not uploaded to MI50, later scratch hit behavior changes and must be measured explicitly.

## Initial measurement

`LLAMA_EXPERT_MISS_HISTOGRAM=1` with `LLAMA_DECODE_TELEMETRY=1` enables cumulative unique upload counts sampled by existing phase boundaries. `target_verify` excludes startup and prefill and reports counts for 0, 1, 2, 3, 4, 5+ misses, the maximum, upload bytes, and bytes in callbacks with three or more misses. Inclusive server-step counters must not be added to child phases.

Run `fork-docs/scripts/measure-expert-misses.ps1` after rebuilding. It runs once with all adopted improvements enabled and keeps raw logs outside Git. The histogram does not change placement or execution. Cache and server translation units passed MSVC syntax checks; full-model measurement remains for the user.
## Experimental RTX execution

At least two unique fresh misses are required per layer and verification microbatch. Select up to min(configured slots, fresh misses minus one) in deterministic router order. Selected RTX Experts keep their RAM originals and are not also uploaded into MI50 scratch. Thus future cache hits and total transfer bytes may change; comparison logs retain both device counters rather than requiring identical physical uploads.

Use pinned host staging and reusable CUDA workspaces keyed by weight layout/type and selected count. No long-lived RTX Expert weight cache is used. Both token lanes can reuse one transferred tuple. CUDA computes compact contributions, and original lane positions are restored before router weighting and reduction. CUDA and Vulkan numerical paths can differ; output differences must be reviewed before interpreting speed as equivalent work.

The default LLAMA_EXPERT_RTX_COLD_VRAM_MIB=128 cap covers explicit side buffers (including its graph tensors), not all internal CUDA allocator pools or graph-capture overhead. Lazy workspace allocation requires 512 MiB reported free headroom in addition to the allocation; it does not shrink the requested context. Unsupported layouts, insufficient buffer budget or busy pinned staging retain MI50 processing. Enabling the experiment adds a contribution merge even when a particular invocation falls back, so this overhead is included in the ON comparison.

Use compare-rtx-cold-experts.ps1 with -Mode speed (then telemetry), -RtxSlots 1, optional -RtxVramMiB 128, -Context 2048 and -Prompt. Scripts disable the separate full-hot experiment and store raw logs outside Git. Native CUDA submissions/graph-launch counters prove work was enqueued, not measured overlap duration on the devices.
## Targeted validation

The edited translation units compiled with project build flags and affected quantized Vulkan shaders regenerated. A small 4070 Ti plus MI50 fixture used quantized Flash-Next Expert weights with one- and two-slot dispatch, duplicate token lanes, stale-output fallback, fresh graph construction after reserve and zero-budget disabling. RTX-assigned ranks matched a separate CUDA reference exactly; untouched ranks matched the Vulkan reference bit-for-bit, and skipped Vulkan ranks were zero. CUDA versus Vulkan on the assigned ranks differed by approximately 1.40-1.70% relative L2 in these fixtures. This is not a model-quality assessment and does not establish generated text equivalence. Full-model outputs, acceptance rates and speed require the user A/B test.
