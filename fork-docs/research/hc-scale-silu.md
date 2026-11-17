# HC Scale/SiLU fusion and operation telemetry

## Scope

`GGML_CUDA_HC_SCALE_SILU_FUSE=1` enables an optional CUDA fusion for contiguous F32 `SCALE -> SILU` after an HC weight matrix multiplication. It supports one or two tokens and up to 4096 intermediate values per token. The original matrix multiplication and its quantization path remain unchanged. The older `GGML_CUDA_HC_MIX_FUSE` stays disabled.

The graph matcher checks consumers with the existing fusion helper. It permits disjoint input/output memory or exact in-place reuse for this elementwise kernel; partial overlap is rejected. Unsupported shapes and concurrent-stream graph regions retain the original path. This optimization removes one kernel launch and one intermediate round trip, not the matrix multiplication work. Speedup is unmeasured.

## Measurement

`start-dialogue-measurement.ps1 -Mode components` emits per-response operation details for the existing `hc` and `other` component classifications. Add `-HcScaleSilu` to enable the new fusion. The launch script checks for its activation marker after exit.

Rows contain `cuda_op`, dispatch/sample counts, GPU stream elapsed time, CPU submit/wait time, and dropped samples. Multiple fused operations use `FUSED_GROUP`; timings cannot be assigned to the individual operations within that group. Existing component classification is unchanged: unnamed HC elementwise operations can appear in `other`.

The CSV parser preserves aggregate rows and puts details under `cuda_component:hc/op:...` or `cuda_component:other/op:...`. These are nested breakdowns, not extra time to add to aggregates. GPU and CPU waits also overlap.

Components mode disables CUDA Graphs and introduces synchronization. Use it to identify operations; use separate `-Mode speed` OFF/ON runs with the same five prompts to judge throughput. Shared Expert overlap and full-hot transfer overlap remain enabled; RTX cold Expert sharing remains disabled by the saved baseline settings. Logs remain outside Git.

## Validation

The two modified CUDA translation units compile, and the server passes the MSVC syntax check. PowerShell parsing and a synthetic response with aggregate and operation-detail rows pass. A standalone Q8_0 graph using the graph allocator reproduced exact input/output buffer reuse. After allowing that reuse, one- and two-token runs activated fusion and produced bitwise-identical OFF/ON output over three graph executions. Full-model numerical equivalence, activation and normal-path performance still await user-run comparison. No integration into the main checkout or commit was performed for this change.
