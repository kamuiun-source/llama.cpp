# Decode bottleneck review - 2026-11-17

HC Scale/SiLU fusion remains OFF. This review changes no product code or default options. Raw logs stay under `G:/AI/llama-hetero-moe-local/logs` and are not copied into Git.

## Evidence and comparability

- `dialogue-speed-20261117-194511`: current normal-path baseline, five responses, 46.580 s decode, 26.105 tok/s using the CLI timed-token convention (generated tokens minus one per response).
- `dialogue-speed-20261117-194833`: HC Scale/SiLU ON, 48.145 s, 25.257 tok/s. Same token counts and draft acceptance; one run per configuration is insufficient to isolate thermal/order effects. Keep OFF.
- `dialogue-telemetry-20261117-182307`: graphs enabled, GPU timestamp and transfer counters enabled. Same operational environment flags as the speed baseline except instrumentation. Different run/time and potentially source revision; do not directly combine its durations with the speed run.
- `dialogue-components-20261117-183225` and `190812`: graphs disabled and per-operation synchronization enabled. Use for structural attribution, not normal-path savings. Response 4's HC/recurrent/attention durations are consistent between these two runs.
- `dialogue-components-20261117-193322`: fusion active, 67,008 fewer dispatches in target verification. This did not establish a normal-path speedup.
- `182550` has incomplete response boundaries; exclude from timing totals. `191445` did not activate fusion; do not label it ON evidence.
- Earlier `gpu-expert-handoff-20261117-142822` records Shared Expert enqueue after return readback. It predates the adopted Shared Expert overlap and is historical evidence, not the current critical path.

## Normal-path telemetry: five-response aggregate

Response decode time is 54.764 s in the instrumented run. Target verification accounts for 49.057 s over 698 calls, draft generation 2.608 s, draft catchup 1.435 s and target decode 0.139 s. These named scopes do not fully cover the response timer; the remaining approximately 1.53 s must not be assigned to a particular operation without evidence. Prefill is separate (target prefill scope 19.501 s).

Within target verification:

| Measurement | Total |
| --- | ---: |
| CUDA stream GPU intervals | 14.525 s |
| Vulkan compute GPU segments | 13.940 s |
| CUDA stream synchronization CPU time | 13.040 s |
| Vulkan backend-sync fence wait | 14.474 s |
| Vulkan sync-read fence wait | 2.733 s |
| Vulkan sync-write fence wait | 1.869 s |
| Vulkan all-cause submit CPU time | 1.333 s |
| Vulkan command recording CPU time | 0.752 s |

GPU intervals may include stalls. CPU waits overlap GPU execution and transfer. Do not add these rows, subtract them from wall time as disjoint components, or treat waits as removable overhead. Sync-read/write totals include multiple causes and cannot be called Expert-only transfer time.

There are 68,404 scheduler splits, exactly 98 per verification call on average. Expert-layer steps total 33,504 (48 per call). This remains a frequent cross-backend handoff structure, but raw call counts alone do not prove unnecessary synchronization.

Expert requests: 670,080; misses: 26,040; hit rate: 96.11%. Nevertheless 42.29% of layer steps have at least one miss. Expert uploads total 45.92 GiB and writebacks 11.82 GiB. Transfer is still material; the high per-request hit rate does not mean nearly all layer steps avoid transfer. Already-overlapped transfer bytes cannot be converted directly into recoverable wall time.

## Component attribution

In the OFF components run `190812`, target verification reports other 16.203 s, HC 9.379 s, recurrent 8.189 s, shared expert 5.615 s, attention 4.781 s. HC matrix multiplication is 7.229 s. Other is heterogeneous: fused groups 2.414 s, matrix multiplication 2.065 s, unary 2.036 s, add 1.633 s, scale 1.603 s, get_rows 1.447 s, plus other operations. It is not one optimizable kernel. HC Scale/SiLU appears in other because classification is name based.

## HC code review

`src/models/qwen4exp.cpp:348` builds HC normalization, down projection, scale/SiLU, up projection and gated mixing, plus injection. The local GGUF header inspection found separate Q8_0 down/up weights in layer 0: [10240,320] and [320,10240], with F32 injection [10240,4]. These are GGML weight dimensions, not row-major notation. This review does not claim every tensor in every layer was inspected.

`ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat`) selects MMVQ for supported quantized small batches before MMQ. `mmvq.cu:317` (`ggml_cuda_should_use_mmvq`) sends Q8_0 with two input columns on Ada to that route under the normal conditions. This is source-derived routing, not an observed per-node kernel trace.

`mmvq.cu:1558` quantizes the F32 activation to Q8_1 before the matrix-vector kernel. The two-column implementation already handles both tokens in one kernel, so it is incorrect to describe the current route as two independent matrix calls. Its inner loop invokes the quantized dot product per column; hardware/compiler reuse must be measured rather than assumed absent.

A concrete candidate is the up projection with reduction dimension K=320. `should_use_small_k` at `mmvq.cu:1143` and launch selection around 1218 provide a small-K path for one column. The two-column case at 1244 uses the standard launch without that selection. The generic parameter table used by Ada chooses four warps and two output rows per block for two columns. This suggests investigating thread utilization/reduction overhead for K=320, not changing all Q8_0 kernels globally. Simply enabling the small_k template flag is not sufficient: row/warp mapping must also support two columns.

## Decision and next work

1. First bounded experiment: Q8_0 HC up projection at two tokens (K=320, output width=10240). Compare current MMVQ against a narrowly scoped small-K mapping, preserving input Q8_1 quantization and arithmetic behavior as far as possible. Validate one/two-token output, fallback, graph replay and performance within at most four test categories. Measure small graph GPU time with graph replay, then the existing five-prompt normal benchmark. Do not implement a broad HC rewrite or promise a speedup.
2. Transfer tail: inspect the remaining read/write waits around dynamic promotion and activation return, with existing full-hot and promotion overlap retained. Optimize only a proven exposed dependency; preserve non-duplicated RAM/VRAM residency and writeback semantics. Counter data establishes opportunity to investigate, not recoverable seconds.
3. Recurrent path and unnamed other matrix/fused groups: map heavy nodes to actual kernels/shapes. Recurrent is comparable in instrumented size to HC matrix work, but current evidence does not isolate its expensive sub-operation. Avoid speculative elementwise fusion solely based on synchronized timings.
4. Draft/catchup: approximately 4.04 s in this telemetry run, smaller than target verification. Longer lookahead remains a separate later experiment; it changes acceptance and rollback cost.

Do not prioritize generic deletion of CUDA/Vulkan synchronization, another Scale/SiLU fusion, or command-buffer recording alone. The next implementation candidate is HC small-K two-token scheduling because it combines measurable component weight with a concrete code asymmetry, not because HC has been proven the largest normal-path bottleneck.
