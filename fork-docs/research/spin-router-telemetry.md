# Spin, Router IDs and decode telemetry

Status: integrated for user-run validation, 2026-11-16. No full build or model run performed during integration. The benchmark base is the experimental MTP batch-2 version; its output difference from serial MTP remains unresolved.

## Controls

- `GGML_CUDA_SCHEDULE_SPIN=0`: existing scheduling policy, including upstream CC 12.1 special case. `1`: request Spin and log effective flags. This does not eliminate driver scheduling or GPU compute time.
- `LLAMA_EXPERT_ROUTER_HOST_IDS=1`: retain the actual CUDA-to-host-to-Vulkan fallback bytes for small contiguous I32 inputs. The cache reads them only within the same split's normal callback with exact destination and size match. Other cases retain Vulkan readback. No mapped-memory polling is used.
- `LLAMA_DECODE_TELEMETRY=1`: count existing API calls and CPU wall times without adding GPU synchronization.
- Expert dynamic residency and writeback are unchanged.

## Comparison

`fork-docs/scripts/compare-spin-router.ps1` provides speed, telemetry and separate GPU diagnostic modes. Speed compares baseline, Spin, Router reuse and both using the same batched MTP settings. Run speed first, then telemetry to obtain phase counts. Full GPU diagnostics are optional and perturb execution.

Check exit status, generated text, MTP acceptance and Expert cache behavior along with tok/s. A change in generated text invalidates a simple same-work speed comparison. Router `host_hits` must be nonzero to claim that B was exercised; `elided_readback_bytes` counts omitted API requests, not measured PCIe traffic.

## Interpretation

The telemetry separates target prefill, target decode, target verification, mixed batches, draft and draft catchup. Divide scheduled_splits by graphs for splits per scheduled graph, and API counts by phase calls for calls per logical processing step. A verification step can contain multiple tokens; it is not a generated token. `server_step_total` is inclusive: do not add it to the other rows. Its graph/split counts cover the target only, whereas native API counts include draft, checkpoint and sampling activity on the same host thread. Loading, warmup and teardown are excluded from phase deltas.

Vulkan compute/transfer submit counts refer to queue roles, not shader/copy contents. CPU record time covers graph-node recording and barriers, excluding nested measured submit/wait calls; it does not cover all graph planning, transfer recording, staging copies or mutex waits. Fence wait and polling time can include outstanding GPU work. It is not driver overhead alone. CUDA synchronization counts report actual instrumented runtime API calls, not GPU-internal graph events or driver-internal operations.

The existing `GGML_VK_PERF_LOGGER` uses GPU timestamps but adds submit/fence/query waits. Its per-operation intervals include GPU timeline effects such as barriers and idle time. They are not pure arithmetic time and must not be subtracted from or added to ordinary-run CPU timings. The concurrent logger is not used because its dependency grouping can omit tail spans.

## Earlier overlap audit

In `pipeline-ab-20261116-102041`, baseline had 7287 nodes / 98 splits and 9.53 tok/s; overlap had 7863 nodes / 146 splits and 8.13 tok/s (waited) or 8.74 tok/s (early). All had 95941 permanent hits, 20122 scratch hits and 18117 uploads. Native transfer staging indicates distinct queue handles; historical logs do not prove different queue families or concurrent device execution.

The resident and cold graph branches both process the full selection list, with inactive entries mapped to zero weights. This adds work and splits. Upload publication also waits before exposing the cold slot map. Earlier `expert-overlap-20261116-095049` additionally disabled scratch reuse in the overlap case, so its 10.01 vs 4.90 tok/s was not an overlap-only comparison.

Sources: `ggml/src/ggml-backend.cpp` scheduler input copying and split loop; `src/llama-moe-cache.cpp` cache copy and upload; `ggml/src/ggml-cuda/ggml-cuda.cu` initialization and synchronization; `ggml/src/ggml-vulkan/ggml-vulkan.cpp` queue wrappers, graph construction and performance logger; `tools/server/server-context.cpp` phase scopes; `tools/server/server-queue.cpp` yield_to_queue executes GPU work on the caller thread.
