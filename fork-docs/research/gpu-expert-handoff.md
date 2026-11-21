# GPU Expert handoff measurements

Build after applying the scheduler patch, then run from the repository root.

```powershell
cmake --build .\build-win-cuda-vulkan --target llama-cli
powershell -NoProfile -ExecutionPolicy Bypass -File .\fork-docs\scripts\measure-gpu-expert-handoff.ps1
```

The default runs the adopted configuration once for throughput and once with tracing. Use `-Mode trace` for the diagnostic alone and `-FullHot` to request the separate full-hot Vulkan experiment for both passes. The script clears RTX cold Expert assist, keeps MTP verification width two and dynamic Expert placement, and saves results under `G:\AI\llama-hetero-moe-local\logs`.

`handoff-transfers.csv` times the existing dependency waits and synchronous host reads and writes in each direction. CPU wait includes earlier queued GPU work. Read and write durations measure elapsed API time, including the backend's staging and synchronization. A path without an observable host read has zero read timestamps.

`handoff-by-layer.csv` joins transfers and Shared Expert enqueue calls to the same graph and layer, including zero fresh misses. The trace samples graphs 17 through 20 among graph executions with more than eight splits for each scheduler. Token width and graph ordinal are recorded because the sample can contain prefill or verification. The Shared timestamp refers to the earliest identified CUDA split containing a Shared Expert anchor. The split may contain other work. A returned tensor is associated by its producing split or by its routed output name.

The ordering compares host API markers. CUDA and Vulkan device spans in the trace log use separate clocks. Logging and timing can affect scheduling in the sampled graphs, so use the speed pass for throughput.
