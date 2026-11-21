# Shared Expert overlap

Apply the patch and build `llama-cli`, then run the comparison from the repository root.

```powershell
cmake --build .\build-win-cuda-vulkan --target llama-cli
powershell -NoProfile -ExecutionPolicy Bypass -File .\fork-docs\scripts\compare-shared-expert-overlap.ps1
```

The comparison runs the current MTP2 configuration with dynamic Expert placement and Vulkan upload and promotion overlap in both cases. RTX cold Expert assist stays disabled. Use `-FullHot` to enable the separate full-hot Vulkan experiment in both cases, or `-Mode telemetry` for bounded handoff traces and device-local timestamps. Logs go under `G:\AI\llama-hetero-moe-local\logs`.

When enabled, the scheduler submits a CUDA Shared Expert prefix after the corresponding MI50 Routed Expert split has been submitted and before the existing Vulkan-to-CUDA return copy. That copy still waits on its existing dependencies before the CUDA merge consumes both results. The prefix must feed the named same-layer merge and its inputs and outputs must have disjoint CUDA memory ranges from the routed return buffer. The scheduler uses the original split when the topology or allocation fails these checks. The experimental CUDA multistream graph optimizer, RTX assist, profiling and evaluation callbacks also use the original split.

The `prefix_submits` counter verifies scheduling activation. The host handoff trace can show submission order, while elapsed CUDA and Vulkan event measurements use their respective device clocks. Use the speed pass for throughput and check the script's output and MTP acceptance comparisons before interpreting it. Full-model numeric equivalence and on-device overlap remain to be measured.
