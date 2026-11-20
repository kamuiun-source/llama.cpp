[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [switch]$ReverseOrder,
    [string]$LogDirectory,
    [ValidateNotNullOrEmpty()]
    [string]$BuildDirectory = 'build-win-cuda-vulkan',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256,
    [ValidateRange(1, 19)]
    [int]$RtxSlots = 1,
    [ValidateRange(1, 16384)]
    [int]$RtxVramMiB = 128,
    [ValidateRange(512, 262144)]
    [int]$Context = 2048,
    [ValidateNotNullOrEmpty()]
    [string]$Prompt = 'Explain how a bicycle works in detail.'
)

$ErrorActionPreference = 'Continue'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (![IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory = Join-Path $root $BuildDirectory }
$cli = Join-Path $BuildDirectory 'bin\llama-cli.exe'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$placement = 'C:\Users\kamui\.codex\visualizations\2026\11\12\01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f\flash-next-top300.txt'
$draft = Join-Path (Split-Path (Split-Path $model)) 'MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $model, $placement, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}

$stamp = Get-Date -Format yyyyMMdd-HHmmss
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "rtx-cold-expert-$Mode-$stamp"
if ($LogDirectory) { $runDir = $LogDirectory }
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_TOKENS', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY', 'LLAMA_EXPERT_ROUTER_HOST_IDS',
    'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_SCHEDULE_SPIN', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_EXPERT_VK_DEFER_SLOT_MAP', 'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS', 'GGML_SCHED_OVERLAP_SHARED_EXPERT', 'GGML_SCHED_TRACE_EXPERT_HANDOFF', 'LLAMA_EXPERT_VK_PAIR_IQ4_NL',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_REUSE_GRAPH_FENCE', 'GGML_VK_COMPACT_EXPERT_SUBMITS', 'LLAMA_EXPERT_VK_DEFER_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_PROMOTION', 'LLAMA_EXPERT_VK_FULL_HOT_OVERLAP', 'LLAMA_EXPERT_RTX_COLD_SLOTS', 'LLAMA_EXPERT_RTX_COLD_VRAM_MIB', 'LLAMA_EXPERT_MISS_HISTOGRAM', 'GGML_CUDA_GPU_TIMESTAMPS_DEVICE', 'GGML_CUDA_PROFILE_COMPONENTS')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }

try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') }
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    $env:LLAMA_EXPERT_PLACEMENT = $placement
    $env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
    $env:LLAMA_EXPERT_SCRATCH_REUSE = '1'
    $env:LLAMA_EXPERT_BATCH_UPLOAD = '1'
    $env:LLAMA_EXPERT_DYNAMIC_RESIDENT = '1'
    $env:LLAMA_EXPERT_BATCH_TOKENS = '2'
    $env:LLAMA_EXPERT_VK_DEFER_SLOT_MAP = '1'
    $env:LLAMA_MTP_REUSE_DRAFT_ANCHOR = '0'
    $env:GGML_SCHED_BATCH_DEVICE_INPUTS = '1'
    $env:GGML_CUDA_SCHEDULE_SPIN = '1'
    $env:LLAMA_EXPERT_ROUTER_HOST_IDS = '1'
    if ($Mode -eq 'telemetry') {
        $env:LLAMA_DECODE_TELEMETRY = '1'
        $env:LLAMA_EXPERT_MISS_HISTOGRAM = '1'
        $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
    }

    $env:GGML_VK_REUSE_GRAPH_FENCE = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_UPLOAD = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_PROMOTION = '1'
    $env:LLAMA_EXPERT_VK_FULL_HOT_OVERLAP = '1'
    $env:GGML_SCHED_OVERLAP_SHARED_EXPERT = '1'

    $env:LLAMA_EXPERT_RTX_COLD_VRAM_MIB = "$RtxVramMiB"
    $promptPath = Join-Path $runDir 'prompt.txt'
    [IO.File]::WriteAllText($promptPath, $Prompt, [Text.UTF8Encoding]::new($false))

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', "$Context", '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-f', $promptPath,
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $cases = @(
        @{ Name='baseline'; Rtx=$false }
        @{ Name='rtx-cold'; Rtx=$true }
    )
    if ($ReverseOrder) { [array]::Reverse($cases) }
    $cases | ForEach-Object { $_.Name } | Set-Content (Join-Path $runDir 'run-order.txt')

    $baseArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    if ($LASTEXITCODE -ne 0) { throw "Could not record HEAD. Logs: $runDir" }
    git -C $root status --short | Set-Content (Join-Path $runDir 'working-status.txt') -Encoding UTF8
    git -C $root diff HEAD --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt') -Encoding UTF8
    $diffPath = Join-Path $runDir 'working-diff.patch'
    git -C $root diff --no-ext-diff --no-color --output="$diffPath" HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not record working diff. Logs: $runDir" }

    $acceptance = @{}
    foreach ($case in $cases) {
        [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_RTX_COLD_SLOTS', $(if ($case.Rtx) { "$RtxSlots" } else { [NullString]::Value }), 'Process')
        $prefix = Join-Path $runDir $case.Name
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        $sharedPolicy = if ($case.Rtx) { 'disabled internally by RTX cold slots' } else { 'enabled' }
        $fullHotPolicy = if ($case.Rtx) { 'first-projection fallback on RTX-excluded layers' } else { 'enabled on supported FFNs' }
        [ordered]@{ Name=$case.Name; Mode=$Mode; Tokens=$Tokens; Context=$Context; RtxSlots=$(if ($case.Rtx) { $RtxSlots } else { 0 });
            ReverseOrder=[bool]$ReverseOrder; RtxMinUniqueMisses=2; SharedPolicy=$sharedPolicy; FullHotPolicy=$fullHotPolicy; Environment=$envSettings } |
            ConvertTo-Json -Depth 3 | Set-Content "$prefix.settings.json" -Encoding UTF8
        $runArgs = @('-lv', '4', '--log-file', "$prefix.log") + $baseArgs
        $runArgs | ConvertTo-Json | Set-Content "$prefix.arguments.json" -Encoding UTF8
        Write-Host "Running $Mode/$($case.Name)"
        & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
        $code = $LASTEXITCODE
        "exit_code=$code" | Set-Content "$prefix.exit.txt"
        if ($code -ne 0) { throw "Failed: $($case.Name). Logs: $runDir" }

        $logs = @("$prefix.log", "$prefix.stderr.txt")
        if (!(@(Select-String -LiteralPath $logs -SimpleMatch 'packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1').Count -gt 0)) {
            throw "Expected target batch width 2 and RS1. Logs: $runDir"
        }
        $draftLine = Select-String -LiteralPath "$prefix.log" -Pattern 'draft acceptance = [0-9.]+ \(\s*([0-9]+) accepted /\s*([1-9][0-9]*) generated\)' | Select-Object -Last 1
        if (!$draftLine) { throw "No positive draft count found. Logs: $runDir" }
        $draftCounts = $draftLine.Matches[0].Groups
        $accepted = [int]$draftCounts[1].Value
        $generated = [int]$draftCounts[2].Value
        $acceptance[$case.Name] = @{ Accepted=$accepted; Generated=$generated; Rate=($accepted / $generated) }

        foreach ($pattern in @('CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1',
                'packed expert router: host_hits=[1-9][0-9]*\b')) {
            if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                throw "Missing adopted feature marker: $pattern. Logs: $runDir"
            }
        }
        if (@(Select-String -LiteralPath $logs -SimpleMatch 'final graph fence reused').Count -eq 0) {
            throw "No final graph fence reuse recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -SimpleMatch 'compact expert submit applied').Count -gt 0) {
            throw "Compact expert submit unexpectedly applied in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert Vulkan upload overlap: submitted=[1-9][0-9]* bytes=[1-9][0-9]*').Count -eq 0) {
            throw "Existing upload overlap did not activate in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'expert overlap transfer_submits=[1-9][0-9]* .*paired_submits=[1-9][0-9]*').Count -eq 0) {
            throw "No hot/cold paired dispatch recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert Vulkan promotion overlap: committed=[1-9][0-9]*').Count -eq 0) {
            throw "Existing Vulkan promotion overlap did not activate in $($case.Name). Logs: $runDir"
        }
        $sharedApplied = @(Select-String -LiteralPath $logs -Pattern 'shared expert overlap: prefix_submits=[1-9][0-9]* tokens=[1-9][0-9]*').Count -gt 0
        if ($sharedApplied -eq $case.Rtx) { throw "Unexpected Shared Expert overlap activation in $($case.Name). Logs: $runDir" }
        $fullHot = Select-String -LiteralPath $logs -Pattern 'Vulkan1 expert full hot overlap split_ffns=([0-9]+) .*fallback_ffns=([0-9]+)' | Select-Object -Last 1
        if (!$fullHot -or (!$case.Rtx -and [int]$fullHot.Matches[0].Groups[1].Value -eq 0)) {
            throw "Missing expected FullHot activation in $($case.Name). Logs: $runDir"
        }
        $cacheRtx = Select-String -LiteralPath "$prefix.log" -Pattern 'packed expert RTX offload: eligible_steps=[1-9][0-9]* submitted_steps=([1-9][0-9]*) fallback_steps=[0-9]+ unique_experts=[1-9][0-9]* ranks=[1-9][0-9]* uploaded_bytes=[1-9][0-9]*' | Select-Object -Last 1
        $nativeRtx = Select-String -LiteralPath "$prefix.log" -Pattern 'packed expert CUDA sidecar: submissions=([1-9][0-9]*) graph_launches=([1-9][0-9]*) device_bytes=[1-9][0-9]*' | Select-Object -Last 1
        if ($case.Rtx) {
            if (!$cacheRtx -or !$nativeRtx) {
                throw "No RTX cold Expert execution recorded in $($case.Name). Logs: $runDir"
            }
            $cacheSteps = $cacheRtx.Matches[0].Groups[1].Value
            $nativeSteps = $nativeRtx.Matches[0].Groups
            if ($cacheSteps -ne $nativeSteps[1].Value -or $cacheSteps -ne $nativeSteps[2].Value) {
                throw "Cache and CUDA sidecar counts differ. Logs: $runDir"
            }
        } elseif ($cacheRtx -or $nativeRtx) {
            throw "RTX cold Expert execution unexpectedly occurred in baseline. Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*\b').Count -eq 0) {
            throw "No batched device inputs recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert slot map updates: deferred=[1-9][0-9]*\b').Count -eq 0) {
            throw "No deferred slot-map updates recorded in $($case.Name). Logs: $runDir"
        }
        @("shared_policy=$sharedPolicy", "full_hot_policy=$fullHotPolicy") | Set-Content "$prefix.activation.txt"
        Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=', 'packed expert slot map updates:', 'final graph fence reused', 'packed expert Vulkan upload overlap:', 'expert overlap transfer_submits=', 'packed expert Vulkan promotion overlap:', 'expert promotion overlap submits=', 'expert full hot overlap split_ffns=', 'shared expert overlap: prefix_submits=', 'packed expert RTX offload:', 'packed expert CUDA sidecar:' |
            ForEach-Object Line | Select-Object -Unique | Add-Content "$prefix.activation.txt"
        if ($Mode -eq 'telemetry') {
            foreach ($pattern in @('decode_telemetry: phase=target_verify .*scheduled_splits=[1-9][0-9]*\b',
                    'decode_telemetry: phase=server_step_total vk_compute_submits=',
                    'decode_telemetry: phase=target_verify vk_cause=',
                    'decode_telemetry: phase=target_verify expert_miss_histogram steps=[1-9][0-9]*',
                    'decode_telemetry: phase=target_verify vk_gpu_compute_segment_samples=[1-9][0-9]*\b')) {
                if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                    throw "Missing Vulkan telemetry row: $pattern. Logs: $runDir"
                }
            }
        }
    }

    foreach ($case in $cases) {
        $prefix = Join-Path $runDir $case.Name
        $patterns = @('packed expert slots:', 'packed expert requests:', 'packed expert dynamic residency: promotions=',
            'packed expert RTX offload:', 'packed expert CUDA sidecar:', 'expert_miss_histogram', 'draft acceptance =', '[|] +eval time =')
        Select-String -LiteralPath "$prefix.log" -Pattern $patterns | ForEach-Object Line | Set-Content "$prefix.expert-summary.txt"
    }
    $texts = @{}
    foreach ($case in $cases) {
        $text = Get-Content -LiteralPath (Join-Path $runDir "$($case.Name).stdout.txt") -Raw
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Missing stdout for $($case.Name). Logs: $runDir" }
        $clean = $text -replace '\x1b\[[0-?]*[ -/]*[@-~]', ''
        $response = [regex]::Match($clean, '(?ms)^> [^\r\n]*\r?\n(?<body>.*?)^\s*\[ Prompt:')
        if (!$response.Success -or [string]::IsNullOrWhiteSpace($response.Groups['body'].Value)) {
            throw "Cannot extract generated response for $($case.Name). Logs: $runDir"
        }
        $texts[$case.Name] = $response.Groups['body'].Value.Trim()
        $texts[$case.Name] | Set-Content (Join-Path $runDir "$($case.Name).response.txt") -Encoding UTF8
    }
    $same = $texts['rtx-cold'] -ceq $texts['baseline']
    $baseline = $acceptance['baseline']
    $rtx = $acceptance['rtx-cold']
    $zeroAcceptance = $rtx.Accepted -eq 0 -and $rtx.Generated -ge 16
    $qualityReview = $baseline.Accepted -eq 0 -or $rtx.Accepted -eq 0 -or $rtx.Rate -lt ($baseline.Rate / 2)
    $repetition = ($texts['rtx-cold'] -replace '\s', '') -match '^([0-9])\1{15,}$'
    @(
        "rtx-cold generated_response_equal_baseline=$same"
        "baseline_mtp_accepted=$($baseline.Accepted) baseline_mtp_generated=$($baseline.Generated) baseline_mtp_rate=$($baseline.Rate)"
        "rtx_mtp_accepted=$($rtx.Accepted) rtx_mtp_generated=$($rtx.Generated) rtx_mtp_rate=$($rtx.Rate)"
        "quality_review=$qualityReview mtp_zero_acceptance_failure=$zeroAcceptance repeated_digit_response=$repetition"
        'The speed delta includes Shared Expert overlap being disabled and FullHot fallback on RTX-excluded layers.'
    ) | Set-Content (Join-Path $runDir 'output-comparison.txt')
    if ($zeroAcceptance -or $repetition) { throw "RTX quality check failed. Inspect output-comparison.txt and responses. Logs: $runDir" }
    if ($qualityReview) { Write-Warning "MTP acceptance requires review. See output-comparison.txt. Logs: $runDir" }
    if (!$same) { Write-Warning "Generated text differs from baseline. Inspect both responses before interpreting speed. Logs: $runDir" }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { if ($null -eq $saved[$key]) { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') } else { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') } }
    Write-Host "Log directory: $runDir"
}
