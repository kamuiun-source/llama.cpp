[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256,
    [string]$BuildDirectory = 'G:\AI\build-prefill32'
)

$ErrorActionPreference = 'Continue'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$cli = Join-Path $BuildDirectory 'bin\llama-cli.exe'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$placement = 'C:\Users\kamui\.codex\visualizations\2026\11\12\01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f\flash-next-top300.txt'
$draft = Join-Path (Split-Path (Split-Path $model)) 'MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $model, $placement, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}

$stamp = Get-Date -Format yyyyMMdd-HHmmss
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "vulkan-submit-$Mode-$stamp"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_BATCH_TOKENS',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY',
    'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_COMPACT_EXPERT_SUBMITS')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }

try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    $env:LLAMA_EXPERT_PLACEMENT = $placement
    $env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
    $env:LLAMA_EXPERT_BATCH_TOKENS = '2'
    $env:LLAMA_MTP_REUSE_DRAFT_ANCHOR = '0'
    if ($Mode -eq 'telemetry') {
        $env:LLAMA_DECODE_TELEMETRY = '1'
        $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
    }

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.', '-lv', '4',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $cases = @(
        @{ Name='baseline'; Compact=$false }
        @{ Name='compact'; Compact=$true }
    )

    $baseArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    if ($LASTEXITCODE -ne 0) { throw "Could not record HEAD. Logs: $runDir" }
    git -C $root status --short | Set-Content (Join-Path $runDir 'working-status.txt') -Encoding UTF8
    git -C $root diff HEAD --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt') -Encoding UTF8
    $diffPath = Join-Path $runDir 'working-diff.patch'
    git -C $root diff --no-ext-diff --no-color --output="$diffPath" HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not record working diff. Logs: $runDir" }

    foreach ($case in $cases) {
        $env:GGML_VK_COMPACT_EXPERT_SUBMITS = if ($case.Compact) { '1' } else { $null }
        $prefix = Join-Path $runDir $case.Name
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        [ordered]@{ Name=$case.Name; Mode=$Mode; Tokens=$Tokens; Compact=$case.Compact; Environment=$envSettings } |
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
        if (!(@(Select-String -LiteralPath $logs -Pattern 'draft acceptance = .* / +[1-9][0-9]* generated').Count -gt 0)) {
            throw "No positive draft count found. Logs: $runDir"
        }

        foreach ($pattern in @('CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1',
                'packed expert router: host_hits=[1-9][0-9]*\b')) {
            if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                throw "Missing adopted feature marker: $pattern. Logs: $runDir"
            }
        }
        $compactApplied = Select-String -LiteralPath $logs -SimpleMatch 'compact expert submit applied' -Quiet
        if ($compactApplied -ne $case.Compact) {
            throw "Unexpected Vulkan submit feature activation in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*\b').Count -eq 0) {
            throw "No batched device inputs recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert slot map updates: deferred=[1-9][0-9]*\b').Count -eq 0) {
            throw "No deferred slot-map updates recorded in $($case.Name). Logs: $runDir"
        }
        Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=', 'packed expert slot map updates:', 'final graph fence reused', 'compact expert submit applied' |
            ForEach-Object Line | Select-Object -Unique | Set-Content "$prefix.activation.txt"
        if ($Mode -eq 'telemetry') {
            foreach ($pattern in @('decode_telemetry: phase=target_verify .*scheduled_splits=[1-9][0-9]*\b',
                    'decode_telemetry: phase=server_step_total vk_compute_submits=',
                    'decode_telemetry: phase=target_verify vk_cause=',
                    'decode_telemetry: phase=target_verify vk_gpu_compute_segment_samples=[1-9][0-9]*\b')) {
                if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                    throw "Missing Vulkan telemetry row: $pattern. Logs: $runDir"
                }
            }
        }
    }

    $texts = @{}
    foreach ($case in $cases) {
        $text = Get-Content -LiteralPath (Join-Path $runDir "$($case.Name).stdout.txt") -Raw
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Missing stdout for $($case.Name). Logs: $runDir" }
        $texts[$case.Name] = ($text -replace '\x1b\[[0-?]*[ -/]*[@-~]', '' -replace '\[ Prompt:.*?\]', '').Trim()
    }
    $allEqual = $true
    $rows = foreach ($case in $cases | Where-Object { $_.Name -ne 'baseline' }) {
        $same = $texts[$case.Name] -ceq $texts['baseline']
        $allEqual = $allEqual -and $same
        "$($case.Name) normalized_stdout_equal_baseline=$same"
    }
    @($rows) + "all_normalized_stdout_equal=$allEqual" | Set-Content (Join-Path $runDir 'output-comparison.txt')
    if (!$allEqual) { throw "Normalized stdout differs from baseline. Logs: $runDir" }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Write-Host "Log directory: $runDir"
}
