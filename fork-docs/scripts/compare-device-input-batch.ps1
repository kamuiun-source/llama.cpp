[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256
)

$ErrorActionPreference = 'Continue'
$root = 'G:\AI\llama-hetero-moe'
$cli = Join-Path $root 'build-win-cuda-vulkan\bin\llama-cli.exe'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$placement = 'C:\Users\kamui\.codex\visualizations\2026\11\12\01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f\flash-next-top300.txt'
$draft = Join-Path (Split-Path (Split-Path $model)) 'MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $model, $placement, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}

$stamp = Get-Date -Format yyyyMMdd-HHmmss
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "device-input-batch-$Mode-$stamp"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_TOKENS', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY', 'LLAMA_EXPERT_ROUTER_HOST_IDS',
    'LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_SCHEDULE_SPIN', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_EXPERT_VK_DEFER_SLOT_MAP', 'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }

try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    $env:LLAMA_EXPERT_PLACEMENT = $placement
    $env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
    $env:LLAMA_EXPERT_SCRATCH_REUSE = '1'
    $env:LLAMA_EXPERT_BATCH_UPLOAD = '1'
    $env:LLAMA_EXPERT_DYNAMIC_RESIDENT = '1'
    $env:LLAMA_EXPERT_BATCH_TOKENS = '2'
    $env:LLAMA_EXPERT_VK_DEFER_SLOT_MAP = '0'
    $env:LLAMA_MTP_REUSE_DRAFT_ANCHOR = '0'
    if ($Mode -eq 'telemetry') { $env:LLAMA_DECODE_TELEMETRY = '1' }

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.', '-lv', '4',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $cases = @(
        @{ Name='batch-off'; Batch='0'; Spin='1'; Router='1' }
        @{ Name='batch-on';  Batch='1'; Spin='1'; Router='1' }
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
        $env:GGML_SCHED_BATCH_DEVICE_INPUTS = $case.Batch
        $env:GGML_CUDA_SCHEDULE_SPIN = $case.Spin
        $env:LLAMA_EXPERT_ROUTER_HOST_IDS = $case.Router
        $prefix = Join-Path $runDir $case.Name
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        [ordered]@{ Name=$case.Name; Mode=$Mode; Tokens=$Tokens; Environment=$envSettings } |
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
        $spinPattern = if ($case.Spin -eq '1') {
            'CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1'
        } else {
            'CUDA device 0 scheduling: request=baseline effective_flags=0x[0-9a-f]+ effective_spin=[01]'
        }
        if (!(@(Select-String -LiteralPath $logs -Pattern $spinPattern).Count -gt 0)) {
            throw "Expected CUDA scheduling flags not found for $($case.Name). Logs: $runDir"
        }
        if ($case.Spin -eq '0' -and (@(Select-String -LiteralPath $logs -Pattern 'CUDA device 0 scheduling: request=baseline .*effective_spin=1').Count -gt 0)) {
            Write-Warning 'CUDA baseline already reports effective spin; the Spin switch may not change its scheduling.'
        }
        if ($case.Router -eq '1' -and !(@(Select-String -LiteralPath $logs -Pattern 'packed expert router: host_hits=[1-9][0-9]*\b').Count -gt 0)) {
            throw "Expected positive router host-ID reuse in $($case.Name). Logs: $runDir"
        }
        if ($case.Batch -eq '1' -and !(@(Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*\b').Count -gt 0)) {
            throw "No batched device inputs recorded. Logs: $runDir"
        }
        if ($Mode -eq 'telemetry') {
            foreach ($pattern in @('decode_telemetry: phase=target_verify .*scheduled_splits=[1-9][0-9]*\b',
                    'decode_telemetry: phase=server_step_total cuda_stream_synchronize=',
                    'decode_telemetry: phase=server_step_total vk_compute_submits=')) {
                if (!(@(Select-String -LiteralPath $logs -Pattern $pattern).Count -gt 0)) {
                    throw "Missing decode telemetry row: $pattern. Logs: $runDir"
                }
            }
        }
    }
    $texts = @{}
    foreach ($case in $cases) {
        $text = Get-Content -LiteralPath (Join-Path $runDir "$($case.Name).stdout.txt") -Raw
        $texts[$case.Name] = ($text -replace '\x1b\[[0-?]*[ -/]*[@-~]', '' -replace '\[ Prompt:.*?\]', '').Trim()
    }
    $same = $texts['batch-off'] -ceq $texts['batch-on']
    "normalized_stdout_equal=$same" | Set-Content (Join-Path $runDir 'output-comparison.txt')
    if (!$same) { Write-Warning 'Generated output differs. Check correctness before interpreting speed.' }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Write-Host "Log directory: $runDir"
}
