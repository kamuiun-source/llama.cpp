[CmdletBinding()]
param(
    [ValidateSet('both', 'cpu', 'gpu')]
    [string]$Mode = 'both',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256
)

function Set-MeasureEnvironment($Name, $Value) {
    if ($null -eq $Value) {
        [Environment]::SetEnvironmentVariable($Name, [NullString]::Value, 'Process')
    } else {
        [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    }
}

function Read-CudaTelemetry([string[]]$Paths) {
    $phases = [ordered]@{}
    foreach ($row in (Select-String -LiteralPath $Paths -SimpleMatch 'decode_telemetry: phase=')) {
        $match = [regex]::Match($row.Line, 'decode_telemetry: phase=(?<phase>\w+) (?<fields>.*)$')
        if (!$match.Success) { continue }
        $phase = $match.Groups['phase'].Value
        if (!$phases.Contains($phase)) { $phases[$phase] = [ordered]@{} }
        $fields = $phases[$phase]
        foreach ($field in [regex]::Matches($match.Groups['fields'].Value,
                '(?:^| )(?<key>wall_ms|calls|target_tokens|graphs|scheduled_splits|cuda_[a-z0-9_]+|vk_gpu_compute_segment_samples)=(?<value>[0-9]+(?:\.[0-9]+)?)(?= |$)')) {
            $key = $field.Groups['key'].Value
            $value = $field.Groups['value'].Value
            if ($value.Contains('.')) {
                $fields[$key] = [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
            } else {
                $fields[$key] = [uint64]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }
    return $phases
}

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
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "cuda-breakdown-$Mode-$stamp"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_TOKENS', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY', 'LLAMA_EXPERT_ROUTER_HOST_IDS',
    'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'GGML_CUDA_GPU_TIMESTAMPS_DEVICE', 'LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_SCHEDULE_SPIN', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_EXPERT_VK_DEFER_SLOT_MAP', 'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_REUSE_GRAPH_FENCE', 'GGML_VK_COMPACT_EXPERT_SUBMITS', 'LLAMA_EXPERT_VK_DEFER_UPLOAD')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }

try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { Set-MeasureEnvironment $key $null }
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
    $env:LLAMA_DECODE_TELEMETRY = '1'

    $env:GGML_VK_REUSE_GRAPH_FENCE = '1'

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $cases = @('cpu', 'gpu')
    if ($Mode -ne 'both') { $cases = @($Mode) }

    $baseArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    if ($LASTEXITCODE -ne 0) { throw "Could not record HEAD. Logs: $runDir" }
    git -C $root status --short | Set-Content (Join-Path $runDir 'working-status.txt') -Encoding UTF8
    git -C $root diff HEAD --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt') -Encoding UTF8
    $diffPath = Join-Path $runDir 'working-diff.patch'
    git -C $root diff --no-ext-diff --no-color --output="$diffPath" HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not record working diff. Logs: $runDir" }

    foreach ($case in $cases) {
        $timestamps = if ($case -eq 'gpu') { 'CUDA0' } else { $null }
        Set-MeasureEnvironment 'GGML_CUDA_GPU_TIMESTAMPS_DEVICE' $timestamps
        $prefix = Join-Path $runDir $case
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        [ordered]@{ Name=$case; Mode=$Mode; Tokens=$Tokens; Environment=$envSettings } |
            ConvertTo-Json -Depth 3 | Set-Content "$prefix.settings.json" -Encoding UTF8
        $runArgs = @('-lv', '4', '--log-file', "$prefix.log") + $baseArgs
        $runArgs | ConvertTo-Json | Set-Content "$prefix.arguments.json" -Encoding UTF8
        Write-Host "Running $Mode/$case"
        & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
        $code = $LASTEXITCODE
        "exit_code=$code" | Set-Content "$prefix.exit.txt"
        if ($code -ne 0) { throw "Failed: $case. Logs: $runDir" }

        $logs = @("$prefix.log", "$prefix.stderr.txt")
        $required = @(
            'packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1',
            'draft acceptance = .* / +[1-9][0-9]* generated',
            'CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1',
            'packed expert router: host_hits=[1-9][0-9]*\b',
            'packed expert dynamic residency: promotions=',
            'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*\b',
            'packed expert slot map updates: deferred=[1-9][0-9]*\b',
            'final graph fence reused'
        )
        $forbidden = @(
            'compact expert submit applied',
            'packed expert upload deferral: deferred=[1-9][0-9]*\b',
            'MTP draft anchor: reused=[1-9][0-9]*\b',
            'vk_gpu_compute_segment_samples=[1-9][0-9]*\b'
        )
        Select-String -LiteralPath $logs -Pattern ($required + $forbidden) |
            ForEach-Object Line | Select-Object -Unique | Set-Content "$prefix.activation.txt"
        foreach ($pattern in $required) {
            if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                throw "Missing required marker '$pattern' in $case. Logs: $runDir"
            }
        }
        foreach ($pattern in $forbidden) {
            if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -gt 0) {
                throw "Unexpected activation '$pattern' in $case. Logs: $runDir"
            }
        }

        $phases = Read-CudaTelemetry $logs
        $phases | ConvertTo-Json -Depth 4 | Set-Content "$prefix.telemetry.json" -Encoding UTF8
        foreach ($phase in @('target_verify', 'draft', 'draft_catchup', 'server_step_total')) {
            $fields = $phases[$phase]
            if (!$fields -or !$fields.Contains('wall_ms')) {
                throw "Missing phase wall measurement '$phase' in $case. Logs: $runDir"
            }
        }
        if ($phases['target_verify']['scheduled_splits'] -le 0 -or
                !$phases['target_verify'].Contains('cuda_stream_synchronize_cpu_ms')) {
            throw "Missing target verify split or CUDA CPU timing in $case. Logs: $runDir"
        }
        $gpuSamples = $phases['target_verify']['cuda_gpu_stream_span_samples']
        if ($case -eq 'gpu' -and $gpuSamples -le 0) {
            throw "No CUDA stream span samples in target_verify. Logs: $runDir"
        }
        if ($case -eq 'cpu' -and @($phases.Values | Where-Object { $_['cuda_gpu_stream_span_samples'] -gt 0 }).Count -gt 0) {
            throw "CUDA stream events unexpectedly collected in CPU-only case. Logs: $runDir"
        }
    }

    $texts = @{}
    foreach ($case in $cases) {
        $text = Get-Content -LiteralPath (Join-Path $runDir "$case.stdout.txt") -Raw
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Missing stdout for $case. Logs: $runDir" }
        $texts[$case] = ($text -replace '\x1b\[[0-?]*[ -/]*[@-~]', '' -replace '\[ Prompt:.*?\]', '').Trim()
        if (!$texts[$case]) { throw "Empty normalized stdout for $case. Logs: $runDir" }
    }
    if ($Mode -eq 'both') {
        $same = $texts['cpu'] -ceq $texts['gpu']
        "normalized_stdout_equal=$same" | Set-Content (Join-Path $runDir 'output-comparison.txt')
        if (!$same) { throw "Normalized stdout differs across CPU and GPU cases. Logs: $runDir" }
    }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { Set-MeasureEnvironment $key $saved[$key] }
    Write-Host "Log directory: $runDir"
}
