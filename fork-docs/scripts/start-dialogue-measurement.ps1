[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry', 'components')]
    [string]$Mode = 'telemetry',
    [switch]$Accepted,
    [switch]$HcScaleSilu,
    [switch]$HcUpPair,
    [switch]$HcDownReuse,
    [switch]$HcUpReuse,
    [switch]$HcInjectSplit,
    [switch]$HcAll,
    [switch]$ExpertRamCopyTiming,
    [switch]$ExpertPrefill32,
    [switch]$ExpertPrefillActiveWaves,
    [switch]$ExpertPrefillTrace,
    [string]$BuildDirectory = 'G:\AI\build-prefill32'
)

$ErrorActionPreference = 'Stop'
if ($Accepted) { $HcAll = $true }
if ($ExpertPrefillActiveWaves -and !$ExpertPrefill32) {
    throw 'ExpertPrefillActiveWaves requires ExpertPrefill32.'
}
$projections = @(
    @{ Name='down'; Enabled=($HcAll -or $HcDownReuse); Key='GGML_CUDA_HC_DOWN_REUSE'; Marker='CUDA HC down reuse applied' }
    @{ Name='up'; Enabled=($HcAll -or $HcUpReuse); Key='GGML_CUDA_HC_UP_REUSE'; Marker='CUDA HC up reuse applied' }
    @{ Name='inject'; Enabled=($HcAll -or $HcInjectSplit); Key='GGML_CUDA_HC_INJECT_SPLIT'; Marker='CUDA HC inject split applied' }
)
$activeProjections = @($projections | Where-Object { $_.Enabled })
if ($ExpertPrefillTrace -and ($Mode -ne 'speed' -or $ExpertPrefill32 -or $ExpertRamCopyTiming -or $HcUpPair -or $HcScaleSilu -or $activeProjections.Count)) {
    throw 'ExpertPrefillTrace requires -Mode speed and the standard Expert and HC settings.'
}
if (($HcUpPair -and $HcScaleSilu) -or (($HcUpPair -or $HcScaleSilu) -and $activeProjections.Count)) {
    throw 'Run the legacy HC experiment separately from the new projection experiments.'
}
$cli = Join-Path $BuildDirectory 'bin\llama-cli.exe'
$settingsPath = 'G:\AI\llama-hetero-moe-local\logs\rtx-cold-repeated-20261117-172614\round-01\baseline.settings.json'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$draft = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $settingsPath, $model, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$retiredKeys = @(
    'LLAMA_EXPERT_SCRATCH_REUSE', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_ROUTER_HOST_IDS', 'GGML_CUDA_SCHEDULE_SPIN', 'LLAMA_EXPERT_VK_DEFER_SLOT_MAP',
    'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_SCHED_OVERLAP_SHARED_EXPERT', 'GGML_VK_REUSE_GRAPH_FENCE',
    'LLAMA_EXPERT_VK_OVERLAP_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_PROMOTION', 'LLAMA_EXPERT_VK_FULL_HOT_OVERLAP',
    'LLAMA_EXPERT_RAM_ROTATION', 'GGML_SCHED_SHARED_HOST_EXPERT_INPUT', 'GGML_SCHED_SHARED_HOST_EXPERT_RETURN',
    'LLAMA_MTP_DEVICE_PENDING', 'GGML_CUDA_SSM_CONV_HISTORY', 'GGML_SCHED_OVERLAP_SHARED_READBACK'
)
$baselineEnvironment = @($settings.Environment.PSObject.Properties | Where-Object { $_.Name -notin $retiredKeys })
$telemetryEnvironment = @('LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_TELEMETRY_PER_RESPONSE', 'LLAMA_EXPERT_MISS_HISTOGRAM',
    'GGML_CUDA_GPU_TIMESTAMPS_DEVICE', 'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_PROFILE_COMPONENTS')
$acceptedEnvironment = [ordered]@{
    LLAMA_EXPERT_PREFILL_512 = '1'
    LLAMA_EXPERT_PREFILL_512_POLICY = 'auto'
    LLAMA_EXPERT_PREFILL_LOGICAL_TICKS = '1'
    LLAMA_EXPERT_PREFILL_OVERLAP_TEN = '1'
    LLAMA_EXPERT_PREFILL_STATS = '1'
    LLAMA_EXPERT_EARLY_DECODE = $null
    LLAMA_MTP_VERIFY_THREE = $null
    LLAMA_MTP_DRAFT_VOCAB = $null
    LLAMA_MTP_DRAFT_VOCAB_STATS = if ($Mode -eq 'speed') { $null } else { '1' }
    LLAMA_MTP_REUSE_DRAFT_ANCHOR = $null
    LLAMA_MTP_LOGIT_TRACE = $null
    LLAMA_MTP_LOGIT_REFERENCE = $null
    LLAMA_MTP_LOGIT_MATCH_DRAFTS = $null
    LLAMA_MTP_ROUTED_DISPATCH_STATS = $null
    GGML_VK_MTP_ROUTED_REDUCE = '1'
    GGML_VK_MTP_ROUTED_WEIGHTED = '1'
    GGML_VK_MTP_ROUTED_GATE_UP = '1'
    GGML_VK_MTP_ROUTED_GATE_UP_PAIR = '0'
    GGML_SCHED_SHARED_INPUT_PRODUCER_COPY = '0'
    GGML_VK_SKIP_EMPTY_POOL_RESET = '0'
    GGML_SCHED_TRACE_EXPERT_HANDOFF = $null
    GGML_VK_PERF_LOGGER = $null
    GGML_VK_PERF_LOGGER_CONCURRENT = $null
    GGML_VK_SERIALIZE_SUBMISSIONS = $null
    LLAMA_DECODE_PROFILE = $null
    LLAMA_DECODE_PROFILE_CUDA = $null
}
if ($Accepted -and ($ExpertPrefill32 -or $ExpertPrefillActiveWaves)) {
    throw 'The accepted preset uses Prefill512. Run older Prefill32 experiments without -Accepted.'
}
$keys = @($baselineEnvironment.Name) + @('PATH', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_HC_SCALE_SILU_FUSE', 'GGML_CUDA_HC_UP_PAIR', 'LLAMA_EXPERT_RTX_COLD_SLOTS', 'LLAMA_EXPERT_PROFILE_RAM_COPY', 'LLAMA_EXPERT_PREFILL_32', 'LLAMA_EXPERT_PREFILL_ACTIVE_WAVES', 'LLAMA_EXPERT_PREFILL_TRACE') + $telemetryEnvironment + @($projections | ForEach-Object { $_.Key })
if ($Accepted) { $keys += @($acceptedEnvironment.Keys) }
$keys = @($keys | Select-Object -Unique)
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
$projectionLabel = if ($activeProjections.Count -eq $projections.Count) { 'hc-all-' }
    elseif ($activeProjections.Count) { 'hc-' + (($activeProjections | ForEach-Object { $_.Name }) -join '-') + '-' }
    else { '' }
$ramLabel = if ($ExpertRamCopyTiming) { 'ram-timing-' } else { '' }
$prefillLabel = if ($ExpertPrefill32) { 'prefill32-' } else { '' }
if ($Accepted) { $prefillLabel += 'accepted-' }
if ($ExpertPrefillActiveWaves) { $prefillLabel += 'active-waves-' }
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' ("dialogue-$Mode-$projectionLabel$ramLabel$prefillLabel" + (Get-Date -Format yyyyMMdd-HHmmss))
if ($ExpertPrefillTrace) {
    $suffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' ("dialogue-prefill-trace-" + (Get-Date -Format yyyyMMdd-HHmmss) + "-$suffix")
}
New-Item -ItemType Directory -Path $runDir | Out-Null
$tracePath = Join-Path $runDir 'expert-prefill.jsonl'

try {
    foreach ($item in $baselineEnvironment) {
        [Environment]::SetEnvironmentVariable($item.Name, $item.Value, 'Process')
    }
    if ($Accepted) {
        foreach ($key in $telemetryEnvironment) {
            [Environment]::SetEnvironmentVariable($key, $null, 'Process')
        }
        foreach ($item in $acceptedEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($item.Key, $item.Value, 'Process')
        }
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_RTX_COLD_SLOTS', $null, 'Process')
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_PROFILE_RAM_COPY', $(if ($ExpertRamCopyTiming) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_PREFILL_32', $(if ($ExpertPrefill32) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_PREFILL_ACTIVE_WAVES', $(if ($ExpertPrefillActiveWaves) { '1' } else { $null }), 'Process')
    $env:LLAMA_EXPERT_PREFILL_TRACE = if ($ExpertPrefillTrace) { $tracePath } else { $null }
    [Environment]::SetEnvironmentVariable('LLAMA_DECODE_TELEMETRY_PER_RESPONSE', $null, 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_MIX_FUSE', $null, 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_SCALE_SILU_FUSE', $(if ($HcScaleSilu) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_UP_PAIR', $(if ($HcUpPair) { '1' } else { $null }), 'Process')
    foreach ($projection in $projections) {
        [Environment]::SetEnvironmentVariable($projection.Key, $(if ($projection.Enabled) { '1' } else { $null }), 'Process')
    }
    if ($Mode -ne 'speed') {
        $env:LLAMA_DECODE_TELEMETRY = '1'
        $env:LLAMA_DECODE_TELEMETRY_PER_RESPONSE = '1'
        $env:LLAMA_EXPERT_MISS_HISTOGRAM = '1'
        $env:GGML_CUDA_GPU_TIMESTAMPS_DEVICE = 'CUDA0'
        $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
    }
    if ($Mode -eq 'components') {
        $env:GGML_CUDA_DISABLE_GRAPHS = '1'
        $env:GGML_CUDA_PROFILE_COMPONENTS = '1'
    }
    $effective = [ordered]@{}
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { $effective[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
    }
    @{ Mode=$Mode; AcceptedPreset=[bool]$Accepted; Cli=$cli; Environment=$effective; Instrumented=($Mode -ne 'speed' -or $ExpertRamCopyTiming -or $ExpertPrefillTrace) } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $runDir 'settings.json') -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\research\dialogue-benchmark-prompts.md') -Destination $runDir
    $batchSize = if ($Accepted) { '512' } else { '256' }
    $runArgs = @('-lv', '4', '--log-file', (Join-Path $runDir 'dialogue.log'), '--simple-io', '--color', 'off',
        '-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', $batchSize, '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234', '-n', '256',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    if ($Accepted) { $runArgs += @('--output-file', (Join-Path $runDir 'transcript.txt')) }
    $runArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    if ($ExpertPrefillTrace) {
        Write-Host 'Send the five saved prompts in order, then /exit.'
        Write-Host 'For a separate long-prompt trace, start a fresh run and enter this command before asking one short question:'
        Write-Host ("/read " + [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\research\dialogue-benchmark-prompts.md')))
        Write-Host 'Enter it twice to give the wider windows enough tokens, then ask one short question and /exit after the answer.'
    } elseif ($Accepted) {
        Write-Host "Mode: $Mode. Enter prompts interactively, then /exit to produce the telemetry summary."
    } else {
        Write-Host "Mode: $Mode. Send the five saved prompts in order, then /exit."
    }
    try {
        $ErrorActionPreference = 'Continue'
        & $cli @runArgs 2> (Join-Path $runDir 'stderr.log')
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
    }
    "exit_code=$code" | Set-Content (Join-Path $runDir 'exit.txt')
    if ($Mode -ne 'speed') {
        $summarize = Join-Path $PSScriptRoot 'summarize-dialogue-telemetry.ps1'
        if ($code -eq 0) {
            & $summarize -LogDirectory $runDir
        } else {
            try {
                & $summarize -LogDirectory $runDir -AllowPartial
            } catch {
                Write-Warning "Telemetry recovery could not summarize any completed turns: $($_.Exception.Message)"
            }
        }
    }
    if ($code -ne 0) { throw "llama-cli failed: $code" }
    if ($ExpertPrefillTrace -and (!(Test-Path -LiteralPath $tracePath -PathType Leaf) -or (Get-Item -LiteralPath $tracePath).Length -eq 0)) {
        throw 'Expert Prefill trace was not recorded. Check the binary and prompt coverage.'
    }
    if ($ExpertPrefillTrace) {
        $compareScript = Join-Path $PSScriptRoot 'compare-expert-prefill.py'
        Write-Host "Compare the trace with: python `"$compareScript`" `"$runDir`" --per-layer"
    }
    $dialogueLog = Join-Path $runDir 'dialogue.log'
    if (!(Select-String -LiteralPath $dialogueLog -SimpleMatch 'CUDA SSM convolution history applied' -Quiet)) {
        Write-Warning 'CUDA SSM convolution history was not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'MTP device pending: enabled=1 copies=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'MTP device pending copies were not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'shared host expert input: batches=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'Shared host Expert input batches were not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'shared host expert return: copies=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'Shared host Expert return copies were not observed.'
    }
    if ($ExpertRamCopyTiming) {
        if (!(Select-String -LiteralPath $dialogueLog -Pattern 'expert_staging_copy: phase=target_verify verify_calls=[1-9][0-9]*\b' -Quiet)) {
            throw 'Expert staging copy timing recorded no target verification calls. Check the binary and MTP draft activity.'
        }
        if (!(Select-String -LiteralPath $dialogueLog -SimpleMatch 'packed expert promotion RAM memcpy:' -Quiet)) {
            throw 'Expert RAM copy timing did not activate. Check the binary and dynamic Expert placement.'
        }
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert RAM rotation: rotations=[1-9][0-9]*' -Quiet)) {
        if (!(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert RAM rotation: .*pool_bytes=[1-9][0-9]*' -Quiet)) {
            Write-Warning 'Expert RAM rotation pool allocation was not observed.'
        } else {
            Write-Warning 'The Expert RAM rotation pool was allocated, but this run performed no rotations.'
        }
    }
    if ($ExpertPrefill32 -and !(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert prefill: layer_batches=[1-9][0-9]* max_tokens=(?:[3-9]|[1-3][0-9]) ' -Quiet)) {
        throw 'The grouped Expert prefill path did not process a batch larger than two tokens.'
    }
    if ($ExpertPrefillActiveWaves -and !(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert prefill waves: active_only=1 executed=[1-9][0-9]* skipped=[1-9][0-9]*' -Quiet)) {
        throw 'Active Expert prefill did not skip any waves. Check the binary and prompt coverage.'
    }
    foreach ($projection in $activeProjections) {
        if (!(Select-String -LiteralPath $dialogueLog -SimpleMatch $projection.Marker -Quiet)) {
            throw "HC $($projection.Name) did not activate. Check the binary, tensor type, shape and layout."
        }
    }
    if ($HcUpPair -and !(Select-String -LiteralPath (Join-Path $runDir 'dialogue.log') -SimpleMatch 'CUDA HC up pair applied' -Quiet)) {
        throw 'HC up pair did not activate. Check the binary and graph layout.'
    }
    if ($HcScaleSilu -and !(Select-String -LiteralPath (Join-Path $runDir 'dialogue.log') -SimpleMatch 'CUDA HC scale SiLU fusion applied' -Quiet)) {
        throw 'HC scale SiLU fusion did not activate. Check the binary and graph layout.'
    }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Write-Host "Log directory: $runDir"
}
