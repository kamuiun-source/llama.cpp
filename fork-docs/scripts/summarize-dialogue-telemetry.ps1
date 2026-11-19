[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$LogDirectory,
    [switch]$AllowPartial
)

$ErrorActionPreference = 'Stop'
$responses = [Collections.Generic.List[object]]::new()
$metrics = [Collections.Generic.List[object]]::new()
$discarded = [Collections.Generic.List[string]]::new()
$completed = @{}
$draftByTask = @{}
$active = $null

function Get-LogFields([string]$Text) {
    $fields = [ordered]@{}
    foreach ($match in [regex]::Matches($Text, '(\w+)=([^\s]+)')) {
        $fields[$match.Groups[1].Value] = $match.Groups[2].Value
    }
    return $fields
}

foreach ($line in Get-Content -LiteralPath (Join-Path $LogDirectory 'dialogue.log') -Encoding UTF8) {
    if ($line -match 'mtp_vocab_request: (.*)$') {
        $draft = Get-LogFields $Matches[1]
        if ($draft['single_slot'] -eq '1' -and $draft.Contains('request')) { $draftByTask[$draft['request']] = $draft }
    }
    if ($line -match 'decode_response_begin: (.*)$') {
        if ($null -ne $active) {
            if (!$AllowPartial) { throw "Response $($active['response']) has no end marker before the next response." }
            $discarded.Add($active['response'])
        }
        $active = Get-LogFields $Matches[1]
        if ($active['response'] -notmatch '^\d+$') { throw 'Missing response ID in telemetry begin marker' }
        $blockMetrics = [Collections.Generic.List[object]]::new()
    } elseif ($line -match 'decode_response_end: response=(\d+)') {
        $end = $Matches[1]
        if ($null -eq $active -or $active['response'] -ne $end) {
            if (!$AllowPartial) { throw 'Response telemetry boundary mismatch' }
            if ($null -ne $active) { $discarded.Add($active['response']) }
            $discarded.Add($end)
            $active = $null
            continue
        }
        if ($completed.ContainsKey($end)) { throw "Duplicate completed telemetry response $end" }
        $completed[$end] = $true
        $responses.Add([pscustomobject]$active)
        $metrics.AddRange($blockMetrics)
        $active = $null
    } elseif ($null -ne $active -and $line -match '(decode_telemetry|expert_staging_copy|decode_pool_reset): (.*)$') {
        $category = $Matches[1]
        $fields = Get-LogFields $Matches[2]
        $group = if ($category -eq 'decode_pool_reset') { "pool_reset:$($fields['device'])/$($fields['pool'])" }
            elseif ($category -eq 'expert_staging_copy') { 'expert_staging_copy' }
            elseif ($fields.Contains('cuda_component')) { 'cuda_component:' + $fields['cuda_component'] }
            elseif ($fields.Contains('vk_cause')) { 'vk_cause:' + $fields['vk_cause'] } else { 'phase_total' }
        if ($fields.Contains('cuda_op')) { $group += '/op:' + $fields['cuda_op'] }
        if ($fields.Contains('hc_projection')) { $group += '/projection:' + $fields['hc_projection'] }
        foreach ($key in ($fields.Keys | Sort-Object)) {
            if ($key -in @('phase', 'cuda_component', 'cuda_op', 'hc_projection', 'vk_cause', 'device', 'pool')) { continue }
            $blockMetrics.Add([pscustomobject]@{
                Response=$active['response']; Phase=$fields['phase']; Group=$group; Metric=$key; Value=$fields[$key]
            })
        }
    }
}
if ($null -ne $active) {
    if (!$AllowPartial) { throw 'Incomplete response telemetry block' }
    $discarded.Add($active['response'])
}
if (!$responses.Count) { throw 'No response telemetry. Rebuild and enable LLAMA_DECODE_TELEMETRY_PER_RESPONSE=1 with one slot.' }

function Convert-TimingValue($Value) {
    if ($null -eq $Value) { return $null }
    return [double]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
}

$phaseTotals = @{}
foreach ($row in $metrics) {
    if ($row.Group -eq 'phase_total') {
        $phaseTotals["$($row.Response)/$($row.Phase)/$($row.Metric)"] = $row.Value
    }
}
$phaseColumns = [ordered]@{
    TargetPrefillMs = 'target_prefill/wall_ms'
    TargetDecodeMs = 'target_decode/wall_ms'
    TargetVerifyMs = 'target_verify/wall_ms'
    TargetVerifyCalls = 'target_verify/calls'
    TargetVerifyPositions = 'target_verify/target_tokens'
    TargetMixedMs = 'target_mixed/wall_ms'
    DraftPhaseMs = 'draft/wall_ms'
    DraftCatchupMs = 'draft_catchup/wall_ms'
    ServerStepMs = 'server_step_total/wall_ms'
    VerifyCudaGpuSpanMs = 'target_verify/cuda_gpu_stream_span_ms'
    VerifyCudaGpuSamples = 'target_verify/cuda_gpu_stream_span_samples'
    VerifyCudaGpuDropped = 'target_verify/cuda_gpu_stream_span_dropped'
    VerifyCudaGpuPending = 'target_verify/cuda_gpu_stream_span_pending'
    VerifyCudaSyncCpuMs = 'target_verify/cuda_stream_synchronize_cpu_ms'
    VerifyVkGpuSegmentMs = 'target_verify/vk_gpu_compute_segment_ms'
    VerifyVkGpuSamples = 'target_verify/vk_gpu_compute_segment_samples'
    VerifyVkGpuDropped = 'target_verify/vk_gpu_compute_segment_dropped'
    VerifyVkFenceWaitCpuMs = 'target_verify/vk_fence_wait_cpu_ms'
    VerifyVkRecordCpuMs = 'target_verify/vk_record_cpu_ms'
    VerifyVkComputeSubmits = 'target_verify/vk_compute_submits'
    VerifyVkTransferSubmits = 'target_verify/vk_transfer_submits'
    VerifyExpertRequests = 'target_verify/expert_requests'
    VerifyExpertMisses = 'target_verify/expert_request_misses'
    VerifyExpertUploadBytes = 'target_verify/uploaded_bytes'
    VerifyExpertWritebackBytes = 'target_verify/expert_writeback_bytes'
}
$turns = @(
    foreach ($response in $responses) {
        $id = $response.response
        $draft = $draftByTask[$response.task]
        $draftMs = if ($null -ne $draft -and $draft['generated'] -eq $response.generated_tokens -and
                      $draft['drafted'] -eq $response.draft_generated -and $draft['accepted'] -eq $response.draft_accepted) {
            (Convert-TimingValue $draft['draft_wall_us']) / 1000
        } else { $null }
        $turn = [ordered]@{
            Response = [int]$id
            Task = [int]$response.task
            PromptTokens = [int]$response.prompt_tokens
            CachedTokens = [int]$response.cached_tokens
            ContextTokens = [int]$response.context_tokens
            GeneratedTokens = [int]$response.generated_tokens
            DecodeTokens = [Math]::Max(0, [int]$response.generated_tokens - 1)
            PromptMs = Convert-TimingValue $response.prompt_ms
            DecodeMs = Convert-TimingValue $response.decode_ms
            DecodeTokS = Convert-TimingValue $response.generation_tps
            DraftProposed = [int]$response.draft_generated
            DraftAccepted = [int]$response.draft_accepted
            DraftAcceptanceRate = if ([int]$response.draft_generated -gt 0) {
                [double]$response.draft_accepted / [int]$response.draft_generated
            } else { $null }
            DraftOnlyMs = $draftMs
            Truncated = $response.truncated -eq '1'
        }
        foreach ($entry in $phaseColumns.GetEnumerator()) {
            $turn[$entry.Key] = Convert-TimingValue $phaseTotals["$id/$($entry.Value)"]
        }
        [pscustomobject]$turn
    }
)

$exitCode = $null
$exitPath = Join-Path $LogDirectory 'exit.txt'
if ((Test-Path -LiteralPath $exitPath -PathType Leaf) -and
        (Get-Content -LiteralPath $exitPath -Raw) -match 'exit_code=(-?\d+)') { $exitCode = [int64]$Matches[1] }
$decodeTokens = ($turns | Measure-Object DecodeTokens -Sum).Sum
$decodeMs = ($turns | Measure-Object DecodeMs -Sum).Sum
$settingsPath = Join-Path $LogDirectory 'settings.json'
$mode = $null
if (Test-Path -LiteralPath $settingsPath -PathType Leaf) {
    try {
        $mode = (Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json).Mode
    } catch {
        Write-Warning 'Could not read settings.json. Summary is based on completed response blocks.'
    }
}
$summary = [ordered]@{
    Mode = $mode
    CompleteTurns = $turns.Count
    DiscardedResponseIds = @($discarded | Select-Object -Unique)
    ExitCode = $exitCode
    Partial = $discarded.Count -gt 0 -or ($null -ne $exitCode -and $exitCode -ne 0)
    DecodeTokens = $decodeTokens
    DecodeMs = $decodeMs
    PooledDecodeTokS = if ($decodeMs -gt 0) { 1000.0 * $decodeTokens / $decodeMs } else { $null }
    Notes = @(
        'Decode excludes the first generated token from its token count because prefill produced that token.'
        'CPU waits, GPU spans and the server-step total can overlap. Keep them separate.'
        'GPU spans cover instrumented stream or command-buffer intervals. Check sample, dropped and pending counts.'
        'Missing metrics are null. The components mode disables CUDA Graphs and synchronizes dispatch timing.'
    )
    Turns = $turns
}
$responses | Export-Csv (Join-Path $LogDirectory 'responses.csv') -NoTypeInformation -Encoding UTF8
$metrics | Export-Csv (Join-Path $LogDirectory 'metrics.csv') -NoTypeInformation -Encoding UTF8
$turns | Export-Csv (Join-Path $LogDirectory 'turn_summary.csv') -NoTypeInformation -Encoding UTF8
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $LogDirectory 'summary.json') -Encoding UTF8
$turns | Format-Table Response, PromptTokens, CachedTokens, GeneratedTokens, DecodeMs, DecodeTokS, DraftAccepted, DraftProposed | Out-Host
Write-Host "Saved $($turns.Count) completed turns and $($metrics.Count) detailed metrics to $LogDirectory"
if ($summary.Partial) { Write-Warning 'Partial run. See summary.json for the exit code and discarded response IDs.' }
