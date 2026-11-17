[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$LogDirectory)

$ErrorActionPreference = 'Stop'
$responses = @()
$metrics = @()
$active = $null
foreach ($line in Get-Content -LiteralPath (Join-Path $LogDirectory 'dialogue.log')) {
    if ($line -match 'decode_response_begin: (.*)$') {
        if ($null -ne $active) { throw "Response $active has no end marker before the next response. Log output may be missing; cannot produce a complete report." }
        $fields = [ordered]@{}
        foreach ($m in [regex]::Matches($Matches[1], '(\w+)=([^\s]+)')) { $fields[$m.Groups[1].Value] = $m.Groups[2].Value }
        $active = $fields['response']
        $responses += [pscustomobject]$fields
    } elseif ($line -match 'decode_response_end: response=(\d+)') {
        if ($null -eq $active -or $active -ne $Matches[1]) { throw 'Response telemetry boundary mismatch' }
        $active = $null
    } elseif ($null -ne $active -and $line -match 'decode_telemetry: (.*)$') {
        $fields = @{}
        foreach ($m in [regex]::Matches($Matches[1], '(\w+)=([^\s]+)')) { $fields[$m.Groups[1].Value] = $m.Groups[2].Value }
        $group = if ($fields.ContainsKey('cuda_component')) { 'cuda_component:' + $fields['cuda_component'] }
            elseif ($fields.ContainsKey('vk_cause')) { 'vk_cause:' + $fields['vk_cause'] } else { 'phase_total' }
        if ($fields.ContainsKey('cuda_op')) { $group += '/op:' + $fields['cuda_op'] }
        if ($fields.ContainsKey('hc_projection')) { $group += '/projection:' + $fields['hc_projection'] }
        foreach ($key in ($fields.Keys | Sort-Object)) {
            if ($key -in @('phase', 'cuda_component', 'cuda_op', 'hc_projection', 'vk_cause')) { continue }
            $metrics += [pscustomobject]@{ Response=$active; Phase=$fields['phase']; Group=$group; Metric=$key; Value=$fields[$key] }
        }
    }
}
if ($null -ne $active) { throw 'Incomplete response telemetry block' }
if (!$responses.Count) { throw 'No response telemetry. Rebuild and enable LLAMA_DECODE_TELEMETRY_PER_RESPONSE=1 with one slot.' }
$responses | Export-Csv (Join-Path $LogDirectory 'responses.csv') -NoTypeInformation -Encoding UTF8
$metrics | Export-Csv (Join-Path $LogDirectory 'metrics.csv') -NoTypeInformation -Encoding UTF8
Write-Host "Extracted $($responses.Count) responses and $($metrics.Count) metrics. GPU spans and CPU waits overlap; do not sum them."
