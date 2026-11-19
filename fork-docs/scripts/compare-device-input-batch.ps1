[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256
)

throw 'This historical comparison requires device input batching to be configurable. Run the original script with a matching build from 21b588e80.'
