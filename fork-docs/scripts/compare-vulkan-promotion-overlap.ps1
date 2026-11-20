[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256
)

throw 'This historical comparison requires Vulkan Expert promotion overlap to be configurable. Run the original script with a matching build from 21b588e80.'
