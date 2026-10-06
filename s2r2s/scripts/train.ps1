# Train a policy. Example: .\scripts\train.ps1 -Run tee_v2 -Minutes 120
param([Parameter(Mandatory=$true)][string]$Run, [double]$Minutes = 90, [string]$Objects = 'tee', [string]$Extra = '')
Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\python.exe -u -m s2r2s.train --run $Run --minutes $Minutes --objects $Objects @($Extra -split ' ' | Where-Object { $_ })
