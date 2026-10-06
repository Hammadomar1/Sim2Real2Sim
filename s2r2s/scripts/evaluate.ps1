# Held-out evaluation, side by side with the scripted baseline. Example: .\scripts\evaluate.ps1 runs\tee_v1\best.pt
param([Parameter(Mandatory=$true)][string]$Checkpoint, [int]$Episodes = 1000)
Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\python.exe -m s2r2s.evaluate $Checkpoint --episodes $Episodes --baseline
