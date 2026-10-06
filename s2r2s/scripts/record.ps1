# Render a 2x2 grid video. Example: .\scripts\record.ps1 runs\tee_v1\best.pt videos\tee.mp4
param([Parameter(Mandatory=$true)][string]$Checkpoint, [string]$Out = 'videos\policy.mp4', [string]$Camera = 'front')
Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\python.exe -m s2r2s.record --checkpoint $Checkpoint --out $Out --grid 2 --camera $Camera
