# Watch a policy in the MuJoCo viewer. Example: .\scripts\play.ps1 runs\tee_v1\best.pt   (or -Scripted)
param([string]$Checkpoint = '', [switch]$Scripted, [int]$Seed = 1)
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($Scripted) { & .\.venv\Scripts\python.exe -m s2r2s.play --scripted --seed $Seed }
else { & .\.venv\Scripts\python.exe -m s2r2s.play --checkpoint $Checkpoint --seed $Seed }
