param([string]$Checkpoint = 'runs/seed0/latest.pt', [ValidateRange(1,5)][int]$Stage = 2, [int]$Seed = 3000000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
$runTag = Split-Path -Leaf (Split-Path -Parent $Checkpoint)
$checkpointTag = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$recording = "artifacts/$runTag/${checkpointTag}_stage${Stage}_seed${Seed}.npz"
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxPath/scripts/record_policy.py" --checkpoint $Checkpoint --stage $Stage --seed $Seed --output $recording
if ($LASTEXITCODE -ne 0) { throw 'Policy recording failed.' }
$viewerArgs = '"{0}" --replay "{1}" --label "PPO {4}/{5} - STAGE {2} - SEED {3}"' -f (Join-Path $PSScriptRoot 'viewer.py'), (Join-Path $projectRoot $recording), $Stage, $Seed, $runTag, $checkpointTag
Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot "artifacts/$runTag/policy-stage$Stage-viewer.log") -RedirectStandardError (Join-Path $projectRoot "artifacts/$runTag/policy-stage$Stage-viewer.error.log")
Write-Host 'Showing the recorded learned-policy rollout. P pauses/resumes; R returns to the start. This is not the scripted demonstration.'
