param([ValidateSet(56,39)][int]$Scene = 56)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recording = Join-Path $projectRoot "artifacts/push_failure_diagnosis/scene_$Scene.npz"
if (-not (Test-Path -LiteralPath $recording)) { throw "Missing diagnostic recording: $recording" }
$viewerArgs = '"{0}" --replay "{1}" --label "PPO FAILURE - SCENE {2} - FILTERED POLICY"' -f (Join-Path $PSScriptRoot 'viewer.py'), $recording, $Scene
Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot "artifacts/push_failure_diagnosis/viewer_$Scene.log") -RedirectStandardError (Join-Path $projectRoot "artifacts/push_failure_diagnosis/viewer_$Scene.error.log")
Write-Host 'Showing the exact failed batch rollout. P pauses/resumes; R restarts. The episode times out at 30 seconds.'
