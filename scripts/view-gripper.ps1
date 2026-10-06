param([ValidateSet('native','warp')][string]$Backend = 'warp')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recording = if ($Backend -eq 'warp') { 'gripper_mpc_precision.npz' } else { 'gripper_precision_independent.npz' }
$trajectory = Join-Path $projectRoot "artifacts/$recording"
if (-not (Test-Path -LiteralPath $trajectory)) { throw "Missing recording: $trajectory" }
$viewerArgs = '"{0}" --replay "{1}" --label "GRIPPER {2}: VERIFIED RECORDED DEMONSTRATION"' -f (Join-Path $PSScriptRoot 'viewer.py'), $trajectory, $Backend.ToUpper()
Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot "artifacts/gripper-$Backend-viewer.log") -RedirectStandardError (Join-Path $projectRoot "artifacts/gripper-$Backend-viewer.error.log")
Write-Host 'Opened MuJoCo recorded gripper demonstration. P pauses; R restarts. This is diagnostic control, not a trained RL policy.'
