param([ValidateSet('gate','gripper','contact','initial','full','feedback','feedback1ms')][string]$Trial = 'gripper')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$viewerPython = Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe'
$viewerScript = Join-Path $PSScriptRoot 'viewer.py'
if ($Trial -notin @('gripper','contact','gate','full')) { throw 'These recordings are not validated for the current gripper. Use -Trial full, gate or contact.' }
$trajectoryName = switch ($Trial) { 'full' { 'gripper_complete_path_audit.npz' } 'gate' { 'gripper_gate_reposition_audit.npz' } 'contact' { 'contact_validated_trial.npz' } default { 'gripper_push.npz' } }
$trajectory = Join-Path $projectRoot ('artifacts/' + $trajectoryName)
if (-not (Test-Path -LiteralPath $trajectory)) { throw 'Controller audit trajectory is missing.' }
$viewerArgs = '"{0}" --replay "{1}"' -f $viewerScript, $trajectory
Start-Process -FilePath $viewerPython -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot "artifacts/$Trial-viewer.log") -RedirectStandardError (Join-Path $projectRoot "artifacts/$Trial-viewer.error.log")
Write-Host "Opened recorded $Trial controller trial. P pauses, R returns to the beginning. This is not a trained policy."
