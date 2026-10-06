$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$recording = Join-Path $projectRoot 'artifacts/push_recovery/recovery.npz'
$viewerArgs = '"{0}" --replay "{1}" --label "DIAGNOSTIC RECOVERY - POLICY UNTIL 20s THEN SEARCHED CONTROL"' -f (Join-Path $PSScriptRoot 'viewer.py'), $recording
Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot 'artifacts/push_recovery/viewer.log') -RedirectStandardError (Join-Path $projectRoot 'artifacts/push_recovery/viewer.error.log')
Write-Host 'Physical feasibility demonstration, not learned recovery. Intervention begins at 20 seconds. P pauses; R restarts.'
