$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$viewerPython = Join-Path $projectRoot '.venv-viewer\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $viewerPython)) {
    throw 'Viewer environment missing. Run scripts/setup-viewer.ps1 first.'
}
$viewerScript = Join-Path $PSScriptRoot 'viewer.py'
$process = Start-Process -FilePath $viewerPython -ArgumentList @('"' + $viewerScript + '"') -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot 'artifacts/viewer.log') -RedirectStandardError (Join-Path $projectRoot 'artifacts/viewer.error.log') -PassThru
Write-Host "MuJoCo viewer launched (PID $($process.Id)). Starts paused. P: physics; R: reset. No trained policy."
