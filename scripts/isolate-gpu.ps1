param([ValidateSet('settling','plane','controller','cold','jaws','native','report','view')][string]$Mode = 'settling')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $python="/home/$env:USERNAME/.venvs/so101-m1/bin/python"
    switch ($Mode) {
        'settling' { & wsl -d Ubuntu-22.04 -- $python scripts/isolate_gpu.py --shared-stream }
        'plane' { & wsl -d Ubuntu-22.04 -- $python scripts/isolate_gpu.py --shared-stream --plane-table }
        'controller' { & wsl -d Ubuntu-22.04 -- $python scripts/isolate_controller.py --mode drive }
        'cold' { & wsl -d Ubuntu-22.04 -- $python scripts/isolate_controller.py --mode drive --no-warmstart }
        'jaws' { & wsl -d Ubuntu-22.04 -- $python scripts/isolate_controller.py --mode drive --box-jaws }
        'native' { & wsl -d Ubuntu-22.04 -- $python scripts/native_settle_reference.py }
        'report' { & wsl -d Ubuntu-22.04 -- $python scripts/report_gpu_isolation.py }
        'view' {
            $viewerArgs='"{0}" --replay "{1}" --label "GPU settling diagnostic - original box table"' -f (Join-Path $PSScriptRoot 'viewer.py'),(Join-Path $projectRoot 'artifacts/isolation_settling_view.npz')
            Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot 'artifacts/isolation-viewer.log') -RedirectStandardError (Join-Path $projectRoot 'artifacts/isolation-viewer.error.log')
            return
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Isolation $Mode command failed." }
} finally { Pop-Location }
