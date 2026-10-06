param([ValidateSet('all','native','warp','replicas','view')][string]$Mode = 'all')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ($Mode -eq 'view') {
        $trajectory = Join-Path $projectRoot 'artifacts/full_replay_warp_single_nominal.npz'
        if (-not (Test-Path -LiteralPath $trajectory)) { throw 'Run the GPU replay first.' }
        $viewerArgs = '"{0}" --replay "{1}" --label "GPU REPLAY - see sensitivity report for outcome"' -f (Join-Path $PSScriptRoot 'viewer.py'), $trajectory
        Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot 'artifacts/gpu-replay-viewer.log') -RedirectStandardError (Join-Path $projectRoot 'artifacts/gpu-replay-viewer.error.log')
        return
    }
    $python = "/home/$env:USERNAME/.venvs/so101-m1/bin/python"
    if ($Mode -in @('all','native')) {
        & wsl -d Ubuntu-22.04 -- $python scripts/replay_sensitivity.py --backend native
        if ($LASTEXITCODE -ne 0) { throw 'Native replay execution failed.' }
    }
    if ($Mode -in @('all','warp')) {
        & wsl -d Ubuntu-22.04 -- $python scripts/replay_sensitivity.py --backend warp --nominal-only
        if ($LASTEXITCODE -ne 0) { throw 'Single GPU replay execution failed.' }
        & wsl -d Ubuntu-22.04 -- $python scripts/replay_sensitivity.py --backend warp
        if ($LASTEXITCODE -ne 0) { throw 'GPU sensitivity execution failed.' }
    }
    if ($Mode -in @('all','replicas')) {
        & wsl -d Ubuntu-22.04 -- $python scripts/replay_sensitivity.py --backend warp --replicas 3
        if ($LASTEXITCODE -ne 0) { throw 'GPU replica execution failed.' }
    }
    & wsl -d Ubuntu-22.04 -- $python scripts/report_replay_sensitivity.py
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed; run all modes first.' }
    Write-Host 'Diagnostics finished. Read GPU_REPLAY_SENSITIVITY.md: command completion does not mean task success.'
} finally { Pop-Location }
