param([ValidateSet('validate','trace-plane','trace-box','repeat-plane','reset-plane','reset-box','report','view')][string]$Mode='validate')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $python="/home/$env:USERNAME/.venvs/so101-m1/bin/python"
    switch ($Mode) {
        'validate' { & wsl -d Ubuntu-22.04 -- $python scripts/validate_table_candidate.py }
        'trace-plane' { & wsl -d Ubuntu-22.04 -- $python scripts/trace_moving_contacts.py --plane }
        'repeat-plane' { & wsl -d Ubuntu-22.04 -- $python scripts/trace_moving_contacts.py --plane --tag _repeat }
        'trace-box' { & wsl -d Ubuntu-22.04 -- $python scripts/trace_moving_contacts.py }
        'reset-plane' { & wsl -d Ubuntu-22.04 -- $python scripts/trace_moving_contacts.py --plane --reset-failed }
        'reset-box' { & wsl -d Ubuntu-22.04 -- $python scripts/trace_moving_contacts.py --reset-failed }
        'report' { & wsl -d Ubuntu-22.04 -- $python scripts/report_table_checkpoint.py }
        'view' {
            $viewerArgs='"{0}" --replay "{1}" --label "Candidate table: gate-contact failure diagnostic"' -f (Join-Path $PSScriptRoot 'viewer.py'),(Join-Path $projectRoot 'artifacts/table_candidate_failure_view.npz')
            Start-Process -FilePath (Join-Path $projectRoot '.venv-viewer/Scripts/pythonw.exe') -ArgumentList $viewerArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $projectRoot 'artifacts/table-candidate-viewer.log') -RedirectStandardError (Join-Path $projectRoot 'artifacts/table-candidate-viewer.error.log')
            return
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Table checkpoint $Mode command failed." }
} finally { Pop-Location }
