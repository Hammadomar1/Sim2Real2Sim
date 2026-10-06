param([switch]$Search)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ($Search) {
        & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/finish_parking.py
        if ($LASTEXITCODE -ne 0) { throw 'Final parking search failed.' }
    }
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_path.py --input gripper_complete_path --timestep .001 --output gripper_complete_path_audit
    if ($LASTEXITCODE -ne 0) { throw 'Independent full-path audit failed.' }
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_complete_env.py
    if ($LASTEXITCODE -ne 0) { throw 'Production environment replay failed.' }
} finally { Pop-Location }
