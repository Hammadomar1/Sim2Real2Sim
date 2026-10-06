$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_gripper.py
    if ($LASTEXITCODE -ne 0) { throw 'Gripper contact check failed. Inspect artifacts/gripper_validation.json.' }
} finally { Pop-Location }
