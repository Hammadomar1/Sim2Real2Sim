param([switch]$Search)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ($Search) {
        & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/gate_reposition.py --prefix 140
        if ($LASTEXITCODE -ne 0) { throw 'Withdrawal search failed.' }
        Copy-Item artifacts/gripper_gate_reposition.npz artifacts/gripper_gate_escape.npz
        Copy-Item artifacts/gripper_gate_reposition.json artifacts/gripper_gate_escape.json
        & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/gate_reposition.py --continue-push
        if ($LASTEXITCODE -ne 0) { throw 'Safe gate passage search failed.' }
    }
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_path.py --input gripper_gate_reposition --timestep .001 --output gripper_gate_reposition_audit --gate-only
    if ($LASTEXITCODE -ne 0) { throw 'Independent gate audit failed.' }
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_gate_env.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment gate replay failed.' }
} finally { Pop-Location }
