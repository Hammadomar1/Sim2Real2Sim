param([switch]$Full)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
function Invoke-Validation([string[]]$PythonArgs) {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Validation failed: $PythonArgs" }
}
if ($Full) {
    Invoke-Validation @('scripts/validate_physics.py','--gpu')
    Invoke-Validation @('scripts/validate_path.py','--input','gripper_precision_path','--output','gripper_precision_independent')
    Invoke-Validation @('scripts/validate_layout.py','--resets-only')
    Invoke-Validation @('scripts/validate_table_candidate.py')
    Invoke-Validation @('-m','pytest','-q','--junitxml=artifacts/gripper_final_rules_tests.xml')
    Invoke-Validation @('scripts/gripper_mpc.py')
    Invoke-Validation @('scripts/numerical_soak.py')
    & ./scripts/run.ps1 train --num-envs 16 --updates 2 --hours 0.1 --run-dir runs/gripper_smoke_final --validation-interval 1000
    & ./scripts/run.ps1 train --num-envs 16 --updates 2 --hours 0.1 --resume runs/gripper_smoke_final/latest.pt --run-dir runs/gripper_resume_final --validation-interval 2
}
Invoke-Validation @('scripts/finalize_gripper_readiness.py')
