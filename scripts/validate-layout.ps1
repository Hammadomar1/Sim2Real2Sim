$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python -m pytest tests -q --junitxml=artifacts/task_rules_tests.xml
    if ($LASTEXITCODE -ne 0) { throw 'Rule tests failed.' }
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python scripts/validate_layout.py --resets-only
    if ($LASTEXITCODE -ne 0) { throw 'Layout validation failed; inspect artifacts/layout_validation.json.' }
} finally { Pop-Location }
