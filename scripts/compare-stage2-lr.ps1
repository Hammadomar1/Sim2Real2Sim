param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python -u "$linuxPath/scripts/compare_stage2_lr.py"
if ($LASTEXITCODE -ne 0) { throw 'Bounded learning-rate comparison failed; inspect saved logs before retrying.' }
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxPath/scripts/report_stage2_lr.py"
if ($LASTEXITCODE -ne 0) { throw 'Comparison reporting failed.' }
Write-Host 'Completed comparison: artifacts/stage2_lr_comparison/comparison.png'
