param([ValidateRange(1,50)][int]$Updates = 50)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python -u "$linuxPath/scripts/train_push_efficiency.py" --updates $Updates
if ($LASTEXITCODE -ne 0) { throw 'Efficiency comparison failed; inspect the saved branch before restarting.' }
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxPath/scripts/report_push_efficiency.py"
if ($LASTEXITCODE -ne 0) { throw 'Efficiency reporting failed.' }
