param(
    [ValidateSet('baseline','corrected','directional')][string]$Variant = 'directional',
    [ValidateRange(1,150)][int]$Updates = 100,
    [string]$Checkpoint = 'artifacts/stage2_reward_audit/baseline_latest.pt',
    [string]$RunDir = '',
    [int]$Seed = 71
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
if (-not $RunDir) { $RunDir = 'runs/stage2_' + $Variant + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python -u "$linuxPath/scripts/stage2_experiment.py" --variant $Variant --updates $Updates --checkpoint $Checkpoint --run-dir $RunDir --seed $Seed
if ($LASTEXITCODE -ne 0) { throw 'Stage-2 diagnostic failed; inspect its logs before further training.' }
Write-Host "Bounded reward experiment saved in $RunDir. This is a warm start with a fresh critic/optimizer, not an exact resume."
