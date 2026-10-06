param([switch]$NativeOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$linuxRoot = (& wsl -d Ubuntu-22.04 -- wslpath -a $projectRoot).Trim()
$scriptPath = "$linuxRoot/scripts/validate_physics.py"
if ($NativeOnly) {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python $scriptPath
} else {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python $scriptPath --gpu
}
if ($LASTEXITCODE -ne 0) { throw 'Physics validation did not pass. Inspect artifacts/physics_validation.json and console output.' }
