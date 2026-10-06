param([double]$Timestep = 0.002)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$linuxRoot = (& wsl -d Ubuntu-22.04 -- wslpath -a $projectRoot).Trim()
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxRoot/scripts/validate_path.py" --timestep $Timestep
if ($LASTEXITCODE -ne 0) { throw 'Path verification failed; inspect artifacts/full_path_validation*.json.' }
