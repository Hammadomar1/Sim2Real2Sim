$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($workspace.Replace('\','/'))).Trim()
& wsl -d Ubuntu-22.04 -- bash "$linuxPath/scripts/setup.sh"
if ($LASTEXITCODE -ne 0) { throw "Setup failed: $LASTEXITCODE" }
