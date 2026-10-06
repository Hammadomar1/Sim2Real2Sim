$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$linuxPath = (& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
Write-Host 'Open http://localhost:6006 in your browser. Ctrl+C stops TensorBoard.'
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/tensorboard --logdir "$linuxPath/runs" --host 127.0.0.1 --port 6006
