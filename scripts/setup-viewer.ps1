$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv-viewer'
if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'Scripts/python.exe'))) {
    $pythonPath = Join-Path $env:APPDATA 'uv/python/cpython-3.12.13-windows-x86_64-none/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Install Python 3.12 with uv, or create .venv-viewer using Python 3.12 first.' }
    & $pythonPath -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create viewer environment.' }
}
& (Join-Path $environmentPath 'Scripts/python.exe') -m pip install -r (Join-Path $projectRoot 'requirements-viewer.txt')
if ($LASTEXITCODE -ne 0) { throw 'Viewer package installation failed.' }
