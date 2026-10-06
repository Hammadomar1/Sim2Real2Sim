# Live training curves at http://localhost:6006
Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\tensorboard.exe --logdir runs
