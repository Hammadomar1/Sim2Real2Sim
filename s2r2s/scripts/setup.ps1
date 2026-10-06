# One-time setup: fetches the SO-101 model, creates .venv with Python 3.12, MuJoCo 3.11, PyTorch (CUDA 12.8)
# and the s2r2s package, then runs the tests.
# Needs git and uv (https://docs.astral.sh/uv/). Without an NVIDIA GPU, PyTorch falls back to CPU automatically.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# The SO-101 model comes from MuJoCo Menagerie, pinned to the revision used for Milestone 1. It is a sparse,
# partial clone (only robotstudio_so101 is downloaded) and is not tracked by this repository.
$menagerie = Join-Path (Split-Path -Parent (Get-Location).Path) 'assets\menagerie'
if (-not (Test-Path (Join-Path $menagerie 'robotstudio_so101\so101.xml'))) {
    git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git $menagerie
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone MuJoCo Menagerie' }
    git -C $menagerie sparse-checkout set robotstudio_so101
    if ($LASTEXITCODE -ne 0) { throw 'Could not configure the Menagerie sparse checkout' }
    git -C $menagerie checkout 4d038b3feae26ec82b46a4d586379114012a8ac7
    if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned Menagerie revision' }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv not found: install it with "pip install uv" or from https://docs.astral.sh/uv/' }
uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
& .\.venv\Scripts\python.exe -c "import mujoco, torch; print('mujoco', mujoco.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
& .\.venv\Scripts\python.exe -m pytest -q
