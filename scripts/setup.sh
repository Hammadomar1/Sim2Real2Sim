#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/so101-m1"
uv python install 3.12
uv sync --python 3.12
uv run python -c 'import torch,mujoco,warp,mujoco_warp,mjlab; print("MuJoCo",mujoco.__version__); print("Torch",torch.__version__); print("GPU",torch.cuda.get_device_name(0)); warp.init(); a=torch.randn(128,128,device="cuda"); assert torch.isfinite(a@a).all(); print("CUDA math OK")'

