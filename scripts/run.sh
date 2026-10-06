#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/so101-m1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
exec uv run --frozen so101-m1 "$@"
