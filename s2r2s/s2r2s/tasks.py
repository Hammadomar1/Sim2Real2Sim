"""Pick the environment class for a configuration."""
from __future__ import annotations

from .clutter import ClutterEnv
from .env import EnvConfig, PushEnv
from .gate import GateEnv


def make_env(cfg: EnvConfig) -> PushEnv:
    if cfg.task.gate and cfg.task.clutter:
        raise ValueError("gate and clutter tasks cannot be combined yet")
    if cfg.task.gate:
        return GateEnv(cfg)
    if cfg.task.clutter:
        return ClutterEnv(cfg)
    return PushEnv(cfg)
