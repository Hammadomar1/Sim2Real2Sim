"""Pick the environment class for a configuration."""
from __future__ import annotations

from .env import EnvConfig, PushEnv
from .gate import GateEnv


def make_env(cfg: EnvConfig) -> PushEnv:
    return GateEnv(cfg) if cfg.task.gate else PushEnv(cfg)
