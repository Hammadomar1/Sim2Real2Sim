"""Episode-level evaluation shared by training, the CLI evaluator and the baselines.

A fixed seed fixes every scenario (start pose, goal, physical parameters,
sensor noise), so different agents can be compared on identical scenes.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch

from .env import EnvConfig, PushEnv
from .scripted import KeypointPusher
from .tasks import make_env

AgentFactory = Callable[[PushEnv], Callable[[dict], np.ndarray]]


def policy_agent(policy, device, deterministic: bool = True) -> AgentFactory:
    def factory(env):
        @torch.no_grad()
        def act(obs):
            x = torch.as_tensor(obs["actor"], device=device)
            a = policy.act_deterministic(x) if deterministic else policy.distribution(x).sample()
            return a.clamp(-1, 1).cpu().numpy()
        return act
    return factory


def scripted_agent() -> AgentFactory:
    def factory(env):
        ctl = KeypointPusher(env)
        return lambda obs: ctl.act()
    return factory


def summarize(eps: dict) -> dict:
    if len(eps.get("success", [])) == 0:
        return {"episodes": 0}
    s = {k: np.concatenate(v) for k, v in eps.items()}
    reached = s["first_success"] > 0
    pos, yaw = s["pos_err"], s["yaw_err"]
    return {
        "episodes": int(len(s["success"])),
        "success": float(s["success"].mean()),
        "failure": float(s["failed"].mean()),
        "tipped": float(s["tipped"].mean()),
        "outside": float(s["outside"].mean()),
        "nonfinite": float(s["nonfinite"].mean()),
        "pos_err_mm_median": float(np.median(pos) * 1e3),
        "yaw_err_deg_median": float(np.degrees(np.median(yaw))),
        "pos_err_mm_p90": float(np.percentile(pos, 90) * 1e3),
        "yaw_err_deg_p90": float(np.degrees(np.percentile(yaw, 90))),
        # Proposal's normalised pose error E = e_pos / 20 mm + e_yaw / 10 deg.
        "E_mean": float(np.mean(pos / 0.020 + np.degrees(yaw) / 10.0)),
        "reached": float(reached.mean()),
        "time_to_goal_s_median": float(np.median(s["first_success"][reached]) * 0.05) if reached.any() else float("nan"),
        "difficulty_mean": float(s["difficulty"].mean()),
        **({"passed_gate": float(s["passed"].mean())} if "passed" in s else {}),
        **({"disturbed": float(s["disturbed"].mean()),
            "disturbed_when_in_way": float(s["disturbed"][s["in_way"]].mean()) if s["in_way"].any() else float("nan"),
            "clutter_moved_mm_median": float(np.median(s["clutter_moved_mm"]))} if "disturbed" in s else {}),
    }


def run_episodes(env_cfg: EnvConfig, agent: AgentFactory, episodes: int, seed: int,
                 difficulty: float = 1.0, record: bool = False) -> dict:
    """Run exactly one episode in each of ``episodes`` worlds."""
    cfg = copy.deepcopy(env_cfg)
    cfg.num_envs, cfg.seed = episodes, seed
    cfg.task.difficulty, cfg.task.easy_fraction = difficulty, 0.0
    env = make_env(cfg)
    act = agent(env)
    goals, starts = env.goal.copy(), env.object_pose().copy()
    params = env.params.copy()
    done_once = np.zeros(env.n, dtype=bool)
    eps, rows = {}, [None] * env.n
    rate = np.zeros(env.n)
    prev_a = np.zeros((env.n, 2))
    obs = env.obs
    for _ in range(env.max_steps):
        a = act(obs)
        rate += np.where(done_once, 0.0, ((a - prev_a) ** 2).sum(-1))
        prev_a = a
        obs, _, _, info = env.step(a)
        if "episode" in info:
            ids = info["terminal_ids"]
            fresh = ~done_once[ids]
            for k, v in info["episode"].items():
                eps.setdefault(k, []).append(np.asarray(v)[fresh])
            if record:
                for j, i in enumerate(ids[fresh]):
                    rows[i] = {k: np.asarray(v)[fresh][j].item() for k, v in info["episode"].items()}
            done_once[ids] = True
        if done_once.all():
            break
    out = summarize(eps)
    out["action_rate_mean"] = float(rate.mean() / env.max_steps)
    if record:
        for i, r in enumerate(rows):
            r.update(start=starts[i].tolist(), goal=goals[i].tolist(),
                     table_mu=float(params[i, 0]), pusher_mu=float(params[i, 1]), mass_scale=float(params[i, 2]))
        out["per_episode"] = rows
    return out
