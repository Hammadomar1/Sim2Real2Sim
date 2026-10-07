"""Watch a policy (or the scripted baseline) push objects in the MuJoCo viewer, in real time.

    python -m s2r2s.play --checkpoint runs/tee_v1/best.pt
    python -m s2r2s.play --scripted
    python -m s2r2s.play --scripted --gate        # Level 1: push the T through a gate

Keys: P pause/resume | N new episode | K kick the object (disturbance) | F faster/slower
Green outline = goal. Yellow outline = the policy's camera estimate (delayed and noisy).
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from .env import EnvConfig, TaskConfig
from .evaluation import policy_agent, scripted_agent
from .tasks import make_env as build_env
from .train import load_policy
from .visual import SceneMirror


def make_env(args):
    if args.checkpoint:
        policy, cfg, _ = load_policy(args.checkpoint, "cpu")
        agent = policy_agent(policy, "cpu", deterministic=not args.stochastic)
    else:
        cfg = EnvConfig(task=TaskConfig(objects=(args.object,), gate=args.gate, clutter=args.clutter,
                                        episode_seconds=30.0 if args.gate else 20.0))
        agent = scripted_agent()
    cfg.num_envs, cfg.num_threads, cfg.seed = 1, 1, args.seed
    cfg.task.difficulty, cfg.task.easy_fraction = args.difficulty, 0.0
    if args.object and args.checkpoint and args.object in cfg.task.objects:
        cfg.task.objects = (args.object,)
    if args.no_randomization:
        cfg.rand.enabled = False
    env = build_env(cfg)
    return env, agent(env)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--scripted", action="store_true", help="run the heuristic baseline instead of a policy")
    p.add_argument("--object", default="tee")
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-randomization", action="store_true")
    p.add_argument("--gate", action="store_true", help="with --scripted: the Level 1 gate puzzle")
    p.add_argument("--clutter", default="", help="with --scripted: add a block to keep in place, e.g. box")
    args = p.parse_args(argv)
    if not args.checkpoint and not args.scripted:
        p.error("give --checkpoint or --scripted")
    torch.set_num_threads(1)
    env, act = make_env(args)
    mirror = SceneMirror(env)
    keys = {"pause": False, "new": False, "kick": False, "speed": 1.0}

    def on_key(code):
        c = chr(code) if 32 <= code < 127 else ""
        if c == "P":
            keys["pause"] = not keys["pause"]
        elif c == "N":
            keys["new"] = True
        elif c == "K":
            keys["kick"] = True
        elif c == "F":
            keys["speed"] = 4.0 if keys["speed"] == 1.0 else 1.0

    rng = np.random.default_rng(args.seed)
    episode = 1
    with mujoco.viewer.launch_passive(mirror.model, mirror.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = [0.19, 0.0, 0.02]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.62, 200.0, -42.0
        obs = env.obs
        while viewer.is_running():
            t0 = time.time()
            if keys["new"]:
                keys["new"] = False
                obs = env.reset_worlds([0])
                episode += 1
            if keys["kick"]:
                keys["kick"] = False
                env.displace_object(0, rng.uniform(-0.025, 0.025, 2), rng.uniform(-0.6, 0.6))
                obs = env.obs
            if not keys["pause"]:
                obs, _, done, _ = env.step(act(obs))
                if done[0]:
                    episode += 1
            mirror.sync()
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"episode {episode}", mirror.status() + ("  [paused]" if keys["pause"] else "")))
            viewer.sync()
            time.sleep(max(0.0, env.cfg.task.control_dt / keys["speed"] - (time.time() - t0)))


if __name__ == "__main__":
    main()
