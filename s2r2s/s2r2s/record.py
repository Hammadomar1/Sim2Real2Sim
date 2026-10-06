"""Render episodes to an MP4 (offscreen), optionally as a grid of simultaneous episodes.

    python -m s2r2s.record --checkpoint runs/tee_v1/best.pt --out videos/tee.mp4 --grid 2
    python -m s2r2s.record --scripted --out videos/scripted.mp4
    python -m s2r2s.record --checkpoint ... --camera d435i      # what the planned real camera sees
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from .env import EnvConfig, PushEnv, TaskConfig
from .evaluation import policy_agent, scripted_agent
from .train import load_policy
from .visual import SceneMirror


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--scripted", action="store_true")
    p.add_argument("--object", default="tee")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=2, help="episodes per tile")
    p.add_argument("--grid", type=int, default=1, help="NxN tiles of different scenarios")
    p.add_argument("--camera", default="front", choices=["front", "d435i", "top"])
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--no-randomization", action="store_true")
    p.add_argument("--kick-every", type=float, default=0.0,
                   help="seconds between random displacements of the object (disturbance demo)")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    if args.checkpoint:
        policy, cfg, _ = load_policy(args.checkpoint, "cpu")
        agent = policy_agent(policy, "cpu")
    else:
        cfg = EnvConfig(task=TaskConfig(objects=(args.object,)))
        agent = scripted_agent()
    tiles = args.grid ** 2
    cfg.num_envs, cfg.num_threads, cfg.seed = tiles, min(tiles, 8), args.seed
    cfg.task.difficulty, cfg.task.easy_fraction = args.difficulty, 0.0
    if args.no_randomization:
        cfg.rand.enabled = False
    env = PushEnv(cfg)
    act = agent(env)
    # The real camera cannot see the policy's estimate marker, so hide it in that view.
    mirrors = [SceneMirror(env, i, show_estimate=args.camera != "d435i") for i in range(tiles)]
    renderers = [mujoco.Renderer(m.model, args.height, args.width) for m in mirrors]
    finished = np.zeros(tiles, dtype=int)
    frames = []
    obs = env.obs
    rng = np.random.default_rng(args.seed)
    kick_steps = round(args.kick_every / cfg.task.control_dt) if args.kick_every else 0
    while finished.min() < args.episodes:
        canvas = []
        for i, (m, r) in enumerate(zip(mirrors, renderers)):
            m.sync()
            r.update_scene(m.data, camera=args.camera)
            img = np.ascontiguousarray(r.render())
            if finished[i] >= args.episodes:
                img = (img * 0.35).astype(np.uint8)
            cv2.rectangle(img, (0, 0), (img.shape[1], 32), (25, 25, 25), -1)
            cv2.putText(img, m.status(), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            canvas.append(img)
        rows = [np.concatenate(canvas[k * args.grid:(k + 1) * args.grid], 1) for k in range(args.grid)]
        frames.append(np.concatenate(rows, 0))
        if kick_steps:
            for i in range(tiles):
                t = env.step_count[i]
                if t > 0 and t % kick_steps == 0 and t < env.max_steps - kick_steps:
                    pose = env.object_pose()[i]
                    for _ in range(20):   # keep clear of the rod and inside the start region
                        shift, turn = rng.uniform(-0.02, 0.02, 2), rng.uniform(-0.8, 0.8)
                        new = np.r_[pose[:2] + shift, pose[2] + turn][None]
                        gap = env._surface_distance(env.tip[i:i + 1, :2], new, env.obj_id[i:i + 1])[0]
                        if gap > 0.01 and env._in_region(new[:, :2], env.cfg.task.obj_r, env.cfg.task.obj_az)[0]:
                            env.displace_object(i, shift, turn)
                            break
            obs = env.obs
        obs, _, done, _ = env.step(act(obs))
        finished += done
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=round(1 / cfg.task.control_dt), quality=7, macro_block_size=8)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) * cfg.task.control_dt:.0f} s)")


if __name__ == "__main__":
    main()
