"""Where should the real camera go?  Measure arm occlusion of the object from candidate mounts.

    python -m s2r2s.camera_study --checkpoint runs/tee_v1/best.pt
    python -m s2r2s.camera_study --scripted

Runs pushing episodes and, at regular intervals, casts rays from each candidate camera to
a grid of points on the object's top face. A point is occluded when robot geometry (arm
meshes or the pusher) blocks the line of sight. Ray casting is exact and does not depend on
the GPU. Also reports ground resolution (mm per pixel) and saves a contact sheet of what
each camera sees.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from .env import EnvConfig, PushEnv, TaskConfig
from .evaluation import policy_agent, scripted_agent
from .objects import OBJECTS
from .scene import _look_at_quat, build_spec, set_marker
from .train import load_policy

TARGET = (0.20, 0.0, 0.0)
# Candidate mounts (position in the robot base frame, metres). All look at the workspace centre.
CANDIDATES = {
    "front_high": (0.46, 0.00, 0.62),    # tripod opposite the robot, ~67 deg elevation (default)
    "overhead": (0.20, 0.00, 0.70),      # straight down above the workspace
    "front_low": (0.62, 0.00, 0.38),     # opposite the robot, ~42 deg elevation
    "side_left": (0.20, 0.42, 0.55),     # robot's left side
    "corner": (0.50, 0.32, 0.55),        # front-left corner
    "behind_robot": (-0.15, 0.00, 0.60), # over the robot's shoulder
}
FOVY = 42.5           # RealSense D435i colour stream (69 x 42 deg); a Logitech C920 is similar (~70 x 43)


def build_study_model(cfg, object_name):
    spec = build_spec(cfg.scene, object_name, visual=True)
    for name, pos in CANDIDATES.items():
        spec.worldbody.add_camera(name=f"cand_{name}", pos=list(pos), quat=list(_look_at_quat(pos, TARGET)), fovy=FOVY)
    model = spec.compile()
    model.body_sameframe[model.body("object").id] = 0
    return model


def top_face_points(shape, n=5):
    """Grid of points (object frame) 0.5 mm above the object's top face."""
    z = shape.height + 0.0005
    pts = []
    if shape.disk_radius > 0:
        for r in np.linspace(0, shape.disk_radius * 0.9, n):
            for a in np.linspace(0, 2 * math.pi, 2 * n, endpoint=False):
                pts.append((r * math.cos(a), r * math.sin(a), z))
    for cx, cy, hx, hy in shape.centred_boxes:
        for u in np.linspace(-0.9, 0.9, n):
            for v in np.linspace(-0.9, 0.9, n):
                pts.append((cx + u * hx, cy + v * hy, z))
    return np.array(pts)


def ground_resolution(pos, height_px, fovy_deg=FOVY):
    """mm per pixel at the workspace centre for a pinhole camera."""
    dist = math.dist(pos, TARGET)
    focal = height_px / 2 / math.tan(math.radians(fovy_deg) / 2)
    return dist / focal * 1e3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--scripted", action="store_true")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--every", type=int, default=2, help="check every k-th control step")
    p.add_argument("--out", default="videos/camera_study")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    if args.checkpoint:
        policy, cfg, _ = load_policy(args.checkpoint, "cpu")
        agent = policy_agent(policy, "cpu")
    else:
        cfg, agent = EnvConfig(task=TaskConfig(objects=("tee",))), scripted_agent()
    cfg.num_envs, cfg.num_threads, cfg.seed = args.episodes, 8, 77
    cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
    env = PushEnv(cfg)
    act = agent(env)
    shape = OBJECTS[cfg.task.objects[0]]
    model = build_study_model(cfg, shape.name)
    data = mujoco.MjData(model)
    obj_geoms = np.array([i for i in range(model.ngeom) if model.geom(i).name.startswith("object_")])
    set_marker(model, data, "estimate", (5.0, 5.0, 0.0))     # keep the estimate marker out of the rays
    local = top_face_points(shape)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)    # objects, pusher, robot meshes
    geomid = np.zeros(len(local), dtype=np.int32)
    dist = np.zeros(len(local))
    cams = {name: np.array(pos) for name, pos in CANDIDATES.items()}
    occl = {k: [] for k in CANDIDATES}
    nq = env.idx.nq
    obs = env.obs
    for step in range(env.max_steps):
        obs, _, done, _ = env.step(act(obs))
        if step % args.every:
            continue
        poses = env.object_pose()
        for i in range(env.n):
            data.qpos[:] = env.state[i, 1:1 + nq]
            mujoco.mj_kinematics(model, data)
            c, s = math.cos(poses[i, 2]), math.sin(poses[i, 2])
            world = np.c_[poses[i, 0] + c * local[:, 0] - s * local[:, 1],
                          poses[i, 1] + s * local[:, 0] + c * local[:, 1], local[:, 2]]
            for name, cam in cams.items():
                vec = (world - cam).ravel()
                mujoco.mj_multiRay(model, data, cam, vec, groups, True, -1, geomid, dist, None, len(local), 10.0)
                blocked = (dist >= 0) & (dist < 0.999) & ~np.isin(geomid, obj_geoms)
                occl[name].append(blocked.mean())
    rows = {}
    print(f"\nArm occlusion of the object's top face, {env.n} episodes, sampled every {args.every * 50} ms "
          f"({len(occl['overhead'])} views per camera):")
    print(f"{'camera':14s}{'mean':>8s}{'>25%':>8s}{'>50%':>8s}{'>90%':>8s}{'mm/px 720p':>12s}{'mm/px 1080p':>13s}")
    for name, v in sorted(occl.items(), key=lambda kv: np.mean(kv[1])):
        v = np.array(v)
        rows[name] = dict(mean=float(v.mean()), over25=float((v > 0.25).mean()), over50=float((v > 0.5).mean()),
                          over90=float((v > 0.9).mean()), mm_px_720=ground_resolution(CANDIDATES[name], 720),
                          mm_px_1080=ground_resolution(CANDIDATES[name], 1080), position=CANDIDATES[name])
        q = rows[name]
        print(f"{name:14s}{q['mean']:8.1%}{q['over25']:8.1%}{q['over50']:8.1%}{q['over90']:8.1%}"
              f"{q['mm_px_720']:12.2f}{q['mm_px_1080']:13.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "occlusion.json").write_text(json.dumps(rows, indent=1))
    # Contact sheet: what each candidate camera sees at the current (mid-push) state of world 0.
    r = mujoco.Renderer(model, 360, 640)
    data.qpos[:] = env.state[0, 1:1 + nq]
    set_marker(model, data, "goal", env.goal[0])
    mujoco.mj_forward(model, data)
    tiles = []
    for name in CANDIDATES:
        r.update_scene(data, camera=f"cand_{name}")
        img = np.ascontiguousarray(r.render())
        label = f"{name}: {rows[name]['mean']:.1%} mean occlusion"
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (25, 25, 25), -1)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    sheet = np.concatenate([np.concatenate(tiles[k:k + 3], 1) for k in (0, 3)], 0)
    cv2.imwrite(str(out / "camera_views.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {out / 'occlusion.json'} and {out / 'camera_views.png'}")


if __name__ == "__main__":
    main()
