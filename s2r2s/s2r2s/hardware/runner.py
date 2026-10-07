"""Run a trained pushing policy on a robot in a 20 Hz loop, with safety limits and trial logs.

    python -m s2r2s.hardware.runner --checkpoint runs/tee_v1/best.pt --robot sim --episodes 20       # dry run
    python -m s2r2s.hardware.runner --checkpoint runs/gate_pilot/best.pt --robot sim --episodes 20
    python -m s2r2s.hardware.runner --checkpoint runs/tee_v1/best.pt --robot so101 --port COM5 \\
        --map calibration/joint_map.json --goal 0.20 0.05 90         # real arm (needs the camera)

The policy, the controller (action -> commanded rod position -> IK) and the observation code are
the ones used in training; only the measurements come from outside (joint encoders, camera).
Safety: joint targets move at most --max-step-deg per tick from the measured joints, the rod stays
inside the reachable annulus at its fixed height, and the arm holds still when the object has not
been seen for --max-pose-age seconds. Ctrl+C lifts the rod and exits.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..env import wrap
from ..evaluate import TEST_SEED
from ..tasks import make_env
from ..train import load_policy
from .joint_map import JointMap
from .poses import CameraPoseSource, SimPoseSource
from .robots import SimRobot, SO101Robot

PROJECT = Path(__file__).resolve().parents[2]


def move_to(robot, jmap: JointMap, kin, xyz, seconds: float = 2.0, dt: float = 0.05, realtime: bool = True):
    """Move the rod tip to ``xyz`` (vertical rod) along a joint-space line, slowly."""
    q0 = jmap.to_sim(robot.read_joints())
    target, err, _ = kin.solve(np.array([xyz]), q0[None], iterations=30)
    if err[0] > 2e-3:
        raise ValueError(f"cannot reach {xyz}")
    for q in np.linspace(q0, target[0], max(2, round(seconds / dt))):
        t0 = time.monotonic()
        robot.send_joints(jmap.to_robot(q))
        if realtime:
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))


class PolicyRunner:
    def __init__(self, checkpoint, robot, poses, jmap: JointMap, device: str = "cpu",
                 max_step_deg: float = 4.0, max_pose_age: float = 0.5, realtime: bool = True):
        self.policy, cfg, _ = load_policy(checkpoint, device)
        cfg = copy.deepcopy(cfg)
        cfg.num_envs, cfg.num_threads = 1, 1
        cfg.rand.enabled = False          # real measurements replace the simulated camera and physics randomisation
        self.cfg, self.device = cfg, device
        self.twin = make_env(cfg)         # runs the controller and observation code; no physics is stepped
        self.robot, self.poses, self.jmap = robot, poses, jmap
        self.max_step = math.radians(max_step_deg)
        self.max_pose_age, self.realtime = max_pose_age, realtime
        self.dt = cfg.task.control_dt

    @property
    def clutter(self) -> bool:
        """A clutter policy: it also needs the camera's pose of the block that must stay in place."""
        return hasattr(self.twin, "clutter_home")

    @staticmethod
    def _first_pose(read, what, timeout):
        deadline = time.monotonic() + timeout
        pose, _ = read()
        while pose is None and time.monotonic() < deadline:
            pose, _ = read()
        if pose is None:
            raise RuntimeError(f"no {what} pose from the camera")
        return pose

    def start(self, goal, gate: dict | None = None, timeout: float = 2.0):
        """Begin a trial: goal (x, y, yaw) and, for gate policies, the gate (centre, angle, width, side).

        Clutter policies: where the block is now is where it must still be at the end.
        """
        self.twin.set_goal(np.asarray(goal, float))
        pose = self._first_pose(self.poses.read, "object", timeout)
        extra = {"block": self._first_pose(self.poses.read_block, "block", timeout)[None]} if self.clutter else {}
        if gate is not None:
            self.twin.configure_gate(gate["centre"], gate["angle"], gate["width"], gate["side"], start_yaw=pose[2])
        q = self.jmap.to_sim(self.robot.read_joints())
        self.twin.sync_measurements(q[None], pose[None], first=True, **extra)
        self.last_seen = time.monotonic()
        self.log = []

    def tick(self):
        t0 = time.monotonic()
        q = self.jmap.to_sim(self.robot.read_joints())
        pose, _ = self.poses.read()
        if pose is not None:
            self.last_seen = t0
        stale = (t0 - self.last_seen) > self.max_pose_age
        extra = {}
        if self.clutter:                 # a missing block estimate keeps the last one
            block, _ = self.poses.read_block()
            extra["block"] = None if block is None else block[None]
        obs = self.twin.sync_measurements(q[None], None if pose is None else pose[None], **extra)
        if stale:                        # object lost: hold still
            a, target = np.zeros(2), q
        else:
            with torch.no_grad():
                action = self.policy.act_deterministic(torch.as_tensor(obs, device=self.device)).cpu().numpy()
            a, q_cmd = self.twin.command(action)
            self.twin.prev_action[:] = a
            a, target = a[0], np.clip(q_cmd[0], q - self.max_step, q + self.max_step)
        self.robot.send_joints(self.jmap.to_robot(target))
        self.log.append(dict(t=t0, q=q.tolist(), target=target.tolist(), tip=self.twin.tip[0].tolist(),
                             cmd=self.twin.cmd_xy[0].tolist(), pose=None if pose is None else pose.tolist(),
                             action=np.asarray(a).tolist(), stale=bool(stale),
                             stage=int(self.twin.stage[0]) if hasattr(self.twin, "stage") else -1,
                             **({"block": None if extra["block"] is None else extra["block"][0].tolist()}
                                if self.clutter else {})))
        if self.realtime:
            time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))

    def errors(self, pose):
        """Final position (m) and symmetry-aware yaw (rad) errors of ``pose`` against the goal."""
        g = self.twin.goal[0]
        yaw_goal = self.twin._symmetric_goal_yaw(np.array([pose[2]]), np.array([g[2]]), self.twin.obj_id[:1])[0]
        return float(np.linalg.norm(pose[:2] - g[:2])), float(abs(wrap(pose[2] - yaw_goal)))

    def block_moved(self, block, home=None):
        """Clutter policies: how far the block at ``block`` is from ``home``, by default where the camera saw it
        at the start (m, rad). In simulation, pass the true start pose: the camera's start estimate is noisy."""
        home = self.twin.clutter_home[0] if home is None else np.asarray(home)
        return (float(np.linalg.norm(block[:2] - home[:2])),
                float(self.twin._clutter_yaw_err(np.array([block[2]]), np.array([home[2]]))[0]))


def save_trial(path: Path, runner: PolicyRunner, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({**meta, "ticks": runner.log}, indent=1))


def dry_run(args):
    """Episodes on SimRobot through the full hardware path (joint map, pose source, safety limits)."""
    _, cfg, _ = load_policy(args.checkpoint, "cpu")
    scenes_cfg = copy.deepcopy(cfg)
    scenes_cfg.num_envs, scenes_cfg.seed, scenes_cfg.num_threads = args.episodes, args.seed, 4
    scenes_cfg.task.difficulty, scenes_cfg.task.easy_fraction = 1.0, 0.0
    scenes = make_env(scenes_cfg)            # only used to draw the same kind of scenes as training
    hidden = JointMap.load(args.hidden_map) if args.hidden_map else JointMap()
    jmap = JointMap.load(args.map) if args.map else JointMap()
    robot = SimRobot(cfg.scene, cfg.task.objects[0], hidden=hidden, encoder_noise_deg=0.05)
    gate_task, clutter_task = hasattr(scenes, "gate_centre"), hasattr(scenes, "clutter_home")
    runner = PolicyRunner(args.checkpoint, robot, None, jmap, max_step_deg=args.max_step_deg, realtime=False)
    stats = []
    for i in range(args.episodes):
        robot.place(scenes.tip[i, :2], scenes.object_pose()[i])
        gate = None
        if gate_task:
            gate = dict(centre=scenes.gate_centre[i], angle=scenes.gate_angle[i], width=scenes.gate_width[i],
                        side=scenes.gate_side[i])
            robot.place_gate(gate["centre"], gate["angle"], gate["width"])
        if clutter_task:
            robot.place_block(scenes.clutter_home[i])
        runner.poses = SimPoseSource(robot, latency_steps=1, seed=args.seed + i)
        runner.start(scenes.goal[i], gate)
        for _ in range(scenes.max_steps):
            runner.tick()
        pos, yaw = runner.errors(robot.object_pose())
        ok = pos <= cfg.task.success_pos and yaw <= cfg.task.success_yaw
        moved, turned = runner.block_moved(robot.block_pose(), scenes.clutter_home[i]) if clutter_task else (0.0, 0.0)
        kept = moved <= cfg.task.clutter_tol and turned <= cfg.task.clutter_yaw_tol
        ok = ok and kept
        stats.append((ok, pos, yaw, runner.twin.stage[0] == 2 if gate_task else kept))
        if args.log_dir:
            save_trial(Path(args.log_dir) / f"sim_{i:03d}", runner, dict(
                checkpoint=args.checkpoint, goal=scenes.goal[i].tolist(), start=scenes.object_pose()[i].tolist(),
                tool_start=scenes.tip[i, :2].tolist(),
                gate=None if gate is None else {k: np.asarray(v).tolist() for k, v in gate.items()},
                final=robot.object_pose().tolist(), pos_err=pos, yaw_err=yaw, success=bool(ok),
                **({"block_home": scenes.clutter_home[i].tolist(), "block_moved": moved} if clutter_task else {})))
        print(f"episode {i + 1:3d}: {'success' if ok else 'miss   '} pos {pos * 1e3:5.1f} mm  yaw {math.degrees(yaw):5.1f} deg"
              + (f"  passed gate {bool(stats[-1][3])}" if gate_task else "")
              + (f"  block moved {moved * 1e3:4.1f} mm" if clutter_task else ""), flush=True)
    s = np.array(stats, dtype=float)
    print(f"\n{args.episodes} episodes through the hardware path: success {s[:, 0].mean():.1%}, "
          f"median {np.median(s[:, 1]) * 1e3:.1f} mm / {math.degrees(np.median(s[:, 2])):.1f} deg"
          + (f", passed gate {s[:, 3].mean():.1%}" if gate_task else "")
          + (f", block kept in place {s[:, 3].mean():.1%}" if clutter_task else ""))


def real_run(args):
    jmap = JointMap.load(args.map)
    robot = SO101Robot(args.port, max_step_deg=args.max_step_deg)
    try:
        poses = CameraPoseSource()
    except NotImplementedError as error:
        robot.close()
        raise SystemExit(f"{error}\nUntil then, rehearse with --robot sim.")
    runner = PolicyRunner(args.checkpoint, robot, poses, jmap, max_step_deg=args.max_step_deg)
    kin = runner.twin.kin
    goal = np.array([args.goal[0], args.goal[1], math.radians(args.goal[2])])
    try:
        move_to(robot, jmap, kin, (*args.tool_start, 0.03))                    # above the start point
        move_to(robot, jmap, kin, (*args.tool_start, runner.cfg.task.tool_height), seconds=1.0)
        runner.start(goal)
        for _ in range(round(runner.cfg.task.episode_seconds / runner.dt)):
            runner.tick()
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        tip = runner.twin.tip[0]
        move_to(robot, jmap, kin, (tip[0], tip[1], 0.04), seconds=1.0)      # lift the rod clear
        pose, _ = poses.read()
        meta = dict(checkpoint=args.checkpoint, goal=goal.tolist(), final=None if pose is None else pose.tolist())
        if pose is not None:
            meta["pos_err"], meta["yaw_err"] = runner.errors(pose)
        save_trial(Path(args.log_dir or PROJECT / "trials") / time.strftime("%Y%m%d_%H%M%S"), runner, meta)
        robot.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="", help="joint map from calibrate.py (required on the real arm)")
    ap.add_argument("--max-step-deg", type=float, default=4.0)
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--episodes", type=int, default=20, help="sim dry run")
    ap.add_argument("--seed", type=int, default=TEST_SEED, help="sim dry run: scene seed")
    ap.add_argument("--hidden-map", default="", help="sim dry run: the simulated arm's true (miscalibrated) map")
    ap.add_argument("--goal", type=float, nargs=3, default=(0.20, 0.05, 90.0), metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--tool-start", type=float, nargs=2, default=(0.15, -0.10), metavar=("X", "Y"))
    args = ap.parse_args(argv)
    if args.robot == "sim":
        dry_run(args)
    else:
        if not args.map:
            ap.error("--map is required on the real arm (run calibrate.py first)")
        real_run(args)


if __name__ == "__main__":
    main()
