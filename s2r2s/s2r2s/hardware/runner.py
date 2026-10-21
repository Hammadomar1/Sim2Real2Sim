"""Run a trained pushing policy on a robot in a 20 Hz loop, with safety limits and trial logs.

    python -m s2r2s.hardware.runner --checkpoint runs/tee_v1/best.pt --robot sim --episodes 20       # dry run
    python -m s2r2s.hardware.runner --checkpoint runs/gate_pilot/best.pt --robot sim --episodes 20
    python -m s2r2s.hardware.runner --checkpoint runs/tee_lightbox_v1/best.pt --robot sim --camera render   # C920 pipeline
    python -m s2r2s.hardware.runner --checkpoint runs/tee_v1/best.pt --robot so101 --port COM5 \\
        --map calibration/joint_map.json --goal 0.20 0.05 90         # real arm (needs the camera)
    python -m s2r2s.hardware.runner --level 2 --robot sim --episodes 50                    # Levels 2 and 3 (levels.py)
    python -m s2r2s.hardware.runner --level 2 --robot sim --camera render --mount back --episodes 8   # C920 on the back wall

The policy, the controller (action -> commanded rod position -> IK) and the observation code are
the ones used in training; only the measurements come from outside (joint encoders, camera).
Safety: joint targets move at most --max-step-deg per tick from the measured joints and the rod stays
inside the reachable annulus at its fixed height. When the object has not been seen for --max-pose-age
seconds (2 s: the policy closes most shorter gaps itself by pushing on), the runner lifts the rod and
moves it a few cm out of the camera's way until the object is seen again, then hands back to the policy
(--on-lost hold: hold still instead); after --recover-seconds it gives up. A trial whose start position
hides the object (or the clutter block) starts from a clear one nearby, after flying the rod (up to 40 mm high)
out of the camera's way to find them. Ctrl+C lifts the rod and exits.
Servo targets go out at about 100 Hz (--ramp-hz), stepping evenly from one tick's target to the next as in
training (TaskConfig.servo_ramp); --no-ramp sends one jump per tick, which the servos take at full speed.
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


def load_skill(checkpoint, device: str = "cpu", lightbox: bool = False):
    """A policy and its twin: the training environment with one world, randomisation off (real measurements
    replace the simulated camera and physics), stepped by no physics. Returns (policy, cfg, twin)."""
    policy, cfg, _ = load_policy(checkpoint, device)
    cfg = copy.deepcopy(cfg)
    cfg.scene.lightbox = cfg.scene.lightbox or lightbox     # the kinematics must know where the work surface is
    cfg.num_envs, cfg.num_threads = 1, 1
    cfg.rand.enabled = False
    return policy, cfg, make_env(cfg)


class PolicyRunner:
    """The policy in the loop, plus what the runner does by itself when the camera loses the object.

    ``on_lost="recover"`` (default): once no pose has arrived for ``max_pose_age`` (2 s) the runner takes over the rod.
    It lifts it to ``lift`` above the mat and moves it a few centimetres away from the last known object pose,
    inside the reach ring (``_retreat_point``: toward the robot and away from the object, or, when the pose source
    can predict the arm's shadow (``view``, ``arm_mask.ViewPredictor``), the nearest point from which the camera
    would see the object again). There it waits for the camera, lowers the rod clear of the object and hands back
    to the policy with fresh measurements. If the object stays unseen it tries further points, and after
    ``recover_seconds`` it gives up: the rod stays lifted and ``tick`` returns ``"lost"``. ``on_lost="hold"``:
    hold the arm still instead (the old behaviour; with the left-wall C920 it can freeze where it hides the T).

    A trial that cannot start because the arm hides the object (or the clutter block) at its start position
    (``start``) moves the rod to the pose that hides least of the band, flying as high as a vertical rod reaches
    (up to ``start_lift``) and around what the camera does see, finds them, then starts from the nearest clear
    point from which the object stays in view.
    """

    def __init__(self, checkpoint, robot, poses, jmap: JointMap, device: str = "cpu",
                 max_step_deg: float = 4.0, max_pose_age: float = 2.0, realtime: bool = True, lightbox: bool = False,
                 on_lost: str = "recover", recover_seconds: float = 8.0, lift: float = 0.025,
                 start_lift: float = 0.04, ramp: bool = True, ramp_hz: float = 100.0):
        # The twin runs the controller and observation code; no physics is stepped.
        self.policy, cfg, self.twin = load_skill(checkpoint, device, lightbox)
        self.cfg, self.device = cfg, device
        self.robot, self.poses, self.jmap = robot, poses, jmap
        self.max_step = math.radians(max_step_deg)
        self.max_pose_age, self.realtime = max_pose_age, realtime
        # The camera sees the object now (to lower the rod, hand back, begin a trial): a pose at most this old.
        self.fresh_age = min(max_pose_age, 0.5)
        self.dt = cfg.task.control_dt
        # Servo targets: each tick's move goes out in equal steps at about ``ramp_hz`` (TaskConfig.servo_ramp, as in
        # training) instead of one jump that the servos take at full speed, 20 times a second.
        self.ramp_steps = max(1, round(self.dt * ramp_hz)) if ramp else 1
        self.last_target = None
        if on_lost not in ("recover", "hold"):
            raise ValueError("on_lost: 'recover' or 'hold'")
        self.on_lost, self.recover_seconds, self.lift, self.start_lift = on_lost, recover_seconds, lift, start_lift
        self.rod_speed = cfg.task.max_speed           # the policy's own speed limit, for the runner's moves too
        self.ticks, self.mode, self.log = 0, "policy", []
        self.events: list[dict] = []              # one per recovery: where, why, how it ended
        self._park = self._sp_goal = self._sp_since = None
        self.last_seen = self.block_seen = self.clear_t0 = 0.0
        self.start_shift = 0.0

    @property
    def clutter(self) -> bool:
        """A clutter policy: it also needs the camera's pose of the block that must stay in place."""
        return hasattr(self.twin, "clutter_home")

    def _first_poses(self, timeout):
        """The object's pose and, for clutter policies, the block's, from frames within ``timeout`` (each None
        while unseen; ``read_block`` gives the block in the frame ``read`` took)."""
        deadline = time.monotonic() + timeout
        pose = block = None
        while True:
            p, _ = self.poses.read()
            pose = pose if p is None else p
            if self.clutter:
                b, _ = self.poses.read_block()
                block = block if b is None else b
            if (pose is not None and (block is not None or not self.clutter)) or time.monotonic() >= deadline:
                return pose, block

    def start(self, goal, gate: dict | None = None, timeout: float = 2.0, clear_seconds: float = 10.0):
        """Begin a trial: goal (x, y, yaw) and, for gate policies, the gate (centre, angle, width, side).

        Clutter policies: where the block is now is where it must still be at the end. When the camera does not
        see the object, or the block, within ``timeout`` (the arm hides it), the rod is moved to a clear start
        first (``_clear_start``); RuntimeError if they stay unseen.
        """
        self.twin.set_goal(np.asarray(goal, float))
        self.log, self.events, self.mode = [], [], "policy"
        self.start_shift, self.block_seen, self.clear_t0 = 0.0, -1e9, self._clock()
        self.last_target = self.jmap.to_sim(self.robot.read_joints())      # the first ramp starts where the arm is
        self._tell_arm(self.last_target)
        pose, block = self._first_poses(timeout)
        if pose is None or (self.clutter and block is None):
            pose = self._clear_start(gate, clear_seconds, pose, block)
            block = self.twin.clutter_obs[0].copy() if self.clutter else None     # seen on the way
        extra = {}
        if self.clutter:
            extra["block"] = block[None]
        if gate is not None:
            self.twin.configure_gate(gate["centre"], gate["angle"], gate["width"], gate["side"], start_yaw=pose[2])
        q = self.jmap.to_sim(self.robot.read_joints())
        self.twin.sync_measurements(q[None], pose[None], first=True, **extra)
        self.last_seen = self.block_seen = self._clock()

    def _clock(self) -> float:
        """Seconds for the lost-object check: wall time on the real arm, control ticks in a dry run (which runs
        faster or slower than real time)."""
        return time.monotonic() if self.realtime else self.ticks * self.dt

    def _tell_arm(self, q):
        """The camera pose source masks the arm out (arm_mask.py) at these joint angles; the simulated one has no use for them."""
        if hasattr(self.poses, "set_arm"):
            self.poses.set_arm(q)

    def _measure(self):
        """Joints, the camera's object pose (None: no fresh one) and, for clutter policies, the block's; the twin
        is updated with them (a missing estimate keeps the last one)."""
        self.ticks += 1
        q = self.jmap.to_sim(self.robot.read_joints())
        self._tell_arm(q)
        pose, _ = self.poses.read()
        if pose is not None:
            self.last_seen = self._clock()
        extra = {}
        if self.clutter:
            block, _ = self.poses.read_block()
            extra["block"] = None if block is None else block[None]
            if block is not None:
                self.block_seen = self._clock()
        obs = self.twin.sync_measurements(q[None], None if pose is None else pose[None], **extra)
        return q, pose, extra, obs

    def _send(self, t0, q, target, pose, extra, a, stale):
        """Send this tick's joint target: in ``ramp_steps`` equal steps from the last one, spread over the tick."""
        start = q if self.last_target is None else self.last_target
        n = self.ramp_steps
        for k in range(1, n + 1):
            self.robot.send_joints(self.jmap.to_robot(start + (target - start) * k / n), seconds=self.dt / n)
            if self.realtime and k < n:
                time.sleep(max(0.0, t0 + self.dt * k / n - time.monotonic()))
        self.last_target = np.array(target, dtype=float)
        self.log.append(dict(t=t0, q=q.tolist(), target=target.tolist(), tip=self.twin.tip[0].tolist(),
                             cmd=self.twin.cmd_xy[0].tolist(), pose=None if pose is None else pose.tolist(),
                             action=np.asarray(a).tolist(), stale=bool(stale), mode=self.mode,
                             stage=int(self.twin.stage[0]) if hasattr(self.twin, "stage") else -1,
                             **({"block": None if extra["block"] is None else extra["block"][0].tolist()}
                                if self.clutter else {}), **self._log_extra()))
        if self.realtime:
            time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))

    def _log_extra(self) -> dict:
        """More fields for each tick's log entry (``levels.LevelRunner``: the current push)."""
        return {}

    def tick(self) -> str:
        """One control period. Returns the mode: "policy", the recovery's "retreat" / "lower", or "lost" (gave up:
        end the trial)."""
        t0 = time.monotonic()
        q, pose, extra, obs = self._measure()
        stale = (self._clock() - self.last_seen) > self.max_pose_age
        if self.mode == "policy" and stale and self.on_lost == "recover":
            self._begin_recovery(q)
        if self.mode != "policy":
            a, target = np.zeros(2), self._recover(q, pose)
        elif stale:                      # on_lost="hold": hold still
            a, target = np.zeros(2), q
        else:
            with torch.no_grad():
                action = self.policy.act_deterministic(torch.as_tensor(obs, device=self.device)).cpu().numpy()
            a, q_cmd = self.twin.command(action)
            self.twin.prev_action[:] = a
            a, target = a[0], np.clip(q_cmd[0], q - self.max_step, q + self.max_step)
        self._send(t0, q, target, pose, extra, a, stale)
        return self.mode

    # ------------------------------------------------------------------ moving the rod out of the camera's way
    @property
    def view(self):
        """The pose source's shadow predictor (``arm_mask.ViewPredictor``), or None (``SimPoseSource``)."""
        return getattr(self.poses, "view", None)

    def _rod_goal(self, xy, z):
        return np.array([xy[0], xy[1], z])

    def _rod_step(self, q, goal, settle: float = 1.0) -> tuple[np.ndarray, bool]:
        """Joint targets for one tick of a straight move of the rod's setpoint towards ``goal`` (x, y, z) at
        ``rod_speed``, joint steps capped at ``max_step`` like the policy's; and whether the rod is there (within
        3 mm, or ``settle`` seconds after the setpoint got there)."""
        d = goal - self.sp
        step = self.rod_speed * self.dt
        dist = float(np.linalg.norm(d))
        self.sp = goal.copy() if dist <= step else self.sp + d * step / dist
        q_ik, _, _ = self.twin.kin.solve(self.sp[None], self.q_sp[None], iterations=8)
        self.q_sp = q_ik[0]
        if dist > step or self._sp_goal is None or not np.array_equal(goal, self._sp_goal):
            self._sp_goal, self._sp_since = (goal.copy(), self._clock()) if dist <= step else (None, None)
        there = self._sp_goal is not None and (float(np.linalg.norm(self.twin.tip[0] - goal)) < 0.003
                                               or self._clock() - self._sp_since >= settle)
        return np.clip(self.q_sp, q - self.max_step, q + self.max_step), there

    def _start_rod_move(self, q):
        self.sp, self.q_sp = self.twin.kin.forward(q[None])[0][0], q.copy()
        self._sp_goal = self._sp_since = None

    def _gaps(self, xy, pose) -> np.ndarray:
        """Distance from the rod's surface at points ``xy`` (n, 2) to the nearest of the object at ``pose``, the
        clutter block and the gate's walls (m)."""
        tw = self.twin
        n = len(xy)
        gap = tw._surface_distance(xy, np.tile(pose, (n, 1)), tw.obj_id[:1].repeat(n))
        if self.clutter:
            gap = np.minimum(gap, tw._block_clearance(xy, np.tile(tw.clutter_obs[0], (n, 1))) - self.cfg.scene.pusher_radius)
        if hasattr(tw, "wall_gap"):
            gap = np.minimum(gap, tw.wall_gap(xy, np.zeros(n, dtype=int)))
        return gap

    def _clear_of_things(self, xy, pose, path_from=None, clear: float = 0.010) -> np.ndarray:
        """The rod at ``xy`` is at least ``clear`` from the object at ``pose``, the clutter block and the gate's
        walls, and the straight path to it from ``path_from`` passes over none of them (the rod is lifted only
        25 mm: 5 mm over a 20 mm object). The path may start touching something (the rod was pushing it), but
        never comes closer than 3 mm, or than where it started."""
        xy = np.atleast_2d(xy)
        ok = self._gaps(xy, pose) >= clear
        if path_from is not None:
            k = 12
            w = np.linspace(0, 1, k)[None, :, None]
            seg = np.asarray(path_from, float)[None, None, :] * (1 - w) + xy[:, None, :] * w
            floor = min(0.003, float(self._gaps(np.atleast_2d(path_from), pose)[0]) - 0.001)
            ok &= (self._gaps(seg.reshape(-1, 2), pose).reshape(len(xy), k) >= floor).all(1)
        return ok

    def _in_reach(self, xy, margin: float = 0.005):
        t = self.cfg.task
        r, a = np.hypot(xy[:, 0], xy[:, 1]), np.abs(np.arctan2(xy[:, 1], xy[:, 0]))
        return (r >= t.tool_r[0] + margin) & (r <= t.tool_r[1] - margin) & (a <= t.tool_az - 0.02)

    def _ik(self, xy, z, seed):
        q, err, _ = self.twin.kin.solve(np.array([[xy[0], xy[1], z]]), seed[None], iterations=30)
        return q[0], float(err[0])

    def _liftable(self, xy, z: float | None = None) -> np.ndarray:
        """Which points (n, 2) a vertical rod reaches at ``z`` (default ``lift``; the vertical-rod workspace narrows
        with height: at 25 mm r 13.5-27 cm straight ahead, at 40 mm 16-24 cm)."""
        xy = np.atleast_2d(xy)
        _, err, _ = self.twin.kin.solve(np.c_[xy, np.full(len(xy), self.lift if z is None else z)],
                                        self.twin.kin.seed(xy), iterations=30)
        return err < 0.001

    def _lift_z(self, xy, top: float | None = None) -> float:
        """The highest rod height up to ``top`` (default ``lift``) that a vertical rod reaches at ``xy``."""
        xy = np.tile(np.asarray(xy, float), (6, 1))
        tool_z = self.twin.tool_z_cmd[0]
        zs = np.linspace(self.lift if top is None else top, tool_z, 6)
        _, err, _ = self.twin.kin.solve(np.c_[xy, zs], self.twin.kin.seed(xy), iterations=30)
        return float(zs[np.argmax(err < 0.001)]) if (err < 0.001).any() else float(tool_z)

    def _visible(self, q_seed, xy, pose) -> float:
        """Predicted share of the object in view with the rod at ``xy``, the lower of lifted and at tool height
        (1 when the pose source cannot predict)."""
        if self.view is None:
            return 1.0
        out = 1.0
        for z in (self.lift, self.twin.tool_z_cmd[0]):
            q, err = self._ik(xy, z, q_seed)
            if err > 0.002:
                return 0.0
            out = min(out, self.view.visible(q, self.cfg.task.objects[0], pose))
        return out

    def _retreat_point(self, q, tip_xy, pose, tried, distances=(0.03, 0.05, 0.08), need: float = 0.6, path_from=None):
        """Where to take the rod when the object is lost: a few cm from ``tip_xy``, inside the reach ring, clear
        of the object (last known ``pose``), the block and the walls, not near a point ``tried`` already. Nearest
        first, preferring the direction away from the object and toward the robot. With a shadow predictor, the
        first such point from which the camera would see at least ``need`` of the object (else the best one);
        without, the most preferred direction. None when no point qualifies. ``path_from``: where the lifted rod
        comes from (default ``tip_xy``), for the walls; False: any way there (the caller finds one)."""
        away = tip_xy - pose[:2]
        away = away / max(np.linalg.norm(away), 1e-9)
        home = -tip_xy / max(np.linalg.norm(tip_xy), 1e-9)
        u = away + home
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([-home[1], home[0]])
        ang = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        dirs = np.c_[np.cos(ang), np.sin(ang)]
        cands = np.concatenate([tip_xy + d * dirs for d in distances])
        dist = np.repeat(distances, len(dirs))
        pref = 1 - np.tile(dirs @ u, len(distances))                      # 0 along u, 2 against it
        ok = self._in_reach(cands) & self._clear_of_things(
            cands, pose, None if path_from is False else tip_xy if path_from is None else path_from)
        ok[ok] &= self._liftable(cands[ok])
        for p in tried:
            ok &= np.linalg.norm(cands - p, axis=-1) > 0.015
        if not ok.any():
            return None
        order = [k for k in np.argsort(dist + (0.01 if self.view is not None else 0.05) * pref) if ok[k]]
        if self.view is None:
            return cands[order[0]]
        best, best_vis = None, -1.0
        for k in order:
            vis = self._visible(q, cands[k], pose)
            if vis >= need:
                return cands[k]
            if vis > best_vis:
                best, best_vis = cands[k], vis
        return best

    def _begin_recovery(self, q):
        pose = self.twin.obs_pose[0].copy()
        self.mode, self.rec_t0, self.rec_tried, self.rec_wait = "retreat", self._clock(), [], None
        self._start_rod_move(q)
        self.events.append(dict(tick=len(self.log), lost_at=self.twin.tip[0, :2].tolist(), last_pose=pose.tolist()))
        self._next_retreat(q)

    def _next_retreat(self, q):
        """Pick the next retreat point; with none left, stay where the rod is (lifted) until the time runs out."""
        pose = self.twin.obs_pose[0]
        wp = self._retreat_point(q, self.twin.tip[0, :2], pose, self.rec_tried)
        self.wp = self.sp[:2].copy() if wp is None else wp
        self.rec_tried.append(self.wp.copy())
        self.mode, self.rec_wait, self.rec_phase = "retreat", None, "up"
        self.up = np.r_[self.sp[:2], max(self.sp[2], self._lift_z(self.sp[:2]))]   # lift in place first

    def _recover(self, q, pose) -> np.ndarray:
        """One tick of the recovery: retreat (lift, move to the point), wait for the camera, lower, resume."""
        now = self._clock()
        if self.mode == "lost" or now - self.rec_t0 > self.recover_seconds:
            if self.mode != "lost":
                self.events[-1].update(result="lost", seconds=now - self.rec_t0)
            self.mode = "lost"                                   # gave up: hold the rod where it is, lifted
            target, _ = self._rod_step(q, self.sp)
            return target
        tool_z = self.twin.tool_z_cmd[0]
        if self.mode == "retreat" and self.rec_phase == "up":
            target, there = self._rod_step(q, self.up)
            if there:
                self.rec_phase = "across"
            return target
        if self.mode == "retreat":
            target, there = self._rod_step(q, self._rod_goal(self.wp, self.lift if self._liftable(self.wp)[0] else self.up[2]))
            if there:
                self.rec_wait = now if self.rec_wait is None else self.rec_wait
                if now - self.last_seen <= self.fresh_age:         # seen again: lower the rod if it is clear there
                    if self._clear_of_things(self.wp[None], self.twin.obs_pose[0])[0]:
                        self.mode = "lower"
                    else:
                        self._next_retreat(q)
                elif now - self.rec_wait > 0.6:                    # still unseen here: try the next point
                    self._next_retreat(q)
            return target
        # lower
        goal = self._rod_goal(self.wp, tool_z)
        target, there = self._rod_step(q, goal)
        if there:
            if now - self.last_seen <= self.fresh_age:
                self._resume(q)
            else:                                                    # lost again on the way down
                self._next_retreat(q)
        return target

    def _resume(self, q):
        """Hand back to the policy: the controller restarts from the measured rod, the newest pose and no motion."""
        tw = self.twin
        tw.cmd_xy = tw.tip[:, :2].copy()
        tw.q_cmd = q[None].copy()
        tw.prev_action[:] = 0.0
        tw.prev_obs_pose = tw.obs_pose.copy()
        tw.pose_hist[:] = tw.obs_pose[:, None, :]
        tw.obs = tw._observe()
        self.events[-1].update(result="resumed", seconds=self._clock() - self.rec_t0, at=tw.tip[0, :2].tolist())
        self.mode = "policy"

    # ------------------------------------------------------------------ a trial the arm hides from the camera
    def _park_points(self) -> np.ndarray:
        """Rod positions (n, 2) at ``start_lift`` on a 1 cm x 5 deg polar grid, best first (cached): with a shadow
        predictor, by the share of the work band the arm hides there; without, by distance from straight ahead near
        the inner edge of the ring a rod that high reaches."""
        if self._park is None:
            t = self.cfg.task
            r, a = np.meshgrid(np.arange(t.tool_r[0] + 0.01, t.tool_r[1] - 0.005, 0.01), np.radians(np.arange(-50, 51, 5)))
            xy = np.c_[(r * np.cos(a)).ravel(), (r * np.sin(a)).ravel()]
            xy = xy[self._liftable(xy, self.start_lift)]
            if self.view is not None:
                key = [self.view.band_hidden(self._ik(p, self.start_lift, self.twin.kin.seed(p[None])[0])[0]) for p in xy]
            else:
                key = np.linalg.norm(xy - (np.hypot(*xy.T).min() + 0.01, 0.0), axis=1)
            self._park = xy[np.argsort(key, kind="stable")]
        return self._park

    def _fly_z(self, xy) -> np.ndarray:
        """Rod heights (n,) for the clear start's flight over points ``xy`` (n, 2): as high as a vertical rod reaches
        there, up to ``start_lift``, in 2.5 mm steps (below ``lift`` only where it does not reach that)."""
        xy = np.atleast_2d(xy)
        zs = np.arange(self.start_lift, self.lift - 1e-9, -0.0025)
        pts = np.repeat(xy, len(zs), 0)
        _, err, _ = self.twin.kin.solve(np.c_[pts, np.tile(zs, len(xy))], self.twin.kin.seed(pts), iterations=30)
        ok = (err < 0.001).reshape(len(xy), len(zs))
        return np.array([zs[np.argmax(k)] if k.any() else self._lift_z(p) for k, p in zip(ok, xy)])

    def _flight(self, tip, pose, block) -> list:
        """The way (x, y points) for the clear start's flight from ``tip`` to the best park point: straight there, or
        through one other park point, passing over neither the object nor the block where the camera saw them at
        the start (``pose``, ``block``; None: unseen) nor the gate's walls (the rod clears objects by 20 mm where it
        flies at 40 mm, by 5 mm at 25 mm). The next best park point when the best cannot be reached that way;
        straight to the best when none can."""
        parks = self._park_points()
        return self._way(tip, parks, pose, block) or [parks[0]]

    def _way(self, frm, targets, pose, block):
        """The way (x, y points) for the lifted rod from ``frm`` to the first of ``targets`` (n, 2) it can reach
        straight, or through one park point, passing over neither the object at ``pose`` nor the block at ``block``
        (None: not known) nor the gate's walls. Like ``_clear_of_things``, it may start near something but never
        comes closer than 3 mm, or than at ``frm``. None when it reaches none of them."""
        tw, parks = self.twin, self._park_points()

        def gaps(xy):
            gap = np.full(len(xy), np.inf)
            if pose is not None:
                gap = tw._surface_distance(xy, np.tile(pose, (len(xy), 1)), tw.obj_id[:1].repeat(len(xy)))
            if self.clutter and block is not None:
                gap = np.minimum(gap, tw._block_clearance(xy, np.tile(block, (len(xy), 1))) - self.cfg.scene.pusher_radius)
            if hasattr(tw, "wall_gap"):
                gap = np.minimum(gap, tw.wall_gap(xy, np.zeros(len(xy), dtype=int)))
            return gap
        # Where a lifted rod goes: the ring of park points, ``frm`` and ``targets``.
        ends = np.vstack([parks, targets, frm[None]])
        r, az = np.hypot(*ends.T), np.abs(np.arctan2(ends[:, 1], ends[:, 0]))
        ring = (r.min() - 0.005, r.max() + 0.005, az.max() + 0.02)
        floor = min(0.003, float(gaps(frm[None])[0]) - 0.001)

        def clear(a, b):                  # the straight lines a[i] -> b[i] (n, 2), checked every 3 mm
            k = max(2, int(np.linalg.norm(b - a, axis=-1).max() / 0.003) + 1)
            w = np.linspace(0, 1, k)[None, :, None]
            seg = (a[:, None] * (1 - w) + b[:, None] * w).reshape(-1, 2)
            r, az = np.hypot(*seg.T), np.abs(np.arctan2(seg[:, 1], seg[:, 0]))
            ok = (gaps(seg) >= floor) & (r >= ring[0]) & (r <= ring[1]) & (az <= ring[2])
            return ok.reshape(len(a), k).all(1)
        direct = clear(np.tile(frm, (len(targets), 1)), targets)
        via = parks[clear(np.tile(frm, (len(parks), 1)), parks)]
        for i, t in enumerate(targets):
            if direct[i]:
                return [t]
            ok = clear(via, np.tile(t, (len(via), 1))) if len(via) else np.zeros(0, bool)
            if ok.any():
                k = np.argmin(np.where(ok, np.linalg.norm(via - frm, axis=1) + np.linalg.norm(via - t, axis=1), np.inf))
                return [via[k], t]
        return None

    def _fly_goals(self, way) -> np.ndarray:
        """Rod setpoints (x, y, z) for a lifted flight along the x, y points ``way``: every centimetre or less, as
        high as a vertical rod reaches (``_fly_z``)."""
        legs = [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.01) + 2))[1:] for a, b in zip(way[:-1], way[1:])]
        xy = np.concatenate([np.asarray(way[:1], float)] + legs)
        return np.c_[xy, self._fly_z(xy)]

    def _move_until(self, goals, seconds, done=None, early: bool = False, glide: bool = False) -> bool:
        """Runner-driven moves through ``goals`` (x, y, z) (no policy), one tick each, until each is reached and
        ``done()`` (if given) holds at the last one (``early``: anywhere on the way), or ``seconds`` pass. Returns
        whether it got there. ``glide``: after the first goal, go on from each one before the last as soon as the
        setpoint is there, not the rod (a path through them)."""
        q = self.jmap.to_sim(self.robot.read_joints())
        self._start_rod_move(q)
        t_end, k = self._clock() + seconds, 0
        while self._clock() < t_end:
            t0 = time.monotonic()
            q, pose, extra, _ = self._measure()
            target, there = self._rod_step(q, goals[k])
            self._send(t0, q, target, pose, extra, np.zeros(2), False)
            if early and done is not None and done():
                return True
            if k < len(goals) - 1 and (there or glide and k > 0 and np.array_equal(self.sp, goals[k])):
                k += 1
            elif there and (done is None or done()):
                return True
        return False

    def _clear_start(self, gate, seconds: float, pose=None, block=None) -> np.ndarray:
        """The arm hides the object (or the clutter block) at its start position: lift the rod straight up and fly
        it where the arm hides least (a park point), as high as a vertical rod reaches on the way (up to
        ``start_lift``: the hidden one may be anywhere under the arm's shadow, and the gripper hides the rod's
        surroundings), around the one the camera saw at the start (``pose``, ``block``; None: unseen; ``_flight``).
        Find both, then lower it at the nearest clear point from which the object should stay in view (the block's
        start pose is known by then)."""
        self.mode = "start"
        if gate is not None:              # the walls must be known before checking clearances
            self.twin.configure_gate(gate["centre"], gate["angle"], gate["width"], gate["side"], start_yaw=0.0)
        q = self.jmap.to_sim(self.robot.read_joints())
        tip0 = self.twin.kin.forward(q[None])[0][0]
        self.last_seen = -1e9
        fresh = lambda: (self._clock() - self.last_seen <= self.fresh_age
                         and (not self.clutter or self.block_seen > self.clear_t0))
        self._move_until(self._fly_goals([tip0[:2]] + self._flight(tip0[:2], pose, block)), seconds, fresh,
                         early=True, glide=True)                                     # straight up first
        if not fresh():
            raise RuntimeError(f"no {'object or block' if self.clutter else 'object'} pose from the camera, also "
                               "with the arm out of the way")
        # With a shadow predictor the planned start may still do (the arm came from elsewhere); without one it is
        # known to hide the object, and the alternatives step out further.
        tried = [] if self.view is not None else [tip0[:2].copy()]
        distances = (0.02, 0.04, 0.06, 0.08, 0.10) if self.view is not None else (0.04, 0.07, 0.10)
        for attempt in range(6):
            pose, here = self.twin.obs_pose[0].copy(), self.twin.tip[0, :2].copy()
            q = self.jmap.to_sim(self.robot.read_joints())
            start = None
            if attempt == 0 and self.view is not None and self._in_reach(tip0[None, :2])[0] \
                    and self._liftable(tip0[:2])[0] and self._clear_of_things(tip0[None, :2], pose, here)[0] \
                    and self._visible(q, tip0[:2], pose) >= 0.6:
                start = tip0[:2].copy()
            if start is None:
                start = self._retreat_point(q, tip0[:2], pose, tried, distances=distances, path_from=here)
            way = None if start is None else [start]
            if start is None:          # every straight way to a start passes over something: one through a park point
                start = self._retreat_point(q, tip0[:2], pose, tried, distances=distances, path_from=False)
                way = None if start is None else self._way(here, start[None], pose,
                                                            self.twin.clutter_obs[0] if self.clutter else None)
            if start is None:
                break
            tried.append(start)
            # As high as on the way here, then straight down.
            if way is not None and self._move_until(self._fly_goals([here] + way), seconds, glide=True) \
                    and self._move_until([np.r_[start, self.twin.tool_z_cmd[0]]], seconds, fresh):
                self.start_shift = float(np.linalg.norm(start - tip0[:2]))
                self.mode = "policy"
                return self.twin.obs_pose[0].copy()
        raise RuntimeError("no start position from which the camera sees the object")

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


def runner_options(args) -> dict:
    """The runner's safety, recovery and ramp settings from the command line."""
    return dict(max_step_deg=args.max_step_deg, lightbox=args.lightbox, max_pose_age=args.max_pose_age,
                on_lost=args.on_lost, recover_seconds=args.recover_seconds, ramp=not args.no_ramp, ramp_hz=args.ramp_hz)


def lift_clear(robot, jmap: JointMap, runner: PolicyRunner):
    """End of a real trial: lift the rod clear, up to 4 cm, as high as a vertical rod reaches there."""
    tip = runner.twin.tip[0]
    try:
        move_to(robot, jmap, runner.twin.kin, (tip[0], tip[1], runner._lift_z(tip[:2], top=0.04)), seconds=1.0)
    except ValueError as error:
        print(f"could not lift the rod: {error}")


def dry_run(args):
    """Episodes on SimRobot through the full hardware path (joint map, pose source, safety limits)."""
    _, cfg, _ = load_policy(args.checkpoint, "cpu")
    cfg.scene.lightbox = cfg.scene.lightbox or args.lightbox
    scenes_cfg = copy.deepcopy(cfg)
    scenes_cfg.num_envs, scenes_cfg.seed, scenes_cfg.num_threads = args.episodes, args.seed, 4
    scenes_cfg.task.difficulty, scenes_cfg.task.easy_fraction = 1.0, 0.0
    scenes = make_env(scenes_cfg)            # only used to draw the same kind of scenes as training
    hidden = JointMap.load(args.hidden_map) if args.hidden_map else JointMap()
    jmap = JointMap.load(args.map) if args.map else JointMap()
    robot = SimRobot(cfg.scene, cfg.task.objects[0], hidden=hidden, encoder_noise_deg=0.05)
    gate_task, clutter_task = hasattr(scenes, "gate_centre"), hasattr(scenes, "clutter_home")
    runner = PolicyRunner(args.checkpoint, robot, None, jmap, realtime=False, **runner_options(args))
    camera = None
    if args.camera == "render":                  # the C920 pipeline on rendered frames (slow: ~0.1 s per tick)
        from .camera_sim import RenderedPoseSource
        camera = RenderedPoseSource(robot, cfg.scene, cfg.task.objects[0], latency_steps=2, dt=runner.dt, seed=args.seed,
                                    mount=args.mount)
    stats, dropped, unseen, shifted, recoveries = [], [], [], [], []
    for i in range(args.episodes):
        robot.place(scenes.tip[i, :2], scenes.object_pose()[i])
        gate = None
        if gate_task:
            gate = dict(centre=scenes.gate_centre[i], angle=scenes.gate_angle[i], width=scenes.gate_width[i],
                        side=scenes.gate_side[i])
            robot.place_gate(gate["centre"], gate["angle"], gate["width"])
            if camera is not None:
                camera.place_gate(gate["centre"], gate["angle"], gate["width"])
        if clutter_task:
            robot.place_block(scenes.clutter_home[i])
        if camera is not None:
            camera.reset()
            runner.poses = camera
        else:
            runner.poses = SimPoseSource(robot, latency_steps=1, seed=args.seed + i)
        try:
            runner.start(scenes.goal[i], gate)
        except RuntimeError as error:          # the rendered camera cannot see the object, even with the arm moved
            unseen.append(i)
            print(f"episode {i + 1:3d}: not started ({error})", flush=True)
            continue
        if runner.start_shift:
            shifted.append(runner.start_shift)
        lost = False
        for _ in range(scenes.max_steps):
            if runner.tick() == "lost":
                lost = True
                break
        recoveries.append(runner.events)
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
                start_shift=runner.start_shift, recoveries=runner.events, gave_up=lost,
                **({"block_home": scenes.clutter_home[i].tolist(), "block_moved": moved} if clutter_task else {})))
        if camera is not None:
            dropped.append(camera.counts["dropped"].get(camera.object_name, 0) / max(camera.counts["frames"], 1))
        print(f"episode {i + 1:3d}: {'success' if ok else 'miss   '} pos {pos * 1e3:5.1f} mm  yaw {math.degrees(yaw):5.1f} deg"
              + (f"  passed gate {bool(stats[-1][3])}" if gate_task else "")
              + (f"  block moved {moved * 1e3:4.1f} mm" if clutter_task else "")
              + (f"  started {runner.start_shift * 1e3:.0f} mm from the planned start" if runner.start_shift else "")
              + (f"  lost {len(runner.events)}x, resumed {sum(e.get('result') == 'resumed' for e in runner.events)}"
                 if runner.events else "") + ("  GAVE UP (object lost)" if lost else ""), flush=True)
    if unseen:
        print(f"\n{len(unseen)} of {args.episodes} episodes could not start: the camera did not see the object even with "
              f"the arm moved out of the way (episodes {[i + 1 for i in unseen]}); the {len(stats)} that started:")
    if not stats:
        return
    events = [e for ep in recoveries for e in ep]
    if shifted:
        print(f"\n{len(shifted)} episodes started away from the planned start, where the arm hid the object "
              f"({np.mean(shifted) * 1e3:.0f} mm on average)")
    if events:
        took = [e["seconds"] for e in events if e.get("result") == "resumed"]
        print(f"object lost {len(events)} times in {sum(bool(ep) for ep in recoveries)} episodes (--on-lost {args.on_lost}): "
              f"resumed {len(took)}" + (f" after {np.median(took):.1f} s median" if took else "")
              + f", gave up {sum(e.get('result') == 'lost' for e in events)}")
    s = np.array(stats, dtype=float)
    print(f"\n{len(stats)} episodes through the hardware path: success {s[:, 0].mean():.1%}, "
          f"median {np.median(s[:, 1]) * 1e3:.1f} mm / {math.degrees(np.median(s[:, 2])):.1f} deg"
          + (f", passed gate {s[:, 3].mean():.1%}" if gate_task else "")
          + (f", block kept in place {s[:, 3].mean():.1%}" if clutter_task else "")
          + (f"; rendered C920 ({args.mount} mount), {np.mean(dropped):.1%} of frames without a pose"
             if camera is not None else ""))


def real_run(args):
    jmap = JointMap.load(args.map)
    robot = SO101Robot(args.port, max_step_deg=args.max_step_deg)
    runner = PolicyRunner(args.checkpoint, robot, None, jmap, **runner_options(args))
    try:      # the camera masks the arm out using the policy's own scene (lightbox, rod)
        poses = runner.poses = CameraPoseSource(scene=runner.cfg.scene, block_name=runner.cfg.task.clutter or "box")
    except FileNotFoundError as error:
        robot.close()
        raise SystemExit(f"{error}\nSee HARDWARE.md section 4. Until then, rehearse with --robot sim.")
    kin = runner.twin.kin
    goal = np.array([args.goal[0], args.goal[1], math.radians(args.goal[2])])
    try:
        move_to(robot, jmap, kin, (*args.tool_start, 0.03))                    # above the start point
        move_to(robot, jmap, kin, (*args.tool_start, runner.cfg.task.tool_height), seconds=1.0)
        runner.start(goal)
        if runner.start_shift:
            print(f"the arm hid the object at the start: started {runner.start_shift * 1e3:.0f} mm from --tool-start")
        for _ in range(round(runner.cfg.task.episode_seconds / runner.dt)):
            if runner.tick() == "lost":
                print(f"object not seen again within {args.recover_seconds:.0f} s: trial ended")
                break
    except KeyboardInterrupt:
        print("stopped by user")
    except RuntimeError as error:
        print(f"trial not started: {error}")
    finally:
        lift_clear(robot, jmap, runner)
        pose, _ = poses.read()
        meta = dict(checkpoint=args.checkpoint, goal=goal.tolist(), final=None if pose is None else pose.tolist(),
                    start_shift=runner.start_shift, recoveries=runner.events)
        if pose is not None:
            meta["pos_err"], meta["yaw_err"] = runner.errors(pose)
        meta["camera"] = poses.summary()
        save_trial(Path(args.log_dir or PROJECT / "trials") / time.strftime("%Y%m%d_%H%M%S"), runner, meta)
        print(f"camera: {meta['camera']}")
        poses.close()
        robot.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="", help="the policy (one per trial); not with --level")
    ap.add_argument("--level", type=int, choices=[2, 3], default=0,
                    help="Level 2 (two blocks to their goals) or 3 (the box nested in the T's corner): a push plan from "
                         "the camera's first view, a skill per push (levels.py)")
    ap.add_argument("--tee", default="runs/clutter_lightbox_v2/best.pt", help="--level: push the T, keep the box")
    ap.add_argument("--box", default="runs/clutter_box_lightbox_v2/best.pt", help="--level: push the box, keep the T")
    ap.add_argument("--dock", default="runs/dock_lightbox_v3/best.pt", help="--level 3: push the box into the T's corner")
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="", help="joint map from calibrate.py (required on the real arm)")
    ap.add_argument("--max-step-deg", type=float, default=4.0)
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--episodes", type=int, default=20, help="sim dry run")
    ap.add_argument("--seed", type=int, default=TEST_SEED, help="sim dry run: scene seed")
    ap.add_argument("--hidden-map", default="", help="sim dry run: the simulated arm's true (miscalibrated) map")
    ap.add_argument("--goal", type=float, nargs=3, default=(0.20, 0.05, 90.0), metavar=("X", "Y", "YAW_DEG"),
                    help="real arm: the object's goal (with --level, the T's)")
    ap.add_argument("--box-goal", type=float, nargs=3, default=(0.20, -0.06, 0.0), metavar=("X", "Y", "YAW_DEG"),
                    help="real arm, --level 2: the box's goal")
    ap.add_argument("--tool-start", type=float, nargs=2, default=(0.15, -0.10), metavar=("X", "Y"))
    ap.add_argument("--max-pose-age", type=float, default=2.0,
                    help="the object is lost when it has not been seen for this long (s); shorter gaps the policy "
                         "rides through on its last pose")
    ap.add_argument("--no-ramp", action="store_true",
                    help="send each tick's joint target in one jump (the servos lurch 20 times a second) instead of "
                         "ramping to it at --ramp-hz")
    ap.add_argument("--ramp-hz", type=float, default=100.0, help="rate of the ramped servo targets (default 100 Hz)")
    ap.add_argument("--on-lost", choices=["recover", "hold"], default="recover",
                    help="lost object: lift and move the rod out of the camera's way until it is seen again (default), "
                         "or hold the arm still")
    ap.add_argument("--recover-seconds", type=float, default=8.0,
                    help="give up (end the trial, rod lifted) when the object stays unseen this long")
    ap.add_argument("--camera", choices=["sim", "render"], default="sim",
                    help="sim dry run: the training camera model, or the C920 pipeline on rendered frames (lightbox)")
    ap.add_argument("--mount", choices=["left", "right", "back"], default="left",
                    help="--camera render: where the C920 is (camera_sim.MOUNTS): the left-wall slot as built, its mirror, "
                         "or the back wall 25 cm above the mat (LIGHTBOX.md C6)")
    ap.add_argument("--lightbox", action="store_true",
                    help="the arm stands in the lightbox, 9.8 mm below the mat (implied by policies trained there)")
    args = ap.parse_args(argv)
    if bool(args.checkpoint) == bool(args.level):
        ap.error("give --checkpoint (one policy) or --level 2|3 (the skills)")
    if args.robot == "so101" and not args.map:
        ap.error("--map is required on the real arm (run calibrate.py first)")
    if args.level:
        from . import levels
        (levels.dry_run if args.robot == "sim" else levels.real_run)(args)
    elif args.robot == "sim":
        dry_run(args)
    else:
        real_run(args)


if __name__ == "__main__":
    main()
