"""Heuristic SE(2) pushing controller.

Used to check that the task is physically solvable and that the reward and
success logic behave, before (and as a baseline for) reinforcement learning.
It reads the true object pose, so it is a diagnostic tool, not a deployable
policy.

Strategy: pick the keypoint farthest from its goal, move the pusher around
the object to the point just behind that keypoint, then push the keypoint
towards its goal. Pushing an extremity both translates and rotates the
object, so repeating this on the worst keypoint reduces pose error.
"""
from __future__ import annotations

import numpy as np

from .env import PushEnv


class KeypointPusher:
    def __init__(self, env: PushEnv, clearance: float = 0.004, approach_tol: float = 0.004):
        self.env = env
        self.clearance = clearance
        self.approach_tol = approach_tol
        self.pushing = np.zeros(env.n, dtype=bool)
        self.focus = np.zeros(env.n, dtype=np.int64)

    def reset(self, ids):
        self.pushing[ids] = False

    def _sdf(self, xy, pose, obj_id):
        return self.env._surface_distance(xy, pose, obj_id)

    def act(self) -> np.ndarray:
        env = self.env
        n = env.n
        pose = env.object_pose()
        obj_id = env.obj_id
        goal = env.goal.copy()
        goal[:, 2] = env._symmetric_goal_yaw(pose[:, 2], goal[:, 2], obj_id)
        kp = env._keypoints(pose, obj_id)
        kpg = env._keypoints(goal, obj_id)
        err = kpg - kp
        errn = np.linalg.norm(err, axis=-1) * (env.obj.keypoint_weight[obj_id] > 0)
        # Keep pushing the same keypoint unless another one is much worse.
        cur = errn[np.arange(n), self.focus]
        switch = (~self.pushing) | (errn.max(1) > 2.0 * cur + 0.004)
        self.focus = np.where(switch, errn.argmax(1), self.focus)
        i = self.focus
        p = kp[np.arange(n), i]
        e = err[np.arange(n), i]
        u = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-9)
        tip = env.tip[:, :2]
        # Pre-push point: walk back from the keypoint along -u until clear of the footprint.
        steps = np.arange(0.0, 0.10, 0.001)
        cand = p[:, None, :] - u[:, None, :] * steps[None, :, None]
        sdf = self._sdf(cand.reshape(-1, 2), np.repeat(pose, len(steps), 0), np.repeat(obj_id, len(steps))).reshape(n, -1)
        first = np.argmax(sdf > self.clearance, axis=1)
        pre = p - u * steps[first][:, None]
        # Already in a pushing configuration?  Pusher behind the keypoint, close to the object.
        behind = ((p - tip) * u).sum(-1) > 0
        near = self._sdf(tip, pose, obj_id) < self.clearance + 0.003
        rel = p - tip
        lateral = np.abs(u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]) < 0.008
        at_pre = np.linalg.norm(tip - pre, axis=-1) < self.approach_tol
        self.pushing = (self.pushing & behind & near & lateral) | at_pre
        # Approach: straight to the pre-push point, or orbit the object if that path would hit it.
        target = pre.copy()
        seg = tip[:, None, :] + (pre - tip)[:, None, :] * np.linspace(0, 1, 12)[None, :, None]
        hit = (self._sdf(seg.reshape(-1, 2), np.repeat(pose, 12, 0), np.repeat(obj_id, 12)).reshape(n, -1)
               < self.clearance * 0.5).any(1)
        centre = pose[:, :2]
        r_orbit = env.obj.radius[obj_id] + env.cfg.scene.pusher_radius + 0.008
        a_tip = np.arctan2(*(tip - centre).T[::-1])
        a_pre = np.arctan2(*(pre - centre).T[::-1])
        da = (a_pre - a_tip + np.pi) % (2 * np.pi) - np.pi
        a_way = a_tip + np.clip(da, -0.6, 0.6)
        way = centre + r_orbit[:, None] * np.stack([np.cos(a_way), np.sin(a_way)], -1)
        target = np.where(hit[:, None], way, target)
        # Push: move the keypoint towards its goal, slowing down near the goal.
        push_speed = np.clip(np.linalg.norm(e, axis=-1) / 0.02, 0.25, 1.0) * 0.6
        approach = (target - tip) / (env.cfg.task.max_speed * env.cfg.task.control_dt)
        approach = approach / np.maximum(np.linalg.norm(approach, axis=-1, keepdims=True), 1.0)
        action = np.where(self.pushing[:, None], u * push_speed[:, None], approach)
        # Stop once the pose is within tolerance.
        done = (np.linalg.norm(pose[:, :2] - goal[:, :2], axis=-1) < 0.5 * env.cfg.task.success_pos) & \
               (np.abs(pose[:, 2] - goal[:, 2]) < 0.5 * env.cfg.task.success_yaw)
        action[done] = 0.0
        return action
