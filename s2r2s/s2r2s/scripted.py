"""Heuristic SE(2) pushing controller.

Used to check that the task is physically solvable and that the reward and
success logic behave, before (and as a baseline for) reinforcement learning.
It reads the true object pose, so it is a diagnostic tool, not a deployable
policy.

Strategy: pick the keypoint farthest from its goal, move the pusher around
the object to the point just behind that keypoint, then push the keypoint
towards its goal. Pushing an extremity both translates and rotates the
object, so repeating this on the worst keypoint reduces pose error.

Gate task, passage stage: push from directly behind the object's centre, offset sideways in
proportion to its yaw error (to steer the rotation) and angled towards the opening's centre
line, so the object goes through straight instead of jamming on a wall edge.
"""
from __future__ import annotations

import numpy as np

from .env import PushEnv, wrap
from .objects import OBJECTS


class KeypointPusher:
    def __init__(self, env: PushEnv, clearance: float = 0.004, approach_tol: float = 0.004,
                 yaw_gain: float = 0.03, centre_gain: float = 4.0):
        self.env = env
        self.clearance = clearance
        self.approach_tol = approach_tol
        self.yaw_gain = yaw_gain            # sideways contact offset per radian of yaw error (m/rad)
        self.centre_gain = centre_gain      # push-direction steering per metre of lateral offset
        self.pushing = np.zeros(env.n, dtype=bool)
        self.focus = np.zeros(env.n, dtype=np.int64)
        # Footprint outlines (object frame), padded to a common length, for the passage push point.
        outlines = [OBJECTS[name].outline() for name in env.cfg.task.objects]
        m = max(len(o) for o in outlines)
        self.outline = np.stack([np.concatenate([o, np.repeat(o[-1:], m - len(o), 0)]) for o in outlines])

    def reset(self, ids):
        self.pushing[ids] = False

    def _sdf(self, xy, pose, obj_id):
        return self.env._surface_distance(xy, pose, obj_id)

    def act(self) -> np.ndarray:
        env = self.env
        n = env.n
        pose = env.object_pose()
        obj_id = env.obj_id
        # Follow the current waypoint when the task has one (gate), else the goal.
        goal = (env.subgoal() if hasattr(env, "subgoal") else env.goal).copy()
        goal[:, 2] = env._symmetric_goal_yaw(pose[:, 2], goal[:, 2], obj_id)
        final = env.goal.copy()
        final[:, 2] = env._symmetric_goal_yaw(pose[:, 2], final[:, 2], obj_id)
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
        target = self._route(np.arange(n), tip, pre, pose)
        # Push: move the keypoint towards its goal, slowing down near the goal.
        push_speed = np.clip(np.linalg.norm(e, axis=-1) / 0.02, 0.25, 1.0) * 0.6
        approach = (target - tip) / (env.cfg.task.max_speed * env.cfg.task.control_dt)
        approach = approach / np.maximum(np.linalg.norm(approach, axis=-1, keepdims=True), 1.0)
        action = np.where(self.pushing[:, None], u * push_speed[:, None], approach)
        if hasattr(env, "stage"):
            rows = np.flatnonzero(env.stage == 1)
            if len(rows):
                action[rows] = self._passage(rows, pose[rows], tip[rows])
        # Stop once the pose is within tolerance of the final goal.
        done = (np.linalg.norm(pose[:, :2] - final[:, :2], axis=-1) < 0.5 * env.cfg.task.success_pos) & \
               (np.abs(pose[:, 2] - final[:, 2]) < 0.5 * env.cfg.task.success_yaw)
        action[done] = 0.0
        return action

    def _passage(self, rows, pose, tip):
        """Push straight through the opening, steering yaw and lateral offset (gate task, stage 1)."""
        env = self.env
        along, passage = env._axes(rows)
        lateral = ((pose[:, :2] - env.gate_centre[rows]) * along).sum(-1)
        yaw_err = wrap(pose[:, 2] - env.target_yaw[rows])
        # Distance from the centre to the object's back edge, along -passage.
        c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
        o = self.outline[env.obj_id[rows]]
        world = np.stack([c * o[..., 0] - s * o[..., 1], s * o[..., 0] + c * o[..., 1]], -1)
        back = (-(world * passage[:, None, :]).sum(-1)).max(1)
        offset = np.clip(-self.yaw_gain * yaw_err * env.gate_side[rows], -0.012, 0.012)
        contact = (pose[:, :2] - passage * (back + env.cfg.scene.pusher_radius + 0.005)[:, None]
                   + along * offset[:, None])
        direction = passage - along * np.clip(self.centre_gain * lateral, -0.5, 0.5)[:, None]
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        step = env.cfg.task.max_speed * env.cfg.task.control_dt
        target = self._route(rows, tip, contact, pose)
        approach = (target - tip) / step
        approach /= np.maximum(np.linalg.norm(approach, axis=-1, keepdims=True), 1.0)
        at_contact = np.linalg.norm(tip - contact, axis=-1) < 0.008
        # While pushing, only correct sideways towards the push line; never pull back to the standoff.
        sideways = along * (((contact - tip) * along).sum(-1) / step)[:, None]
        return np.where(at_contact[:, None], 0.4 * direction + 0.5 * sideways, approach)

    def _route(self, rows, tip, dest, pose):
        """Next tool target towards ``dest``: straight there, around the object if the straight path
        would hit it (the side that avoids walls), and through the opening if ``dest`` is behind a wall."""
        env = self.env
        n = len(rows)
        obj_id = env.obj_id[rows]
        # The last stretch is excluded: ``dest`` itself is meant to be next to the object.
        seg = tip[:, None, :] + (dest - tip)[:, None, :] * np.linspace(0, 0.85, 12)[None, :, None]
        hit = (self._sdf(seg.reshape(-1, 2), np.repeat(pose, 12, 0), np.repeat(obj_id, 12)).reshape(n, -1)
               < self.clearance * 0.5).any(1)
        centre = pose[:, :2]
        r_orbit = env.obj.radius[obj_id] + env.cfg.scene.pusher_radius + 0.008
        a_tip = np.arctan2(*(tip - centre).T[::-1])
        a_dst = np.arctan2(*(dest - centre).T[::-1])
        da = (a_dst - a_tip + np.pi) % (2 * np.pi) - np.pi
        a_way = a_tip + np.clip(da, -0.6, 0.6)
        way = centre + r_orbit[:, None] * np.stack([np.cos(a_way), np.sin(a_way)], -1)
        walls = hasattr(env, "wall_gap")
        if walls:
            # Orbit the other way round when this way runs into a wall.
            a_alt = a_tip - np.sign(da) * 0.6
            alt = centre + r_orbit[:, None] * np.stack([np.cos(a_alt), np.sin(a_alt)], -1)
            way = np.where((env.wall_gap(way, rows) < 0.003)[:, None], alt, way)
        target = np.where(hit[:, None], way, dest)
        if walls:
            # A target behind the wall: go through the opening first.
            seg = tip[:, None, :] + (target - tip)[:, None, :] * np.linspace(0, 1, 12)[None, :, None]
            wall_hit = (env.wall_gap(seg.reshape(-1, 2), np.repeat(rows, 12)) < 0.002).reshape(n, -1).any(1)
            other_side = np.sign(env.passage_progress(tip, rows)) != np.sign(env.passage_progress(target, rows))
            target = np.where((wall_hit & other_side)[:, None], env.gate_centre[rows], target)
        return target
