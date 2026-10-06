"""Level 1 puzzle: push the object through an opening in a wall, then to its goal pose.

A wall crosses the work band along a radial line, with an opening near r = 0.2 m. The
object starts on one side and its goal is on the other. The wall reaches beyond the
workspace, so the opening is the only way through. The T only fits when turned so its
bar runs along the wall (about +-25 deg of slack for a 64 mm opening; 56-71 mm wide
depending on orientation).

The policy is guided along a path: A, an aligned pose just before the wall; B, the same
pose just past it; G, the goal. The reward uses the remaining path length *through the
opening*, so pushing straight at the wall never looks like progress. The observation's
goal slots show the current waypoint, so a Milestone 1 policy (no gate) already knows how
to follow it; new gate features are appended after the Milestone 1 features (warm start).
"""
from __future__ import annotations

import math

import numpy as np

from .env import EnvConfig, PushEnv, wrap
from .scene import GATE_WALL_HALF_THICKNESS, GATE_WALL_LENGTH, set_gate

CENTRE = np.array([0.2, 0.0])


class GateEnv(PushEnv):
    def __init__(self, cfg: EnvConfig):
        cfg.task.gate = True
        cfg.scene.gate = True
        n = cfg.num_envs
        self.gate_centre = np.zeros((n, 2))
        self.gate_angle = np.zeros(n)                  # wall direction (radial), rad
        self.gate_width = np.zeros(n)
        self.gate_side = np.ones(n)                    # +1: start below the wall's azimuth, goal above
        self.target_yaw = np.zeros(n)                  # an orientation that fits through the opening
        self.stage = np.zeros(n, dtype=np.int64)       # 0 align before the wall, 1 pass through, 2 go to goal
        super().__init__(cfg)

    # ----------------------------------------------------------- gate frame
    def _axes(self, ids):
        """Unit vectors along the wall (radial) and along the passage (towards the goal side)."""
        a = self.gate_angle[ids]
        along = np.stack([np.cos(a), np.sin(a)], -1)
        passage = self.gate_side[ids, None] * np.stack([-np.sin(a), np.cos(a)], -1)
        return along, passage

    def _waypoints(self, ids):
        t = self.cfg.task
        _, passage = self._axes(ids)
        c = self.gate_centre[ids]
        a = np.c_[c - passage * t.gate_pre, self.target_yaw[ids]]
        b = np.c_[c + passage * t.gate_post, self.target_yaw[ids]]
        return a, b

    def wall_gap(self, xy, ids):
        """Distance from the rod surface at ``xy`` to the nearest wall (<= 0: touching)."""
        a = self.gate_angle[ids]
        rel = xy - self.gate_centre[ids]
        lx = rel[:, 0] * np.cos(a) + rel[:, 1] * np.sin(a)
        ly = -rel[:, 0] * np.sin(a) + rel[:, 1] * np.cos(a)
        half_w = self.gate_width[ids] / 2
        d = np.full(len(ids), np.inf)
        for sign, length in ((-1, GATE_WALL_LENGTH[0]), (1, GATE_WALL_LENGTH[1])):
            qx = np.abs(lx - sign * (half_w + length / 2)) - length / 2
            qy = np.abs(ly) - GATE_WALL_HALF_THICKNESS
            sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
            d = np.minimum(d, sdf)
        return d - self.cfg.scene.pusher_radius

    def passage_progress(self, xy, ids):
        """Signed distance of points past the wall, along the passage (negative: start side)."""
        _, passage = self._axes(ids)
        return ((xy - self.gate_centre[ids]) * passage).sum(-1)

    # ------------------------------------------------------------- episodes
    def _randomize(self, i):
        super()._randomize(i)
        r = self.cfg.rand
        mu = self.param_rng.uniform(*r.gate_friction)
        if r.enabled:
            m = self.models[i]
            for name in ("gate_inner", "gate_outer"):
                m.geom_friction[m.geom(name).id, 0] = mu

    def _sample_scene(self, ids):
        t = self.cfg.task
        n = len(ids)
        rng = self.rng
        c = np.full(n, self.difficulty)
        easy = rng.random(n) < t.easy_fraction
        c[easy] = rng.uniform(0, self.difficulty, easy.sum())
        self.difficulty_ep[ids] = c
        # Gate: the opening narrows with difficulty (80 mm lets any orientation through).
        width = t.gate_width_easy + c * (rng.uniform(*t.gate_width, n) - t.gate_width_easy)
        angle = rng.uniform(-t.gate_az, t.gate_az, n)
        radius = rng.uniform(*t.gate_r, n)
        side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        self.gate_width[ids], self.gate_angle[ids], self.gate_side[ids] = width, angle, side
        self.gate_centre[ids] = radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], -1)
        for j, i in enumerate(ids):
            set_gate(self.models[i], self.gate_centre[i], angle[j], width[j])
        # Start on one side of the wall, goal on the other, both inside the object band.
        below = lambda u: -t.obj_az + u * (angle - t.gate_clear_az + t.obj_az)
        above = lambda u: angle + t.gate_clear_az + u * (t.obj_az - angle - t.gate_clear_az)
        u1, u2 = rng.random(n), rng.random(n)
        start_az = np.where(side > 0, below(u1), above(u1))
        goal_az = np.where(side > 0, above(u2), below(u2))
        start = rng.uniform(*t.obj_r, n)[:, None] * np.stack([np.cos(start_az), np.sin(start_az)], -1)
        goal = rng.uniform(*t.obj_r, n)[:, None] * np.stack([np.cos(goal_az), np.sin(goal_az)], -1)
        start_yaw = rng.uniform(-math.pi, math.pi, n)
        self.goal[ids] = np.c_[goal, rng.uniform(-math.pi, math.pi, n)]
        # The orientation that fits: bar along the wall (gate-frame yaw 0 or pi), whichever is closer.
        cand = np.stack([angle, angle + math.pi], -1)
        pick = np.abs(wrap(cand - start_yaw[:, None])).argmin(1)
        self.target_yaw[ids] = wrap(cand[np.arange(n), pick])
        self.stage[ids] = 0
        return start, start_yaw

    def _tool_clear(self, ids, cand, start, start_yaw):
        return super()._tool_clear(ids, cand, start, start_yaw) & (self.wall_gap(cand, ids) > 0.006)

    # ----------------------------------------------------------- task hooks
    def _task_distance(self, pose, ids=None, dist=None):
        """Remaining path length (keypoint distance) through the opening; also advances the stage."""
        t = self.cfg.task
        ids = np.arange(self.n) if ids is None else ids
        obj = self.obj_id[ids]
        a, b = self._waypoints(ids)
        g = self.goal[ids]
        d_a = self._keypoint_distance(pose, a, obj)
        d_b = self._keypoint_distance(pose, b, obj)
        d_g = self._keypoint_distance(pose, g, obj) if dist is None else dist
        leg_ab = self._keypoint_distance(a, b, obj)
        leg_bg = self._keypoint_distance(b, g, obj)
        s = self.passage_progress(pose[:, :2], ids)
        st = self.stage[ids]
        st = np.where((st == 0) & (d_a < t.gate_align_tol), 1, st)
        st = np.where(s >= t.gate_cross, 2, st)
        st = np.where((st == 2) & (s < 0.0), 1, st)      # pushed back through the opening
        self.stage[ids] = st
        return np.where(st == 0, d_a + leg_ab + leg_bg, np.where(st == 1, d_b + leg_bg, d_g))

    def _feature_goal(self, ids):
        a, b = self._waypoints(ids)
        st = self.stage[ids][:, None]
        return np.where(st == 0, a, np.where(st == 1, b, self.goal[ids]))

    def _extra_obs(self, ids, pose, tool_xy):
        along, passage = self._axes(ids)
        c = self.gate_centre[ids]
        a, b = self._waypoints(ids)
        g = self.goal[ids]
        obj = self.obj_id[ids]
        st = self.stage[ids]
        leg_bg = self._keypoint_distance(b, g, obj)
        rest = np.where(st == 0, self._keypoint_distance(a, b, obj) + leg_bg, np.where(st == 1, leg_bg, 0.0))
        rel_obj, rel_tool = pose[:, :2] - c, tool_xy - c
        return np.concatenate([
            (c - CENTRE) / 0.1, passage, ((self.gate_width[ids] - 0.067) / 0.003)[:, None],
            (g[:, :2] - CENTRE) / 0.1, np.cos(g[:, 2:3]), np.sin(g[:, 2:3]),
            np.eye(3)[st],
            np.stack([(rel_obj * along).sum(-1), (rel_obj * passage).sum(-1)], -1) / 0.05,
            np.stack([(rel_tool * along).sum(-1), (rel_tool * passage).sum(-1)], -1) / 0.05,
            rest[:, None] / 0.1,
        ], -1)

    def _episode_extras(self, ids):
        return {"passed": self.stage[ids] == 2, "gate_width": self.gate_width[ids].copy()}

    def subgoal(self):
        """Current waypoint or goal for every world (used by the scripted controller)."""
        return self._feature_goal(np.arange(self.n))
