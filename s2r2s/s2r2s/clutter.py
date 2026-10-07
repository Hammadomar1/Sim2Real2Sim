"""Push the object to its goal pose while a second block stays where it is.

The block (``TaskConfig.clutter``, e.g. a 40 mm box) is placed within 25 mm of the path the
object sweeps from its start to its goal in ``clutter_in_way`` of the episodes, elsewhere in the
work band otherwise. It never blocks that straight path: in this narrow work band there is often
no room to detour around a block, which made such scenes unsolvable without touching it. Moving it costs reward
(``w_disturb`` per cm, symmetric, so pushing it back is rewarded) plus a per-step cost while it
is out of tolerance, large enough that a disturbed block spoils the episode (a weaker penalty was
too small, next to the goal reward, for the policy to learn from). Success needs the object at its goal *and* the block within
``clutter_tol`` / ``clutter_yaw_tol`` of where it started.

This is the proposal's "clutter" scene factor and the skill the block-connecting puzzle needs:
place one piece without disturbing the other. The block's pose reaches the policy through the
same camera model as the object's; the critic sees the truth.
"""
from __future__ import annotations

import math

import numpy as np

from .env import EnvConfig, PushEnv, wrap
from .objects import OBJECTS, object_tables

CENTRE = np.array([0.2, 0.0])


def densified_outline(name: str, step: float = 0.002) -> np.ndarray:
    """Footprint boundary points (object frame), at most ``step`` apart."""
    poly = OBJECTS[name].outline()
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * k / n for k in range(n)]
    return np.array(pts)


def footprint_sdf(points, pose, boxes, disk):
    """Signed distance of world points (n, P, 2) to footprints at ``pose`` (n, 3); boxes (n, B, 4), disk (n,)."""
    rel = points - pose[:, None, :2]
    c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
    local = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1)
    q = np.abs(local[:, :, None, :] - boxes[:, None, :, :2]) - boxes[:, None, :, 2:]
    sdf = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    sdf = np.where(boxes[:, None, :, 2] > 0, sdf, np.inf).min(-1)
    return np.where(disk[:, None] > 0, np.linalg.norm(local, axis=-1) - disk[:, None], sdf)


def _to_world(local, pose):
    c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
    return np.stack([c * local[..., 0] - s * local[..., 1], s * local[..., 0] + c * local[..., 1]], -1) + pose[:, None, :2]


class ClutterEnv(PushEnv):
    def __init__(self, cfg: EnvConfig):
        if not cfg.task.clutter:
            raise ValueError("ClutterEnv needs TaskConfig.clutter (an object name)")
        cfg.scene.clutter = cfg.task.clutter
        n = cfg.num_envs
        self.clutter_tab = object_tables([cfg.task.clutter])
        self.clutter_outline = densified_outline(cfg.task.clutter, step=0.004)
        self.object_outlines = [densified_outline(name, step=0.004) for name in cfg.task.objects]
        self.clutter_home = np.zeros((n, 3))
        self.clutter_obs = np.zeros((n, 3))
        self.clutter_hist = np.zeros((n, max(cfg.rand.obs_latency_steps[1], 1) + 2, 3))
        self.prev_disturb = np.zeros(n)
        self.in_way = np.zeros(n, dtype=bool)
        self.blocking = np.zeros(n, dtype=bool)
        super().__init__(cfg)

    # ------------------------------------------------------------- geometry
    def clutter_pose(self):
        o = self.qpos_adr + self.idx.clutter_qpos
        q = self.state[:, o + 3:o + 7]
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return np.c_[self.state[:, o:o + 2], yaw]

    def _clutter_yaw_err(self, yaw, home_yaw):
        s = self.clutter_tab.symmetry[0]
        period = 2 * math.pi / s if s > 0 else 2 * math.pi
        d = (yaw - home_yaw + period / 2) % period - period / 2
        return np.abs(d) if s != 0 else np.zeros_like(d)

    def disturbance(self, ids=None):
        """Displacement of the block from where it started: position (m) and yaw (rad)."""
        ids = np.arange(self.n) if ids is None else ids
        pose, home = self.clutter_pose()[ids], self.clutter_home[ids]
        return np.linalg.norm(pose[:, :2] - home[:, :2], axis=-1), self._clutter_yaw_err(pose[:, 2], home[:, 2])

    def _gap_to_object(self, clutter_pose, object_pose, obj_id):
        """Clearance between the block and the object footprints (negative: overlapping)."""
        n = len(obj_id)
        tab = self.clutter_tab
        c_pts = _to_world(np.broadcast_to(self.clutter_outline, (n, *self.clutter_outline.shape)), clutter_pose)
        gap_a = footprint_sdf(c_pts, object_pose, self.obj.boxes[obj_id], self.obj.disk_radius[obj_id]).min(1)
        o_pts = np.stack([self.object_outlines[k][np.arange(len(self.object_outlines[0])) % len(self.object_outlines[k])]
                          for k in obj_id]) if len(self.object_outlines) > 1 else \
            np.broadcast_to(self.object_outlines[0], (n, *self.object_outlines[0].shape))
        o_pts = _to_world(o_pts, object_pose)
        gap_b = footprint_sdf(o_pts, clutter_pose, np.repeat(tab.boxes, n, 0), np.repeat(tab.disk_radius, n)).min(1)
        return np.minimum(gap_a, gap_b)

    def _sweep_gap(self, clutter_pose, start_pose, goal, obj_id, steps: int = 10):
        """Smallest clearance between the block and the object moved straight (and turned) from start to goal."""
        gap = np.full(len(obj_id), np.inf)
        for f in np.linspace(0, 1, steps):
            mid = np.c_[start_pose[:, :2] + f * (goal[:, :2] - start_pose[:, :2]),
                        start_pose[:, 2] + f * wrap(goal[:, 2] - start_pose[:, 2])]
            gap = np.minimum(gap, self._gap_to_object(clutter_pose, mid, obj_id))
        return gap

    # ------------------------------------------------------------- episodes
    def _randomize(self, i):
        super()._randomize(i)
        r = self.cfg.rand
        scale = self.param_rng.uniform(*r.mass_scale)
        if r.enabled:
            m, b, nominal = self.models[i], self.idx.clutter_body, self.base_models[self.obj_id[i]]
            m.body_mass[b] = nominal.body_mass[b] * scale
            m.body_inertia[b] = nominal.body_inertia[b] * scale

    def _sample_scene(self, ids):
        t = self.cfg.task
        start, start_yaw = super()._sample_scene(ids)
        n = len(ids)
        obj = self.obj_id[ids]
        start_pose, goal = np.c_[start, start_yaw], self.goal[ids]
        want = self.rng.random(n) < t.clutter_in_way
        home = np.zeros((n, 3))
        sweep_gap = np.full(n, np.inf)
        todo = np.arange(n)
        m = 8                                          # candidates per scene and round, checked in one batch
        for attempt in range(8):
            if len(todo) == 0:
                break
            k = len(todo)
            rows = np.repeat(todo, m)
            # Near the path: around the corridor the object sweeps from start to goal.
            seg = goal[rows, :2] - start[rows]
            normal = np.stack([-seg[:, 1], seg[:, 0]], -1) / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
            corridor = (start[rows] + self.rng.uniform(-0.25, 1.25, k * m)[:, None] * seg
                        + normal * self.rng.uniform(-0.075, 0.075, k * m)[:, None])
            anywhere = self._sample_region(k * m, (0.16, 0.24), t.obj_az + 0.1)
            near = np.repeat(want[todo] & (attempt < 6), m)
            cand = np.where(near[:, None], corridor, anywhere)
            cand_pose = np.c_[cand, self.rng.uniform(-math.pi, math.pi, k * m)]
            ok = self._in_region(cand, (0.155, 0.245), t.obj_az + 0.12)
            gap = np.full(k * m, -1.0)
            gap[ok] = self._sweep_gap(cand_pose[ok], start_pose[rows[ok]], goal[rows[ok]], obj[rows[ok]])
            # Never block the straight route (the band is too narrow for reliable detours);
            # "near" blocks sit within 25 mm of the swept path, where careless pushing clips them.
            ok &= (gap > 0.004) & (~near | (gap < 0.025))
            ok = ok.reshape(k, m)
            found = ok.any(1)
            pick = np.arange(k) * m + ok.argmax(1)
            home[todo[found]] = cand_pose[pick[found]]
            sweep_gap[todo[found]] = gap[pick[found]]
            todo = todo[~found]
        if len(todo):
            # Rare: start, goal and path cover the band. Park the block out of reach at the table edge.
            side = -np.sign(np.arctan2(start[todo, 1], start[todo, 0]) + 1e-9)
            home[todo] = np.c_[0.29 * np.cos(side * 0.82), 0.29 * np.sin(side * 0.82), np.zeros(len(todo))]
            sweep_gap[todo] = np.inf
        self.blocking[ids] = False
        in_way = sweep_gap < 0.025
        self.clutter_home[ids] = home
        self.in_way[ids] = in_way
        return start, start_yaw

    def _tool_clear(self, ids, cand, start, start_yaw):
        ok = super()._tool_clear(ids, cand, start, start_yaw)
        home = self.clutter_home[ids]
        tab = self.clutter_tab
        sdf = footprint_sdf(cand[:, None, :], home, np.repeat(tab.boxes, len(ids), 0),
                            np.repeat(tab.disk_radius, len(ids)))[:, 0]
        return ok & (sdf > self.cfg.scene.pusher_radius + 0.01)

    def _place(self, ids, start, start_yaw, tool):
        super()._place(ids, start, start_yaw, tool)
        home = self.clutter_home[ids]
        o = self.qpos_adr + self.idx.clutter_qpos
        h = OBJECTS[self.cfg.task.clutter].height / 2
        self.state[ids, o:o + 7] = np.c_[home[:, :2], np.full(len(ids), h), np.cos(home[:, 2] / 2),
                                         np.zeros(len(ids)), np.zeros(len(ids)), np.sin(home[:, 2] / 2)]
        self.clutter_hist[ids] = home[:, None, :]
        self.clutter_obs[ids] = home
        self.prev_disturb[ids] = 0.0

    # ----------------------------------------------------------- task hooks
    def _after_physics(self):
        r = self.cfg.rand
        pose = self.clutter_pose()
        self.clutter_hist = np.roll(self.clutter_hist, 1, axis=1)
        self.clutter_hist[:, 0] = pose
        seen = self.clutter_hist[np.arange(self.n), self.latency].copy()
        if r.enabled:
            seen[:, :2] += self.noise_rng.normal(0, r.obs_pos_noise, (self.n, 2))
            seen[:, 2] = wrap(seen[:, 2] + self.noise_rng.normal(0, r.obs_yaw_noise, self.n))
            keep = self.noise_rng.random(self.n) < r.obs_dropout
            seen[keep] = self.clutter_obs[keep]
        self.clutter_obs = seen

    def _reward_extra(self):
        t = self.cfg.task
        pos, yaw = self.disturbance()
        d = pos + 0.02 * yaw
        out = -t.w_disturb * (d - self.prev_disturb) / 0.01 - t.w_displaced * ((pos > t.clutter_tol) | (yaw > t.clutter_yaw_tol))
        self.prev_disturb = d
        return out

    def _task_success(self, success):
        t = self.cfg.task
        pos, yaw = self.disturbance()
        return success & (pos <= t.clutter_tol) & (yaw <= t.clutter_yaw_tol)

    def _extra_obs(self, ids, pose, tool_xy, actor=False):
        c = self.clutter_obs[ids] if actor else self.clutter_pose()[ids]
        home, goal = self.clutter_home[ids], self.goal[ids]
        return np.concatenate([
            (c[:, :2] - CENTRE) / 0.1, np.cos(c[:, 2:3]), np.sin(c[:, 2:3]),
            (c[:, :2] - tool_xy) / 0.05, (c[:, :2] - pose[:, :2]) / 0.05,
            (c[:, :2] - home[:, :2]) / 0.01, (self._clutter_yaw_err(c[:, 2], home[:, 2]) / 0.1)[:, None],
            (home[:, :2] - goal[:, :2]) / 0.05,
            # Clearances (the danger signals): object footprint to block, rod to block.
            np.clip(self._gap_to_object(c, pose, self.obj_id[ids]), -0.01, 0.05)[:, None] / 0.02,
            np.clip(footprint_sdf(tool_xy[:, None, :], c, np.repeat(self.clutter_tab.boxes, len(ids), 0),
                                  np.repeat(self.clutter_tab.disk_radius, len(ids)))[:, 0]
                    - self.cfg.scene.pusher_radius, -0.01, 0.05)[:, None] / 0.02,
        ], -1)

    def _episode_extras(self, ids):
        pos, yaw = self.disturbance(ids)
        t = self.cfg.task
        return {"clutter_moved_mm": pos * 1e3, "disturbed": (pos > t.clutter_tol) | (yaw > t.clutter_yaw_tol),
                "in_way": self.in_way[ids].copy(), "blocking": self.blocking[ids].copy()}
