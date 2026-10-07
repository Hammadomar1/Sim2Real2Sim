"""Level 3 skill: dock the box into an inner corner of the T, touching both the stem and the bar.

    python -m s2r2s.train --run dock_a --objects box --clutter tee --dock --init runs/clutter_box_v2/best.pt \\
        --init-std 0.15 --no-curriculum --episode-seconds 10 --minutes 75
    python -m s2r2s.level3 --record-dock-starts runs/dock_b/start_bank.npz      # where docking starts in Level 3
    python -m s2r2s.train --run dock_b --objects box --clutter tee --dock --init runs/dock_a/best.pt --init-std 0.12 \\
        --no-curriculum --episode-seconds 10 --minutes 40 --set task.dock_start_bank=runs/dock_b/start_bank.npz

The T stays where it is; the box starts 12-45 mm out from one of the T's two inner corners (where the bar
overhangs the stem), roughly along the corner's diagonal, and must be pushed in until it touches both faces.
With a start bank (``dock_start_bank``), a share of the episodes (``dock_bank_share``) starts instead where docking
starts in Level 3: there the box skill often leaves the rod between the box and the T, close to the T.
The target is the corner where the T was when docking started, and it stays there: a target that follows the T
lets the skill chase it, shoving both blocks along once the box presses on the T. Success: the box within
``dock_pos_tol`` / ``dock_yaw_tol`` of the corner of the T where it is at the end (so the pieces touch), with the
T moved at most ``clutter_tol`` / ``clutter_yaw_tol`` (5 mm / 5 deg here).

Everything else is the clutter task (``ClutterEnv``): the camera model for both blocks, the rod guard (so only
the box, never the rod, touches the T), and the costs for moving the T. The proximity cost applies to the rod
only (``near_object`` off): the box has to touch the T. The pieces slide easily along each other (low friction
between them), so the corner's two faces guide the box in, like a chamfer would on printed parts.
"""
from __future__ import annotations

import math

import numpy as np

from .clutter import ClutterEnv
from .env import EnvConfig
from .objects import OBJECTS


def corner_offset(base: str, piece: str, side) -> np.ndarray:
    """Pose (x, y, yaw) of the square ``piece`` nested in the inner corner of the T ``base`` on ``side`` (+1 or -1),
    in the base's frame: against the stem's side and under the bar."""
    boxes = OBJECTS[base].centred_boxes
    bar, stem = sorted(boxes, key=lambda b: b[2])[::-1]          # the bar is the wider box
    half = OBJECTS[piece].centred_boxes[0][2]
    side = np.asarray(side, dtype=float)
    return np.stack([stem[0] + side * (stem[2] + half), np.full_like(side, bar[1] - bar[3] - half),
                     np.zeros_like(side)], -1)


def compose(pose, offset):
    """``offset`` (n, 3), given in the frame of ``pose`` (n, 3), in the world."""
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    return np.c_[pose[:, 0] + c * offset[:, 0] - s * offset[:, 1], pose[:, 1] + s * offset[:, 0] + c * offset[:, 1],
                 pose[:, 2] + offset[:, 2]]


class DockEnv(ClutterEnv):
    def __init__(self, cfg: EnvConfig):
        if cfg.task.objects != ("box",) or cfg.task.clutter != "tee":
            raise ValueError("DockEnv docks the box (objects=('box',)) into the T (clutter='tee')")
        t = cfg.task
        t.dock, t.near_object = True, False
        t.clutter_tol, t.clutter_yaw_tol = 0.005, math.radians(5)     # the T may move at most this while docking
        self.side = np.ones(cfg.num_envs)
        self.bank = dict(np.load(t.dock_start_bank)) if t.dock_start_bank else None
        self.bank_rod = np.full((cfg.num_envs, 2), np.nan)        # rod start from the bank, per world
        super().__init__(cfg)

    def docked_pose(self, tee_pose, side):
        """Where the box sits when docked in the corner on ``side`` of a T at ``tee_pose``."""
        return compose(tee_pose, corner_offset(self.cfg.task.clutter, self.cfg.task.objects[0], side))

    def approach(self, tee_pose, side):
        """Unit vector pointing out of the corner along its diagonal, in the world."""
        a = tee_pose[:, 2] + np.arctan2(-1.0, side)
        return np.stack([np.cos(a), np.sin(a)], -1)

    def _sample_scene(self, ids):
        t = self.cfg.task
        n = len(ids)
        tee, side, start = np.zeros((n, 3)), np.ones(n), np.zeros((n, 3))
        todo = np.arange(n)
        for _ in range(100):
            if len(todo) == 0:
                break
            k = len(todo)
            tp = np.c_[self._sample_region(k, t.obj_r, t.obj_az), self.rng.uniform(-math.pi, math.pi, k)]
            sd = np.where(self.rng.random(k) < 0.5, 1.0, -1.0)
            dock = self.docked_pose(tp, sd)
            ang = np.arctan2(*self.approach(tp, sd).T[::-1]) + self.rng.uniform(-0.7, 0.7, k)
            dist = self.rng.uniform(0.012, 0.045, k)
            st = np.c_[dock[:, :2] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1),
                       dock[:, 2] + self.rng.uniform(-0.45, 0.45, k)]
            ok = self._in_region(dock[:, :2], (t.obj_r[0] - 0.015, t.obj_r[1] + 0.015), t.obj_az + 0.1)
            ok &= self._in_region(st[:, :2], (t.obj_r[0] - 0.02, t.obj_r[1] + 0.02), t.obj_az + 0.12)
            ok &= self._gap_to_object(tp, st, np.zeros(k, dtype=np.int64)) > 0.003
            tee[todo[ok]], side[todo[ok]], start[todo[ok]] = tp[ok], sd[ok], st[ok]
            todo = todo[~ok]
        if len(todo):
            raise RuntimeError(f"no docking scene for {len(todo)} worlds")
        if self.bank is not None:
            # Starts recorded in Level 3 (where the box skill leaves the box and the rod), for some episodes.
            use = self.rng.random(n) < t.dock_bank_share
            j = self.rng.integers(len(self.bank["side"]), size=n)
            tee[use], side[use], start[use] = self.bank["tee"][j[use]], self.bank["side"][j[use]], self.bank["box"][j[use]]
            self.bank_rod[ids] = np.where(use[:, None], self.bank["rod"][j], np.nan)
        self.clutter_home[ids], self.side[ids] = tee, side
        self.goal[ids] = self.docked_pose(tee, side)
        self.difficulty_ep[ids] = 1.0
        self.sweep_gap[ids] = self._gap_to_object(tee, start, np.zeros(n, dtype=np.int64))
        self.in_way[ids] = True
        return start[:, :2], start[:, 2]

    def _sample_tool(self, ids, start, start_yaw):
        """Where the rod is when docking begins in Level 3: right beside the box (its surface 0.5-12 mm away),
        behind it (30 %) or on any side, often the T's side (50 %), and at least 4 mm from the T, which the rod
        guard allows; otherwise anywhere clear of both."""
        tool = super()._sample_tool(ids, start, start_yaw)
        banked = np.isfinite(self.bank_rod[ids, 0])
        t, rod = self.cfg.task, self.cfg.scene.pusher_radius
        u = self.rng.random(len(ids))
        pose = np.c_[start, start_yaw]
        for rows, behind in ((np.flatnonzero(u < 0.3), True), (np.flatnonzero((u >= 0.3) & (u < 0.8)), False)):
            if len(rows) == 0:
                continue
            if behind:
                out = self.approach(self.clutter_home[ids[rows]], self.side[ids[rows]])
                ang = np.arctan2(out[:, 1], out[:, 0]) + self.rng.uniform(-0.5, 0.5, len(rows))
            else:
                ang = self.rng.uniform(-math.pi, math.pi, len(rows))
            # Walk out from the box's centre until the rod's surface is the drawn distance from the box.
            want = self.rng.uniform(0.0005, 0.012, len(rows))
            direction = np.stack([np.cos(ang), np.sin(ang)], -1)
            d = np.zeros(len(rows))
            for _ in range(30):
                cand = pose[rows, :2] + d[:, None] * direction
                d += np.maximum(want - self._surface_distance(cand, pose[rows], self.obj_id[ids[rows]]), 0.0) + 1e-4 * (d == 0)
            cand = pose[rows, :2] + d[:, None] * direction
            ok = self._block_clearance(cand, self.clutter_home[ids[rows]]) - rod > 0.004
            ok &= self._in_region(cand, (t.tool_r[0] + 0.01, t.tool_r[1] - 0.01), t.tool_az - 0.1)
            tool[rows[ok]] = cand[ok]
        tool[banked] = self.bank_rod[ids[banked]]
        return tool

    def _task_success(self, success):
        t = self.cfg.task
        pose = self.object_pose()
        target = self.docked_pose(self.clutter_pose(), self.side)        # touching the T where it is now
        yaw_goal = self._symmetric_goal_yaw(pose[:, 2], target[:, 2], self.obj_id)
        pos = np.linalg.norm(pose[:, :2] - target[:, :2], axis=-1)
        yaw = np.abs((pose[:, 2] - yaw_goal + math.pi) % (2 * math.pi) - math.pi)
        moved, turned = self.disturbance()
        return (pos <= t.dock_pos_tol) & (yaw <= t.dock_yaw_tol) & (moved <= t.clutter_tol) & (turned <= t.clutter_yaw_tol)
