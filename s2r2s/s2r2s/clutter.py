"""Push the object to its goal pose while a second block stays where it is.

The block (``TaskConfig.clutter``, e.g. a 40 mm box) sits close beside the path the object sweeps
from its start to its goal (within ``clutter_near``) in ``clutter_in_way`` of the episodes, and
anywhere in the work band otherwise. It never comes closer to that straight sweep than
``clutter_min_gap``: the rod's diameter plus the guard margin below, so the rod always fits
between the block and the object's path. (A first version let it block the path: in this narrow
band there is often no room to detour, and those scenes were unsolvable without touching it.)

The rod, not the object, is what usually hits the block: early in an episode it drives straight
at the object, through a block in the way. So the controller has a **rod guard**
(``_guard_command``, shared with the real arm): it never commands the rod closer than
``rod_guard`` to the block's estimated footprint, and the rod slides along the block instead.
A guard alone lets the rod get stuck leaning on it, so the part of each command the guard has to
remove costs reward (``w_guard``): the policy learns to plan around the block.

Moving the block costs reward (``w_disturb`` per cm, symmetric, so pushing it back is rewarded)
plus a per-step cost while it is out of tolerance, large enough that a disturbed block spoils the
episode. A smaller cost grows as the rod or the object comes within ``near_margin`` of the block.
Success needs the object at its goal *and* the block within ``clutter_tol`` / ``clutter_yaw_tol``
of where it started.

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


def box_sdf(px, py, hx, hy):
    """Signed distance from points (px, py), given in a box's own frame, to the box with half sizes hx, hy."""
    qx, qy = np.abs(px) - hx, np.abs(py) - hy
    return np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2) + np.minimum(np.maximum(qx, qy), 0.0)


def rect_gap(a, b):
    """Signed distance between oriented rectangles ``a`` and ``b`` (elementwise; a zero half size is a point).

    Each rectangle is a tuple (x, y, cos, sin, half x, half y) of broadcastable arrays. Apart, the result is
    the exact clearance: the closest points of two convex polygons include a corner of one of them.
    Overlapping, it is minus the penetration depth, from the separating-axis test over the four edge
    normals (corners alone would miss two bars crossing like a plus sign).
    """
    ax, ay, ac, as_, ahx, ahy = a
    bx, by, bc, bs, bhx, bhy = b
    dx, dy = bx - ax, by - ay
    c, s = ac * bc + as_ * bs, ac * bs - as_ * bc              # cos, sin of (angle b - angle a)
    abs_c, abs_s = np.abs(c), np.abs(s)
    bxa, bya = dx * ac + dy * as_, dy * ac - dx * as_          # b's centre in a's frame
    axb, ayb = -(dx * bc + dy * bs), dx * bs - dy * bc         # a's centre in b's frame
    sep = np.maximum(np.maximum(np.abs(bxa) - ahx - bhx * abs_c - bhy * abs_s, np.abs(bya) - ahy - bhx * abs_s - bhy * abs_c),
                     np.maximum(np.abs(axb) - bhx - ahx * abs_c - ahy * abs_s, np.abs(ayb) - bhy - ahx * abs_s - ahy * abs_c))
    apart = np.inf
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            apart = np.minimum(apart, box_sdf(bxa + sx * bhx * c - sy * bhy * s, bya + sx * bhx * s + sy * bhy * c, ahx, ahy))
            apart = np.minimum(apart, box_sdf(axb + sx * ahx * c + sy * ahy * s, ayb - sx * ahx * s + sy * ahy * c, bhx, bhy))
    return np.where(sep > 0, apart, sep)


def footprint_parts(name: str):
    """A footprint as rectangles (centre x, y, half x, half y; object frame) grown by a radius.

    Boxes and box unions are themselves with radius 0; a disk is its centre point grown by its radius.
    """
    shape = OBJECTS[name]
    if shape.disk_radius > 0:
        return np.zeros((1, 4)), shape.disk_radius
    return np.array(shape.centred_boxes, dtype=float), 0.0


def place_parts(parts, pose):
    """Footprint rectangles (k, 4) or (n, k, 4) placed at ``pose`` (n, 3): (x, y, cos, sin, half x, half y), each (n, k)."""
    c, s = np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3])
    px, py = parts[..., 0], parts[..., 1]
    x, y = pose[:, 0:1] + c * px - s * py, pose[:, 1:2] + s * px + c * py
    return x, y, np.broadcast_to(c, x.shape), np.broadcast_to(s, x.shape), np.broadcast_to(parts[..., 2], x.shape), \
        np.broadcast_to(parts[..., 3], x.shape)


class ClutterEnv(PushEnv):
    def __init__(self, cfg: EnvConfig):
        if not cfg.task.clutter:
            raise ValueError("ClutterEnv needs TaskConfig.clutter (an object name)")
        cfg.scene.clutter = cfg.task.clutter
        n = cfg.num_envs
        self.clutter_tab = object_tables([cfg.task.clutter])
        self.clutter_parts, self.clutter_grow = footprint_parts(cfg.task.clutter)
        # Object footprints padded to a common number of rectangles (repeating a rectangle changes no distance).
        parts = [footprint_parts(name) for name in cfg.task.objects]
        k = max(len(p) for p, _ in parts)
        self.object_parts = np.stack([np.concatenate([p, np.repeat(p[:1], k - len(p), 0)]) for p, _ in parts])
        self.object_grow = np.array([g for _, g in parts])
        self.clutter_home = np.zeros((n, 3))
        self.sweep_gap = np.zeros(n)
        self.clutter_obs = np.zeros((n, 3))
        self.clutter_hist = np.zeros((n, max(cfg.rand.obs_latency_steps[1], 1) + 2, 3))
        self.prev_disturb = np.zeros(n)
        self.in_way = np.zeros(n, dtype=bool)
        self.first_touch = np.zeros(n, dtype=np.int64)    # what moved the block first: 0 nothing, 1 rod, 2 object
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
        """Clearance between the block and the object footprints (exact when apart; negative: overlapping)."""
        obj = place_parts(self.object_parts[obj_id], object_pose)
        block = place_parts(self.clutter_parts, clutter_pose)
        gap = rect_gap([v[:, :, None] for v in obj], [v[:, None, :] for v in block])
        return gap.min((1, 2)) - self.object_grow[obj_id] - self.clutter_grow

    def _block_frame(self, xy, clutter_pose):
        """Points (n, 2) in the frame of each block rectangle, and the rectangles."""
        x, y, c, s, hx, hy = block = place_parts(self.clutter_parts, clutter_pose)
        dx, dy = xy[:, 0:1] - x, xy[:, 1:2] - y
        return dx * c + dy * s, dy * c - dx * s, block

    def _block_clearance(self, xy, clutter_pose):
        """Distance from points (n, 2) to the block footprint (negative: inside)."""
        lx, ly, (_, _, _, _, hx, hy) = self._block_frame(xy, clutter_pose)
        return box_sdf(lx, ly, hx, hy).min(1) - self.clutter_grow

    def _block_nearest(self, xy, clutter_pose):
        """The block footprint's nearest point to each point (n, 2) (the point itself when inside)."""
        lx, ly, (x, y, c, s, hx, hy) = self._block_frame(xy, clutter_pose)
        lx, ly = np.clip(lx, -hx, hx), np.clip(ly, -hy, hy)
        near = np.stack([x + lx * c - ly * s, y + lx * s + ly * c], -1)                 # per rectangle (n, k, 2)
        d = xy[:, None, :] - near
        dist = np.sqrt((d ** 2).sum(-1))
        if self.clutter_grow > 0:
            near = near + d * np.minimum(self.clutter_grow / np.maximum(dist, 1e-9), 1.0)[..., None]
        return near[np.arange(len(xy)), dist.argmin(1)]

    def _guard_command(self, cmd_xy):
        """The controller never commands the rod closer than ``rod_guard`` to the block: such commands are
        moved out to that distance, so the rod slides along the block instead of pushing it.

        Uses the block pose the robot believes (the camera estimate), so it runs unchanged on the real arm.
        """
        t = self.cfg.task
        if t.rod_guard <= 0:
            return cmd_xy
        need = self.cfg.scene.pusher_radius + t.rod_guard + self.clutter_grow
        r = np.arange(len(cmd_xy))
        # A block made of several rectangles (a T): leaving one may enter another, so repeat a few times.
        for _ in range(1 if len(self.clutter_parts) == 1 else 3):
            lx, ly, (x, y, c, s, hx, hy) = self._block_frame(cmd_xy, self.clutter_obs)
            # Nearest point on each rectangle's outline and the outward normal there (inside: exit by the nearest side).
            inside = (np.abs(lx) < hx) & (np.abs(ly) < hy)
            by_x = hx - np.abs(lx) < hy - np.abs(ly)
            sx, sy = np.where(lx >= 0, 1.0, -1.0), np.where(ly >= 0, 1.0, -1.0)
            px = np.where(inside & by_x, sx * hx, np.clip(lx, -hx, hx))
            py = np.where(inside & ~by_x, sy * hy, np.clip(ly, -hy, hy))
            dist = np.hypot(lx - px, ly - py)
            nx = np.where(inside, np.where(by_x, sx, 0.0), np.where(dist > 1e-12, (lx - px) / np.maximum(dist, 1e-12), 1.0))
            ny = np.where(inside, np.where(by_x, 0.0, sy), np.where(dist > 1e-12, (ly - py) / np.maximum(dist, 1e-12), 0.0))
            signed = np.where(inside, -dist, dist)
            k = signed.argmin(1)
            bx, by = px[r, k] + nx[r, k] * need, py[r, k] + ny[r, k] * need
            target = np.stack([x[r, k] + c[r, k] * bx - s[r, k] * by, y[r, k] + s[r, k] * bx + c[r, k] * by], -1)
            cmd_xy = np.where((signed[r, k] < need)[:, None], target, cmd_xy)
        return cmd_xy

    def _sweep_gap(self, clutter_pose, start_pose, goal, obj_id, steps: int = 33):
        """Smallest clearance between the block and the object moved straight (and turned) from start to goal.

        33 poses: at most 4 mm of travel (or 6 deg of turn) between neighbours at full difficulty.
        """
        n = len(obj_id)
        f = np.linspace(0, 1, steps)[:, None, None]
        turn = wrap(goal[:, 2] - start_pose[:, 2])
        mid = np.concatenate([start_pose[None, :, :2] + f * (goal[None, :, :2] - start_pose[None, :, :2]),
                              start_pose[None, :, 2:] + f * turn[None, :, None]], -1).reshape(-1, 3)
        gap = self._gap_to_object(np.tile(clutter_pose, (steps, 1)), mid, np.tile(obj_id, steps))
        return gap.reshape(steps, n).min(0)

    # ------------------------------------------------------------- episodes
    def _randomize(self, i):
        # The block's mass first: the base class then recomputes the model's derived constants (mj_setConst).
        r = self.cfg.rand
        scale = self.param_rng.uniform(*r.mass_scale)
        if r.enabled:
            m, b, nominal = self.models[i], self.idx.clutter_body, self.base_models[self.obj_id[i]]
            m.body_mass[b] = nominal.body_mass[b] * scale
            m.body_inertia[b] = nominal.body_inertia[b] * scale
        super()._randomize(i)

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
        m = 4                                          # candidates per scene and round, checked in one batch
        for attempt in range(6):
            if len(todo) == 0:
                break
            rows = np.repeat(todo, m)
            k = len(rows)
            near = want[rows] & (attempt < 4)
            yaw = self.rng.uniform(-math.pi, math.pi, k)
            # Near: from a point on the object's path, slide out along a random direction until the clearance
            # to the swept path reaches a random target between clutter_min_gap and clutter_near (secant steps).
            anchor = start[rows] + self.rng.random(k)[:, None] * (goal[rows, :2] - start[rows])
            phi = self.rng.uniform(-math.pi, math.pi, k)
            direction = np.stack([np.cos(phi), np.sin(phi)], -1)
            target = self.rng.uniform(t.clutter_min_gap, t.clutter_near, k)
            cand = np.where(near[:, None], anchor + 0.07 * direction, self._sample_region(k, (0.15, 0.25), t.obj_az + 0.12))
            gap = self._sweep_gap(np.c_[cand, yaw], start_pose[rows], goal[rows], obj[rows])
            r = np.flatnonzero(near)
            off, slope = np.full(len(r), 0.07), np.ones(len(r))
            for _ in range(2):
                new = np.clip(off + np.clip((target[r] - gap[r]) / slope, -0.05, 0.05), 0.0, 0.15)
                g = self._sweep_gap(np.c_[anchor[r] + new[:, None] * direction[r], yaw[r]], start_pose[rows[r]],
                                    goal[rows[r]], obj[rows[r]])
                moved = np.abs(new - off) > 1e-5
                slope = np.where(moved, np.clip((g - gap[r]) / np.where(moved, new - off, 1.0), 0.2, 1.5), slope)
                off, gap[r] = new, g
            cand[r] = anchor[r] + off[:, None] * direction[r]
            # Never closer to the straight route than clutter_min_gap (the guarded rod's width); "near" blocks
            # sit within clutter_near of the swept path, where careless pushing or circling clips them.
            ok = (self._in_region(cand, (0.14, 0.26), t.obj_az + 0.15) & (gap > t.clutter_min_gap)
                  & (~near | (gap < t.clutter_near))).reshape(-1, m)
            found = ok.any(1)
            pick = np.arange(len(todo)) * m + ok.argmax(1)
            home[todo[found]] = np.c_[cand, yaw][pick[found]]
            sweep_gap[todo[found]] = gap[pick[found]]
            todo = todo[~found]
        if len(todo):
            # Rare: start, goal and path cover the band. Park the block beyond the rod's reach (tip 0.275 m plus
            # its radius), its local x axis pointing away from the robot.
            side = -np.sign(np.arctan2(start[todo, 1], start[todo, 0]) + 1e-9)
            p = self.clutter_parts
            park = 0.29 + float(np.max(p[:, 2] - p[:, 0])) + self.clutter_grow     # 0.31 m for the 40 mm box
            home[todo] = np.c_[park * np.cos(side * 0.82), park * np.sin(side * 0.82), side * 0.82]
            sweep_gap[todo] = np.inf
        self.clutter_home[ids] = home
        self.in_way[ids] = sweep_gap < t.clutter_near
        self.sweep_gap[ids] = sweep_gap
        return start, start_yaw

    def _tool_clear(self, ids, cand, start, start_yaw):
        ok = super()._tool_clear(ids, cand, start, start_yaw)
        return ok & (self._block_clearance(cand, self.clutter_home[ids]) > self.cfg.scene.pusher_radius + 0.01)

    def _sample_tool(self, ids, start, start_yaw):
        tool = super()._sample_tool(ids, start, start_yaw)
        t = self.cfg.task
        if t.tool_near_block <= 0:
            return tool
        # Some episodes start with the rod right beside the block, as after placing it (its surface 4-12 mm away).
        near = np.flatnonzero(self.rng.random(len(ids)) < t.tool_near_block)
        rod, todo = self.cfg.scene.pusher_radius, near
        for _ in range(20):
            if len(todo) == 0:
                break
            home = self.clutter_home[ids[todo]]
            ang = self.rng.uniform(-math.pi, math.pi, len(todo))
            # Walk out from the block's centre until the rod's surface is the drawn distance from it.
            want = rod + self.rng.uniform(0.004, 0.012, len(todo))
            d = np.zeros(len(todo))
            for _ in range(30):
                cand = home[:, :2] + d[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
                d += np.maximum(want - self._block_clearance(cand, home), 0.0) + 1e-4 * (d == 0)
            cand = home[:, :2] + d[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
            obj_clear = PushEnv._tool_clear(self, ids[todo], cand, start[todo], start_yaw[todo])
            reach = self._in_region(cand, (t.tool_r[0] + 0.01, t.tool_r[1] - 0.01), t.tool_az - 0.1)
            ok = obj_clear & reach
            tool[todo[ok]] = cand[ok]
            todo = todo[~ok]
        return tool

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
        self.first_touch[ids] = 0

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
        # Keep a margin: cost grows linearly as the rod or the object closes in on the block.
        block = self.clutter_pose()
        rod = self._block_clearance(self.tip[:, :2], block) - self.cfg.scene.pusher_radius
        obj = self._gap_to_object(block, self.object_pose(), self.obj_id)
        # Bookkeeping for evaluation: was the rod or the object closer when the block first moved 2 mm?
        self.first_touch = np.where((self.first_touch == 0) & (pos > 0.002), np.where(rod < obj, 1, 2), self.first_touch)
        out -= t.w_near * np.clip(1.0 - (np.minimum(rod, obj) if t.near_object else rod) / t.near_margin, 0.0, 1.0)
        # Leaning on the rod guard (commanding into the block) gets the rod stuck: plan around the block instead.
        return out - t.w_guard * self.cmd_deflection / (t.max_speed * t.control_dt)

    def _task_success(self, success):
        t = self.cfg.task
        pos, yaw = self.disturbance()
        return success & (pos <= t.clutter_tol) & (yaw <= t.clutter_yaw_tol)

    def _extra_obs(self, ids, pose, tool_xy, actor=False):
        c = self.clutter_obs[ids] if actor else self.clutter_pose()[ids]
        home, goal = self.clutter_home[ids], self.goal[ids]
        s = self.clutter_tab.symmetry[0]          # yaw features that look the same for symmetric poses
        sym = np.c_[np.cos(s * c[:, 2]), np.sin(s * c[:, 2])] if s > 0 else np.zeros((len(ids), 2))
        return np.concatenate([
            (c[:, :2] - CENTRE) / 0.1, sym,
            (c[:, :2] - tool_xy) / 0.05, (c[:, :2] - pose[:, :2]) / 0.05,
            (c[:, :2] - home[:, :2]) / 0.01, (self._clutter_yaw_err(c[:, 2], home[:, 2]) / 0.1)[:, None],
            (home[:, :2] - goal[:, :2]) / 0.05,
            # The danger signals: object footprint to block and rod to block clearances, and where on
            # the block the rod would hit.
            np.clip(self._gap_to_object(c, pose, self.obj_id[ids]), -0.01, 0.05)[:, None] / 0.02,
            np.clip(self._block_clearance(tool_xy, c) - self.cfg.scene.pusher_radius, -0.01, 0.05)[:, None] / 0.02,
            (self._block_nearest(tool_xy, c) - tool_xy) / 0.05,
        ], -1)

    def _episode_extras(self, ids):
        pos, yaw = self.disturbance(ids)
        t = self.cfg.task
        return {"clutter_moved_mm": pos * 1e3, "disturbed": (pos > t.clutter_tol) | (yaw > t.clutter_yaw_tol),
                "touched": pos > 0.002, "touched_by_rod": self.first_touch[ids] == 1, "in_way": self.in_way[ids].copy(),
                "sweep_gap_mm": np.minimum(self.sweep_gap[ids], 1.0) * 1e3}

    # ------------------------------------------------------------- deployment
    def sync_measurements(self, qarm, pose, first=False, block=None):
        """Deployment (see ``PushEnv.sync_measurements``), plus ``block`` (n, 3) or None: the camera's block pose.

        At the start of a trial (``first``) the block's pose also fixes where it belongs.
        """
        if block is not None:
            block = np.atleast_2d(np.asarray(block, dtype=np.float64))
            o = self.qpos_adr + self.idx.clutter_qpos
            self.state[:, o:o + 7] = np.c_[block[:, :2], np.full(len(block), OBJECTS[self.cfg.task.clutter].height / 2),
                                           np.cos(block[:, 2] / 2), 0 * block[:, :2], np.sin(block[:, 2] / 2)]
            self.clutter_obs = block.copy()
            if first:
                self.clutter_home = block.copy()
                self.clutter_hist[:] = block[:, None, :]
                self.prev_disturb[:] = 0.0
        elif first:
            raise ValueError("a clutter policy needs the block's pose at the start of a trial")
        return super().sync_measurements(qarm, pose, first)
