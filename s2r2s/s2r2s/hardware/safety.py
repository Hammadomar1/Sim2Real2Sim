"""Safety for the SO-101: every joint command is checked before it reaches the motors.

LeRobot writes whatever goal position it is given. The servos clamp each joint to the range swept during
LeRobot's calibration, but that range ends at the mechanical stops, and no single-joint limit stops a combination
of angles from driving the gripper into the mat, the base, a wall or the arm itself. ``SafetyGuard`` sits between
every command source (the policy runner, the bring-up tools, teleoperation) and the motors, on the real arm and on
the simulated one alike (robots.py):

  before a command  joint limits (inside both the model's and the calibration's ranges, a margin from the stops),
                    step and speed limits that shrink near anything (the arm brakes inside its margin), and a
                    collision check of the target and of the straight joint-space path to it: arm links, rod,
                    base, mat, floor, table, lightbox walls and camera, each with a clearance. An unsafe target is
                    cut back to the last safe point on that path, and a refused command holds the previous one
                    (not the measured pose, which would let gravity win). From a pose that is already too close
                    (the arm folded at rest, the rod pressed into the mat) only moves that make nothing worse go
                    through, so the arm can always be moved out.
  while moving      ``ServoMonitor``: error flags, temperature, full load without moving, a joint told to move
                    that does not (blocked), and a joint far from where it was sent (falling or pushed). A fault
                    holds the arm where it is (torque stays on: the arm does not drop) and raises ``SafetyStop``.
  at start          ``SafetyGuard.selftest()``: known-unsafe targets (rod into the mat, a joint past its range,
                    the arm into itself, a wall, a jump, NaN) must be refused and safe moves let through.
                    ``guard_for`` refuses to build a guard that fails it, and refuses a joint map calibrated for
                    another workspace (with or without ``--lightbox``).

    python -m s2r2s.hardware.safety                                       # self-test of the guard (no robot)
    python -m s2r2s.hardware.safety --robot so101 --port COM5 --map calibration\\joint_map.json --lightbox
    python -m s2r2s.hardware.safety --robot so101 ... --wiggle            # plus small, slow moves of every joint

Angles are LeRobot degrees (the robot frame) at the interface and MuJoCo radians inside, through the joint map.
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..scene import (ARM_JOINTS, LIGHTBOX_BOARD, LIGHTBOX_FLOOR_X, LIGHTBOX_HALF_WIDTH, LIGHTBOX_MAT_HALF_WIDTH,
                     LIGHTBOX_MAT_X, MENAGERIE_SO101, SceneConfig)
from .joint_map import JointMap


@dataclass
class SafetyLimits:
    joint_margin_deg: float = 3.0       # stay this far inside each joint's calibrated range (the real stops);
                                        # the model's range is a limit too, without margin (pushing uses the
                                        # wrist up to 94.9 of its 95 deg)
    rod_floor: float = 0.0              # lowest rod tip relative to the surface below it (m): it may touch the
                                        # mat, never be commanded into it (a position servo pressing on a hard
                                        # surface reaches full torque within a fraction of a degree)
    rod_clearance: float = 0.010        # rod from the walls and the camera (m)
    link_clearance: float = 0.010       # arm links from the mat, floor, walls and camera (m)
    self_clearance: float = 0.005       # arm links and rod from each other and from the base (m)
    max_step_deg: float = 5.0           # largest change per command, from the measured joints
    max_speed_deg_s: float = 120.0      # largest joint speed (the STS3215 does about 270 deg/s unloaded)
    slow_zone: float = 0.020            # closer than this to any clearance (m) ...
    min_speed_fraction: float = 0.2     # ... speed and step shrink, down to this share at the clearance, so the
                                        # arm brakes inside its margin (not for the rod over the mat: it pushes
                                        # 5 mm above it)
    path_step_deg: float = 1.0          # the path to a target is checked at least this finely
    escape_slack: float = 1.0           # leaving a pose that is already too close: how much deeper (mm or deg)
                                        # the way out may dip on the way (the command itself may not be deeper)
    # ServoMonitor (real arm; the simulated arm reports load from its actuator forces)
    max_load: float = 0.80              # share of the servo's maximum torque, held load_seconds while the joint
    load_seconds: float = 0.3           # moves less than 2 deg: something is resisting the arm (a fast move
                                        # loads the servos fully too, but they get somewhere)
    max_temperature_c: float = 65.0     # the STS3215 is rated to 70 C
    stall_deg: float = 3.0              # a joint this far from its goal ...
    stall_seconds: float = 1.0          # ... that has not moved 1 deg in this long is blocked
    max_tracking_deg: float = 10.0      # a joint this far from where it was last sent (commands are at most
                                        # max_step_deg away) is falling, overloaded or being pushed
    health_hz: float = 5.0              # how often the real arm's load, temperature and status are read


class SafetyStop(Exception):
    """The arm was stopped (it holds its position). Not a RuntimeError, so no tool mistakes it for a routine
    failure; clear it with ``robot.resume()`` after checking the arm."""


SHORT = ("pan", "lift", "elbow", "wrist", "roll")      # ARM_JOINTS in printouts
# Geom kinds in the collision model and the clearance each pair of kinds needs (SafetyLimits field names).
_PAIRS = {("rod", "surface"): "rod_floor", ("rod", "wall"): "rod_clearance", ("link", "rod"): "self_clearance",
          ("base", "rod"): "self_clearance", ("link", "surface"): "link_clearance", ("link", "wall"): "link_clearance",
          ("link", "link"): "self_clearance", ("base", "link"): "self_clearance"}
_LABELS = {"upper_arm": "upper arm", "lower_arm": "lower arm", "camera_mount": "wrist camera mount",
           "moving_jaw_so101_v1": "moving jaw", "wall_back": "back wall", "wall_left": "left wall",
           "wall_right": "right wall"}
_LINK, _BASE, _ENV = 1, 2, 4             # contype bits: links touch links, the base and the environment


def _base_boxes():
    """One oriented box per base part (base plate, motor holder, pan motor, controller-board plate), each the
    bounding box of its mesh: the Menagerie model gives the base no collision geometry. (pos, quat, half-size)."""
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE_SO101))
    out = []
    for g in np.flatnonzero(m.geom_bodyid == m.body("base").id):
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        out.append((m.geom_pos[g] + rot.reshape(3, 3) @ m.geom_aabb[g, :3], m.geom_quat[g].copy(),
                    m.geom_aabb[g, 3:].copy()))
    return out


_BASE_BOXES = None
_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


class CollisionModel:
    """The arm's collision boxes (MuJoCo Menagerie), a box per base part, the rod, and the workspace's fixed
    surfaces (the simulator's mat, floor and walls, ``SceneConfig``), for checking configurations before they
    are commanded.

    The arm against itself, the base and the camera: MuJoCo's collision detection with a margin (boxes of similar
    size). The arm against the mat, floor, table and walls: exact distances to those flat, axis-aligned surfaces,
    from each box's corners and each capsule's end points. (MuJoCo's box-box test misses the 1 mm jaw pads
    against a large block, and a thin board can be passed through unseen; a surface is solid behind it.)"""

    def __init__(self, scene: SceneConfig, limits: SafetyLimits, rod: bool = True, camera_xyz=None):
        global _BASE_BOXES
        if _BASE_BOXES is None:
            _BASE_BOXES = _base_boxes()
        spec = mujoco.MjSpec.from_file(str(MENAGERIE_SO101))
        for geom in list(spec.geoms):        # keep the collision primitives; drop meshes and the 1 mm jaw tips
            tiny = geom.type in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE) and geom.size[0] < 0.003
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH or geom.group != 3 or tiny:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
        for k, (pos, quat, half) in enumerate(_BASE_BOXES):
            spec.body("base").add_geom(name=f"base_box{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos,
                                       quat=quat)
        # The base is welded to the world, so MuJoCo's parent-child filter does not cover it and the shoulder.
        spec.add_exclude(bodyname1="base", bodyname2="shoulder")
        grip = spec.body("gripper")
        if rod:
            px, py = scene.pusher_xy
            grip.add_geom(name="rod", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[scene.pusher_radius, 0, 0],
                          fromto=[px, py, scene.pusher_top_z, px, py, scene.pusher_tip_z + scene.pusher_radius])
            grip.add_site(name="rod_tip", pos=[px, py, scene.pusher_tip_z])
        if scene.lightbox:
            spec.body("base").pos = [0.0, 0.0, -scene.mat_height]
        if camera_xyz is not None:            # the camera body, whatever its mount: a 10 cm cube around the lens
            spec.worldbody.add_geom(name="camera", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05] * 3,
                                    pos=list(camera_xyz))
        m = spec.compile()

        self.model, self.data = m, mujoco.MjData(m)
        self.arm_qpos = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self.gripper_qpos = m.jnt_qposadr[m.joint("gripper").id]
        self.range = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS]].copy()     # radians, (5, 2)
        self.gripper_range = m.jnt_range[m.joint("gripper").id].copy()
        self.gripper_hold = scene.gripper_hold
        self.rod_tip = m.site("rod_tip").id if rod else None
        kinds, labels = [], []
        for g in range(m.ngeom):
            name, body = m.geom(g).name, m.body(m.geom_bodyid[g]).name
            if m.geom_bodyid[g] == 0:
                kind = "wall" if name == "camera" else "ignore"
            else:
                kind = "rod" if name == "rod" else "base" if name.startswith("base_box") else "link"
            kinds.append(kind)
            # Name a contact by the link (body), not by the box: "gripper", "wrist", "base", "rod", "camera", ...
            label = name if kind in ("wall", "rod") else body
            labels.append(_LABELS.get(label, label).replace("_", " "))
            thin = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_size[g].min() < 0.002   # jaw pads
            m.geom_contype[g], m.geom_conaffinity[g] = (0, 0) if thin else {
                "link": (_LINK, _LINK | _BASE), "rod": (_LINK, _LINK | _BASE), "base": (_BASE, 0),
                "wall": (_ENV, _LINK), "ignore": (0, 0)}[kind]
        self.kinds, self.labels = kinds, labels
        need = {k: getattr(limits, v) for k, v in _PAIRS.items()}
        self.need = need
        self.required = np.full((m.ngeom, m.ngeom), np.nan)
        for a, b in itertools.product(range(m.ngeom), repeat=2):
            self.required[a, b] = need.get(tuple(sorted((kinds[a], kinds[b]))), np.nan)
        m.geom_margin[:] = max(need.values()) + limits.slow_zone + 0.002     # see far enough to slow down

        # The fixed surfaces, for the exact distances: every arm geom (boxes by corners, capsules by end points)
        # but the shoulder's, which only turns on the base, always 24 mm over the floor.
        moving = [g for g in range(m.ngeom) if kinds[g] in ("link", "rod") and m.body(m.geom_bodyid[g]).name != "shoulder"]
        self.boxes = np.array([g for g in moving if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX], int)
        self.caps = np.array([g for g in moving if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE], int)
        self.lightbox, self.floor = scene.lightbox, -scene.mat_height

    def _surfaces(self, pts: np.ndarray, radius: np.ndarray):
        """Distances (n, k) from points (n, k, 3) with radii (n, k) to the surface below them and to each wall,
        and the name of the surface below each point. Lightbox: the mat top (z = 0) over the mat, the floor board
        (mat_height lower) elsewhere in the box, the table under the box (a board lower) outside it."""
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        out = {}
        if not self.lightbox:
            out[("surface", "table")] = (z - radius, None)
            return out
        on_mat = ((x >= LIGHTBOX_MAT_X[0]) & (x <= LIGHTBOX_MAT_X[1]) & (np.abs(y) <= LIGHTBOX_MAT_HALF_WIDTH))
        in_box = (x >= LIGHTBOX_FLOOR_X[0]) & (x <= LIGHTBOX_FLOOR_X[1]) & (np.abs(y) <= LIGHTBOX_HALF_WIDTH)
        height = np.where(on_mat, 0.0, np.where(in_box, self.floor, self.floor - LIGHTBOX_BOARD))
        where = np.where(on_mat, "mat", np.where(in_box, "floor", "table"))
        out[("surface", None)] = (z - height - radius, where)
        inner_x, inner_y = LIGHTBOX_FLOOR_X[1] - LIGHTBOX_BOARD, LIGHTBOX_HALF_WIDTH - LIGHTBOX_BOARD
        along = (x >= LIGHTBOX_FLOOR_X[0]) & (x <= LIGHTBOX_FLOOR_X[1])     # the side walls end at the open front
        out[("wall", "back wall")] = (inner_x - x - radius, None)
        out[("wall", "left wall")] = (np.where(along, inner_y - y - radius, np.inf), None)
        out[("wall", "right wall")] = (np.where(along, y + inner_y - radius, np.inf), None)
        return out

    def deficits(self, q_rad, gripper_rad=None) -> tuple[dict, float]:
        """Clearance shortfalls (m) at arm configuration ``q_rad`` (5,): {(thing, thing): (shortfall, distance,
        (kind, kind))}, and the smallest spare clearance of anything but the rod over the mat (m; inf: far)."""
        m, d = self.model, self.data
        d.qpos[self.arm_qpos] = q_rad
        d.qpos[self.gripper_qpos] = self.gripper_hold if gripper_rad is None else gripper_rad
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        out, spare = {}, math.inf

        def note(key, kinds, need, dist):
            nonlocal spare
            if kinds != ("rod", "surface"):
                spare = min(spare, dist - need)
            if dist < need and need - dist > out.get(key, (0.0,))[0]:
                out[key] = (need - dist, dist, kinds)

        for c in d.contact[:d.ncon]:                 # the arm against itself, the base and the camera
            need = self.required[c.geom1, c.geom2]
            if not np.isnan(need):
                note(tuple(sorted((self.labels[c.geom1], self.labels[c.geom2]))),
                     tuple(sorted((self.kinds[c.geom1], self.kinds[c.geom2]))), need, c.dist)

        # The arm against the fixed surfaces: corners of boxes, end points of capsules.
        geoms, pts, rad = [], [], []
        if len(self.boxes):
            rot = d.geom_xmat[self.boxes].reshape(-1, 3, 3)
            local = _CORNERS[None] * m.geom_size[self.boxes][:, None, :]
            pts.append(d.geom_xpos[self.boxes][:, None, :] + np.einsum("nij,nkj->nki", rot, local))
            rad.append(np.zeros((len(self.boxes), 8)))
            geoms += list(self.boxes)
        if len(self.caps):
            axis = d.geom_xmat[self.caps].reshape(-1, 3, 3)[:, :, 2] * m.geom_size[self.caps][:, 1:2]
            ends = np.stack([d.geom_xpos[self.caps] - axis, d.geom_xpos[self.caps] + axis], 1)
            pts.append(np.pad(ends, ((0, 0), (0, 6), (0, 0)), mode="edge"))
            rad.append(np.repeat(m.geom_size[self.caps][:, :1], 8, 1))
            geoms += list(self.caps)
        pts, rad = np.concatenate(pts), np.concatenate(rad)
        for (kind, name), (dist, where) in self._surfaces(pts, rad).items():
            k = np.argmin(dist, 1)
            for i, g in enumerate(geoms):
                gk = self.kinds[g]
                kinds = tuple(sorted((gk, kind)))
                need = self.need.get(kinds)
                if need is None:
                    continue
                label = name or where[i, k[i]]
                note(tuple(sorted((self.labels[g], label))), kinds, need, float(dist[i, k[i]]))
        return out, spare

    def tip(self, q_rad) -> np.ndarray:
        """Rod tip position for arm configuration ``q_rad`` (5,)."""
        self.data.qpos[self.arm_qpos] = q_rad
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.rod_tip].copy()


@dataclass
class Check:
    """What is wrong with a configuration: shortfall per item (mm for clearances, degrees for joint limits),
    a description, and for clearances the kinds of thing involved ("link", "rod", "base", "surface", "wall")."""
    bad: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)
    kinds: dict = field(default_factory=dict)
    spare: float = math.inf             # smallest spare clearance (m), the rod over the mat aside

    @property
    def ok(self) -> bool:
        return not self.bad

    def worse_than(self, other: "Check", slack: float = 0.0) -> bool:
        """True if something is new, or deeper than in ``other`` by more than ``slack`` (mm or degrees; new
        problems get none). Commands must end with no slack: a controller repeating a command from wherever the
        last one stopped must not creep in by a tolerance at a time."""
        return any(v > (other.bad[k] + slack if k in other.bad else 1e-9) for k, v in self.bad.items())


class SafetyGuard:
    """Checks and limits joint commands (LeRobot degrees). See the module docstring."""

    def __init__(self, scene: SceneConfig | None = None, jmap: JointMap | None = None,
                 limits: SafetyLimits | None = None, joint_range_deg=None, rod: bool = True, camera_xyz=None):
        self.scene = scene or SceneConfig()
        self.jmap = jmap or JointMap()
        self.limits = limits or SafetyLimits()
        self.collision = CollisionModel(self.scene, self.limits, rod=rod, camera_xyz=camera_xyz)
        self.calibrated = None if joint_range_deg is None else np.asarray(joint_range_deg, float)
        self._kin, self._examples = None, None
        self._set_range()

    def _set_range(self):
        """Joint limits in robot degrees: the model's range, cut to the calibrated one less the margin."""
        a, b = self.jmap.to_robot(self.collision.range[:, 0]), self.jmap.to_robot(self.collision.range[:, 1])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if self.calibrated is not None:
            m = self.limits.joint_margin_deg
            lo, hi = np.maximum(lo, self.calibrated[:, 0] + m), np.minimum(hi, self.calibrated[:, 1] - m)
        self.lo, self.hi = lo, hi
        self._examples = None

    def set_map(self, jmap: JointMap):
        """A new joint map (calibrate.py fits one, then moves the arm with it)."""
        self.jmap = jmap
        self._kin = None
        self._set_range()

    def set_calibrated_range(self, joint_range_deg):
        """The range each servo was calibrated to (robot degrees, (5, 2)); the real arm reads it at connection."""
        self.calibrated = np.asarray(joint_range_deg, float)
        self._set_range()

    # -- checks
    def check(self, deg, gripper_rad=None) -> Check:
        """Everything wrong with arm configuration ``deg`` (robot degrees, (5,))."""
        deg = np.asarray(deg, float)
        out = Check()
        for j, name in enumerate(ARM_JOINTS):
            over = max(self.lo[j] - deg[j], deg[j] - self.hi[j], 0.0)
            if over > 0:
                out.bad[name] = over
                out.text[name] = (f"{name.replace('_', ' ')} at {deg[j]:.1f} deg, limit "
                                  f"{self.lo[j]:.1f} to {self.hi[j]:.1f}")
        deficits, out.spare = self.collision.deficits(self.jmap.to_sim(deg), gripper_rad)
        for (a, b), (short, dist, kinds) in deficits.items():
            out.bad[(a, b)] = short * 1e3
            out.kinds[(a, b)] = kinds
            out.text[(a, b)] = (f"{a} {abs(dist) * 1e3:.1f} mm {'into' if dist < 0 else 'from'} {b} "
                                f"(needs {(dist + short) * 1e3:+.1f} mm)")
        return out

    def tip(self, deg) -> np.ndarray:
        """Rod tip position (m, robot base frame) at ``deg``."""
        return self.collision.tip(self.jmap.to_sim(deg))

    def filter(self, present, target, dt: float | None = None, previous=None,
               gripper_rad=None) -> tuple[np.ndarray, list[str]]:
        """The safe part of a command: ``target`` (robot degrees) limited to the joint ranges, to
        ``max_speed_deg_s`` from the ``previous`` command (``dt`` seconds ago), to ``max_step_deg`` from the
        measured joints ``present``, and cut back to the last safe point on the straight path from ``present``.
        Returns the command to send and what was limited (empty when it went through unchanged, apart from the
        speed and step limits, which smooth every fast command)."""
        present, target = np.asarray(present, float), np.asarray(target, float)
        if not np.all(np.isfinite(target)) or target.shape != present.shape:
            return present.copy(), ["target not finite: holding"]
        lim = self.limits
        events = []
        safe = np.clip(target, np.minimum(self.lo, present), np.maximum(self.hi, present))   # never further out
        for j in np.flatnonzero(np.abs(safe - target) > 1e-9):
            events.append(f"{ARM_JOINTS[j].replace('_', ' ')} held to {safe[j]:.1f} deg (limit)")
        # Where a refused command leaves the arm: the previous command when it is no worse than the measured
        # pose (holding the measured pose would let gravity win, a sag at a time), else the measured pose.
        anchor = present
        start = self.check(present, gripper_rad)
        if previous is not None and np.max(np.abs(np.asarray(previous) - present)) <= lim.max_step_deg:
            c = self.check(previous, gripper_rad)
            if not c.worse_than(start):
                anchor, start = np.asarray(previous, float), c
        # Near anything, slower: speed and step shrink with the spare clearance where the step starts and where it
        # would end, so the arm arrives slowly and brakes inside its margin.
        wanted = safe
        slow = float(np.clip(start.spare / lim.slow_zone, lim.min_speed_fraction, 1.0))
        for _ in range(2):
            safe = wanted
            if previous is not None and dt is not None:  # speed: the command stream, not the arm's lag behind it
                v = max(0.05, lim.max_speed_deg_s * dt * slow)
                safe = np.clip(safe, np.asarray(previous) - v, np.asarray(previous) + v)
            step = lim.max_step_deg * slow
            safe = np.clip(safe, present - step, present + step)
            end = float(np.clip(self.check(safe, gripper_rad).spare / lim.slow_zone, lim.min_speed_fraction, 1.0))
            if end >= slow:
                break
            slow = end
        # From a clean pose nothing may go wrong anywhere on the path. From a pose that is already too close, the
        # way out may dip up to escape_slack deeper (a joint-space line is not straight in space), but the
        # command must end no deeper than it started.
        slack = 0.0 if start.ok else lim.escape_slack
        n = max(1, math.ceil(np.max(np.abs(safe - anchor)) / lim.path_step_deg))
        last_ok, last_ok_k = anchor, 0            # the furthest point that may be commanded (no deeper than start)
        for k in range(1, n + 1):
            p = anchor + (safe - anchor) * k / n
            c = self.check(p, gripper_rad)
            strict = not c.worse_than(start)
            if strict:
                last_ok, last_ok_k = p, k
            if (k == n and not strict) or c.worse_than(start, slack):
                if last_ok_k == k - 1:             # refine the boundary, to 1/64 of a path step
                    good, bad = last_ok, p
                    for _ in range(6):
                        mid = (good + bad) / 2
                        if self.check(mid, gripper_rad).worse_than(start):
                            bad = mid
                        else:
                            good = mid
                    last_ok = good
                events += [c.text[k2] for k2 in c.bad if c.bad[k2] > start.bad.get(k2, 0.0) + 1e-9]
                return last_ok, events
        return safe, events

    # -- self-test
    def ik(self, xyz) -> np.ndarray:
        """Robot degrees that put the (vertical) rod tip at ``xyz``."""
        if self._kin is None:
            from ..kinematics import PusherKinematics
            from ..scene import build_model
            self._kin = PusherKinematics(build_model(self.scene, visual=False))
        q, _, _ = self._kin.solve(np.array([xyz], float), self._kin.seed(np.array([xyz[:2]])), iterations=40)
        return self.jmap.to_robot(q[0])

    def drive(self, start, target, ticks: int = 200, dt: float = 0.05) -> np.ndarray:
        """Command ``target`` again and again from ``start``, as a controller would, with an arm that follows
        every command: where the guard lets it get to."""
        q = np.array(start, float)
        for _ in range(ticks):
            q, _ = self.filter(q, target, dt, previous=q)
        return q

    def examples(self) -> dict:
        """Named configurations (robot degrees) for the self-test and rehearsals: a safe ``hover`` (rod 30 mm over
        the mat), and targets inside the joint limits that drive the rod into the mat (``mat``), a link into the
        mat or floor (``link_mat``), the arm into the base (``base``), into itself (``self``) and, in the lightbox,
        into a wall (``wall``). Found by a seeded search, so always the same."""
        if self._examples is None:
            rng = np.random.default_rng(7)
            kinds = {"link_mat": lambda k: k == ("link", "surface"), "base": lambda k: "base" in k,
                     "self": lambda k: k in (("link", "link"), ("link", "rod"))}
            if self.scene.lightbox:
                kinds["wall"] = lambda k: "wall" in k
            out = {"hover": self.ik((0.20, 0.0, 0.03))}
            if self.collision.rod_tip is not None:
                out["mat"] = self.ik((0.20, 0.0, -0.03))
            for _ in range(20000):
                q = rng.uniform(self.lo, self.hi)
                found = self.check(q).kinds.values()
                for name, kind in kinds.items():
                    if name not in out and any(kind(k) for k in found):
                        out[name] = q
                if len(out) == len(kinds) + 1 + ("mat" in out):
                    break
            self._examples = out
        return self._examples

    def selftest(self) -> list[tuple[str, bool, str]]:
        """Known-unsafe commands must be refused and safe ones let through. Returns (test, passed, detail)."""
        ex = self.examples()
        out = []
        hover = ex["hover"]
        c = self.check(hover)
        out.append(("safe pose accepted", c.ok, "rod 30 mm above the mat" if c.ok else "; ".join(c.text.values())))
        near = self.ik((0.203, 0.003, 0.03))
        got, ev = self.filter(hover, near)
        out.append(("safe move unchanged", bool(np.allclose(got, near)) and not ev, "; ".join(ev)))
        if "mat" in ex:
            got = self.drive(hover, ex["mat"])
            z = self.tip(got)[2]
            out.append(("rod into the mat refused", z >= self.limits.rod_floor - 5e-4 and not np.allclose(got, ex["mat"]),
                        f"stopped with the rod tip at {z * 1e3:+.1f} mm"))
            pressed = self.ik((0.20, 0.0, self.limits.rod_floor - 0.002))
            up, deeper = self.ik((0.20, 0.0, 0.02)), self.ik((0.20, 0.0, self.limits.rod_floor - 0.006))
            z0, z_up, z_down = (self.tip(q)[2] for q in (pressed, self.drive(pressed, up, 60),
                                                           self.drive(pressed, deeper, 60)))
            out.append(("out of a bad pose allowed, deeper refused", z_up > 0.015 and z_down > z0 - 5e-4,
                        f"from {z0 * 1e3:+.1f} mm: up to {z_up * 1e3:+.1f}, down to {z_down * 1e3:+.1f}"))
        past = hover.copy()                   # the wrist roll: turning it moves nothing into anything
        past[4] = self.hi[4] + 20.0
        got = self.drive(hover, past, 100)
        out.append(("joint past its range refused", abs(got[4] - self.hi[4]) < 1e-6,
                    f"wrist roll stopped at {got[4]:.1f} deg (limit {self.hi[4]:.1f})"))
        jump = hover.copy()
        jump[0] += 60.0
        got, _ = self.filter(hover, jump)
        out.append(("jump limited to one step", abs(got[0] - hover[0]) <= self.limits.max_step_deg + 1e-6,
                    f"pan moved {got[0] - hover[0]:.1f} deg of 60"))
        got, ev = self.filter(hover, np.full(5, np.nan))
        out.append(("non-finite target refused", bool(np.allclose(got, hover)) and bool(ev), "held"))
        names = {"link_mat": "arm into the mat or floor refused", "base": "arm into the base refused",
                 "self": "arm into itself refused", "wall": "arm into a lightbox wall refused"}
        for key, name in names.items():
            if key == "wall" and not self.scene.lightbox:
                continue
            if key not in ex:
                out.append((name, False, "no such configuration found to test with"))
                continue
            got = self.drive(hover, ex[key], 400)
            c = self.check(got)
            out.append((name, c.ok and not np.allclose(got, ex[key]), "stopped short, nothing too close" if c.ok
                        else "; ".join(c.text.values())))
        return out


def require_selftest(guard: SafetyGuard):
    """Raise SafetyStop unless every self-test passes."""
    failed = [f"{name}: {detail}" for name, ok, detail in guard.selftest() if not ok]
    if failed:
        raise SafetyStop("safety self-test failed:\n  " + "\n  ".join(failed))


def guard_for(scene: SceneConfig, jmap: JointMap, limits: SafetyLimits | None = None, camera: bool = True,
              selftest: bool = True, rod: bool = True) -> SafetyGuard:
    """The guard for a tool: the scene the tool runs in, the joint map it uses, the calibrated camera as an
    obstacle (lightbox), after checking that the map was calibrated in this workspace and the self-test passes.
    ``rod=False``: no pusher rod in the gripper (teleoperation with the gripper free)."""
    if jmap.lightbox is not None and jmap.lightbox != scene.lightbox:
        raise SafetyStop(f"the joint map was calibrated {'in' if jmap.lightbox else 'outside'} the lightbox, but this "
                         f"run is {'in' if scene.lightbox else 'outside'} it: "
                         f"{'add' if jmap.lightbox else 'remove'} --lightbox, or recalibrate")
    camera_xyz = None
    if camera and scene.lightbox:
        try:
            from .vision import CameraModel
            camera_xyz = CameraModel.load().position
        except (FileNotFoundError, OSError, KeyError, ValueError):
            camera_xyz = None
    guard = SafetyGuard(scene, jmap, limits, rod=rod, camera_xyz=camera_xyz)
    if selftest:
        require_selftest(guard)
    return guard


_STATUS_BITS = ("voltage", "angle sensor", "overheat", "overcurrent", "angle limit", "overload")


class ServoMonitor:
    """Watches the servos while they move. ``update`` returns why the arm must stop, or None."""

    def __init__(self, limits: SafetyLimits, names=ARM_JOINTS):
        self.limits, self.names = limits, list(names)
        self.load_since = np.full(len(self.names), np.nan)
        self.load_from = np.zeros(len(self.names))
        self.stall_since = np.full(len(self.names), np.nan)
        self.history = deque()
        self.last_goal = None

    def reset_motion(self):
        """The arm was moved by hand, held, or put somewhere: forget where it was last sent."""
        self.last_goal = None
        self.history.clear()
        self.stall_since[:] = np.nan
        self.load_since[:] = np.nan

    def update(self, t: float, present, goal, health: dict | None = None) -> str | None:
        lim = self.limits
        present, goal = np.asarray(present, float), np.asarray(goal, float)
        if self.last_goal is not None:
            away = np.abs(present - self.last_goal)
            if (away > lim.max_tracking_deg).any():
                j = int(np.argmax(away))
                return (f"{self.names[j]} is {away[j]:.0f} deg from where it was sent: the arm is falling, "
                        f"overloaded or being pushed")
        self.last_goal = goal
        if health is not None:
            status = np.asarray(health.get("status", np.zeros(len(self.names))), int)
            for j in np.flatnonzero(status):
                bits = [n for b, n in enumerate(_STATUS_BITS) if status[j] >> b & 1] or [f"code {status[j]}"]
                return f"{self.names[j]} servo reports an error: {', '.join(bits)}"
            temp = np.asarray(health.get("temperature", np.zeros(len(self.names))), float)
            if (temp >= lim.max_temperature_c).any():
                j = int(np.argmax(temp))
                return f"{self.names[j]} servo at {temp[j]:.0f} C (limit {lim.max_temperature_c:.0f} C): let it cool"
            load = np.abs(np.asarray(health.get("load", np.zeros(len(self.names))), float))
            over = load >= lim.max_load
            new = over & np.isnan(self.load_since)
            self.load_from = np.where(new, present, self.load_from)
            self.load_since = np.where(over, np.where(new, t, self.load_since), np.nan)
            stuck = over & (t - self.load_since >= lim.load_seconds) & (np.abs(present - self.load_from) < 2.0)
            if np.any(stuck):
                j = int(np.flatnonzero(stuck)[0])
                return (f"{self.names[j]} servo at {load[j]:.0%} load for {t - self.load_since[j]:.1f} s without "
                        f"moving: something is resisting the arm")
        self.history.append((t, present.copy()))
        while self.history and t - self.history[0][0] > lim.stall_seconds:
            self.history.popleft()
        oldest_t, oldest = self.history[0]
        far = np.abs(goal - present) > lim.stall_deg
        self.stall_since = np.where(far, np.where(np.isnan(self.stall_since), t, self.stall_since), np.nan)
        still = np.abs(present - oldest) < 1.0
        stuck = far & still & (t - self.stall_since >= lim.stall_seconds) & (t - oldest_t >= 0.8 * lim.stall_seconds)
        if stuck.any():
            j = int(np.flatnonzero(stuck)[0])
            return (f"{self.names[j]} has not moved toward its goal for {lim.stall_seconds:.1f} s "
                    f"({abs(goal[j] - present[j]):.1f} deg away): blocked")
        return None


def wiggle(robot, guard: SafetyGuard, amplitude_deg: float = 3.0, speed_deg_s: float = 10.0, realtime: bool = True):
    """Move each joint by +-``amplitude_deg`` and back, slowly, through the guard and the monitor. Returns
    the largest tracking error per joint (deg)."""
    import time
    dt = 0.05
    worst = np.zeros(5)
    home = robot.read_joints()
    for j in range(5):
        for target in (amplitude_deg, -amplitude_deg, 0.0):
            goal = home.copy()
            goal[j] += target
            start = robot.read_joints()
            n = max(2, math.ceil(abs(goal[j] - start[j]) / (speed_deg_s * dt)))
            for q in np.linspace(start, goal, n):
                t0 = time.monotonic()
                robot.send_joints(q, seconds=dt)
                worst = np.maximum(worst, np.abs(robot.read_joints() - q))
                if realtime:
                    time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["none", "sim", "so101"], default="none",
                    help="none: test the guard only; sim / so101: also check the arm's pose and servos")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="", help="joint map from calibrate.py")
    ap.add_argument("--lightbox", action="store_true", help="the arm stands in the lightbox")
    ap.add_argument("--wiggle", action="store_true", help="move each joint +-3 deg, slowly, and watch the servos")
    args = ap.parse_args(argv)
    scene = SceneConfig(timestep=0.005, lightbox=args.lightbox)
    jmap = JointMap.load(args.map) if args.map else JointMap()
    guard = guard_for(scene, jmap, selftest=False)
    print(f"Safety self-test ({'lightbox' if args.lightbox else 'flat table'}, joint limits "
          + ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, lo, hi in zip(SHORT, guard.lo, guard.hi)) + " deg):")
    failed = 0
    for name, ok, detail in guard.selftest():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} {detail}")
        failed += not ok
    if args.robot == "none":
        raise SystemExit(1 if failed else 0)
    from .robots import SimRobot, SO101Robot
    if args.robot == "so101":
        if not args.map:
            print("NOTE  no joint map yet: the collision model uses LeRobot's zero (calibrate.py measures the offsets)")
        robot = SO101Robot(args.port, guard=guard)
    else:
        robot = SimRobot(scene, guard=guard)
    try:
        present = robot.read_joints()
        c = guard.check(present)
        print("Arm now: " + ", ".join(f"{n} {v:+.1f}" for n, v in zip(SHORT, present)) + " deg")
        print("  " + ("PASS  pose within limits and clearances" if c.ok else
                      "NOTE  " + "; ".join(c.text.values()) + " (moves out of this are allowed, deeper ones not)"))
        health = robot.health()
        if health is not None:
            reason = ServoMonitor(guard.limits).update(0.0, present, present, health)
            print("Servos: load " + ", ".join(f"{abs(v):.0%}" for v in health["load"])
                  + " | temperature " + ", ".join(f"{v:.0f}" for v in health["temperature"]) + " C"
                  + (" | voltage " + ", ".join(f"{v:.1f}" for v in health["voltage"]) + " V" if "voltage" in health else ""))
            print(f"  {'FAIL  ' + reason if reason else 'PASS  no servo errors, load and temperature normal'}")
            failed += bool(reason)
        if args.robot == "so101":
            for line in robot.limit_report():
                print("  " + line)
        if args.wiggle and not failed:
            if args.robot == "so101":
                input("Each joint will move +-3 deg, slowly. Keep a hand near the power switch. Enter to start...")
            worst = wiggle(robot, guard, realtime=args.robot == "so101")
            print("Wiggle: largest tracking error " + ", ".join(f"{n} {e:.1f}" for n, e in zip(SHORT, worst))
                  + " deg")
            print("  PASS  every joint moved and came back, no servo fault")
    except SafetyStop as stop:
        print(f"  FAIL  {stop}")
        failed += 1
    finally:
        robot.close()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
