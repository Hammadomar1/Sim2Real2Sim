"""Arm occlusion for the simulated camera: how much of an object's top face a mounted camera sees past the arm.

The actor's camera model (``RandomizationConfig.obs_*``) drops pose estimates at random. A real camera loses
the object in bursts, while the arm is between the camera and the object. With ``rand.occlusion_camera`` set,
the environment casts rays every control step from that camera to a grid of points on each object's top face
(``mj_multiRay`` against the arm's visual meshes, the rod and the wrist camera mount, posed by the world's
joint angles). A frame whose visible share is below ``rand.occlusion_threshold`` counts as dropped: the actor
keeps the last estimate. ``rand.occlusion_bias`` shifts the remaining estimates towards the visible part of the
face (1 = the centroid of the visible points, what a naive segment-centroid estimator reports; 0 = a template
fit that is not pulled by the missing part).

Mount names are the lightbox positions studied in ``camera_study`` (mat frame: z = 0 on the mat top); "c920" is
the scene's measured C920 (``SceneConfig.c920_pos``). Any position can be given as "x,y,z" in metres.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os

import mujoco
import numpy as np

from .objects import OBJECTS
from .scene import SceneConfig, build_model

# Mounts the lightbox allows (inside its walls; z above the mat, which is 9.8 mm above the floor board).
LIGHTBOX_MOUNTS = {
    "c920_left_wall": (0.2022, 0.3662, 0.3902),   # as built: slot in the left wall, 40 cm above the floor
    "right_wall": (0.2022, -0.3662, 0.3902),      # the same slot in the right wall
    "back_wall_low": (0.465, 0.00, 0.25),         # centre of the back wall, opposite the robot (~43 deg down)
    "back_wall_high": (0.465, 0.00, 0.45),        # ... at the top of the wall
    "back_left_corner": (0.465, 0.37, 0.40),      # where the back and left walls meet
    "overhead_rim": (0.20, 0.00, 0.49),           # straight down from the walls' top edge
}
RAY_GROUP = 5          # the ray model's arm geoms are moved to this geom group; rays see nothing else


def mount_position(name: str, scene: SceneConfig) -> np.ndarray:
    """A mount name (LIGHTBOX_MOUNTS or "c920") or "x,y,z" (m, world frame) -> camera position."""
    if name == "c920":
        return np.array([*scene.c920_pos[:2], scene.c920_pos[2] - (scene.mat_height if scene.lightbox else 0.0)])
    if name in LIGHTBOX_MOUNTS:
        return np.array(LIGHTBOX_MOUNTS[name], dtype=float)
    try:
        pos = np.array([float(v) for v in name.split(",")])
    except ValueError:
        pos = np.zeros(0)
    if pos.shape != (3,):
        raise ValueError(f"occlusion camera {name!r}: not a mount ({', '.join(['c920', *LIGHTBOX_MOUNTS])}) or 'x,y,z'")
    return pos


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


class FacePoints:
    """Top-face points of several object types, padded to one count (padding repeats a point: no effect)."""

    def __init__(self, names, grid: int):
        pts = [top_face_points(OBJECTS[name], grid) for name in names]
        k = max(len(p) for p in pts)
        self.local = np.stack([np.concatenate([p, np.repeat(p[:1], k - len(p), 0)]) for p in pts])   # (types, k, 3)
        self.weight = np.stack([np.r_[np.ones(len(p)), np.zeros(k - len(p))] / len(p) for p in pts])  # (types, k)

    def world(self, pose, kind):
        """(n, k, 3) world points for planar poses (n, 3) of object types ``kind`` (n,)."""
        loc = self.local[kind]
        c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
        return np.stack([pose[:, :1] + c * loc[..., 0] - s * loc[..., 1],
                         pose[:, 1:2] + s * loc[..., 0] + c * loc[..., 1], loc[..., 2]], -1)


class OcclusionCamera:
    """Ray-cast visibility of object top faces from one camera position, for a batch of worlds.

    The ray model is the scene's full-mesh build (same joints and qpos layout as the training model) with the
    arm's visual geoms (visual meshes, rod, wrist camera mount) moved to ``RAY_GROUP``; ``fast`` uses the arm's
    collision primitives (Menagerie's boxes and spheres, and the rod) instead: 6x cheaper rays, and at the
    left-wall mount they hide slightly less (27 % of views more than half hidden against 30 %; the same drop
    decision in 97 % of frames). Worlds are split over a thread pool (MuJoCo releases the GIL inside its
    functions; the Python loop around them, about 10 us per world, does not, and sets the floor).
    """

    def __init__(self, scene: SceneConfig, camera: str, object_name: str = "tee", nthread: int = 0, fast: bool = False):
        self.pos = mount_position(camera, scene)
        self.model = build_model(scene, object_name, visual=True)
        m = self.model
        root = m.body("base").id
        for g in range(m.ngeom):
            on_arm = m.body_rootid[m.geom_bodyid[g]] == root
            ray = (m.geom_group[g] == 3 or m.geom(g).name == "pusher") if fast else m.geom_group[g] in (1, 2)
            m.geom_group[g] = RAY_GROUP if on_arm and ray else min(m.geom_group[g], RAY_GROUP - 1)
        self.groups = np.zeros(6, dtype=np.uint8)
        self.groups[RAY_GROUP] = 1
        # The per-world Python work holds the GIL: more than about 8 threads gains nothing.
        self.nthread = min(nthread or os.cpu_count() or 1, 8)
        self.datas = [mujoco.MjData(m) for _ in range(self.nthread)]
        self.pool = ThreadPoolExecutor(self.nthread) if self.nthread > 1 else None

    def _cast(self, data, qpos, vec, geomid, dist):
        """Rays for a contiguous block of worlds (row views of the caller's arrays; results land in place)."""
        m, cam, groups, k = self.model, self.pos, self.groups, geomid.shape[1]
        kinematics, multi_ray, qp = mujoco.mj_kinematics, mujoco.mj_multiRay, data.qpos
        for q, v, g, d in zip(qpos, vec, geomid, dist):
            qp[:] = q
            kinematics(m, data)
            multi_ray(m, data, cam, v, groups, 1, -1, g, d, None, k, 10.0)

    def hidden(self, qpos, points):
        """Which points (n, k, 3) are hidden from the camera by the arm, given the worlds' qpos (n, nq)."""
        n, k = points.shape[:2]
        vec = np.ascontiguousarray((points - self.pos).reshape(n, 3 * k))
        geomid = np.zeros((n, k), dtype=np.int32)
        dist = np.full((n, k), -1.0)
        cuts = np.linspace(0, n, min(self.nthread, n) + 1).astype(int)
        blocks = [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        if self.pool is None or len(blocks) <= 1:
            self._cast(self.datas[0], qpos, vec, geomid, dist)
        else:
            jobs = [self.pool.submit(self._cast, d, qpos[s], vec[s], geomid[s], dist[s]) for d, s in zip(self.datas, blocks)]
            for f in jobs:
                f.result()
        return (dist >= 0) & (dist < 0.999)

    def view(self, qpos, targets, face: FacePoints):
        """What the camera sees of each target, one ray batch per world.

        ``targets``: list of (planar poses (n, 3), face-point types (n,)). Returns (n, targets, 3): the visible
        share of each top face, and the shift (x, y) from the face's centre to the centroid of its visible part.
        """
        pts = [face.world(pose, kind) for pose, kind in targets]
        hidden = np.split(self.hidden(qpos, np.concatenate(pts, 1)), np.cumsum([p.shape[1] for p in pts])[:-1], 1)
        out = np.zeros((len(qpos), len(targets), 3))
        for j, ((_, kind), p, h) in enumerate(zip(targets, pts, hidden)):
            w = face.weight[kind] * ~h
            frac = w.sum(-1)
            centre = (face.weight[kind][..., None] * p[..., :2]).sum(1)
            seen = (w[..., None] * p[..., :2]).sum(1) / np.maximum(frac, 1e-9)[:, None]
            out[:, j] = np.c_[frac, np.where(frac[:, None] > 0, seen - centre, 0.0)]
        return out
