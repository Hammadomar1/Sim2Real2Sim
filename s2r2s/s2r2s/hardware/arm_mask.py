"""Where the arm hides the mat, from its joint angles: the pose estimator's occlusion mask.

    mask = ArmMask(camera, estimator.grid, SceneConfig(lightbox=True))(q_sim)    # bool, on the estimator's grid
    estimator.estimate(frame, hidden=mask)

A dark servo or the rod in front of the T looks like the black mat, so without this the fit pushes the T away
from it, and the visible share counts the hidden part as missing. With it, pixels behind the arm carry no
evidence either way. Each link mesh is reduced to its extreme vertices in ~100 directions; per frame those are
placed by forward kinematics, projected through the camera centre onto the estimator's plane (z = 20 mm), and
the 2D hull of each part is filled, then grown by ``margin`` to cover joint-reading and calibration error and
the time between the joint reading and the frame.
"""
from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np

from ..scene import SceneConfig, SceneIndex, build_model
from .vision import CameraModel, PlaneGrid

_NOT_ROBOT = ("world", "object", "clutter", "goal", "estimate", "clutter_home", "gate")


def _directions(n: int) -> np.ndarray:
    """n roughly even unit vectors (Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    a = math.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z ** 2)
    return np.c_[r * np.cos(a), r * np.sin(a), z]


def _geom_points(model: mujoco.MjModel, g: int, dirs: np.ndarray) -> np.ndarray:
    """Points (geom frame) whose convex hull is (nearly) the geom's: extreme mesh vertices, or the primitive's
    support points in every direction."""
    t, size = model.geom_type[g], model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        return v[np.unique(np.argmax(v @ dirs.T, axis=0))]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.vstack([dirs * size[0] + (0, 0, s * size[1]) for s in (-1, 1)])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return dirs * size[0]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ring = np.c_[size[0] * np.cos(a), size[0] * np.sin(a)]
        return np.vstack([np.c_[ring, np.full(24, s * size[1])] for s in (-1, 1)])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([(sx * size[0], sy * size[1], sz * size[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return np.zeros((1, 3))


class ArmMask:
    def __init__(self, camera: CameraModel, grid: PlaneGrid, scene: SceneConfig | None = None, margin: float = 0.004,
                 directions: int = 96):
        scene = scene or SceneConfig(lightbox=True)
        self.model = build_model(scene, visual=True)
        self.data = mujoco.MjData(self.model)
        self.idx = SceneIndex(self.model)
        self.gripper = scene.gripper_hold
        m = self.model
        dirs = _directions(directions)
        self.geoms = [g for g in range(m.ngeom) if m.geom_group[g] in (1, 2) and m.geom_bodyid[g] > 0
                      and m.body(m.geom_bodyid[g]).name not in _NOT_ROBOT]
        self.points = [_geom_points(m, g, dirs) for g in self.geoms]
        self.grid = grid
        c = camera.position
        self.foot, self.cz = c[:2], float(c[2])
        r = max(1, int(round(margin / grid.res)))
        self.grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.canvas = np.zeros((grid.rows, grid.cols), np.uint8)

    def __call__(self, q_arm, gripper: float | None = None) -> np.ndarray:
        """Bool mask on the grid: True where the arm (at simulator joint angles ``q_arm``, radians) is between the
        camera and the plane."""
        d = self.data
        d.qpos[self.idx.arm_qpos] = q_arm
        d.qpos[self.idx.gripper_qpos] = self.gripper if gripper is None else gripper
        mujoco.mj_kinematics(self.model, d)
        self.canvas[:] = 0
        z, hulls = self.grid.z, []
        for g, pts in zip(self.geoms, self.points):
            w = d.geom_xpos[g] + pts @ d.geom_xmat[g].reshape(3, 3).T
            w = w[w[:, 2] < self.cz - 0.02]            # parts at the camera's height cannot hide the mat
            if len(w) < 3:
                continue
            xy = self.foot + ((self.cz - z) / (self.cz - w[:, 2]))[:, None] * (w[:, :2] - self.foot)
            px = np.round(self.grid.to_px(xy) * 16).astype(np.int32)
            hulls.append(cv2.convexHull(px))
        for h in hulls:                                  # one by one: fillPoly fills several by the even-odd rule
            cv2.fillPoly(self.canvas, [h], 1, cv2.LINE_8, shift=4)
        return cv2.dilate(self.canvas, self.grow) > 0
