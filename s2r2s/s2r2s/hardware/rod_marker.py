"""An ArUco marker on the rod, seen by the C920: the rod's actual motion, for the backlash (``backlash.py``) and
camera latency (``latency.py``) measurements.

Print it with ``python -m s2r2s.hardware.camera_calibrate boards`` (``calibration/print/rod_marker.pdf``: id 20 of
the mat markers' dictionary, a 30 mm black square in a 5 mm white border), glue it to thin card (40 x 40 mm) and
tape the card flat to the rod 5-10 mm above its tip, printed side up, sticking out to the side the camera looks
from (the robot's left for the left-wall slot). Take it off before pushing anything. Where exactly does not
matter: the marker's position comes from the image (``locate``: solvePnP on its corners) and only its motion is
used. Above it, the gripper (and a wrist-camera mount, if fitted) must leave it in view: ``backlash --check-marker``
shows the camera's view at every test pose. The simulated card (``SimCamera(rod_marker=ROD_MARKER)``) sticks out
30 mm to the gripper's left, 7 mm above the tip; from the left-wall slot it is seen at 6 of 7 test poses.

    corners = find(frame)                                  # (4, 2) pixels or None
    uv, xyz = locate(corners, camera)                      # centre pixel, centre in the robot frame
    arm = ArmPoints(scene); g = arm.image_jacobian(q, xyz, camera)   # pixels per radian of each joint
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from ..scene import SceneConfig, SceneIndex, build_model


@dataclass
class RodMarker:
    id: int = 20                             # not one of the mat markers (0-7)
    size: float = 0.030                      # black square (m)
    margin: float = 0.005                    # white border
    sim_pos: tuple = (0.0, 0.030, -0.124)    # simulated card: gripper frame (rod tip at z = -0.131; +y the robot's left)


ROD_MARKER = RodMarker()


def find(image, marker: RodMarker = ROD_MARKER) -> np.ndarray | None:
    """The marker's corners (4, 2) in ArUco order (sub-pixel), or None."""
    from .camera_calibrate import detect_markers
    return detect_markers(image).get(marker.id)


def centre_pixel(corners: np.ndarray) -> np.ndarray:
    """Where the diagonals cross: the image of the square's centre (under perspective, not the corners' mean)."""
    a, b, c, d = np.asarray(corners, float)
    m = np.array([c - a, b - d]).T
    s = np.linalg.solve(m, b - a)[0]
    return a + s * (c - a)


def locate(corners: np.ndarray, camera, marker: RodMarker = ROD_MARKER) -> tuple[np.ndarray, np.ndarray]:
    """Centre pixel and the centre's position in the robot frame (solvePnP on the four corners)."""
    h = marker.size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    ok, _, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float64).reshape(4, 1, 2), camera.K, camera.dist,
                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed on the rod marker")
    return centre_pixel(corners), camera.R.T @ (tvec.ravel() - camera.t)


def average(frames, marker: RodMarker = ROD_MARKER) -> np.ndarray | None:
    """Median corners over several frames of a still marker (None when it is seen in fewer than half)."""
    seen = [c for c in (find(f, marker) for f in frames) if c is not None]
    return np.median(seen, 0) if len(seen) >= max(1, len(frames) // 2) else None


class ArmPoints:
    """Where a point fixed to the gripper goes when the joints move: the scene's arm kinematics (MuJoCo)."""

    def __init__(self, scene: SceneConfig | None = None):
        self.model = build_model(scene or SceneConfig(timestep=0.005, lightbox=True), visual=False)
        self.data = mujoco.MjData(self.model)
        self.idx = SceneIndex(self.model)
        self.body = self.model.body("gripper").id

    def _frame(self, q):
        self.data.qpos[self.idx.arm_qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.body].copy(), self.data.xmat[self.body].reshape(3, 3).copy()

    def attach(self, q, xyz) -> np.ndarray:
        """``xyz`` (robot frame) with the arm at ``q`` -> the same point in the gripper frame."""
        pos, rot = self._frame(q)
        return rot.T @ (np.asarray(xyz, float) - pos)

    def where(self, q, local) -> np.ndarray:
        pos, rot = self._frame(q)
        return pos + rot @ local

    def jacobian(self, q, xyz, eps: float = 1e-4) -> np.ndarray:
        """(3, 5): how the point at ``xyz`` (robot frame, arm at ``q``) moves per radian of each joint."""
        local, q = self.attach(q, xyz), np.asarray(q, float)
        return np.stack([(self.where(q + eps * e, local) - self.where(q - eps * e, local)) / (2 * eps)
                         for e in np.eye(len(q))], 1)

    def image_jacobian(self, q, xyz, camera, eps: float = 1e-4) -> np.ndarray:
        """(2, 5): how the point's image moves (pixels) per radian of each joint."""
        local, q = self.attach(q, xyz), np.asarray(q, float)
        cols = []
        for e in np.eye(len(q)):
            uv = camera.project(np.stack([self.where(q + eps * e, local), self.where(q - eps * e, local)]))
            cols.append((uv[0] - uv[1]) / (2 * eps))
        return np.stack(cols, 1)
