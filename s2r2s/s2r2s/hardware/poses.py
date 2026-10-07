"""Where the object pose comes from: read() -> ((x, y, yaw) in the robot base frame or None, timestamp).

None means "no fresh estimate" (occluded or dropped frame); the runner then keeps the last pose and
stops the arm if estimates stay missing for too long.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .robots import SimRobot


class SimPoseSource:
    """Simulated camera on a SimRobot: the same delay / noise / dropout model as training."""

    def __init__(self, robot: SimRobot, latency_steps: int = 1, pos_noise: float = 0.001,
                 yaw_noise: float = math.radians(1.0), dropout: float = 0.03, seed: int = 0):
        self.robot, self.latency = robot, latency_steps
        self.pos_noise, self.yaw_noise, self.dropout = pos_noise, yaw_noise, dropout
        self.rng = np.random.default_rng(seed)
        self.history = []

    def read(self):
        self.history.insert(0, self.robot.object_pose())
        del self.history[self.latency + 1:]
        if self.rng.random() < self.dropout:
            return None, time.monotonic()
        pose = self.history[-1].copy()
        pose[:2] += self.rng.normal(0, self.pos_noise, 2)
        pose[2] = (pose[2] + self.rng.normal(0, self.yaw_noise) + math.pi) % (2 * math.pi) - math.pi
        return pose, time.monotonic()


class CameraPoseSource:
    """RealSense D435i object-pose estimator (Milestone 2 perception task, not implemented yet).

    Must return the object's (x, y, yaw) in the robot base frame at about 30 Hz, with 2 mm / 2 deg
    accuracy and under 150 ms latency (the requirements measured by ``python -m s2r2s.sensitivity``).
    See ROADMAP.md, section 3, for the recommended pipeline (depth + colour segmentation, template fit).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Camera pose estimation is the Milestone 2 perception task; see ROADMAP.md.")

    def read(self):
        raise NotImplementedError
