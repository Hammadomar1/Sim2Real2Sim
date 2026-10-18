"""Where the object pose comes from: read() -> ((x, y, yaw) in the robot base frame or None, timestamp).

None means "no fresh estimate" (occluded or dropped frame); the runner then keeps the last pose and
stops the arm if estimates stay missing for too long. Clutter policies also need read_block(): the
same for the block that must stay in place.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..scene import SceneConfig
from .robots import SimRobot


class SimPoseSource:
    """Simulated camera on a SimRobot: the same delay / noise / dropout model as training."""

    def __init__(self, robot: SimRobot, latency_steps: int = 1, pos_noise: float = 0.001,
                 yaw_noise: float = math.radians(1.0), dropout: float = 0.03, seed: int = 0):
        self.robot, self.latency = robot, latency_steps
        self.pos_noise, self.yaw_noise, self.dropout = pos_noise, yaw_noise, dropout
        self.rng = np.random.default_rng(seed)
        self.history = {"object": [], "block": []}

    def read(self):
        return self._seen("object", self.robot.object_pose())

    def read_block(self):
        return self._seen("block", self.robot.block_pose())

    def _seen(self, name, truth):
        history = self.history[name]
        history.insert(0, truth)
        del history[self.latency + 1:]
        if self.rng.random() < self.dropout:
            return None, time.monotonic()
        pose = history[-1].copy()
        pose[:2] += self.rng.normal(0, self.pos_noise, 2)
        pose[2] = (pose[2] + self.rng.normal(0, self.yaw_noise) + math.pi) % (2 * math.pi) - math.pi
        return pose, time.monotonic()


class CameraPoseSource:
    """The lightbox C920 and the colour-and-shape pose estimator (``vision.PoseEstimator``).

    A capture thread keeps the newest frame (``camera.C920``); a processing thread runs the estimator on it as
    soon as the previous frame is done, so results are never queued behind old frames. read() / read_block()
    return the newest result: ((x, y, yaw) or None, stamp), where stamp is when that frame reached the computer
    (time.monotonic()). None: the object was missing or too hidden in the newest frame, or nothing newer than
    ``max_age`` has been processed. Needs the files ``python -m s2r2s.hardware.camera_calibrate`` writes
    (settings, intrinsics, extrinsics, colours); see HARDWARE.md section 4.

    set_arm(q) tells it the arm's joint angles (simulator radians, as the runner reads them every tick): each
    frame is then fitted with the arm masked out (``arm_mask.ArmMask``), and poses are returned down to a
    quarter of the object visible instead of 60 %. ``scene`` must match the real setup (lightbox, rod).

    read_block() asks more of a frame (``block_min_visible``, 60 % of the block visible): a missing block estimate
    costs nothing (the runner keeps the last one and the block should not move), while a wrong one moves the
    controller's rod guard.

    ``source``: any object with the camera's ``wait(after, timeout)`` (a ``camera.ReplaySource`` in tests).
    """

    def __init__(self, source=None, camera=None, colours=None, object_name: str = "tee", block_name: str = "box",
                 config=None, max_age: float = 0.3, scene=None, arm_margin: float = 0.006, max_arm_age: float = 0.2,
                 block_min_visible: float = 0.6):
        from .arm_mask import ArmMask
        from .camera import C920
        from .vision import EXTRINSICS, INTRINSICS, COLOURS, CameraModel, Colours, PoseEstimator
        needed = ([(INTRINSICS, "intrinsics"), (EXTRINSICS, "extrinsics")] if camera is None else []) + \
                 ([(COLOURS, "colours")] if colours is None else [])
        for path, step in needed:
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing: run python -m s2r2s.hardware.camera_calibrate {step}")
        camera = camera or CameraModel.load()
        colours = colours or Colours.load()
        names = [n for n in (object_name, block_name) if n in colours.objects]
        self.object_name, self.block_name = object_name, block_name
        self.estimator = PoseEstimator(camera, colours, names, config)
        self.arm_mask = ArmMask(camera, self.estimator.grid, scene or SceneConfig(timestep=0.005, lightbox=True), arm_margin)
        self.max_age, self.max_arm_age, self.block_min_visible = max_age, max_arm_age, block_min_visible
        self._arm = None                      # (joint angles, stamp) from set_arm
        self.source = source if source is not None else C920()
        self._lock = threading.Lock()
        self._latest = None                   # (detections, stamp, processing seconds)
        self.stats = {"frames": 0, "dropped": {n: 0 for n in names}, "process_ms": [], "age_ms": []}
        self._stop = False
        self._thread = threading.Thread(target=self._process, name="pose-estimator", daemon=True)
        self._thread.start()

    def _process(self):
        count = 0
        while not self._stop:
            got = self.source.wait(count, timeout=0.5)
            if got is None:
                continue
            frame, stamp, count = got
            t0 = time.monotonic()
            arm = self._arm
            hidden = self.arm_mask(arm[0]) if arm is not None and abs(stamp - arm[1]) <= self.max_arm_age else None
            found = self.estimator.estimate(frame, stamp, hidden)
            done = time.monotonic()
            with self._lock:
                self._latest = (found, stamp, done - t0)
                s = self.stats
                s["frames"] += 1
                for name, det in found.items():
                    s["dropped"][name] += det.pose is None
                s["process_ms"] = (s["process_ms"] + [(done - t0) * 1e3])[-300:]
                s["age_ms"] = (s["age_ms"] + [(done - stamp) * 1e3])[-300:]

    def set_arm(self, q_sim, stamp: float | None = None):
        """The arm's joint angles (simulator radians) now, or at ``stamp`` (time.monotonic())."""
        self._arm = (np.array(q_sim, float), time.monotonic() if stamp is None else stamp)

    def read(self):
        return self._get(self.object_name)

    def read_block(self):
        return self._get(self.block_name, self.block_min_visible)

    def detections(self):
        """The newest frame's full results (``vision.Detection`` per object) and its stamp, or (None, 0)."""
        with self._lock:
            return (None, 0.0) if self._latest is None else self._latest[:2]

    def _get(self, name, min_visible: float = 0.0):
        with self._lock:
            latest = self._latest
        now = time.monotonic()
        if latest is None or now - latest[1] > self.max_age or name not in latest[0]:
            return None, now
        det = latest[0][name]
        return (None if det.pose is None or det.visible < min_visible else det.pose.copy()), latest[1]

    def summary(self) -> str:
        with self._lock:
            s = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)) for k, v in self.stats.items()}
        if not s["frames"]:
            return "no frames processed"
        drops = ", ".join(f"{n} {d / s['frames']:.1%}" for n, d in s["dropped"].items())
        return (f"{s['frames']} frames, dropped: {drops}; processing {np.median(s['process_ms']):.0f} ms median "
                f"({np.max(s['process_ms']):.0f} max), frame arrival to result {np.median(s['age_ms']):.0f} ms median "
                f"(+ the camera's own delay)")

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        self.source.close()
