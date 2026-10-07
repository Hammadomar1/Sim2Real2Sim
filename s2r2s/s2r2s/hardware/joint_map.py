"""LeRobot joint readings (degrees) <-> MuJoCo joint angles (radians).

LeRobot (``use_degrees=True``) reports each arm joint in degrees, with 0 at the middle of the
range swept during LeRobot's own calibration. The MuJoCo model (MuJoCo Menagerie, derived from
TheRobotStudio's "new calibration" model) is meant to share that zero, but real arms differ by
a few degrees and a joint may turn the other way. Per joint:

    q_mujoco [rad] = sign * radians(reading [deg] - offset [deg])

``calibrate.py`` measures sign and offset on your arm; until then the defaults (sign +1,
offset 0) apply.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..scene import ARM_JOINTS


@dataclass
class JointMap:
    sign: list[float] = field(default_factory=lambda: [1.0] * 5)
    offset_deg: list[float] = field(default_factory=lambda: [0.0] * 5)
    tool_length_error: float = 0.0        # measured rod length minus the model's (m), informational
    names: tuple[str, ...] = ARM_JOINTS

    def to_sim(self, reading_deg) -> np.ndarray:
        """Robot readings (..., 5) in degrees -> MuJoCo radians."""
        return np.asarray(self.sign) * np.radians(np.asarray(reading_deg, dtype=float) - np.asarray(self.offset_deg))

    def to_robot(self, q_rad) -> np.ndarray:
        """MuJoCo radians (..., 5) -> robot command in degrees."""
        return np.degrees(np.asarray(q_rad, dtype=float) * np.asarray(self.sign)) + np.asarray(self.offset_deg)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["names"] = list(self.names)
        path.write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path) -> "JointMap":
        d = json.loads(Path(path).read_text())
        d["names"] = tuple(d.get("names", ARM_JOINTS))
        return JointMap(**d)
