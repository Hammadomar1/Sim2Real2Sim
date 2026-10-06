"""Mirror a simulated world into the full-mesh model for viewing and video."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .env import PushEnv
from .scene import build_model, set_gate, set_marker


class SceneMirror:
    """Copies world ``i`` of a ``PushEnv`` (stripped model) into the visual model (same dynamics/layout)."""

    def __init__(self, env: PushEnv, world: int = 0, show_estimate: bool = True):
        self.env, self.i, self.show_estimate = env, world, show_estimate
        name = env.cfg.task.objects[env.obj_id[world]]
        self.model = build_model(env.cfg.scene, name, visual=True)
        self.data = mujoco.MjData(self.model)
        assert self.model.nq == env.idx.nq and self.model.nv == env.idx.nv
        self.sync()

    def sync(self):
        e, i, nq, nv = self.env, self.i, self.env.idx.nq, self.env.idx.nv
        self.data.qpos[:] = e.state[i, 1:1 + nq]
        self.data.qvel[:] = e.state[i, 1 + nq:1 + nq + nv]
        self.data.time = e.step_count[i] * e.cfg.task.control_dt
        if hasattr(e, "gate_centre"):
            set_gate(self.model, e.gate_centre[i], e.gate_angle[i], e.gate_width[i])
        set_marker(self.model, self.data, "goal", e.goal[i])
        set_marker(self.model, self.data, "estimate", e.obs_pose[i] if self.show_estimate else (5.0, 5.0, 0.0))
        mujoco.mj_forward(self.model, self.data)

    def status(self) -> str:
        e, i = self.env, self.i
        pose = e.object_pose()[i]
        g = e.goal[i]
        sym_goal = e._symmetric_goal_yaw(pose[None, 2], g[None, 2], e.obj_id[i:i + 1])[0]
        pos = np.linalg.norm(pose[:2] - g[:2]) * 1e3
        yaw = abs((pose[2] - sym_goal + math.pi) % (2 * math.pi) - math.pi)
        ok = pos <= e.cfg.task.success_pos * 1e3 and yaw <= e.cfg.task.success_yaw
        stage = f"  [{('align', 'pass', 'goal')[e.stage[i]]}]" if hasattr(e, "stage") else ""
        return (f"t {e.step_count[i] * e.cfg.task.control_dt:4.1f}s  pos err {pos:5.1f} mm  "
                f"yaw err {math.degrees(yaw):5.1f} deg{stage}  {'AT GOAL' if ok else ''}")
