"""Batched forward/inverse kinematics of the SO-101 pusher, in NumPy.

Parameters are read from the compiled MuJoCo model, so FK matches MuJoCo
exactly. The same code is meant to run on the real robot: the policy outputs a
tool-tip displacement, IK turns it into joint targets, and those targets go to
the MuJoCo position actuators (simulation) or the Feetech servos (hardware).

The tool is kept vertical (gripper +z axis = world +z) at a fixed tip height.
Wrist roll does not move the tip (the rod lies on the roll axis to within
1.4 mm), so IK solves for pan, lift, elbow and wrist flex and holds roll fixed.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _axis_angle(axis, angle):
    """Batched Rodrigues rotation matrices for a fixed unit axis. angle: (N,) -> (N, 3, 3)."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    return np.stack([
        np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
        np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
        np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1)], -2)


class PusherKinematics:
    def __init__(self, model: mujoco.MjModel, roll_ref: float = 0.0):
        tool = model.site("tool_tip").id
        body = model.site_bodyid[tool]
        chain = []
        while body != 0:
            chain.append(body)
            body = model.body_parentid[body]
        chain.reverse()
        self.links = []   # (pos, R, None | (axis, anchor, joint index, z-axis fast path))
        names = list(ARM_JOINTS)
        for b in chain:
            jnt = None
            if model.body_jntnum[b] == 1:
                j = model.body_jntadr[b]
                axis, anchor = model.jnt_axis[j].copy(), model.jnt_pos[j].copy()
                fast = not anchor.any() and np.allclose(axis, [0, 0, 1])
                jnt = (axis, anchor, names.index(model.joint(j).name), fast)
            elif model.body_jntnum[b] > 1:
                raise ValueError("expected at most one joint per arm body")
            self.links.append((model.body_pos[b].copy(), _quat_to_mat(model.body_quat[b]), jnt))
        self.tip_local = model.site_pos[tool].copy()
        if np.linalg.norm(model.site_quat[tool] - [1, 0, 0, 0]) > 1e-9:
            raise ValueError("tool site must not be rotated relative to the gripper body")
        jid = [model.joint(n).id for n in ARM_JOINTS]
        self.lower = model.jnt_range[jid, 0].copy()
        self.upper = model.jnt_range[jid, 1].copy()
        self.roll_ref = roll_ref
        # Which way does positive pan turn the arm? (The Menagerie shoulder frame is flipped.)
        branch = np.array([[0.0, -0.17, 0.47, 1.32, roll_ref], [0.2, -0.17, 0.47, 1.32, roll_ref]])
        tip, _ = self.forward(branch)
        self.pan_sign = float(np.sign(np.arctan2(tip[1, 1], tip[1, 0]) - np.arctan2(tip[0, 1], tip[0, 0])))
        self.branch = branch[0]

    def seed(self, xy: np.ndarray) -> np.ndarray:
        """Initial guess on the elbow-up branch used everywhere in the workspace."""
        q = np.tile(self.branch, (len(xy), 1))
        q[:, 0] = self.pan_sign * np.arctan2(xy[:, 1], xy[:, 0])
        return q

    def forward(self, q: np.ndarray, with_jacobian: bool = False):
        """q: (N, 5) arm joints. Returns tip (N,3), tool axis (N,3) [, Jpos (N,3,5), Jrot (N,3,5)]."""
        q = np.atleast_2d(q)
        n = q.shape[0]
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        p = np.zeros((n, 3))
        axes = np.zeros((n, 5, 3))
        origins = np.zeros((n, 5, 3))
        for pos, Rb, jnt in self.links:
            p = p + R @ pos
            R = R @ Rb
            if jnt is not None:
                axis, anchor, k, fast = jnt
                axes[:, k] = R @ axis
                origins[:, k] = p + R @ anchor
                if not fast:
                    Rq = _axis_angle(axis, q[:, k])
                    p = p + np.einsum("nij,nj->ni", R, anchor - Rq @ anchor)
                    R = R @ Rq
                else:  # fast path: rotation about the local z axis through the body origin
                    c, s = np.cos(q[:, k])[:, None], np.sin(q[:, k])[:, None]
                    x, y = R[:, :, 0], R[:, :, 1]
                    R = np.stack([c * x + s * y, c * y - s * x, R[:, :, 2]], axis=-1)
        tip = p + R @ self.tip_local
        tool_axis = R[:, :, 2]
        if not with_jacobian:
            return tip, tool_axis
        jpos = np.cross(axes, tip[:, None, :] - origins).transpose(0, 2, 1)
        jrot = axes.transpose(0, 2, 1)
        return tip, tool_axis, jpos, jrot

    def solve(self, target: np.ndarray, q_init: np.ndarray, iterations: int = 3,
              damping: float = 1e-3, rot_weight: float = 0.05, report: bool = True):
        """Damped least squares for a vertical tool at ``target`` (N, 3).

        Solves pan, lift, elbow, wrist flex; wrist roll is held at ``roll_ref``.
        Returns (q (N, 5), position error norm (N,), tool axis z-component (N,)); the last two
        are None when ``report`` is False (saves one FK pass in the control loop).
        """
        q = np.array(q_init, dtype=float, copy=True)
        q[:, 4] = self.roll_ref
        lo, hi = self.lower[:4] + 1e-3, self.upper[:4] - 1e-3
        eye = np.eye(4) * damping ** 2
        for _ in range(iterations):
            tip, axis, jpos, jrot = self.forward(q, with_jacobian=True)
            ang = np.cross(axis, np.array([0.0, 0.0, 1.0]))[:, :2]
            err = np.concatenate([target - tip, rot_weight * ang], -1)
            J = np.concatenate([jpos[:, :, :4], rot_weight * jrot[:, :2, :4]], 1)
            Jt = J.transpose(0, 2, 1)
            dq = np.linalg.solve(Jt @ J + eye, (Jt @ err[..., None]))[..., 0]
            q[:, :4] = np.clip(q[:, :4] + np.clip(dq, -0.5, 0.5), lo, hi)
        if not report:
            return q, None, None
        tip, axis = self.forward(q)
        return q, np.linalg.norm(tip - target, axis=-1), axis[:, 2]
