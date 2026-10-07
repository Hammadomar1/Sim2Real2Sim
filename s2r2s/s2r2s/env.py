"""Batched SE(2) pushing environment for the SO-101 (CPU MuJoCo, many threads).

One ``PushEnv`` simulates ``num_envs`` independent worlds. Physics runs in C
through ``mujoco.rollout`` (a persistent thread pool); everything else -
kinematics, observations, rewards, resets - is vectorised NumPy.

Action  : 2D tool-tip velocity command in [-1, 1]^2 (scaled by ``max_speed``),
          integrated into a commanded tool position, converted by IK into
          joint position targets for the servos. The same controller code is
          intended to run on the real arm.
Actor   : only quantities the real system can measure - joint encoders (tool
          position via FK) and a camera estimate of the object pose with
          latency, noise and dropouts.
Critic  : the true state plus physical parameters (asymmetric actor-critic).
Episode : fixed horizon; the policy must bring the object to the goal pose
          and keep it there. Episodes end early only on unrecoverable failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import os

import mujoco
from mujoco import rollout
import numpy as np

from .kinematics import PusherKinematics
from .objects import OBJECTS, object_tables
from .scene import SceneConfig, SceneIndex, build_model

TWO_PI = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


@dataclass
class TaskConfig:
    objects: tuple[str, ...] = ("tee",)
    control_dt: float = 0.05
    episode_seconds: float = 20.0
    tool_height: float = 0.005            # tool tip above the table (m)
    max_speed: float = 0.08               # tool speed at |action| = 1 (m/s)
    tracking_limit: float = 0.015         # anti-windup: command stays this close to the measured tip
    tool_r: tuple[float, float] = (0.12, 0.275)   # reachable annulus for the tool tip (m)
    tool_az: float = math.radians(55)
    obj_r: tuple[float, float] = (0.175, 0.225)   # object start/goal centre region
    obj_az: float = math.radians(30)
    fail_r: tuple[float, float] = (0.135, 0.265)  # object centre outside this -> unrecoverable
    fail_az: float = math.radians(42)
    # Goal difficulty: offset limits interpolate from easy (c=0) to full (c=1).
    difficulty: float = 1.0
    goal_shift: tuple[float, float] = (0.03, 0.12)               # max translation (m)
    goal_turn: tuple[float, float] = (math.radians(30), math.pi)  # max rotation (rad)
    easy_fraction: float = 0.25           # share of episodes drawn below the current difficulty
    # Success = final pose within these tolerances (also used for logging).
    success_pos: float = 0.010
    success_yaw: float = math.radians(10)
    # Reward weights.
    w_coarse: float = 1.0
    w_fine: float = 1.0
    sigma_coarse: float = 0.05
    sigma_fine: float = 0.01
    w_progress: float = 0.5               # per cm of keypoint-distance reduction
    w_reach: float = 0.2
    sigma_reach: float = 0.02
    w_action: float = 0.01
    w_action_rate: float = 0.05
    failure_penalty: float = 10.0
    # Level 1 puzzle (GateEnv only): push the object through an opening in a wall, then to the goal.
    gate: bool = False
    gate_az: float = math.radians(6)              # wall azimuth uniformly within +-gate_az
    gate_r: tuple[float, float] = (0.195, 0.205)  # opening centre radius (m)
    gate_width: tuple[float, float] = (0.064, 0.070)  # opening width at full difficulty (m)
    gate_width_easy: float = 0.080                # opening width at difficulty 0 (any orientation passes)
    gate_clear_az: float = math.radians(17)       # start/goal centres at least this far (azimuth) from the wall
    gate_pre: float = 0.05                        # aligned waypoint this far before the wall (m)
    gate_post: float = 0.05                       # ... and this far after it
    gate_align_tol: float = 0.012                 # keypoint distance to the pre-gate pose that starts the passage
    gate_cross: float = 0.035                     # object centre this far past the wall: head for the goal
    # Clutter task (ClutterEnv only): a second block that must stay where it is.
    clutter: str = ""                             # object name from objects.py, e.g. "box"
    clutter_in_way: float = 0.6                   # share of episodes with the block close beside the object's path
    clutter_min_gap: float = 0.015                # block to the object's straight sweep, at least (m): the rod's
                                                  # diameter plus rod_guard, so the guarded rod always fits between
    clutter_near: float = 0.030                   # "close beside": block within this of the sweep (m)
    clutter_tol: float = 0.010                    # allowed displacement of the block (m) ...
    clutter_yaw_tol: float = math.radians(10)     # ... and rotation
    w_disturb: float = 3.0                        # reward per cm the block is moved (negative)
    w_displaced: float = 1.0                      # per-step cost while the block is out of tolerance
    w_near: float = 0.5                           # per-step cost as the rod or the object comes within ...
    near_margin: float = 0.010                    # ... this distance of the block (m), growing linearly to contact
                                                  # (the rod covers up to 4-5 mm per step: 10 mm is 2-3 steps of warning)
    rod_guard: float = 0.003                      # controller: never command the rod closer than this to the
                                                  # (estimated) block, slide along it instead; 0 = off
    w_guard: float = 0.5                          # per-step cost per full step of command (max_speed * control_dt)
                                                  # the guard had to remove: plan around the block, don't lean on it


@dataclass
class RandomizationConfig:
    enabled: bool = True
    mass_scale: tuple[float, float] = (0.7, 1.5)
    com_offset: float = 0.003             # object centre-of-mass shift (m, per axis)
    table_friction: tuple[float, float] = (0.20, 0.50)
    pusher_friction: tuple[float, float] = (0.15, 0.50)
    servo_kp_scale: tuple[float, float] = (0.8, 1.2)
    damping_scale: tuple[float, float] = (0.8, 1.2)
    tool_height_error: float = 0.002      # IK height calibration error (m)
    max_action_delay: int = 5             # physics substeps the previous command persists
    # Camera model for the actor's object pose estimate.
    obs_latency_steps: tuple[int, int] = (0, 2)
    obs_pos_noise: float = 0.001
    obs_yaw_noise: float = math.radians(1.0)
    obs_dropout: float = 0.03
    tool_noise: float = 0.0005
    gate_friction: tuple[float, float] = (0.20, 0.50)


@dataclass
class EnvConfig:
    num_envs: int = 4096
    seed: int = 0
    num_threads: int = 0                  # 0 = all logical CPUs
    task: TaskConfig = field(default_factory=TaskConfig)
    rand: RandomizationConfig = field(default_factory=RandomizationConfig)
    scene: SceneConfig = field(default_factory=lambda: SceneConfig(timestep=0.005))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        task = TaskConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("task").items()})
        rand = RandomizationConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("rand").items()})
        scene = SceneConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("scene").items()})
        return EnvConfig(task=task, rand=rand, scene=scene, **d)


def apply_overrides(cfg: EnvConfig, items) -> EnvConfig:
    """Apply ``section.field=value`` overrides, e.g. ``rand.obs_pos_noise=0.003`` or ``rand.obs_latency_steps=2,4``."""
    for item in items:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        target = getattr(cfg, section)
        old = getattr(target, name)
        parse = type(old[0]) if isinstance(old, tuple) else type(old)
        if parse is bool:
            new = value.lower() in ("1", "true", "yes")
        elif isinstance(old, tuple):
            new = tuple(parse(v) for v in value.split(","))
        else:
            new = parse(value)
        setattr(target, name, new)
    return cfg


class PushEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        t, r = cfg.task, cfg.rand
        self.n = n = cfg.num_envs
        # Independent streams: scenes depend only on the seed, never on whether physical
        # randomisation or sensor noise is enabled, so evaluations on one seed are paired.
        self.rng, self.param_rng, self.noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(3)]
        if len(set(OBJECTS[o].height for o in t.objects)) != 1:
            raise ValueError("all objects in one environment must share a height")
        # One compiled model per object type, then one copy per world for randomisation.
        self.base_models = [build_model(cfg.scene, name, visual=False) for name in t.objects]
        m0 = self.base_models[0]
        self.idx = SceneIndex(m0)
        self.kin = PusherKinematics(m0)
        self.obj = object_tables(list(t.objects))
        self.obj_height = OBJECTS[t.objects[0]].height
        self.nsub = round(t.control_dt / cfg.scene.timestep)
        if abs(self.nsub * cfg.scene.timestep - t.control_dt) > 1e-9:
            raise ValueError("control_dt must be a multiple of the physics timestep")
        self.max_steps = round(t.episode_seconds / t.control_dt)
        self.obj_id = self.rng.integers(len(t.objects), size=n)
        self.models = [self._copy_model(self.base_models[k]) for k in self.obj_id]
        nthread = cfg.num_threads or os.cpu_count()
        self.datas = [mujoco.MjData(m0) for _ in range(nthread)]
        self.pool = rollout.Rollout(nthread=nthread)
        self.scratch = mujoco.MjData(m0)
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(m0, self.spec)
        self.qpos_adr = 1
        self.qvel_adr = 1 + m0.nq
        self.state = np.zeros((n, self.nstate))
        self.out = np.zeros((n, self.nsub, self.nstate))
        self.ctrl = np.zeros((n, self.nsub, m0.nu))
        self.warmstart = np.zeros((n, m0.nv))
        # Nominal physical parameters for randomisation.
        self.arm_dofs = self.idx.arm_qvel
        self.nom_kp = m0.actuator_gainprm[self.idx.arm_act, 0].copy()
        self.nom_kv = -m0.actuator_biasprm[self.idx.arm_act, 2].copy()
        self.nom_damp = m0.dof_damping[self.arm_dofs].copy()
        self.nom_mass = np.array([bm.body_mass[self.idx.object_body] for bm in self.base_models])
        self.nom_inertia = np.array([bm.body_inertia[self.idx.object_body] for bm in self.base_models])
        # Per-world task state.
        z = lambda *s: np.zeros((n, *s))
        self.goal = z(3)
        self.cmd_xy = z(2)
        self.q_cmd = z(5)
        self.prev_ctrl = z(m0.nu)
        self.prev_action = z(2)
        self.cmd_deflection = z()           # how far the last command was moved by _guard_command (m)
        self.prev_dist = z()
        self.step_count = np.zeros(n, dtype=np.int64)
        self.difficulty_ep = z()
        self.tool_z_cmd = np.full(n, t.tool_height)
        self.delay = np.zeros(n, dtype=np.int64)
        self.params = z(4)                 # table mu, pusher mu, mass scale, kp scale (critic only)
        self.latency = np.zeros(n, dtype=np.int64)
        # True object pose history (newest first), long enough for the largest camera latency.
        self.pose_hist = z(max(r.obs_latency_steps[1], 1) + 2, 3)
        self.obs_pose = z(3)               # camera estimate shown to the actor
        self.prev_obs_pose = z(3)
        self.first_success = np.full(n, -1)
        self.tip = z(3)                    # cached FK of the measured joints
        self.qarm = z(5)
        self.difficulty = t.difficulty
        self.num_actions = 2
        self.reset_all()
        self.num_obs = self.obs["actor"].shape[1]
        self.num_critic_obs = self.obs["critic"].shape[1]

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _copy_model(m):
        return m.__copy__()

    def _randomize(self, i):
        """Write fresh physical parameters into world i's model."""
        r = self.cfg.rand
        k = self.obj_id[i]
        m = self.models[i]
        b = self.base_models[k]
        u = lambda lo_hi: self.param_rng.uniform(*lo_hi)
        mu_t, mu_p, ms, kp, damp = (u(r.table_friction), u(r.pusher_friction), u(r.mass_scale),
                                    u(r.servo_kp_scale), u(r.damping_scale))
        com = self.param_rng.uniform(-r.com_offset, r.com_offset, 2)
        if not r.enabled:
            self.params[i] = [b.geom_friction[self.idx.table_geom, 0], b.geom_friction[self.idx.pusher_geom, 0], 1, 1]
            return
        m.geom_friction[self.idx.table_geom, 0] = mu_t
        m.geom_friction[self.idx.pusher_geom, 0] = mu_p
        ob = self.idx.object_body
        m.body_mass[ob] = self.nom_mass[k] * ms
        m.body_inertia[ob] = self.nom_inertia[k] * ms
        m.body_ipos[ob] = b.body_ipos[ob] + np.r_[com, 0.0]
        a = self.idx.arm_act
        m.actuator_gainprm[a, 0] = self.nom_kp * kp
        m.actuator_biasprm[a, 1] = -self.nom_kp * kp
        m.dof_damping[self.arm_dofs] = self.nom_damp * damp
        mujoco.mj_setConst(m, self.scratch)
        self.params[i] = [mu_t, mu_p, ms, kp]

    # ----------------------------------------------------------------- resets
    def reset_all(self):
        self.reset(np.arange(self.n))
        self.obs = self._observe()
        return self.obs

    def _sample_region(self, n, r_rng, az):
        rad = self.rng.uniform(*r_rng, n)
        ang = self.rng.uniform(-az, az, n)
        return np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)

    def _in_region(self, xy, r_rng, az):
        rad = np.linalg.norm(xy, axis=-1)
        ang = np.arctan2(xy[..., 1], xy[..., 0])
        return (rad >= r_rng[0]) & (rad <= r_rng[1]) & (np.abs(ang) <= az)

    def reset(self, ids):
        if len(ids) == 0:
            return
        for i in ids:
            self._randomize(i)
        start, start_yaw = self._sample_scene(ids)
        tool = self._sample_tool(ids, start, start_yaw)
        self._place(ids, start, start_yaw, tool)

    def _sample_scene(self, ids):
        """Object start pose and goal (written to self.goal) for new episodes."""
        t = self.cfg.task
        n = len(ids)
        start = self._sample_region(n, t.obj_r, t.obj_az)
        start_yaw = self.rng.uniform(-math.pi, math.pi, n)
        # Goal pose relative to the start, limited by the curriculum difficulty.
        c = np.full(n, self.difficulty)
        easy = self.rng.random(n) < t.easy_fraction
        c[easy] = self.rng.uniform(0, self.difficulty, easy.sum())
        self.difficulty_ep[ids] = c
        shift = t.goal_shift[0] + c * (t.goal_shift[1] - t.goal_shift[0])
        turn = t.goal_turn[0] + c * (t.goal_turn[1] - t.goal_turn[0])
        goal = start.copy()
        todo = np.arange(n)
        for _ in range(50):
            if len(todo) == 0:
                break
            ang = self.rng.uniform(0, TWO_PI, len(todo))
            dist = shift[todo] * np.sqrt(self.rng.random(len(todo)))
            cand = start[todo] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1)
            ok = self._in_region(cand, t.obj_r, t.obj_az)
            goal[todo[ok]] = cand[ok]
            todo = todo[~ok]
        goal_yaw = wrap(start_yaw + self.rng.uniform(-1, 1, n) * turn)
        self.goal[ids] = np.c_[goal, goal_yaw]
        return start, start_yaw

    def _tool_clear(self, ids, cand, start, start_yaw):
        """Which candidate tool positions are free (base: clear of the object)."""
        clear = self.obj.radius[self.obj_id[ids]] + self.cfg.scene.pusher_radius + 0.01
        return np.linalg.norm(cand - start, axis=-1) > clear

    def _sample_tool(self, ids, start, start_yaw):
        """Tool start: reachable and clear of the object (and of obstacles, in subclasses)."""
        t = self.cfg.task
        n = len(ids)
        tool = np.zeros((n, 2))
        todo = np.arange(n)
        for _ in range(200):
            if len(todo) == 0:
                break
            cand = self._sample_region(len(todo), (t.tool_r[0] + 0.01, t.tool_r[1] - 0.01), t.tool_az - 0.1)
            ok = self._tool_clear(ids[todo], cand, start[todo], start_yaw[todo])
            tool[todo[ok]] = cand[ok]
            todo = todo[~ok]
        if len(todo):
            raise RuntimeError("could not place the tool clear of the object")
        return tool

    def _place(self, ids, start, start_yaw, tool):
        """Write the physics state and per-episode bookkeeping for new episodes."""
        t, r = self.cfg.task, self.cfg.rand
        n = len(ids)
        # Calibration error in tool height (the IK believes the nominal height).
        self.tool_z_cmd[ids] = t.tool_height + self.param_rng.uniform(-1, 1, n) * r.tool_height_error * r.enabled
        q, err, _ = self.kin.solve(np.c_[tool, self.tool_z_cmd[ids]], self.kin.seed(tool), iterations=30)
        if err.max() > 1e-3:
            raise RuntimeError(f"reset IK failed: {err.max():.4f} m")
        # Physics state: time, qpos, qvel.
        qpos = np.zeros((n, self.idx.nq))
        qpos[:, self.idx.arm_qpos] = q
        qpos[:, self.idx.gripper_qpos] = self.cfg.scene.gripper_hold
        o = self.idx.obj_qpos
        qpos[:, o:o + 2] = start
        qpos[:, o + 2] = self.obj_height / 2
        qpos[:, o + 3] = np.cos(start_yaw / 2)
        qpos[:, o + 6] = np.sin(start_yaw / 2)
        self.state[ids] = 0.0
        self.state[ids, self.qpos_adr:self.qpos_adr + self.idx.nq] = qpos
        self.tip[ids], _ = self.kin.forward(q)
        self.qarm[ids] = q
        self.q_cmd[ids] = q
        self.prev_ctrl[ids, :5] = q
        self.prev_ctrl[ids, 5] = self.cfg.scene.gripper_hold
        self.cmd_xy[ids] = tool
        self.prev_action[ids] = 0
        self.step_count[ids] = 0
        self.first_success[ids] = -1
        self.delay[ids] = self.param_rng.integers(0, r.max_action_delay + 1, n) * r.enabled
        lo, hi = r.obs_latency_steps if r.enabled else (0, 0)
        self.latency[ids] = self.param_rng.integers(lo, hi + 1, n)
        pose = np.c_[start, start_yaw]
        self.pose_hist[ids] = pose[:, None, :]
        self.obs_pose[ids] = pose
        self.prev_obs_pose[ids] = pose
        self.prev_dist[ids] = self._task_distance(pose, ids)

    # --------------------------------------------------------- task hooks
    def _task_distance(self, pose, ids=None, dist=None):
        """Distance that drives the reward (base: keypoint distance to the goal)."""
        if dist is not None:
            return dist
        ids = np.arange(self.n) if ids is None else ids
        return self._keypoint_distance(pose, self.goal[ids], self.obj_id[ids])

    def _feature_goal(self, ids):
        """Goal pose shown in the observation (base: the goal; subclasses may show a subgoal)."""
        return self.goal[ids]

    def _extra_obs(self, ids, pose, tool_xy, actor=False):
        """Task-specific features appended to actor and critic observations (base: none)."""
        return np.zeros((len(ids), 0))

    def _episode_extras(self, ids):
        """Task-specific per-episode statistics (base: none)."""
        return {}

    def _after_physics(self):
        """Task-specific bookkeeping right after each physics step (base: none)."""

    def _reward_extra(self):
        """Task-specific reward terms for every world (base: none)."""
        return 0.0

    def _task_success(self, success):
        """Task-specific conditions added to "object at its goal pose" (base: none)."""
        return success

    def _guard_command(self, cmd_xy):
        """Task-specific limits on the commanded rod position (base: none)."""
        return cmd_xy

    # ------------------------------------------------------------- geometry
    def _symmetric_goal_yaw(self, yaw, goal_yaw, obj_id):
        """Goal yaw among the object's symmetric equivalents closest to ``yaw``."""
        s = self.obj.symmetry[obj_id]
        period = np.where(s > 0, TWO_PI / np.maximum(s, 1), TWO_PI)
        k = np.round((goal_yaw - yaw) / period)
        eff = goal_yaw - k * period
        return np.where(s == 0, yaw, eff)

    def _keypoints(self, pose, obj_id):
        kp = self.obj.keypoints[obj_id]                      # (n, K, 2)
        c, s = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
        return np.stack([c * kp[..., 0] - s * kp[..., 1], s * kp[..., 0] + c * kp[..., 1]], -1) + pose[:, None, :2]

    def _keypoint_distance(self, pose, goal, obj_id):
        g = goal.copy()
        g[:, 2] = self._symmetric_goal_yaw(pose[:, 2], goal[:, 2], obj_id)
        d = np.linalg.norm(self._keypoints(pose, obj_id) - self._keypoints(g, obj_id), axis=-1)
        return (d * self.obj.keypoint_weight[obj_id]).sum(-1)

    def _surface_distance(self, tool_xy, pose, obj_id):
        """Distance from the pusher surface to the object footprint (<= 0 means touching)."""
        rel = tool_xy - pose[:, :2]
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1)
        boxes = self.obj.boxes[obj_id]                       # (n, B, 4)
        q = np.abs(local[:, None, :] - boxes[..., :2]) - boxes[..., 2:]
        sdf = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
        valid = boxes[..., 2] > 0
        sdf = np.where(valid, sdf, np.inf).min(-1)
        disk = self.obj.disk_radius[obj_id]
        sdf = np.where(disk > 0, np.linalg.norm(local, axis=-1) - disk, sdf)
        return sdf - self.cfg.scene.pusher_radius

    def _object_pose(self):
        o = self.qpos_adr + self.idx.obj_qpos
        q = self.state[:, o + 3:o + 7]
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        up_z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)          # z component of the object z axis
        return np.c_[self.state[:, o:o + 2], yaw], self.state[:, o + 2], up_z

    def _update_tool(self):
        self.qarm = self.state[:, self.qpos_adr + self.idx.arm_qpos]
        self.tip, _ = self.kin.forward(self.qarm)

    def _project_workspace(self, xy):
        t = self.cfg.task
        rad = np.clip(np.linalg.norm(xy, axis=-1), *t.tool_r)
        ang = np.clip(np.arctan2(xy[:, 1], xy[:, 0]), -t.tool_az, t.tool_az)
        return np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)

    # ------------------------------------------------------------------- step
    def command(self, action):
        """The controller, shared with the real robot: action -> commanded tool position -> joint targets.

        Integrates the action into the commanded tool position (kept within ``tracking_limit`` of the
        measured tip and inside the reachable annulus) and runs one warm-started IK step.
        Returns (clipped action (n, 2), arm joint targets in MuJoCo radians (n, 5)).
        """
        t = self.cfg.task
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        tip = self.tip
        cmd = self.cmd_xy + a * t.max_speed * t.control_dt
        cmd = self._project_workspace(np.clip(cmd, tip[:, :2] - t.tracking_limit, tip[:, :2] + t.tracking_limit))
        self.cmd_xy = self._guard_command(cmd)
        self.cmd_deflection = np.linalg.norm(self.cmd_xy - cmd, axis=-1)
        q, _, _ = self.kin.solve(np.c_[self.cmd_xy, self.tool_z_cmd], self.q_cmd, iterations=1, report=False)
        self.q_cmd = q
        return a, q

    def step(self, action):
        t, r = self.cfg.task, self.cfg.rand
        a, q = self.command(action)
        new_ctrl = np.c_[q, np.full(self.n, self.cfg.scene.gripper_hold)]
        # Servo command latency: the previous command persists for ``delay`` substeps.
        sub = np.arange(self.nsub)[None, :, None]
        self.ctrl[:] = np.where(sub < self.delay[:, None, None], self.prev_ctrl[:, None, :], new_ctrl[:, None, :])
        self.prev_ctrl = new_ctrl
        self.pool.rollout(self.models, self.datas, self.state, self.ctrl, nstep=self.nsub,
                          initial_warmstart=self.warmstart, state=self.out, skip_checks=True)
        self.state[:] = self.out[:, -1]
        self.step_count += 1

        pose, height, up_z = self._object_pose()
        self._update_tool()
        self._after_physics()
        tip = self.tip
        dist = self._keypoint_distance(pose, self.goal, self.obj_id)
        task_dist = self._task_distance(pose, dist=dist)
        surf = self._surface_distance(tip[:, :2], pose, self.obj_id)
        pos_err = np.linalg.norm(pose[:, :2] - self.goal[:, :2], axis=-1)
        yaw_err = np.abs(wrap(pose[:, 2] - self._symmetric_goal_yaw(pose[:, 2], self.goal[:, 2], self.obj_id)))
        success = self._task_success((pos_err <= t.success_pos) & (yaw_err <= t.success_yaw))
        self.first_success = np.where((self.first_success < 0) & success, self.step_count, self.first_success)

        finite = np.isfinite(self.state).all(-1)
        tipped = (up_z < math.cos(math.radians(25))) | (height > self.obj_height / 2 + 0.01)
        outside = ~self._in_region(pose[:, :2], t.fail_r, t.fail_az)
        failed = tipped | outside | ~finite
        timeout = self.step_count >= self.max_steps

        reward = (t.w_coarse * np.exp(-task_dist / t.sigma_coarse) + t.w_fine * np.exp(-task_dist / t.sigma_fine)
                  + t.w_progress * (self.prev_dist - task_dist) / 0.01
                  + t.w_reach * np.exp(-np.maximum(surf, 0) / t.sigma_reach)
                  - t.w_action * (a ** 2).sum(-1) - t.w_action_rate * ((a - self.prev_action) ** 2).sum(-1)
                  - t.failure_penalty * failed + self._reward_extra())
        reward = np.where(finite, reward, -t.failure_penalty)
        self.prev_dist = task_dist
        self.prev_action = a

        # Camera model: delayed, noisy, sometimes missing object pose estimates.
        self.pose_hist = np.roll(self.pose_hist, 1, axis=1)
        self.pose_hist[:, 0] = np.where(finite[:, None], pose, self.pose_hist[:, 1])
        self.prev_obs_pose = self.obs_pose.copy()
        seen = self.pose_hist[np.arange(self.n), self.latency].copy()
        if r.enabled:
            seen[:, :2] += self.noise_rng.normal(0, r.obs_pos_noise, (self.n, 2))
            seen[:, 2] = wrap(seen[:, 2] + self.noise_rng.normal(0, r.obs_yaw_noise, self.n))
            keep = self.noise_rng.random(self.n) < r.obs_dropout
            seen[keep] = self.obs_pose[keep]
        self.obs_pose = seen

        done = failed | timeout
        info = {
            "time_outs": timeout & ~failed,
            "success": success, "failed": failed, "pos_err": pos_err, "yaw_err": yaw_err,
            "dist": dist, "tipped": tipped, "outside": outside, "nonfinite": ~finite,
        }
        ids = np.flatnonzero(done)
        if len(ids):
            # Terminal statistics and critic observation (for bootstrapping truncated episodes).
            info["terminal_ids"] = ids
            info["terminal_critic_obs"] = self._observe(subset=ids)["critic"]
            info["episode"] = {
                "success": success[ids], "failed": failed[ids], "timeout": timeout[ids] & ~failed[ids],
                "pos_err": pos_err[ids], "yaw_err": yaw_err[ids], "dist": dist[ids],
                "first_success": self.first_success[ids], "difficulty": self.difficulty_ep[ids],
                "tipped": tipped[ids], "outside": outside[ids], "nonfinite": ~finite[ids],
                "obj_id": self.obj_id[ids].copy(),
                **self._episode_extras(ids),
            }
            self.state[ids[~finite[ids]]] = 0.0
            self.reset(ids)
        self.obs = self._observe()
        return self.obs, reward, done, info

    # ------------------------------------------------------------ observations
    def _features(self, tool_xy, cmd_xy, pose, prev_pose, q, ids):
        """Shared actor/critic features in the robot base frame, roughly unit scale."""
        centre = np.array([0.2, 0.0])
        goal = self._feature_goal(ids)
        obj_id = self.obj_id[ids]
        g = goal.copy()
        g[:, 2] = self._symmetric_goal_yaw(pose[:, 2], goal[:, 2], obj_id)
        kp = self._keypoints(pose, obj_id)
        kp_goal = self._keypoints(g, obj_id)
        k = kp.shape[1]
        parts = [
            (tool_xy - centre) / 0.1,
            (cmd_xy - tool_xy) / 0.01,
            (pose[:, :2] - centre) / 0.1, np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3]),
            (goal[:, :2] - centre) / 0.1, np.cos(goal[:, 2:3]), np.sin(goal[:, 2:3]),
            ((kp - tool_xy[:, None, :]) / 0.05).reshape(-1, 2 * k),
            ((kp_goal - kp) / 0.05).reshape(-1, 2 * k),
            (pose[:, :2] - prev_pose[:, :2]) / 0.005, wrap(pose[:, 2:3] - prev_pose[:, 2:3]) / 0.1,
            self.prev_action[ids],
            q[:, :4],
        ]
        if len(self.cfg.task.objects) > 1:
            parts.append(np.eye(len(self.cfg.task.objects))[obj_id])
        return np.concatenate(parts, -1)

    def _observe(self, subset=None):
        ids = np.arange(self.n) if subset is None else subset
        r = self.cfg.rand
        tip, q = self.tip[ids], self.qarm[ids]
        tool_meas = tip[:, :2] + (self.noise_rng.normal(0, r.tool_noise, (len(ids), 2)) if r.enabled else 0)
        actor = self._features(tool_meas, self.cmd_xy[ids], self.obs_pose[ids], self.prev_obs_pose[ids], q, ids)
        actor = np.concatenate([actor, self._extra_obs(ids, self.obs_pose[ids], tool_meas, actor=True)], -1)
        pose, _, _ = self._object_pose()
        pose = pose[ids]
        true_feat = self._features(tip[:, :2], self.cmd_xy[ids], pose, self.pose_hist[ids, 1], q, ids)
        qv = self.state[ids, self.qvel_adr:]
        o = self.idx.obj_qvel
        surf = self._surface_distance(tip[:, :2], pose, self.obj_id[ids])
        dist = self._keypoint_distance(pose, self.goal[ids], self.obj_id[ids])
        critic = np.concatenate([
            true_feat,
            qv[:, o:o + 2] / 0.05, qv[:, o + 5:o + 6] / 0.5,      # object planar velocity, yaw rate
            self.params[ids] - np.array([0.35, 0.3, 1.0, 1.0]),
            (self.tool_z_cmd[ids, None] - self.cfg.task.tool_height) / 0.002,
            self.latency[ids, None] / 2.0, self.delay[ids, None] / 5.0,
            np.clip(surf, -0.01, 0.05)[:, None] / 0.02, dist[:, None] / 0.05,
            self._extra_obs(ids, pose, tip[:, :2]),
        ], -1)
        return {"actor": actor.astype(np.float32), "critic": critic.astype(np.float32)}

    # ------------------------------------------------------------- utilities
    def set_difficulty(self, c):
        self.difficulty = float(np.clip(c, 0.0, 1.0))

    def reset_worlds(self, ids):
        """Start new episodes in the given worlds and refresh observations."""
        self.reset(np.asarray(ids))
        self.obs = self._observe()
        return self.obs

    def displace_object(self, i, shift, turn):
        """Teleport world i's object by (dx, dy) and rotate by ``turn`` (disturbance tests)."""
        o = self.qpos_adr + self.idx.obj_qpos
        pose = self.object_pose()[i]
        new = np.r_[pose[:2] + shift, wrap(pose[2] + turn)]
        self.state[i, o:o + 2] = new[:2]
        self.state[i, o + 3:o + 7] = [math.cos(new[2] / 2), 0, 0, math.sin(new[2] / 2)]
        self.state[i, self.qvel_adr:] = 0.0
        self.prev_dist[i] = self._task_distance(new[None], np.array([i]))[0]
        # The camera history is left alone: the policy sees the jump only after its latency.

    def set_goal(self, goal, i=0):
        """Deployment: set world i's goal pose (x, y, yaw) in the robot base frame."""
        self.goal[i] = goal

    def sync_measurements(self, qarm, pose, first=False):
        """Deployment: replace the simulated state with measurements and return the actor observation.

        ``qarm`` (n, 5): measured arm joints in MuJoCo radians. ``pose`` (n, 3) or None: the camera's
        object pose (x, y, yaw) in the robot base frame; None keeps the last one (a dropped frame).
        ``first``: start of a trial (command, history and previous action are initialised).
        This runs exactly the observation code used in training, with randomisation off.
        """
        qarm = np.atleast_2d(np.asarray(qarm, dtype=np.float64))
        self.state[:, self.qpos_adr + self.idx.arm_qpos] = qarm
        self.qarm = qarm
        self.tip, _ = self.kin.forward(qarm)
        if pose is not None:
            pose = np.atleast_2d(np.asarray(pose, dtype=np.float64))
            o = self.qpos_adr + self.idx.obj_qpos
            self.state[:, o:o + 2] = pose[:, :2]
            self.state[:, o + 2] = self.obj_height / 2
            self.state[:, o + 3:o + 7] = np.stack([np.cos(pose[:, 2] / 2), 0 * pose[:, 2], 0 * pose[:, 2],
                                                   np.sin(pose[:, 2] / 2)], -1)
            self.prev_obs_pose = pose.copy() if first else self.obs_pose.copy()
            self.obs_pose = pose.copy()
            self.pose_hist = np.roll(self.pose_hist, 1, axis=1)
            self.pose_hist[:, 0] = pose
        if first:
            self.pose_hist[:] = self.obs_pose[:, None, :]
            self.cmd_xy = self.tip[:, :2].copy()
            self.q_cmd = qarm.copy()
            self.prev_action[:] = 0.0
            self.step_count[:] = 0
        self.prev_dist = self._task_distance(self.object_pose())    # also advances task stages (gate)
        self.obs = self._observe()
        return self.obs["actor"]

    def object_pose(self):
        return self._object_pose()[0]

    def tool_position(self):
        return self.tip.copy()
