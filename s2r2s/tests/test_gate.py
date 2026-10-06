import math

import mujoco
import numpy as np
import torch

from s2r2s.env import EnvConfig, RandomizationConfig, TaskConfig
from s2r2s.gate import GateEnv
from s2r2s.ppo import ActorCritic, PPOConfig
from s2r2s.scene import SceneConfig, build_model, set_gate
from s2r2s.train import warm_start


def make(n=64, seed=0, difficulty=1.0, randomize=True):
    cfg = EnvConfig(num_envs=n, seed=seed, num_threads=4,
                    task=TaskConfig(gate=True, episode_seconds=30.0, difficulty=difficulty, easy_fraction=0.0),
                    rand=RandomizationConfig(enabled=randomize))
    return GateEnv(cfg)


def test_walls_collide_where_the_gate_is_placed():
    m = build_model(SceneConfig(timestep=0.005, gate=True), visual=False)
    d = mujoco.MjData(m)
    q = m.jnt_qposadr[m.joint("object_free").id]

    def wall_contact(xy):
        d.qpos[:] = 0
        d.qpos[q:q + 3] = [*xy, 0.01]
        d.qpos[q + 3] = 1
        mujoco.mj_forward(m, d)
        return any("gate" in m.geom(c.geom1).name + m.geom(c.geom2).name for c in d.contact[:d.ncon])

    set_gate(m, (0.2, 0.0), 0.0, 0.066)
    assert not wall_contact((0.2, 0.0)) and wall_contact((0.24, 0.0))
    set_gate(m, (0.2, 0.05), 0.0, 0.066)
    assert not wall_contact((0.24, 0.0)) and wall_contact((0.24, 0.05))


def test_scenes_start_before_and_end_beyond_the_wall():
    env = make(256, seed=1)
    ids = np.arange(env.n)
    assert (env.passage_progress(env.object_pose()[:, :2], ids) < -0.045).all()
    assert (env.passage_progress(env.goal[:, :2], ids) > 0.045).all()
    assert (env.wall_gap(env.tip[:, :2], ids) > 0.005).all()
    assert ((env.gate_width >= 0.064 - 1e-9) & (env.gate_width <= 0.070 + 1e-9)).all()
    assert (env.stage == 0).all()


def test_the_object_cannot_pass_through_a_wall():
    env = make(16, seed=2, randomize=False)
    ids = np.arange(env.n)
    along, passage = env._axes(ids)
    # Put the object right in front of the inner wall (well off the opening) and push it into the wall.
    for i in ids:
        target = env.gate_centre[i] - along[i] * 0.06 - passage[i] * 0.06
        env.displace_object(i, target - env.object_pose()[i, :2], 0.0)
    for _ in range(80):
        tip = env.tip[:, :2]
        behind = env.object_pose()[:, :2] - passage * 0.05
        a = np.clip((behind - tip) / 0.004, -1, 1)
        near = np.linalg.norm(behind - tip, axis=1) < 0.006
        a[near] = passage[near]
        env.step(a)
    assert (env.passage_progress(env.object_pose()[:, :2], ids) < 0.0).all()


def test_stage_advances_after_the_wall_and_falls_back():
    env = make(4, seed=3, randomize=False)
    ids = np.arange(env.n)
    _, passage = env._axes(ids)
    goal_side = env.gate_centre + passage * 0.06
    for i in ids:
        env.displace_object(i, goal_side[i] - env.object_pose()[i, :2], 0.0)
    assert (env.stage == 2).all()
    start_side = env.gate_centre - passage * 0.01
    for i in ids:
        env.displace_object(i, start_side[i] - env.object_pose()[i, :2], 0.0)
    assert (env.stage == 1).all()


def test_warm_start_reproduces_the_old_policy_on_old_inputs():
    old = ActorCritic(33, 45, 2, PPOConfig())
    new = ActorCritic(50, 62, 2, PPOConfig())
    warm_start(new, old.state_dict(), init_std=0.25)
    x, xc = torch.randn(64, 50), torch.randn(64, 62)
    assert torch.allclose(new.act_deterministic(x), old.act_deterministic(x[:, :33]), atol=1e-6)
    assert torch.allclose(new.value(xc), old.value(xc[:, :45]), atol=1e-6)
    assert math.isclose(float(new.log_std.detach().exp()[0]), 0.25, rel_tol=1e-6)
