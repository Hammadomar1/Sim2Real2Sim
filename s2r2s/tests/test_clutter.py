import numpy as np

from s2r2s.clutter import ClutterEnv
from s2r2s.env import EnvConfig, RandomizationConfig, TaskConfig


def make(n=64, seed=0, randomize=True):
    return ClutterEnv(EnvConfig(num_envs=n, seed=seed, num_threads=4,
                                task=TaskConfig(clutter="box", difficulty=1.0, easy_fraction=0.0),
                                rand=RandomizationConfig(enabled=randomize)))


def test_block_never_overlaps_the_object_start_or_goal():
    env = make(512, seed=1)
    assert (env._gap_to_object(env.clutter_home, env.object_pose(), env.obj_id) > 0.0039).all()
    assert (env._gap_to_object(env.clutter_home, env.goal, env.obj_id) > 0.0039).all()
    assert 0.5 < env.in_way.mean() < 0.8                     # most blocks sit right beside the path
    gap = env._sweep_gap(env.clutter_home, env.object_pose(), env.goal, env.obj_id)
    assert (gap > 0.0039).all()                                # ... but never block it
    assert np.allclose(env.clutter_pose(), env.clutter_home)


def test_resting_block_stays_put():
    env = make(64, seed=2, randomize=False)
    for _ in range(40):
        env.step(np.zeros((env.n, 2)))
    pos, yaw = env.disturbance()
    assert pos.max() < 1e-4 and yaw.max() < 1e-3


def test_moving_the_block_costs_reward_and_success():
    env = make(4, seed=3, randomize=False)
    o = env.qpos_adr + env.idx.clutter_qpos
    _, r0, _, _ = env.step(np.zeros((env.n, 2)))
    env.state[:, o] += 0.02                     # someone slides the block 2 cm
    _, r1, _, info = env.step(np.zeros((env.n, 2)))
    expected = env.cfg.task.w_disturb * 2.0 + env.cfg.task.w_displaced
    assert np.allclose(r0 - r1, expected, atol=0.05)
    # Even with the object exactly at its goal, a displaced block means no success.
    env.goal[:] = np.c_[env.object_pose()]
    assert not env._task_success(np.ones(env.n, dtype=bool)).any()
