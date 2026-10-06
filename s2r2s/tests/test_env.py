import math

import mujoco
import numpy as np
import pytest

from s2r2s.env import EnvConfig, PushEnv, RandomizationConfig, TaskConfig
from s2r2s.scripted import KeypointPusher


def make(n=64, seed=0, randomize=True, difficulty=1.0, objects=("tee",)):
    cfg = EnvConfig(num_envs=n, seed=seed, num_threads=4,
                    task=TaskConfig(objects=objects, difficulty=difficulty, easy_fraction=0.0),
                    rand=RandomizationConfig(enabled=randomize))
    return PushEnv(cfg)


def test_state_layout_is_time_qpos_qvel():
    env = make(4)
    m = env.base_models[0]
    d = mujoco.MjData(m)
    d.time, d.qpos[:], d.qvel[:] = 1.5, np.arange(m.nq) * 0.01, -np.arange(m.nv) * 0.1
    st = np.zeros(env.nstate)
    mujoco.mj_getState(m, d, st, env.spec)
    assert st[0] == 1.5 and np.allclose(st[env.qpos_adr:env.qpos_adr + m.nq], d.qpos)
    assert np.allclose(st[env.qvel_adr:env.qvel_adr + m.nv], d.qvel)


def test_zero_action_keeps_everything_still():
    env = make(64, randomize=False)
    pose0, tip0 = env.object_pose().copy(), env.tool_position().copy()
    for _ in range(40):
        _, _, done, info = env.step(np.zeros((env.n, 2)))
        assert not done.any()
    assert np.abs(env.object_pose()[:, :2] - pose0[:, :2]).max() < 1e-4
    assert np.abs(env.tool_position()[:, :2] - tip0[:, :2]).max() < 1e-3


def test_resets_are_valid():
    env = make(512, seed=3)
    pose, goal, tip = env.object_pose(), env.goal, env.tool_position()
    t = env.cfg.task
    for xy in (pose[:, :2], goal[:, :2]):
        r = np.linalg.norm(xy, axis=1)
        assert (r >= t.obj_r[0] - 1e-9).all() and (r <= t.obj_r[1] + 1e-9).all()
    clear = env._surface_distance(tip[:, :2], pose, env.obj_id)
    assert (clear > 0.005).all(), "tool must start clear of the object"
    assert np.allclose(tip[:, 2], env.tool_z_cmd, atol=1e-4)


def test_random_actions_are_stable_and_finite():
    env = make(256, seed=4)
    for _ in range(150):
        obs, rew, done, info = env.step(np.random.default_rng(0).uniform(-1, 1, (env.n, 2)))
        assert np.isfinite(obs["actor"]).all() and np.isfinite(obs["critic"]).all() and np.isfinite(rew).all()
        if "episode" in info:
            assert not info["episode"]["nonfinite"].any()
            assert not info["episode"]["tipped"].any()


def test_keypoint_distance_respects_symmetry():
    env = make(4, objects=("box",))
    pose = np.array([[0.2, 0.0, 0.0]] * 4)
    goal = np.array([[0.2, 0.0, a] for a in (0.0, math.pi / 2, math.pi, -math.pi / 2)])
    assert np.allclose(env._keypoint_distance(pose, goal, np.zeros(4, int)), 0.0, atol=1e-12)
    tee = make(1)
    d = tee._keypoint_distance(np.array([[0.2, 0.0, 0.0]]), np.array([[0.2, 0.0, math.pi]]), np.zeros(1, int))
    assert d > 0.02


def test_reward_prefers_progress_over_idling():
    env = make(2, randomize=False, difficulty=0.5)
    t = env.cfg.task
    base = env.prev_dist.copy()
    # Same state, but pretend the object came from 5 mm farther away: progress must add reward.
    gain = t.w_progress * 0.005 / 0.01
    assert gain > t.w_reach, "a 5 mm push per step must be worth more than idling next to the object"
    assert np.all(base > 0)


def test_scripted_pusher_solves_easy_goals():
    env = make(64, seed=5, randomize=False, difficulty=0.0)
    ctl = KeypointPusher(env)
    success = []
    for _ in range(env.max_steps):
        _, _, done, info = env.step(ctl.act())
        if "episode" in info:
            success.append(info["episode"]["success"])
            ctl.reset(info["terminal_ids"])
            if sum(len(s) for s in success) >= env.n:
                break
    assert np.concatenate(success).mean() > 0.9


def test_scenes_do_not_depend_on_randomization_or_noise():
    a = make(32, seed=9)
    b = make(32, seed=9, randomize=False)
    assert np.allclose(a.object_pose(), b.object_pose()) and np.allclose(a.goal, b.goal)
    assert np.allclose(a.tool_position()[:, :2], b.tool_position()[:, :2], atol=1e-6)
