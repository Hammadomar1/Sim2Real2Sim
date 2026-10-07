import math

import numpy as np

from s2r2s.dock import DockEnv, compose, corner_offset
from s2r2s.env import EnvConfig, TaskConfig
from s2r2s.level2 import DOCK
from s2r2s.level3 import PREDOCK, Level3Env


def dock_env(n=64, seed=0):
    return DockEnv(EnvConfig(num_envs=n, seed=seed, num_threads=4,
                             task=TaskConfig(objects=("box",), clutter="tee", dock=True, episode_seconds=10.0)))


def put_box(env, pose):
    o = env.qpos_adr + env.idx.obj_qpos
    env.state[:, o:o + 2] = pose[:, :2]
    env.state[:, o + 3:o + 7] = np.c_[np.cos(pose[:, 2] / 2), np.zeros((len(pose), 2)), np.sin(pose[:, 2] / 2)]


def test_docked_box_touches_the_stem_and_the_bar():
    off = corner_offset("tee", "box", np.array([1.0, -1.0]))
    assert np.allclose(off[0], [-off[1, 0], off[1, 1], 0.0])         # the two corners mirror each other
    env = dock_env()
    zeros = np.zeros(env.n, dtype=np.int64)
    assert np.allclose(env._gap_to_object(env.clutter_home, env.goal, zeros), 0.0, atol=1e-9)
    # It touches both faces: moved 1 mm away from one, it still touches the other; moved away from both, it is free.
    off = corner_offset("tee", "box", env.side)
    for dx, dy, touching in ((0.001, 0.0, True), (0.0, -0.001, True), (0.001, -0.001, False)):
        moved = compose(env.clutter_home, off + np.c_[env.side * dx, np.full(env.n, dy), np.zeros(env.n)])
        gap = env._gap_to_object(env.clutter_home, moved, zeros)
        assert np.allclose(gap, 0.0, atol=1e-9) if touching else (gap > 0.0009).all()


def test_docking_scenes_and_success():
    env = dock_env(256, seed=1)
    start = env.object_pose()
    dist = np.linalg.norm(start[:, :2] - env.goal[:, :2], axis=1)
    assert (dist > 0.011).all() and (dist < 0.046).all()
    zeros = np.zeros(env.n, dtype=np.int64)
    assert (env._gap_to_object(env.clutter_home, start, zeros) > 0.003).all()
    assert not env._task_success(np.ones(env.n, dtype=bool)).any()
    put_box(env, env.goal)                                              # seated in the corner: docked
    assert env._task_success(np.ones(env.n, dtype=bool)).all()
    o = env.qpos_adr + env.idx.clutter_qpos
    env.state[:, o] += 0.006                                            # ... unless the T was pushed 6 mm
    assert not env._task_success(np.ones(env.n, dtype=bool)).any()


def test_the_dock_target_stays_but_docked_means_touching():
    env = dock_env(8, seed=2)
    before = env.goal.copy()
    o = env.qpos_adr + env.idx.clutter_qpos
    env.state[:, o:o + 2] += 0.002                    # the T nudged 2 mm (diagonally)
    env.step(np.zeros((env.n, 2)))
    assert np.allclose(env.goal, before)              # the target does not chase the T
    put_box(env, env.docked_pose(env.clutter_pose(), env.side))
    assert env._task_success(np.ones(env.n, dtype=bool)).all()      # touching the T where it is: docked


def test_assembly_plans_end_with_docking():
    cfg = EnvConfig(num_envs=128, seed=3, num_threads=4, task=TaskConfig(clutter="box", episode_seconds=80.0, easy_fraction=0.0))
    w = Level3Env(cfg, parking=True)
    r = np.arange(w.n)
    assert (w.plan_who[r, w.plan_len - 1] == DOCK).all() and set(np.unique(w.plan_len)) <= {3, 4}
    zeros = np.zeros(w.n, dtype=np.int64)
    gap = w._gap_to_object(w.box_goal, w.goal, zeros)                  # the pre-dock spot: 25 mm out along the diagonal
    assert np.allclose(gap, PREDOCK / math.sqrt(2), atol=1e-6) and (gap > w.cfg.task.clutter_min_gap).all()
    last = w.plan_len - 1
    assert np.allclose(w.push_target(r, last), w.docked(w.obs_pose, w.side))   # follows the camera's T


def test_docking_can_start_from_recorded_level3_states(tmp_path):
    src = dock_env(16, seed=7)                         # stand-in for starts recorded in Level 3
    bank = tmp_path / "bank.npz"
    np.savez(bank, tee=src.clutter_home, box=src.object_pose(), rod=src.tool_position()[:, :2], side=src.side)
    env = DockEnv(EnvConfig(num_envs=64, seed=8, num_threads=4, task=TaskConfig(
        objects=("box",), clutter="tee", dock=True, episode_seconds=10.0, dock_start_bank=str(bank), dock_bank_share=1.0)))
    tee, box, rod = env.clutter_home, env.object_pose(), env.tool_position()[:, :2]
    match = [np.flatnonzero(np.all(np.isclose(src.clutter_home, t), axis=1)) for t in tee]
    assert all(len(m) == 1 for m in match)              # every scene is one of the recorded starts ...
    j = np.array([m[0] for m in match])
    assert np.allclose(box, src.object_pose()[j]) and np.allclose(rod, src.tool_position()[j, :2], atol=2e-4)
    assert np.allclose(env.goal, env.docked_pose(tee, src.side[j]))   # ... with its corner as the target
