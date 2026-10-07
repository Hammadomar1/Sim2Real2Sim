import numpy as np
import torch

from s2r2s.env import EnvConfig, TaskConfig
from s2r2s.level2 import BOX, TEE, Level2Env, Skill, run_level2
from s2r2s.ppo import ActorCritic, PPOConfig
from s2r2s.tasks import make_env


def world(n=256, seed=0, **task):
    task = {"clutter": "box", "episode_seconds": 40.0, "easy_fraction": 0.0, **task}
    return Level2Env(EnvConfig(num_envs=n, seed=seed, num_threads=4, task=TaskConfig(**task)))


def untrained(path, objects, clutter):
    """A random clutter policy with the layout of (objects pushed, clutter kept), saved as a checkpoint."""
    cfg = EnvConfig(num_envs=1, num_threads=1, task=TaskConfig(objects=objects, clutter=clutter))
    env = make_env(cfg)
    dims = (env.num_obs, env.num_critic_obs, env.num_actions)
    torch.manual_seed(0)
    torch.save({"ppo": {"policy": ActorCritic(*dims, PPOConfig()).state_dict()}, "env_cfg": cfg.to_dict(),
                "ppo_cfg": PPOConfig().to_dict(), "iteration": 0, "samples": 0, "dims": dims}, path)
    return str(path)


def test_scenes_can_be_done_in_the_chosen_order():
    w = world()
    t = w.cfg.task
    tee, box = w.object_pose(), w.clutter_pose()
    clear = w.order_clearances(tee, w.goal, box, w.box_goal)
    chosen = clear[np.arange(w.n), w.order]
    assert (chosen > t.clutter_min_gap).all()               # both pushes keep the skills' gap to the block that stays
    assert np.allclose(clear.max(1), w.plan_clearance) and (w.plan_len == 2).all()
    goals = np.stack([w.goal, w.box_goal], 1)
    assert np.allclose(w.plan_goal[:, 1], goals[np.arange(w.n), 1 - w.order])     # the second push: the other block
    assert w._in_region(w.box_goal[:, :2], t.obj_r, t.obj_az).all() and w._in_region(w.goal[:, :2], t.obj_r, t.obj_az).all()
    assert 0.2 < np.mean(w.order == TEE) < 0.8              # the sequencer uses both orders


def test_sweep_ends_match_the_pairwise_gaps():
    w = world(64, seed=1)
    tee, box = w.object_pose(), w.clutter_pose()
    zeros = np.zeros(w.n, dtype=np.int64)
    ends = np.minimum(w._gap_to_object(box, tee, zeros), w._gap_to_object(box, w.goal, zeros))
    assert np.allclose(w._sweep(TEE, tee, w.goal, box, steps=2), ends)
    ends = np.minimum(w._gap_to_object(box, tee, zeros), w._gap_to_object(w.box_goal, tee, zeros))
    assert np.allclose(w._sweep(BOX, box, w.box_goal, tee, steps=2), ends)
    assert (w._sweep(BOX, box, w.box_goal, tee) <= w._sweep(BOX, box, w.box_goal, tee, steps=2) + 1e-12).all()


def test_skills_take_turns_and_guard_the_block_that_stays(tmp_path):
    tee = Skill(untrained(tmp_path / "tee.pt", ("tee",), "box"), 8, "cpu")
    box = Skill(untrained(tmp_path / "box.pt", ("box",), "tee"), 8, "cpu")
    cfg = EnvConfig.from_dict(torch.load(tmp_path / "tee.pt", weights_only=False)["env_cfg"])
    cfg.task.episode_seconds = 3.0
    r = run_level2(cfg, tee, box, 8, seed=4)
    assert r["episodes"] == 8 and 0.0 <= r["success"] <= 1.0
    rows = r["per_episode"]
    assert all(row["order_valid"] for row in rows)           # the sequencer only picks orders that leave the gap
    # Each skill's twin guards the other block: its clutter is the block that must stay put.
    w = world(8, seed=4)
    first_box = np.flatnonzero(w.order == BOX)
    if len(first_box):
        box.start(first_box, w.qarm[first_box], w.clutter_pose()[first_box], w.object_pose()[first_box],
                  w.box_goal[first_box], w.tool_z_cmd[first_box])
        assert np.allclose(box.twin.clutter_home[first_box], w.object_pose()[first_box])
        assert np.allclose(box.twin.cmd_xy[first_box], w.tool_position()[first_box, :2])


def test_parking_plans_keep_every_push_clear():
    w = world(1024, seed=2, episode_seconds=60.0)
    w.parking = True
    w.reset(np.arange(w.n))
    t = w.cfg.task
    three = np.flatnonzero(w.plan_len == 3)
    assert 0 < len(three) < 0.1 * w.n and set(np.unique(w.plan_len)) <= {2, 3}   # parking is the exception
    a = w.plan_who[three, 0]
    assert (w.plan_who[three, 2] == a).all() and (w.plan_who[three, 1] == 1 - a).all()
    tee, box = w.object_pose()[three], w.clutter_pose()[three]
    starts, goals = np.stack([tee, box], 1), np.stack([w.goal[three], w.box_goal[three]], 1)
    r = np.arange(len(three))
    a_start, a_goal, b_start, b_goal = starts[r, a], goals[r, a], starts[r, 1 - a], goals[r, 1 - a]
    spot = w.plan_goal[three, 0]
    for k in range(len(three)):                                  # each of the three pushes keeps the 15 mm gap
        A, B, one = a[k], 1 - a[k], slice(k, k + 1)
        assert w._sweep(A, a_start[one], spot[one], b_start[one])[0] > t.clutter_min_gap
        assert w._sweep(B, b_start[one], b_goal[one], spot[one])[0] > t.clutter_min_gap
        assert w._sweep(A, spot[one], a_goal[one], b_goal[one])[0] > t.clutter_min_gap
    assert (np.linalg.norm(spot[:, :2] - a_start[:, :2], axis=1) <= t.goal_shift[1]).all()
    assert w._in_region(spot[:, :2], t.obj_r, t.obj_az).all()
