import numpy as np

from s2r2s.clutter import ClutterEnv, rect_gap
from s2r2s.env import EnvConfig, RandomizationConfig, TaskConfig
from s2r2s.objects import OBJECTS


def make(n=64, seed=0, randomize=True, **task):
    task = {"clutter": "box", "difficulty": 1.0, "easy_fraction": 0.0, **task}
    return ClutterEnv(EnvConfig(num_envs=n, seed=seed, num_threads=4, task=TaskConfig(**task),
                                rand=RandomizationConfig(enabled=randomize)))


def _outline(name, pose, step=0.0005):
    """Densely sampled footprint boundary in the world (brute-force reference)."""
    poly = OBJECTS[name].outline()
    pts = np.concatenate([a + (b - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step)), endpoint=False)[:, None]
                          for a, b in zip(poly, np.roll(poly, -1, 0))])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def _inside(points, name, pose):
    rel = points - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    lx, ly = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    shape = OBJECTS[name]
    if shape.disk_radius > 0:
        return np.hypot(lx, ly) < shape.disk_radius
    return np.any([(np.abs(lx - cx) < hx) & (np.abs(ly - cy) < hy) for cx, cy, hx, hy in shape.centred_boxes], axis=0)


def test_clearance_is_exact():
    o = np.ones(1)
    bar = lambda x, y, a, hx, hy: (x * o, y * o, np.cos(a) * o, np.sin(a) * o, hx * o, hy * o)
    # Two bars crossing like a plus sign: no corner of either lies inside the other.
    assert np.isclose(rect_gap(bar(0, 0, 0, 0.03, 0.005), bar(0, 0, np.pi / 2, 0.03, 0.005)), -0.035)
    assert np.isclose(rect_gap(bar(0, 0, 0, 0.01, 0.01), bar(0.03, 0.03, 0, 0.01, 0.01)), np.hypot(0.01, 0.01))
    rng = np.random.default_rng(0)
    for clutter in ("box", "disk"):
        env = make(4, clutter=clutter)
        n = 60
        a = np.c_[rng.uniform(0.15, 0.25, n), rng.uniform(-0.05, 0.05, n), rng.uniform(-np.pi, np.pi, n)]
        b = np.c_[rng.uniform(0.15, 0.25, n), rng.uniform(-0.05, 0.05, n), rng.uniform(-np.pi, np.pi, n)]
        gap = env._gap_to_object(a, b, np.zeros(n, dtype=int))
        for i in range(n):
            pa, pb = _outline(clutter, a[i]), _outline("tee", b[i])
            if _inside(pa, "tee", b[i]).any() or _inside(pb, clutter, a[i]).any():
                assert gap[i] < 0
            else:
                assert abs(gap[i] - np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1)).min()) < 2e-4


def test_the_guarded_rod_fits_between_the_block_and_the_path():
    env = make(512, seed=1)
    t = env.cfg.task
    assert np.isclose(t.clutter_min_gap, 2 * env.cfg.scene.pusher_radius + t.rod_guard)    # the guarded rod fits
    assert (env._gap_to_object(env.clutter_home, env.object_pose(), env.obj_id) > t.clutter_min_gap - 1e-9).all()
    assert (env._gap_to_object(env.clutter_home, env.goal, env.obj_id) > t.clutter_min_gap - 1e-9).all()
    gap = env._sweep_gap(env.clutter_home, env.object_pose(), env.goal, env.obj_id)
    assert (gap > t.clutter_min_gap - 1e-9).all()                      # the rod always fits between ...
    assert 0.6 < env.in_way.mean() < 0.9                               # ... though most blocks sit close beside the path
    assert np.allclose(env.in_way, gap < t.clutter_near)
    assert (env._block_clearance(env.tool_position()[:, :2], env.clutter_home) > env.cfg.scene.pusher_radius).all()
    assert np.allclose(env.clutter_pose(), env.clutter_home)
    # Scenes depend on the seed only, so evaluations with randomisation on and off are paired.
    assert np.allclose(make(512, seed=1, randomize=False).clutter_home, env.clutter_home)


def test_resting_block_stays_put():
    env = make(64, seed=2, randomize=False)
    for _ in range(40):
        env.step(np.zeros((env.n, 2)))
    pos, yaw = env.disturbance()
    assert pos.max() < 1e-4 and yaw.max() < 1e-3


def test_moving_the_block_costs_reward_and_success():
    env = make(4, seed=3, randomize=False, w_near=0.0)
    o = env.qpos_adr + env.idx.clutter_qpos
    _, r0, _, _ = env.step(np.zeros((env.n, 2)))
    env.state[:, o] += 0.02                     # someone slides the block 2 cm
    _, r1, _, info = env.step(np.zeros((env.n, 2)))
    expected = env.cfg.task.w_disturb * 2.0 + env.cfg.task.w_displaced
    assert np.allclose(r0 - r1, expected, atol=0.05)
    # Even with the object exactly at its goal, a displaced block means no success.
    env.goal[:] = np.c_[env.object_pose()]
    assert not env._task_success(np.ones(env.n, dtype=bool)).any()


def test_coming_close_to_the_block_costs_reward():
    env = make(8, seed=4, randomize=False)
    t, rod = env.cfg.task, env.cfg.scene.pusher_radius
    tip = env.tool_position()[:, :2]
    o = env.qpos_adr + env.idx.obj_qpos
    env.state[:, o:o + 2] = tip + np.array([0.0, 0.12])          # the T well out of the way
    o = env.qpos_adr + env.idx.clutter_qpos

    def block_at(gap):
        """Put the block (and its home, so it is not "disturbed") ``gap`` from the rod's surface."""
        home = np.c_[tip + np.array([rod + gap + 0.020, 0.0]), np.zeros(env.n)]
        env.clutter_home[:] = home
        env.state[:, o:o + 2] = home[:, :2]
        env.state[:, o + 3:o + 7] = [1, 0, 0, 0]
        env.prev_disturb[:] = 0.0
        return home

    block_at(2 * t.near_margin)
    assert np.allclose(env._reward_extra(), 0.0)                  # outside the margin: free
    home = block_at(0.001)
    assert np.allclose(env._reward_extra(), -t.w_near * (1 - 0.001 / t.near_margin))
    assert np.allclose(env._block_nearest(tip, home), tip + np.array([rod + 0.001, 0.0]))


def test_rod_guard_slides_the_rod_along_the_block():
    moved = {}
    for guard in (0.0, 0.003):
        # Wide failure limits: the T is moved far away below, and must not end (and reset) the episode.
        env = make(16, seed=5, randomize=False, rod_guard=guard, fail_r=(0.0, 1.0), fail_az=np.pi)
        tip = env.tool_position()[:, :2]
        o = env.qpos_adr + env.idx.obj_qpos
        env.state[:, o:o + 2] = tip + 2.0 * (tip - env.clutter_home[:, :2])      # the T well behind the rod
        for _ in range(40):                       # drive straight at the block's centre for 2 s
            aim = env.clutter_pose()[:, :2] - env.tool_position()[:, :2]
            env.step(aim / np.linalg.norm(aim, axis=1, keepdims=True))
        moved[guard] = env.disturbance()[0]
        if guard:
            need = env.cfg.scene.pusher_radius + guard
            assert (env._block_clearance(env.cmd_xy, env.clutter_obs) > need - 1e-6).all()
            # Leaning on the guard is what the w_guard cost sees: about a full step removed per tick.
            full_step = env.cfg.task.max_speed * env.cfg.task.control_dt
            assert np.median(env.cmd_deflection) > 0.5 * full_step
    assert np.median(moved[0.0]) > 0.02                # unguarded, the rod shoves the block away ...
    assert moved[0.003].max() < 0.001                  # ... guarded, it never moves it
