import math

import mujoco
import numpy as np
import pytest

from s2r2s.kinematics import PusherKinematics
from s2r2s.scene import MENAGERIE_SO101, SceneIndex, build_model


@pytest.fixture(scope="module")
def model():
    return build_model(visual=False)


def test_stripped_model_keeps_menagerie_dynamics(model):
    orig = mujoco.MjModel.from_xml_path(str(MENAGERIE_SO101))
    visual = build_model(visual=True)
    for i in range(1, orig.nbody):
        name = orig.body(i).name
        for m in (model, visual):
            assert np.allclose(orig.body_mass[i], m.body(name).mass)
            assert np.allclose(orig.body_inertia[i], m.body(name).inertia)
    assert np.allclose(orig.actuator_gainprm[:, 0], model.actuator_gainprm[:6, 0])
    assert model.nq == visual.nq and model.nv == visual.nv


def test_fk_and_jacobian_match_mujoco(model):
    kin, idx, d = PusherKinematics(model), SceneIndex(model), mujoco.MjData(model)
    q = np.random.default_rng(0).uniform(kin.lower * 0.8, kin.upper * 0.8, size=(50, 5))
    tip, axis, jp, jr = kin.forward(q, with_jacobian=True)
    for k in range(len(q)):
        d.qpos[idx.arm_qpos] = q[k]
        mujoco.mj_forward(model, d)
        Jp, Jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d, Jp, Jr, idx.tool_site)
        assert np.allclose(d.site_xpos[idx.tool_site], tip[k], atol=1e-12)
        assert np.allclose(Jp[:, idx.arm_qvel], jp[k], atol=1e-9)
        assert np.allclose(Jr[:, idx.arm_qvel], jr[k], atol=1e-9)


def test_ik_covers_the_pushing_workspace(model):
    kin = PusherKinematics(model)
    rng = np.random.default_rng(1)
    n = 20000
    r = rng.uniform(0.12, 0.275, n)
    a = rng.uniform(-math.radians(55), math.radians(55), n)
    tgt = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(0.003, 0.007, n)], -1)
    q, err, axis_z = kin.solve(tgt, kin.seed(tgt[:, :2]), iterations=30)
    assert (err < 1e-4).mean() > 0.999
    assert (axis_z > 0.9999).mean() > 0.999


def test_single_ik_iteration_tracks_small_moves(model):
    kin = PusherKinematics(model)
    rng = np.random.default_rng(2)
    n = 500
    r, a = rng.uniform(0.15, 0.25, n), rng.uniform(-0.6, 0.6, n)
    tgt = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, 0.005)], -1)
    q, _, _ = kin.solve(tgt, kin.seed(tgt[:, :2]), iterations=30)
    for _ in range(50):
        ang = rng.uniform(0, 2 * np.pi, n)
        tgt[:, :2] += 0.004 * np.stack([np.cos(ang), np.sin(ang)], -1)
        q, err, _ = kin.solve(tgt, q, iterations=1)
    assert np.percentile(err, 99) < 3e-4
