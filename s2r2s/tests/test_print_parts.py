import numpy as np
import pytest

from s2r2s.objects import OBJECTS
from s2r2s.print_parts import ROD_LENGTH, check, extrude, parts, rod, rod_below_jaws, write_stl
from s2r2s.scene import GATE_WALL_LENGTH, SceneConfig


def test_every_part_is_watertight_and_matches_the_simulator():
    cfg, items = parts(all_objects=True)
    for stem, _, tris, _ in items:
        volume = check(tris)
        if stem in OBJECTS:
            s = OBJECTS[stem]
            o = s.outline()                                  # the union of the boxes (the plus's bars overlap)
            area = (np.pi * s.disk_radius ** 2 if s.disk_radius > 0 else
                    0.5 * abs(np.sum(o[:, 0] * np.roll(o[:, 1], -1) - np.roll(o[:, 0], -1) * o[:, 1])))
            assert volume == pytest.approx(area * s.height * 1e9, rel=2e-3)
    names = [stem for stem, *_ in items]
    assert {"tee", "box", "rod", "gate_wall_inner", "gate_wall_outer"} <= set(names)
    walls = {stem: tris for stem, _, tris, _ in items if stem.startswith("gate_wall")}
    for (stem, tris), length in zip(sorted(walls.items()), GATE_WALL_LENGTH):
        size = np.ptp(tris.reshape(-1, 3), axis=0)
        assert size == pytest.approx([length * 1e3, 10, 25])


def test_rod_has_the_simulated_radius_and_a_rounded_tip():
    cfg = SceneConfig()
    tris = rod(cfg.pusher_radius * 1e3, ROD_LENGTH)
    pts = tris.reshape(-1, 3)
    assert np.ptp(pts, axis=0) == pytest.approx([12, 12, ROD_LENGTH], abs=1e-6)
    top = pts[pts[:, 2] > ROD_LENGTH - cfg.pusher_radius * 1e3 + 1e-9]
    assert np.allclose(np.linalg.norm(top - [0, 0, ROD_LENGTH - 6], axis=1), 6)   # the tip is a hemisphere
    assert check(tris) == pytest.approx(np.pi * 36 * (ROD_LENGTH - 6) + 2 / 3 * np.pi * 216, rel=5e-3)
    assert 25 < rod_below_jaws(cfg) < 35                     # the documented ~30 mm below the jaw tips


def test_concave_outline_and_stl_file(tmp_path):
    tris = extrude(OBJECTS["tee"].outline() * 1e3, 20)
    assert check(tris) == pytest.approx(31500)
    write_stl(tmp_path / "t.stl", tris, "t")
    data = (tmp_path / "t.stl").read_bytes()
    assert len(data) == 84 + 50 * len(tris)
