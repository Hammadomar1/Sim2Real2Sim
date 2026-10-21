"""Ready-to-print STL files for the real setup, from the simulator's own dimensions.

    python -m s2r2s.print_parts                  # print/: the T, the box, the rod, the gate walls, README.txt, preview.png
    python -m s2r2s.print_parts --all-objects    # also the other four objects of the six-object set

Every size comes from the code the policies were trained with: the footprints and height in objects.py, the rod's
radius and reach in SceneConfig (scene.py), and the gate wall constants in scene.py. The rod's depth below the jaw
tips is measured on the Menagerie gripper model. If a printed part comes out different, measure it and change those
values (then retrain or at least re-check with `python -m s2r2s.lightbox`), not the STL.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

from .env import RandomizationConfig, TaskConfig
from .objects import OBJECTS
from .scene import GATE_WALL_HALF_HEIGHT, GATE_WALL_HALF_THICKNESS, GATE_WALL_LENGTH, SceneConfig, build_model

PROJECT = Path(__file__).resolve().parents[1]
PLA = 1.24e-3                                    # g / mm^3
ROD_LENGTH = 60.0                                # mm: the part below the jaw tips plus about 30 mm for the jaws to hold


# ---------------------------------------------------------------- meshes (triangles in mm, outward normals)
def _ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles (index triples, counter-clockwise) of a simple counter-clockwise polygon."""
    idx = list(range(len(poly)))
    tris = []
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i], poly[j], poly[l]
            if cross(a, b, c) <= 1e-12:
                continue                                       # reflex corner
            inside = any(cross(a, b, poly[m]) >= 0 and cross(b, c, poly[m]) >= 0 and cross(c, a, poly[m]) >= 0
                         for m in idx if m not in (i, j, l))
            if not inside:
                tris.append((i, j, l))
                idx.pop(k)
                break
        else:
            raise ValueError("polygon is not simple")
    tris.append(tuple(idx))
    return tris


def extrude(poly_mm: np.ndarray, height: float) -> np.ndarray:
    """A prism: the polygon (any orientation) from z = 0 to z = height. Returns (N, 3, 3)."""
    p = np.asarray(poly_mm, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:
        p = p[::-1]
    bottom = np.c_[p, np.zeros(len(p))]
    top = np.c_[p, np.full(len(p), height)]
    tris = []
    for i, j, k in _ear_clip(p):
        tris.append([top[i], top[j], top[k]])                  # counter-clockwise seen from above: normal +z
        tris.append([bottom[i], bottom[k], bottom[j]])         # reversed: normal -z
    for i in range(len(p)):
        j = (i + 1) % len(p)
        tris.append([bottom[i], bottom[j], top[j]])
        tris.append([bottom[i], top[j], top[i]])
    return np.array(tris)


def rod(radius: float, length: float, segments: int = 96, rings: int = 24) -> np.ndarray:
    """A cylinder standing on a flat end at z = 0, with a hemispherical end (the tip) at z = length."""
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = lambda r, z: np.c_[r * np.cos(a), r * np.sin(a), np.full(segments, z)]
    z0 = length - radius
    levels = [ring(radius, 0.0), ring(radius, z0)]
    for k in range(1, rings):
        t = k / rings * math.pi / 2
        levels.append(ring(radius * math.cos(t), z0 + radius * math.sin(t)))
    apex = np.array([0.0, 0.0, length])
    centre = np.zeros(3)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append([centre, levels[0][j], levels[0][i]])      # flat end, normal -z
        for lo, hi in zip(levels[:-1], levels[1:]):
            tris.append([lo[i], lo[j], hi[j]])
            tris.append([lo[i], hi[j], hi[i]])
        tris.append([levels[-1][i], levels[-1][j], apex])
    return np.array(tris)


def check(tris: np.ndarray) -> float:
    """Watertight and consistently oriented (every edge used once in each direction); returns the volume (mm^3)."""
    key = lambda v: tuple(np.round(v, 6))
    edges = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (key(a), key(b))
            edges[e] = edges.get(e, 0) + 1
    for (a, b), n in edges.items():
        if n != 1 or edges.get((b, a), 0) != 1:
            raise ValueError(f"mesh is not watertight at edge {a} -> {b}")
    volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    if volume <= 0:
        raise ValueError("mesh is inside out")
    return float(volume)


def write_stl(path: Path, tris: np.ndarray, name: str):
    """Binary STL in millimetres."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for normal, t in zip(n, tris):
            f.write(struct.pack("<12fH", *normal, *t.ravel(), 0))


# ---------------------------------------------------------------- the parts
def rod_below_jaws(cfg: SceneConfig) -> float:
    """How far (mm) the rod's tip reaches below the fixed jaw's lowest point, in the gripper model."""
    import mujoco
    m = build_model(cfg, visual=True)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    g = m.body("gripper").id
    R, p = d.xmat[g].reshape(3, 3), d.xpos[g]
    lowest = min((R.T @ (d.geom_xpos[k] - p))[2] - m.geom_size[k, 0]
                 for k in range(m.ngeom) if m.geom(k).name.startswith("fixed_jaw_sph_tip"))
    return (lowest - cfg.pusher_tip_z) * 1e3


def parts(all_objects: bool = False):
    """(file stem, title, triangles, notes) for every part."""
    cfg = SceneConfig()
    out = []
    names = ["tee", "box"] + ([n for n in OBJECTS if n not in ("tee", "box")] if all_objects else [])
    for name in names:
        shape = OBJECTS[name]
        if shape.disk_radius > 0:
            a = np.linspace(0, 2 * math.pi, 128, endpoint=False)
            outline = shape.disk_radius * np.c_[np.cos(a), np.sin(a)]
        else:
            outline = shape.outline()
        out.append((name, f"{name} block", extrude(outline * 1e3, shape.height * 1e3), shape))
    out.append(("rod", "pusher rod", rod(cfg.pusher_radius * 1e3, ROD_LENGTH), None))
    for label, length in zip(("inner", "outer"), GATE_WALL_LENGTH):
        rect = np.array([[0, 0], [length, 0], [length, 2 * GATE_WALL_HALF_THICKNESS],
                         [0, 2 * GATE_WALL_HALF_THICKNESS]]) * 1e3
        out.append((f"gate_wall_{label}", f"gate wall, {label}", extrude(rect, 2 * GATE_WALL_HALF_HEIGHT * 1e3), None))
    return cfg, out


def readme(cfg: SceneConfig, made: list, below: float) -> str:
    rand, task = RandomizationConfig(), TaskConfig()
    lo, hi = rand.mass_scale
    t, b = OBJECTS["tee"], OBJECTS["box"]
    ty = (t.centroid - np.array([0.0, max(cy + hy for _, cy, _, hy in t.boxes)])) * 1e3   # centroid below the bar's top edge
    lines = [
        "Printed parts for the SO-101 lightbox, generated by `python -m s2r2s.print_parts` from the simulator's",
        "own dimensions. Units: millimetres. Each STL sits on the print bed (z = 0), no supports needed.",
        "",
        "FILES",
    ]
    for stem, title, tris, vol, bbox in made:
        lines.append(f"  {stem + '.stl':24s} {title:22s} {bbox[0]:6.1f} x {bbox[1]:5.1f} x {bbox[2]:5.1f}   "
                     f"solid {vol / 1e3:5.1f} cm3 = {vol * PLA:5.1f} g PLA")
    lines += [
        "",
        f"T BLOCK (tee.stl): half-scale Push-T. Bar {2 * t.boxes[0][2] * 1e3:.0f} x {2 * t.boxes[0][3] * 1e3:.0f}, "
        f"stem {2 * t.boxes[1][2] * 1e3:.0f} x {2 * t.boxes[1][3] * 1e3:.0f} below it",
        f"  (60 x 60 overall), {t.height * 1e3:.0f} tall. Centre of mass on the symmetry axis, "
        f"{-ty[1]:.1f} below the top edge of the bar.",
        f"BOX (box.stl): {2 * b.boxes[0][2] * 1e3:.0f} x {2 * b.boxes[0][3] * 1e3:.0f} x {b.height * 1e3:.0f}.",
        f"  Mass: {t.mass * 1e3:.0f} g each (training covered {lo * t.mass * 1e3:.0f}-{hi * t.mass * 1e3:.0f} g). Solid "
        f"PLA would be ~39 g, so print about 60 % infill",
        "  (gyroid or grid, 3 walls), weigh, and adjust. Keep the infill even: the centre of mass may sit at most",
        f"  {rand.com_offset * 1e3:.0f} mm from the middle. Accuracy: within 0.3 mm of the sizes above; if a part is "
        "off by more, put",
        "  its measured size in objects.py.",
        "  Colour: matte, bright and different for the T and the box (for example orange and green). Not black,",
        "  white or grey (the mat is black, the walls white), not silk or glossy. No text or logo on the top face:",
        "  the camera finds the blocks by the colour of their top face. Keep the top edges sharp.",
        "",
        f"ROD (rod.stl): diameter {2 * cfg.pusher_radius * 1e3:.0f}, length {ROD_LENGTH:.0f}, rounded end "
        f"(radius {cfg.pusher_radius * 1e3:.0f}) = the tip.",
        "  Print standing on its flat end, 100 % infill (or use a 12 mm wooden dowel or metal rod with a rounded end).",
        f"  Mounting: rod vertical, rounded end down, clamped in the closed gripper and touching the FIXED jaw,",
        "  centred between the jaw tips. The tip must reach exactly",
        f"  {below:.1f} mm below the lowest point of the fixed jaw (the simulator's design figure is 30 mm). Mark",
        f"  {below:.0f} mm from the tip with a pen, line the mark up with the jaw tips, close the gripper firmly.",
        "  The policy pushes with the rod's tip 5 mm above the mat. If your rod ends up different, measure it and set",
        "  SceneConfig.pusher_radius / pusher_tip_z, then re-run the calibration (HARDWARE.md 2).",
        "",
        f"GATE WALLS (Level 1 only): {2 * GATE_WALL_HALF_THICKNESS * 1e3:.0f} thick, "
        f"{2 * GATE_WALL_HALF_HEIGHT * 1e3:.0f} tall, {GATE_WALL_LENGTH[0] * 1e3:.0f} (inner) and "
        f"{GATE_WALL_LENGTH[1] * 1e3:.0f} (outer) long.",
        f"  Stand them in one line pointing straight away from the robot, with an opening of "
        f"{task.gate_width[0] * 1e3:.0f}-{task.gate_width[1] * 1e3:.0f} between their ends,",
        f"  the shorter one nearer the robot, the opening's centre {task.gate_r[0] * 1e3:.0f}-{task.gate_r[1] * 1e3:.0f} "
        "from the base origin (38.8 behind the pan axis). Fix them",
        "  to the mat (double-sided tape) so they cannot move. Any infill.",
        "",
        "AFTER PRINTING: measure every part with calipers; weigh the T and the box; tilt-test the T on the mat for",
        "friction (LIGHTBOX.md A4). Paper printouts (ChArUco board, mat markers, rod marker) come from",
        "`python -m s2r2s.hardware.camera_calibrate boards`.",
    ]
    return "\n".join(lines) + "\n"


def preview(path: Path, made_tris: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(4 * len(made_tris), 4))
    for k, (title, tris) in enumerate(made_tris):
        ax = fig.add_subplot(1, len(made_tris), k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolor="#e8823a", edgecolor="#5a3a1a", linewidths=0.05, alpha=1.0))
        lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(c[2] - r, c[2] + r), title=title)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(PROJECT / "print"))
    p.add_argument("--all-objects", action="store_true", help="also the ell, rect, disk and plus blocks")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, items = parts(args.all_objects)
    below = rod_below_jaws(cfg)
    made, pics = [], []
    for stem, title, tris, _ in items:
        vol = check(tris)
        flat = tris.reshape(-1, 3)
        bbox = flat.max(0) - flat.min(0)
        write_stl(out / f"{stem}.stl", tris, f"s2r2s {title}")
        made.append((stem, title, tris, vol, bbox))
        pics.append((title, tris))
        print(f"{stem + '.stl':24s} {len(tris):6d} triangles  {bbox[0]:6.1f} x {bbox[1]:5.1f} x {bbox[2]:5.1f} mm  "
              f"{vol / 1e3:6.2f} cm3", flush=True)
    (out / "README.txt").write_text(readme(cfg, made, below), encoding="utf-8")
    preview(out / "preview.png", pics)
    print(f"rod tip {below:.1f} mm below the fixed jaw's lowest point | wrote {out}")


if __name__ == "__main__":
    main()
