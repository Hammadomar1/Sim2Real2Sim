"""Planar pushing objects: geometry, keypoints and rotational symmetry.

Every object is an extruded 2D shape built from boxes (or a single disk) and
sits flat on the table. The object frame origin is the 2D centroid of its
footprint, which is also the centre of mass (uniform density). Keypoints are
used for the pose-error reward and for symmetry-aware evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class ObjectShape:
    name: str
    # (centre_x, centre_y, half_x, half_y) in metres, in an arbitrary design frame.
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    disk_radius: float = 0.0
    height: float = 0.020
    mass: float = 0.030
    # Rotational symmetry order: 1 none, 2 = 180 deg, 4 = 90 deg, 0 = continuous.
    symmetry: int = 1
    # Keypoints in the design frame; shifted to the centroid frame on access.
    design_keypoints: tuple[tuple[float, float], ...] = ()
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)

    @property
    def centroid(self) -> np.ndarray:
        if self.disk_radius > 0:
            return np.zeros(2)
        areas = np.array([4 * hx * hy for _, _, hx, hy in self.boxes])
        centres = np.array([(cx, cy) for cx, cy, _, _ in self.boxes])
        return (areas[:, None] * centres).sum(0) / areas.sum()

    @property
    def centred_boxes(self) -> list[tuple[float, float, float, float]]:
        c = self.centroid
        return [(cx - c[0], cy - c[1], hx, hy) for cx, cy, hx, hy in self.boxes]

    @property
    def keypoints(self) -> np.ndarray:
        """(K, 2) keypoints in the object (centroid) frame."""
        if self.disk_radius > 0:
            return np.zeros((1, 2))
        return np.asarray(self.design_keypoints, dtype=float) - self.centroid

    @property
    def radius(self) -> float:
        """Largest distance from the centroid to any point of the footprint."""
        if self.disk_radius > 0:
            return self.disk_radius
        corners = [(cx + sx * hx, cy + sy * hy) for cx, cy, hx, hy in self.centred_boxes
                   for sx in (-1, 1) for sy in (-1, 1)]
        return float(np.max(np.linalg.norm(corners, axis=1)))

    def outline(self) -> np.ndarray:
        """Polygon outline (M, 2) for drawing; disks are sampled."""
        if self.disk_radius > 0:
            t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            return self.disk_radius * np.stack([np.cos(t), np.sin(t)], -1)
        return _union_outline(self.centred_boxes)


def _union_outline(boxes):
    """Outline of a union of axis-aligned boxes, traced on the grid of their edges."""
    # Snap to 1 nm: shared edges computed from different boxes can differ in the last bit, which
    # would leave a zero-height row of cells and drop corners from the outline.
    xs = sorted({round(cx + s * hx, 9) for cx, _, hx, _ in boxes for s in (-1, 1)})
    ys = sorted({round(cy + s * hy, 9) for _, cy, _, hy in boxes for s in (-1, 1)})
    inside = lambda x, y: any(abs(x - cx) < hx and abs(y - cy) < hy for cx, cy, hx, hy in boxes)
    edges = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mx, my = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if not inside(mx, my):
                continue
            cell = [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            nbrs = [(mx, ys[j] - 1e-6), (xs[i + 1] + 1e-6, my), (mx, ys[j + 1] + 1e-6), (xs[i] - 1e-6, my)]
            for k in range(4):
                if not inside(*nbrs[k]):
                    edges.append((cell[k], cell[(k + 1) % 4]))
    # chain edges into a loop
    loop = [edges[0][0], edges[0][1]]
    rest = edges[1:]
    while rest:
        for n, (a, b) in enumerate(rest):
            if np.allclose(a, loop[-1]):
                loop.append(b); rest.pop(n); break
        else:
            break
    pts = np.array(loop[:-1])
    # drop collinear points
    def turn(a, b):
        return a[0] * b[1] - a[1] * b[0]
    keep = [k for k in range(len(pts)) if abs(turn(pts[k] - pts[k - 1], pts[(k + 1) % len(pts)] - pts[k])) > 1e-12]
    return pts[keep]


# Half-scale version of the Diffusion Policy Push-T block (bar 120x30, stem 30x90 mm).
T_BLOCK = ObjectShape(
    name="tee",
    boxes=((0.0, 0.0225, 0.030, 0.0075),     # bar: 60 x 15 mm
           (0.0, -0.0075, 0.0075, 0.0225)),  # stem: 15 x 45 mm
    design_keypoints=((-0.0225, 0.0225), (0.0225, 0.0225), (0.0, -0.0225)),
    rgba=(0.30, 0.45, 0.85, 1.0),
)

# The remaining shapes complete the proposal's six-object set.
OBJECTS: dict[str, ObjectShape] = {
    "tee": T_BLOCK,
    "ell": ObjectShape("ell", boxes=((-0.0150, 0.0, 0.0075, 0.0300), (0.0075, -0.0225, 0.0150, 0.0075)),
                       design_keypoints=((-0.0150, 0.0225), (-0.0150, -0.0225), (0.0150, -0.0225)),
                       rgba=(0.90, 0.55, 0.15, 1.0)),
    "box": ObjectShape("box", boxes=((0.0, 0.0, 0.020, 0.020),), symmetry=4,
                       design_keypoints=((0.015, 0.015), (-0.015, 0.015), (-0.015, -0.015), (0.015, -0.015)),
                       rgba=(0.35, 0.70, 0.35, 1.0)),
    "rect": ObjectShape("rect", boxes=((0.0, 0.0, 0.030, 0.0125),), symmetry=2,
                        design_keypoints=((0.025, 0.0), (-0.025, 0.0), (0.0, 0.008)),
                        rgba=(0.60, 0.40, 0.80, 1.0)),
    "disk": ObjectShape("disk", disk_radius=0.022, symmetry=0, rgba=(0.85, 0.80, 0.25, 1.0)),
    "plus": ObjectShape("plus", boxes=((0.0, 0.0, 0.025, 0.0075), (0.0, 0.0, 0.0075, 0.025)), symmetry=4,
                        design_keypoints=((0.020, 0.0), (0.0, 0.020), (-0.020, 0.0), (0.0, -0.020)),
                        rgba=(0.20, 0.75, 0.80, 1.0)),
}


@dataclass
class ObjectTables:
    """Padded numpy tables for vectorised per-environment object lookups."""
    names: list[str]
    keypoints: np.ndarray        # (n_obj, K, 2)  padded by repeating the last keypoint
    keypoint_weight: np.ndarray  # (n_obj, K)     1/K_i for real keypoints, 0 for padding
    symmetry: np.ndarray         # (n_obj,)
    radius: np.ndarray           # (n_obj,)
    boxes: np.ndarray            # (n_obj, B, 4)  (cx, cy, hx, hy); padded with hx = hy = 0
    disk_radius: np.ndarray      # (n_obj,)
    nominal_mass: np.ndarray     # (n_obj,)
    extra: dict = field(default_factory=dict)


def object_tables(names: list[str]) -> ObjectTables:
    shapes = [OBJECTS[n] for n in names]
    k = max(len(s.keypoints) for s in shapes)
    b = max(max(len(s.boxes), 1) for s in shapes)
    kp = np.zeros((len(shapes), k, 2))
    kw = np.zeros((len(shapes), k))
    bx = np.zeros((len(shapes), b, 4))
    for i, s in enumerate(shapes):
        p = s.keypoints
        kp[i] = np.concatenate([p, np.repeat(p[-1:], k - len(p), 0)])
        kw[i, :len(p)] = 1.0 / len(p)
        cb = s.centred_boxes
        if cb:
            bx[i, :len(cb)] = cb
    return ObjectTables(names=list(names), keypoints=kp, keypoint_weight=kw,
                        symmetry=np.array([s.symmetry for s in shapes]),
                        radius=np.array([s.radius for s in shapes]), boxes=bx,
                        disk_radius=np.array([s.disk_radius for s in shapes]),
                        nominal_mass=np.array([s.mass for s in shapes]))
