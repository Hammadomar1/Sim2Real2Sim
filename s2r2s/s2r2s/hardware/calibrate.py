"""Measure joint offsets (and check joint directions) by touching known points with the rod tip.

    python -m s2r2s.hardware.calibrate --robot sim --sim-offsets 4,-6,3,8 --sim-signs 1,1,-1,1   # rehearsal
    python -m s2r2s.hardware.calibrate --robot so101 --port COM5                                 # real arm

Real-arm procedure:
 1. Mark the points on the table (``--points file.csv`` with x,y in metres, or the default 9-point
    grid printed at start). Coordinates are in the robot base frame: x forward, y to the robot's
    left, origin 38.8 mm behind the shoulder-pan axis (the centre of the rotating base).
 2. Torque turns off: support the arm. For each point, rest the rod tip on the mark with the rod
    roughly vertical and press Enter (10 readings are averaged).
 3. The fit tries every combination of joint directions, solves for the joint offsets and the rod
    length error, and saves calibration/joint_map.json. Residuals above ~3 mm mean a mismeasured
    point, a slipping rod, or a wrong rod length in SceneConfig. Rod length trades off against the
    elbow and wrist offsets (tip positions stay right either way); for the cleanest offsets, measure
    the rod with calipers, put it in SceneConfig and use --no-fit-rod.
 4. With --verify, the arm then hovers 15 mm above each point so you can check alignment by eye.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from ..kinematics import PusherKinematics
from ..scene import SceneConfig, build_model
from .joint_map import JointMap
from .robots import SimRobot, SO101Robot

PROJECT = Path(__file__).resolve().parents[2]


def default_points() -> np.ndarray:
    """Nine points spread over the pushing workspace (robot base frame, on the table)."""
    pts = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
           for r in (0.15, 0.20, 0.25) for a in (-35, 0, 35)]
    return np.array([(x, y, 0.0) for x, y in pts])


def _tips(kin: PusherKinematics, q: np.ndarray, rod_delta: float) -> np.ndarray:
    """Rod-tip positions for joint angles q (N, 5); a longer rod (rod_delta > 0) moves the tip along -tool axis."""
    tip, axis = kin.forward(q)
    return tip - rod_delta * axis


def _solve(f, x0, iterations: int = 50):
    """Small Levenberg-Marquardt with finite differences (a handful of parameters)."""
    x, lam = np.array(x0, float), 1e-3
    r = f(x)
    for _ in range(iterations):
        eps = 1e-6
        J = np.stack([(f(x + eps * e) - r) / eps for e in np.eye(len(x))], 1)
        step = np.linalg.solve(J.T @ J + lam * np.eye(len(x)), -J.T @ r)
        r_new = f(x + step)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
    return x, r


def fit(readings_deg: np.ndarray, points: np.ndarray, kin: PusherKinematics, fit_rod: bool = True):
    """Find joint signs (pan, lift, elbow, wrist flex), offsets and the rod length error.

    Returns (JointMap, per-point errors in metres, ranking of all sign combinations by RMS error).
    Wrist roll barely moves the tip, so its sign and offset keep their defaults.
    """
    results = []
    for signs in itertools.product((1.0, -1.0), repeat=4):
        sign = np.array([*signs, 1.0])

        def residual(p):
            offset = np.r_[p[:4], 0.0]
            q = sign * np.radians(readings_deg - offset)
            return (_tips(kin, q, p[4] if fit_rod else 0.0) - points).ravel()

        p, r = _solve(residual, np.zeros(5))
        results.append((float(np.sqrt(np.mean(r.reshape(-1, 3) ** 2) * 3)), sign, p))
    results.sort(key=lambda t: t[0])
    rms, sign, p = results[0]
    jmap = JointMap(sign=sign.tolist(), offset_deg=[*map(float, p[:4]), 0.0], tool_length_error=float(p[4]))
    errors = np.linalg.norm(_tips(kin, jmap.to_sim(readings_deg), jmap.tool_length_error) - points, axis=1)
    ranking = [(rms_k, s.tolist()) for rms_k, s, _ in results]
    return jmap, errors, ranking


def simulate_touches(robot: SimRobot, points: np.ndarray, rng=None, tilt_deg: float = 3.0) -> np.ndarray:
    """A simulated person resting the rod tip on each point, rod within a few degrees of vertical."""
    rng = rng or np.random.default_rng(0)
    kin = robot.kin
    readings = []
    for p in points:
        upright, _, _ = kin.solve(p[None], kin.seed(p[None, :2]), iterations=30)
        tilt = math.radians(rng.uniform(-tilt_deg, tilt_deg))
        for _ in range(3):
            # A slightly tilted rod: change wrist flex, then restore the tip position with the other
            # joints (damped, bounded steps within joint limits). Halve the tilt if that fails.
            q = upright.copy()
            q[0, 3] += tilt
            for _ in range(60):
                tip, _, jp, _ = kin.forward(q, with_jacobian=True)
                J = jp[0][:, :3]
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ (p - tip[0]))
                q[0, :3] = np.clip(q[0, :3] + np.clip(dq, -0.1, 0.1), kin.lower[:3], kin.upper[:3])
            if np.linalg.norm(kin.forward(q)[0][0] - p) < 1e-5:
                break
            tilt /= 2
        else:
            q = upright
        robot.move_by_hand(q[0])
        readings.append(np.mean([robot.read_joints() for _ in range(10)], 0))
    return np.array(readings)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="", help="serial port of the follower arm, e.g. COM5")
    ap.add_argument("--points", default="", help="CSV with x,y (m) per line; default: 9-point grid")
    ap.add_argument("--out", default=str(PROJECT / "calibration" / "joint_map.json"))
    ap.add_argument("--verify", action="store_true", help="afterwards, hover 15 mm above each point")
    ap.add_argument("--no-fit-rod", action="store_true",
                    help="trust the rod length in SceneConfig (measure it with calipers): better-conditioned offsets")
    ap.add_argument("--sim-offsets", default="4,-6,3,8", help="sim rehearsal: hidden offsets (deg)")
    ap.add_argument("--sim-signs", default="1,1,-1,1", help="sim rehearsal: hidden joint directions")
    ap.add_argument("--sim-rod", type=float, default=0.002, help="sim rehearsal: rod longer than modelled (m)")
    args = ap.parse_args(argv)

    points = default_points()
    if args.points:
        xy = np.loadtxt(args.points, delimiter=",", ndmin=2)
        points = np.c_[xy[:, :2], np.zeros(len(xy))]
    kin = PusherKinematics(build_model(SceneConfig(timestep=0.005), visual=False))
    print("Touch points (robot base frame, m):")
    for k, p in enumerate(points):
        print(f"  {k + 1}: x={p[0]:.3f}  y={p[1]:+.3f}")

    if args.robot == "sim":
        hidden = JointMap(sign=[*map(float, args.sim_signs.split(",")), 1.0],
                          offset_deg=[*map(float, args.sim_offsets.split(",")), 0.0])
        scene = SceneConfig(timestep=0.005)
        scene.pusher_tip_z -= args.sim_rod
        robot = SimRobot(scene, hidden=hidden, encoder_noise_deg=0.1)
        readings = simulate_touches(robot, points)
    else:
        robot = SO101Robot(args.port)
        input("Torque will turn OFF. Support the arm with one hand, then press Enter...")
        robot.set_torque(False)
        readings = []
        for k, p in enumerate(points):
            input(f"Rest the rod tip on point {k + 1} (x={p[0]:.3f}, y={p[1]:+.3f}), rod vertical, then Enter...")
            samples = []
            for _ in range(10):
                samples.append(robot.read_joints())
                time.sleep(0.02)
            readings.append(np.mean(samples, 0))
        readings = np.array(readings)
        input("Hold the arm in a safe pose; torque turns back ON when you press Enter...")
        robot.set_torque(True)

    jmap, errors, ranking = fit(readings, points, kin, fit_rod=not args.no_fit_rod)
    print("\nJoint directions tried (best first): " + ", ".join(f"{s[:4]} -> {rms * 1e3:.1f} mm" for rms, s in ranking[:3]))
    print("Fitted joint map:")
    for name, s, o in zip(jmap.names, jmap.sign, jmap.offset_deg):
        print(f"  {name:14s} sign {s:+.0f}  offset {o:+7.2f} deg")
    print(f"  rod length error {jmap.tool_length_error * 1e3:+.1f} mm (positive: rod longer than SceneConfig says)")
    print("Per-point error (mm): " + " ".join(f"{e * 1e3:.1f}" for e in errors) +
          f" | RMS {np.sqrt(np.mean(errors ** 2)) * 1e3:.2f} mm")
    if args.robot == "sim":
        print(f"(sim rehearsal: hidden signs {args.sim_signs}, offsets {args.sim_offsets} deg, rod +{args.sim_rod * 1e3:.1f} mm)")
    if errors.max() > 0.003:
        print("WARNING: errors above 3 mm. Re-measure the points or check that the rod is not slipping.")
    jmap.save(args.out)
    Path(args.out).with_suffix(".points.json").write_text(json.dumps(
        {"points": points.tolist(), "readings_deg": np.asarray(readings).tolist(), "errors_m": errors.tolist()}, indent=1))
    print(f"saved {args.out}")

    if args.verify:
        hover = points + np.array([0, 0, 0.015])
        q = jmap.to_sim(robot.read_joints())[None]
        for k, p in enumerate(hover):
            target, _, _ = kin.solve(p[None], q, iterations=30)
            path = np.linspace(q[0], target[0], 40)          # about 2 s per move at 20 Hz
            for qk in path:
                robot.send_joints(jmap.to_robot(qk))
                if args.robot == "so101":
                    time.sleep(0.05)
            q = target
            if args.robot == "so101":
                input(f"Hovering above point {k + 1}: is the rod tip right above the mark? Enter for the next...")
    robot.close()


if __name__ == "__main__":
    main()
