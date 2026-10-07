"""How well does the rod tip follow commands?  Joint tracking, tip error and lag on a test path.

    python -m s2r2s.hardware.tracking --robot sim
    python -m s2r2s.hardware.tracking --robot so101 --port COM5 --map calibration/joint_map.json

The rod moves around a square in the work area (at --height above the table, 30 mm by default, so
nothing is touched) at --speed, using the same IK as the policy. Every tick records the commanded
and the measured joints; tip positions come from forward kinematics of each. Reports the tip
tracking error, the lag that best aligns measured to commanded, and per-joint steady errors, and
saves a plot and the raw data.

On the real arm, forward kinematics of the *measured* joints cannot see backlash or link flex:
film the same run with the camera (Milestone 2) to get the true tip error.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..kinematics import PusherKinematics
from ..scene import SceneConfig, build_model
from .joint_map import JointMap
from .robots import SimRobot, SO101Robot
from .runner import move_to

PROJECT = Path(__file__).resolve().parents[2]


def square_path(centre=(0.20, 0.0), half: float = 0.04, speed: float = 0.04, dt: float = 0.05, loops: int = 2):
    corners = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]) * half + np.asarray(centre)
    pts = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(1, round(np.linalg.norm(b - a) / (speed * dt)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(pts * loops)


def analyse(t, q_cmd, q_meas, kin: PusherKinematics, dt: float):
    tip_cmd = kin.forward(q_cmd)[0]
    tip_meas = kin.forward(q_meas)[0]
    lags = range(0, 11)
    rms = [np.sqrt(np.mean(np.sum((tip_meas[s:] - tip_cmd[:len(tip_cmd) - s]) ** 2, -1))) for s in lags]
    best = int(np.argmin(rms))
    err = np.linalg.norm(tip_meas - tip_cmd, axis=-1)
    aligned = np.linalg.norm(tip_meas[best:] - tip_cmd[:len(tip_cmd) - best], axis=-1)
    return dict(tip_err_rms_mm=float(np.sqrt(np.mean(err ** 2)) * 1e3), tip_err_max_mm=float(err.max() * 1e3),
                lag_ms=best * dt * 1e3, tip_err_after_lag_rms_mm=float(np.sqrt(np.mean(aligned ** 2)) * 1e3),
                joint_err_rms_deg=np.degrees(np.sqrt(np.mean((q_meas[best:] - q_cmd[:len(q_cmd) - best]) ** 2, 0))).tolist(),
                tip_cmd=tip_cmd, tip_meas=tip_meas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["sim", "so101"], default="sim")
    ap.add_argument("--port", default="")
    ap.add_argument("--map", default="")
    ap.add_argument("--height", type=float, default=0.03, help="rod tip height above the table (m)")
    ap.add_argument("--speed", type=float, default=0.04, help="m/s (the policy uses up to 0.08)")
    ap.add_argument("--half", type=float, default=0.04, help="half side of the square (m)")
    ap.add_argument("--out", default=str(PROJECT / "calibration" / "tracking"))
    args = ap.parse_args(argv)
    dt = 0.05
    jmap = JointMap.load(args.map) if args.map else JointMap()
    kin = PusherKinematics(build_model(SceneConfig(timestep=0.005), visual=False))
    realtime = args.robot == "so101"
    robot = SO101Robot(args.port) if realtime else SimRobot(encoder_noise_deg=0.05)
    path = square_path(half=args.half, speed=args.speed, dt=dt)
    move_to(robot, jmap, kin, (*path[0], args.height), seconds=3.0, realtime=realtime)
    q = jmap.to_sim(robot.read_joints())[None]
    t, q_cmd, q_meas = [], [], []
    for xy in path:
        t0 = time.monotonic()
        q, _, _ = kin.solve(np.array([[*xy, args.height]]), q, iterations=2)
        robot.send_joints(jmap.to_robot(q[0]))
        t.append(t0)
        q_cmd.append(q[0].copy())
        q_meas.append(jmap.to_sim(robot.read_joints()))
        if realtime:
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    move_to(robot, jmap, kin, (*path[-1], 0.05), seconds=1.0, realtime=realtime)
    robot.close()
    r = analyse(np.array(t), np.array(q_cmd), np.array(q_meas), kin, dt)
    print(f"path: square {2 * args.half * 1e3:.0f} mm at {args.speed * 1e3:.0f} mm/s, {args.height * 1e3:.0f} mm above the table, {len(path)} ticks")
    print(f"tip tracking error: RMS {r['tip_err_rms_mm']:.2f} mm, max {r['tip_err_max_mm']:.2f} mm")
    print(f"best-fit lag: {r['lag_ms']:.0f} ms; error after removing the lag: RMS {r['tip_err_after_lag_rms_mm']:.2f} mm")
    print("joint error after lag (deg): " + ", ".join(f"{e:.2f}" for e in r["joint_err_rms_deg"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), t=np.array(t), q_cmd=np.array(q_cmd), q_meas=np.array(q_meas))
    out.with_suffix(".json").write_text(json.dumps({k: v for k, v in r.items() if not k.startswith("tip_")}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r["tip_cmd"][:, 1] * 1e3, r["tip_cmd"][:, 0] * 1e3, label="commanded")
    ax[0].plot(r["tip_meas"][:, 1] * 1e3, r["tip_meas"][:, 0] * 1e3, label="measured", alpha=0.8)
    ax[0].set(xlabel="y (mm)", ylabel="x (mm)", title="Rod tip path (FK)", aspect="equal")
    ax[0].legend()
    ax[1].plot(np.arange(len(path)) * dt, np.linalg.norm(r["tip_meas"] - r["tip_cmd"], axis=-1) * 1e3)
    ax[1].set(xlabel="time (s)", ylabel="tip error (mm)", title="Tracking error (no lag removed)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"saved {out.with_suffix('.png')}, .npz, .json")


if __name__ == "__main__":
    main()
