"""Will a trained policy work in the real lightbox?  Same held-out scenes, training scene vs lightbox, and
the lightbox under the quantities we have not measured yet (mat height, mat friction, camera latency), the
hardware errors, and the arm hiding the object from the camera at each candidate mount (occlusion.py;
"hidden" = share of the object's frames lost to the arm, "gap p90 s" = 90th percentile of each episode's
longest run of them).

    python -m s2r2s.lightbox runs/tee_v1/best.pt
    python -m s2r2s.lightbox runs/clutter_v7/best.pt --episodes 1000
    python -m s2r2s.lightbox runs/tee_v1/best.pt --set rand.table_friction=0.6,0.6   # after measuring the mat

The lightbox scene (``SceneConfig.lightbox``) has the measured geometry: the mat top is the work surface,
9.8 mm above the surface the robot stands on, with a near edge 7.9 cm in front of the pan axis. Every row
runs the same scenes (start, goal, physical parameters and sensor noise come from one seed), so rows
differ only by the condition. Every row starts from the standard randomisation (RandomizationConfig()),
whatever the checkpoint trained with, so different checkpoints are graded under identical conditions.
Results go next to the checkpoint as lightbox_<name>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .env import RandomizationConfig, apply_overrides
from .evaluate import TEST_SEED
from .evaluation import policy_agent, run_episodes
from .train import load_policy

LIGHTBOX = ["scene.lightbox=true"]
# (label, overrides). Mat height error e: the mat top is e higher than modelled, so the rod tip, commanded
# 5 mm above the modelled mat, ends up 5 mm - e above the real one (negative: pressed into the mat).
CONDITIONS = [
    ("training scene (flat table, robot on the table)", []),
    ("lightbox (measured geometry)", LIGHTBOX),
    ("lightbox, IK not told about the mat: rod 4.8 mm into it", LIGHTBOX + ["task.tool_height=-0.0048"]),
    ("mat 4 mm lower than modelled: rod 9 mm above it", LIGHTBOX + ["task.tool_height=0.009"]),
    ("mat 2 mm higher than modelled: rod 3 mm above it", LIGHTBOX + ["task.tool_height=0.003"]),
    ("mat 4 mm higher than modelled: rod 1 mm above it", LIGHTBOX + ["task.tool_height=0.001"]),
    ("mat 6 mm higher than modelled: rod 1 mm into it", LIGHTBOX + ["task.tool_height=-0.001"]),
    ("mat friction 0.20 (trained on 0.20-0.50)", LIGHTBOX + ["rand.table_friction=0.2,0.2"]),
    ("mat friction 0.35", LIGHTBOX + ["rand.table_friction=0.35,0.35"]),
    ("mat friction 0.50", LIGHTBOX + ["rand.table_friction=0.5,0.5"]),
    ("mat friction 0.70", LIGHTBOX + ["rand.table_friction=0.7,0.7"]),
    ("mat friction 0.90", LIGHTBOX + ["rand.table_friction=0.9,0.9"]),
    ("camera latency 100-150 ms (trained on 0-100)", LIGHTBOX + ["rand.obs_latency_steps=2,3"]),
    ("camera latency 150-200 ms", LIGHTBOX + ["rand.obs_latency_steps=3,4"]),
    ("camera latency 200-300 ms", LIGHTBOX + ["rand.obs_latency_steps=4,6"]),
    ("camera noise 2 mm / 2 deg (trained on 1 / 1)", LIGHTBOX + ["rand.obs_pos_noise=0.002", "rand.obs_yaw_noise=0.0349"]),
    ("joint calibration error up to 0.5 deg per joint", LIGHTBOX + ["rand.joint_offset_deg=0.5"]),
    ("joint calibration error up to 1 deg per joint", LIGHTBOX + ["rand.joint_offset_deg=1.0"]),
    ("joint calibration error up to 2 deg per joint", LIGHTBOX + ["rand.joint_offset_deg=2.0"]),
    ("backlash +-0.5 deg, pan only", LIGHTBOX + ["rand.backlash_deg=1,0,0,0"]),
    ("backlash +-0.5 deg, every joint (Menagerie's figure)", LIGHTBOX + ["rand.backlash_deg=1,1,1,1"]),
    ("backlash +-1 deg, every joint", LIGHTBOX + ["rand.backlash_deg=2,2,2,2"]),
    ("all at once (cal 1 deg, backlash 0.5, mu .2-.9, 0-150 ms)", LIGHTBOX + [
        "rand.joint_offset_deg=1.0", "rand.backlash_deg=1,1,1,1", "rand.table_friction=0.2,0.9", "rand.obs_latency_steps=0,3"]),
    # Arm occlusion (occlusion.py): frames with less than half the top face visible are dropped (by default).
    ("30 % of frames dropped at random (trained on 3 %)", LIGHTBOX + ["rand.obs_dropout=0.3"]),
    ("arm occlusion, C920 in the left-wall slot (as built)", LIGHTBOX + ["rand.occlusion_camera=c920"]),
    ("arm occlusion, C920 in the right-wall slot", LIGHTBOX + ["rand.occlusion_camera=right_wall"]),
    ("arm occlusion, C920 on the back wall, 25 cm up", LIGHTBOX + ["rand.occlusion_camera=back_wall_low"]),
    ("left wall, estimator copes with 25 % of the T visible", LIGHTBOX + ["rand.occlusion_camera=c920", "rand.occlusion_threshold=0.25"]),
    ("left wall, estimator needs 75 % visible", LIGHTBOX + ["rand.occlusion_camera=c920", "rand.occlusion_threshold=0.75"]),
    ("left wall, estimate pulled to the visible part", LIGHTBOX + ["rand.occlusion_camera=c920", "rand.occlusion_bias=1.0"]),
    ("back wall, estimator needs 75 % visible", LIGHTBOX + ["rand.occlusion_camera=back_wall_low", "rand.occlusion_threshold=0.75"]),
    ("back wall, estimate pulled to the visible part", LIGHTBOX + ["rand.occlusion_camera=back_wall_low", "rand.occlusion_bias=1.0"]),
    ("right wall, estimator needs 75 % visible", LIGHTBOX + ["rand.occlusion_camera=right_wall", "rand.occlusion_threshold=0.75"]),
    ("right wall, estimate pulled to the visible part", LIGHTBOX + ["rand.occlusion_camera=right_wall", "rand.occlusion_bias=1.0"]),
]
COLUMNS = [("success", "solved", "{:7.1%}"), ("pos_err_mm_median", "mm", "{:6.1f}"), ("yaw_err_deg_median", "deg", "{:6.1f}"),
           ("failure", "failed", "{:7.1%}"), ("dropped", "off mat", "{:8.1%}"), ("disturbed", "block moved", "{:12.1%}"),
           ("occluded", "hidden", "{:7.1%}"), ("occluded_longest_s_p90", "gap p90 s", "{:10.1f}")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--only", nargs="*", type=int, default=[], help="row numbers to run (default: all)")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE", help="applied to every row")
    p.add_argument("--threads", type=int, default=0, help="simulation threads (default: all; fewer next to a training run)")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, base, ck = load_policy(args.checkpoint, device)
    base.rand, base.scene.lightbox = RandomizationConfig(), False      # rows set the conditions, not the checkpoint
    base.num_threads = args.threads
    agent = policy_agent(policy, device)
    rows = [(k, c) for k, c in enumerate(CONDITIONS) if not args.only or k in args.only]
    print(f"{args.checkpoint}: {args.episodes} held-out scenes (seed {args.seed}) per row, same scenes in every row")
    print(f"{'':4s}{'condition':58s}" + "".join(f"{h:>{len(f.format(0.0))}s}" for _, h, f in COLUMNS))
    results = {}
    for k, (label, overrides) in rows:
        cfg = apply_overrides(copy.deepcopy(base), overrides + args.set)
        r = run_episodes(cfg, agent, args.episodes, args.seed)
        results[label] = {"overrides": overrides + args.set, **r}
        cells = "".join(fmt.format(r[key]) if key in r else " " * len(fmt.format(0.0)) for key, _, fmt in COLUMNS)
        print(f"{k:3d} {label:58s}{cells}", flush=True)
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(f"lightbox_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "episodes": args.episodes, "seed": args.seed,
                               "results": results}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
