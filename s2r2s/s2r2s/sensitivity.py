"""How good must the real camera pipeline be?  Degrade the simulated camera and re-evaluate.

    python -m s2r2s.sensitivity runs/tee_v1/best.pt

Sweeps object-pose noise, latency and dropout rate of the camera model (one factor at a
time, others at their training values) and reports success and final error. The policy is
fixed; only the sensor model changes. Use the result as the acceptance target for the
perception pipeline in Milestone 2.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from .evaluate import TEST_SEED
from .evaluation import policy_agent, run_episodes
from .train import load_policy

SWEEPS = {
    "noise (mm / deg, 1 sigma)": [("obs_pos_noise", "obs_yaw_noise", v) for v in (0.0, 1.0, 2.0, 4.0, 8.0)],
    "latency (50 ms steps)": [("obs_latency_steps", None, v) for v in ((0, 0), (0, 2), (2, 4), (4, 6), (6, 8))],
    "dropout (fraction of frames)": [("obs_dropout", None, v) for v in (0.0, 0.03, 0.10, 0.25, 0.50)],
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, base, _ = load_policy(args.checkpoint, device)
    rows = []
    for title, settings in SWEEPS.items():
        print(f"\n{title}")
        print(f"  {'setting':>14s} {'success':>8s} {'pos mm':>8s} {'yaw deg':>8s} {'E mean':>7s} {'failures':>9s}")
        for a, b, v in settings:
            cfg = copy.deepcopy(base)
            if a == "obs_pos_noise":
                cfg.rand.obs_pos_noise, cfg.rand.obs_yaw_noise = v / 1000, math.radians(v)
                label = f"{v:.0f} mm/{v:.0f} deg"
            elif a == "obs_latency_steps":
                cfg.rand.obs_latency_steps = v
                label = f"{v[0] * 50}-{v[1] * 50} ms"
            else:
                cfg.rand.obs_dropout = v
                label = f"{v:.0%}"
            r = run_episodes(cfg, policy_agent(policy, device), args.episodes, args.seed)
            rows.append({"sweep": title, "setting": label, **r})
            print(f"  {label:>14s} {r['success']:8.1%} {r['pos_err_mm_median']:8.1f} {r['yaw_err_deg_median']:8.1f} "
                  f"{r['E_mean']:7.2f} {r['failure']:9.1%}", flush=True)
    out = Path(args.checkpoint).with_name(f"sensitivity_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
