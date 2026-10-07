"""Evaluate a trained policy on held-out scenarios (and optionally the scripted baseline).

    python -m s2r2s.evaluate runs/tee_v1/best.pt --episodes 1000 --baseline
    python -m s2r2s.evaluate runs/tee_v1/best.pt --timestep 0.002     # physics-step robustness
    python -m s2r2s.evaluate runs/tee_v1/best.pt --no-randomization   # nominal physics, clean camera

The test seed differs from the seed used for checkpoint selection during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluation import policy_agent, run_episodes, scripted_agent
from .train import load_policy

TEST_SEED = 20_000_003


def apply_overrides(cfg, items):
    """Apply ``section.field=value`` overrides, e.g. ``rand.obs_pos_noise=0.003`` or ``rand.obs_latency_steps=2,4``."""
    for item in items:
        key, value = item.split("=", 1)
        section, field = key.split(".")
        target = getattr(cfg, section)
        old = getattr(target, field)
        parse = type(old[0]) if isinstance(old, tuple) else type(old)
        if parse is bool:
            new = value.lower() in ("1", "true", "yes")
        elif isinstance(old, tuple):
            new = tuple(parse(v) for v in value.split(","))
        else:
            new = parse(value)
        setattr(target, field, new)
    return cfg

ROWS = [("passed_gate", "passed the gate", "{:.1%}"), ("success", "success rate", "{:.1%}"), ("pos_err_mm_median", "position error median (mm)", "{:.1f}"),
        ("pos_err_mm_p90", "position error p90 (mm)", "{:.1f}"), ("yaw_err_deg_median", "yaw error median (deg)", "{:.1f}"),
        ("yaw_err_deg_p90", "yaw error p90 (deg)", "{:.1f}"), ("E_mean", "mean E = pos/20mm + yaw/10deg", "{:.2f}"),
        ("reached", "ever within tolerance", "{:.1%}"), ("time_to_goal_s_median", "time to tolerance median (s)", "{:.1f}"),
        ("failure", "failures (tipped/escaped)", "{:.1%}"),
        ("disturbed", "clutter block disturbed (>10 mm or >10 deg)", "{:.1%}"),
        ("disturbed_when_in_way", "  ... when it was on the path", "{:.1%}"), ("action_rate_mean", "action rate (smoothness, lower=smoother)", "{:.4f}")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--timestep", type=float, default=0.0, help="override the physics timestep")
    p.add_argument("--no-randomization", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--baseline", action="store_true", help="also run the scripted pusher on the same scenes")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE",
                   help="config overrides, e.g. rand.obs_pos_noise=0.003 task.max_speed=0.06")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg, ck = load_policy(args.checkpoint, device)
    if args.timestep:
        cfg.scene.timestep = args.timestep
    if args.no_randomization:
        cfg.rand.enabled = False
    apply_overrides(cfg, args.set)
    results = {"policy": run_episodes(cfg, policy_agent(policy, device, not args.stochastic),
                                      args.episodes, args.seed, args.difficulty, record=True)}
    if args.baseline:
        results["scripted"] = run_episodes(cfg, scripted_agent(), args.episodes, args.seed, args.difficulty, record=True)
    names = list(results)
    print(f"\n{args.checkpoint} (iteration {ck['iteration']}, {ck['samples'] / 1e6:.0f}M samples) | "
          f"{args.episodes} episodes, seed {args.seed}, difficulty {args.difficulty}, timestep "
          f"{cfg.scene.timestep * 1e3:.0f} ms, randomization {'on' if cfg.rand.enabled else 'off'}")
    print(f"{'':44s}" + "".join(f"{n:>12s}" for n in names))
    for key, label, fmt in ROWS:
        if all(key not in results[n] for n in names):
            continue
        print(f"{label:44s}" + "".join(f"{fmt.format(results[n].get(key, float('nan'))):>12s}" for n in names))
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        f"eval_{Path(args.checkpoint).stem}_s{args.seed}_d{args.difficulty}"
        f"{'_dt' + str(args.timestep) if args.timestep else ''}{'_norand' if args.no_randomization else ''}.json")
    out.write_text(json.dumps({"args": vars(args), **results}, indent=1))
    print(f"\nper-episode results: {out}")


if __name__ == "__main__":
    main()
