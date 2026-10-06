"""Train a pushing policy with PPO.

    python -m s2r2s.train --run tee_v1 --minutes 90

Writes runs/<run>/: config.json, progress.jsonl, eval.jsonl, latest.pt,
best.pt and TensorBoard logs (tb/). Resume with --resume runs/<run>/latest.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .env import EnvConfig, PushEnv, TaskConfig
from .evaluation import policy_agent, run_episodes, summarize as episode_summary
from .ppo import PPO, PPOConfig, ActorCritic

PROJECT = Path(__file__).resolve().parents[1]


def collect(eps: dict, info: dict):
    for k, v in info["episode"].items():
        eps.setdefault(k, []).append(np.asarray(v))


def save(path: Path, ppo: PPO, env_cfg: EnvConfig, ppo_cfg: PPOConfig, it, samples, difficulty, best, dims):
    tmp = path.with_suffix(".tmp")
    torch.save({"ppo": ppo.state_dict(), "env_cfg": env_cfg.to_dict(), "ppo_cfg": ppo_cfg.to_dict(),
                "iteration": it, "samples": samples, "difficulty": difficulty, "best": best, "dims": dims}, tmp)
    tmp.replace(path)


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    env_cfg = EnvConfig.from_dict(ck["env_cfg"])
    ppo_cfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["ppo_cfg"].items()})
    n_actor, n_critic, n_act = ck["dims"]
    policy = ActorCritic(n_actor, n_critic, n_act, ppo_cfg).to(device)
    policy.load_state_dict(ck["ppo"]["policy"])
    policy.eval()
    return policy, env_cfg, ck


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--objects", default="tee", help="comma-separated object names")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", type=float, default=0.0, help="initial curriculum difficulty")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-randomization", action="store_true")
    p.add_argument("--timestep", type=float, default=0.005)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=512)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    run_dir = PROJECT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        env_cfg = EnvConfig.from_dict(ck["env_cfg"])
        ppo_cfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["ppo_cfg"].items()})
        env_cfg.seed = args.seed + 1000 * ck["iteration"]
        env_cfg.task.difficulty = ck["difficulty"]
    else:
        env_cfg = EnvConfig(num_envs=args.num_envs, seed=args.seed,
                            task=TaskConfig(objects=tuple(args.objects.split(",")),
                                            difficulty=1.0 if args.no_curriculum else args.difficulty))
        env_cfg.scene.timestep = args.timestep
        env_cfg.rand.enabled = not args.no_randomization
        ppo_cfg = PPOConfig()
    (run_dir / "config.json").write_text(json.dumps({"env": env_cfg.to_dict(), "ppo": ppo_cfg.to_dict(),
                                                    "args": vars(args)}, indent=2))
    env = PushEnv(env_cfg)
    # Desynchronise episode boundaries so resets spread evenly over training.
    env.step_count[:] = env.rng.integers(0, env.max_steps, env.n)
    dims = (env.num_obs, env.num_critic_obs, env.num_actions)
    ppo = PPO(*dims, env.n, ppo_cfg, device)
    it, samples, best = 0, 0, -1.0
    if ck is not None:
        ppo.load_state_dict(ck["ppo"])
        it, samples, best = ck["iteration"], ck["samples"], ck["best"]
    writer = SummaryWriter(str(run_dir / "tb"))
    eval_seed = 10_000_019            # fixed held-out evaluation scenarios
    obs = env.obs
    window = {}
    t_start = time.time()
    last_print = time.time()
    print(f"run {run_dir} | envs {env.n} | actor obs {dims[0]} critic obs {dims[1]} | device {device}", flush=True)
    try:
        while it < args.iterations and (time.time() - t_start) < args.minutes * 60:
            t0 = time.time()
            for _ in range(ppo_cfg.rollout_steps):
                a_obs = torch.as_tensor(obs["actor"], device=device)
                c_obs = torch.as_tensor(obs["critic"], device=device)
                action = ppo.act(a_obs, c_obs)
                obs, reward, done, info = env.step(action.cpu().numpy())
                r = torch.as_tensor(reward, dtype=torch.float32, device=device)
                if "terminal_ids" in info:
                    ids = info["terminal_ids"]
                    trunc = info["time_outs"][ids]
                    if trunc.any():
                        v = ppo.bootstrap_value(torch.as_tensor(info["terminal_critic_obs"][trunc], device=device))
                        r[torch.as_tensor(ids[trunc], device=device)] += ppo_cfg.gamma * v
                    collect(window, info)
                ppo.record(r, torch.as_tensor(done, dtype=torch.float32, device=device))
            t1 = time.time()
            stats = ppo.update(torch.as_tensor(obs["critic"], device=device))
            ppo.observe_normalizers()
            it += 1
            samples += env.n * ppo_cfg.rollout_steps
            t2 = time.time()
            for k, v in stats.items():
                writer.add_scalar(f"ppo/{k}", v, samples)
            writer.add_scalar("perf/samples_per_s", env.n * ppo_cfg.rollout_steps / (t2 - t0), samples)
            writer.add_scalar("perf/collect_s", t1 - t0, samples)
            writer.add_scalar("perf/update_s", t2 - t1, samples)
            if it % 10 == 0:
                summ = episode_summary(window)
                cur = {}
                if summ["episodes"]:
                    # Curriculum: judge only episodes drawn at the current difficulty.
                    s_all = {k: np.concatenate(v) for k, v in window.items()}
                    at_level = s_all["difficulty"] >= env.difficulty - 1e-6
                    cur_success = float(s_all["success"][at_level].mean()) if at_level.any() else 0.0
                    cur = {"level_success": cur_success, "level_episodes": int(at_level.sum())}
                    if not args.no_curriculum and at_level.sum() >= 200 and cur_success >= 0.8 and env.difficulty < 1.0:
                        env.set_difficulty(round(env.difficulty + 0.1, 3))
                row = {"iteration": it, "samples": samples, "minutes": (time.time() - t_start) / 60,
                       "difficulty": env.difficulty, **summ, **cur, **stats}
                with (run_dir / "progress.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                for k, v in row.items():
                    if isinstance(v, (int, float)) and k not in ("iteration",):
                        writer.add_scalar(f"train/{k}", v, samples)
                window = {}
                if time.time() - last_print > 20 or it % 50 == 0:
                    last_print = time.time()
                    print(f"it {it:5d} | {samples / 1e6:7.1f}M samples | {row['minutes']:5.1f} min | diff {env.difficulty:.1f} "
                          f"| succ {summ.get('success', float('nan')):.2f} (level {cur.get('level_success', float('nan')):.2f}) "
                          f"| fail {summ.get('failure', float('nan')):.3f} | pos {summ.get('pos_err_mm_median', float('nan')):5.1f} mm "
                          f"yaw {summ.get('yaw_err_deg_median', float('nan')):5.1f} deg | std {stats['action_std']:.3f} "
                          f"| lr {stats['lr']:.1e} | {env.n * ppo_cfg.rollout_steps / (t2 - t0):6.0f} sps", flush=True)
            if it % args.eval_every == 0:
                ev = run_episodes(env_cfg, policy_agent(ppo.policy, device), args.eval_episodes, eval_seed)
                ev.update(iteration=it, samples=samples, difficulty_train=env.difficulty)
                with (run_dir / "eval.jsonl").open("a") as f:
                    f.write(json.dumps(ev) + "\n")
                for k, v in ev.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"eval/{k}", v, samples)
                score = ev["success"] - 1e-3 * ev["pos_err_mm_median"]
                print(f"   EVAL (full difficulty, {ev['episodes']} fixed scenes): success {ev['success']:.3f} | "
                      f"pos {ev['pos_err_mm_median']:.1f} mm (p90 {ev['pos_err_mm_p90']:.1f}) | yaw {ev['yaw_err_deg_median']:.1f} deg "
                      f"(p90 {ev['yaw_err_deg_p90']:.1f}) | failures {ev['failure']:.3f} | time-to-goal {ev['time_to_goal_s_median']:.1f} s", flush=True)
                if score > best:
                    best = score
                    save(run_dir / "best.pt", ppo, env_cfg, ppo_cfg, it, samples, env.difficulty, best, dims)
                save(run_dir / "latest.pt", ppo, env_cfg, ppo_cfg, it, samples, env.difficulty, best, dims)
    except KeyboardInterrupt:
        print("interrupted - saving latest.pt", flush=True)
    save(run_dir / "latest.pt", ppo, env_cfg, ppo_cfg, it, samples, env.difficulty, best, dims)
    writer.close()
    print(f"done: {it} iterations, {samples / 1e6:.1f}M samples, {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
