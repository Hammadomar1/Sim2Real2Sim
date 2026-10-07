"""Proximal Policy Optimization with an asymmetric actor-critic (PyTorch).

Self-contained so the team can read every line: Gaussian policy, running
observation normalisation, value-target normalisation, GAE with correct
bootstrapping of time-limit truncations, clipped surrogate and value losses,
and a KL-adaptive learning rate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn


@dataclass
class PPOConfig:
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.002
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 3e-3
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (512, 256, 128)
    init_std: float = 0.5
    min_std: float = 0.03

    def to_dict(self):
        return asdict(self)


class RunningNorm(nn.Module):
    """Running mean/variance (parallel Welford), used for observations and value targets.

    The sample count is kept per input, so inputs appended by a warm start (``train.warm_start``) adapt
    to their data at once while the copied inputs keep their statistics.
    """

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.full((dim,), 1e-4))
        self.clip = clip

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "count"
        if key in state_dict and state_dict[key].dim() == 0:      # checkpoints with one count for all inputs
            state_dict[key] = state_dict[key].expand(state_dict[prefix + "mean"].shape).clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, self.mean.shape[0])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean


def mlp(n_in, hidden, n_out, out_gain):
    layers, last = [], n_in
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        last = h
    out = nn.Linear(last, n_out)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class ActorCritic(nn.Module):
    def __init__(self, n_actor: int, n_critic: int, n_act: int, cfg: PPOConfig):
        super().__init__()
        self.actor_norm = RunningNorm(n_actor)
        self.critic_norm = RunningNorm(n_critic)
        self.value_norm = RunningNorm(1)
        self.actor = mlp(n_actor, cfg.hidden, n_act, 0.01)
        self.critic = mlp(n_critic, cfg.hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), math.log(cfg.init_std)))
        self.min_log_std = math.log(cfg.min_std)

    def distribution(self, actor_obs):
        mean = self.actor(self.actor_norm(actor_obs))
        std = self.log_std.clamp(min=self.min_log_std).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, critic_obs):
        """Normalised value prediction."""
        return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, actor_obs):
        return self.actor(self.actor_norm(actor_obs))


class PPO:
    def __init__(self, n_actor, n_critic, n_act, n_envs, cfg: PPOConfig, device):
        self.cfg = cfg
        self.device = device
        self.policy = ActorCritic(n_actor, n_critic, n_act, cfg).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        T, N = cfg.rollout_steps, n_envs
        z = lambda *s: torch.zeros((T, N, *s), device=device)
        self.buf = dict(actor=z(n_actor), critic=z(n_critic), action=z(n_act), logp=z(), value=z(),
                        reward=z(), done=z())
        self.t = 0

    @torch.no_grad()
    def act(self, actor_obs, critic_obs):
        dist = self.policy.distribution(actor_obs)
        action = dist.sample()
        b = self.buf
        b["actor"][self.t] = actor_obs
        b["critic"][self.t] = critic_obs
        b["action"][self.t] = action
        b["logp"][self.t] = dist.log_prob(action).sum(-1)
        b["value"][self.t] = self.policy.value_norm.denormalize(self.policy.value(critic_obs)[:, None])[:, 0]
        return action

    @torch.no_grad()
    def bootstrap_value(self, critic_obs):
        return self.policy.value_norm.denormalize(self.policy.value(critic_obs)[:, None])[:, 0]

    @torch.no_grad()
    def record(self, reward, done):
        self.buf["reward"][self.t] = reward
        self.buf["done"][self.t] = done
        self.t += 1

    @torch.no_grad()
    def observe_normalizers(self):
        self.policy.actor_norm.update(self.buf["actor"])
        self.policy.critic_norm.update(self.buf["critic"])

    def update(self, last_critic_obs):
        cfg, b = self.cfg, self.buf
        with torch.no_grad():
            last_value = self.bootstrap_value(last_critic_obs)
            adv = torch.zeros_like(b["reward"])
            gae = torch.zeros_like(last_value)
            for t in reversed(range(cfg.rollout_steps)):
                next_value = last_value if t == cfg.rollout_steps - 1 else b["value"][t + 1]
                nonterminal = 1.0 - b["done"][t]
                delta = b["reward"][t] + cfg.gamma * next_value * nonterminal - b["value"][t]
                gae = delta + cfg.gamma * cfg.lam * nonterminal * gae
                adv[t] = gae
            ret = adv + b["value"]
            self.policy.value_norm.update(ret)
            ret_n = self.policy.value_norm(ret[..., None])[..., 0]
            old_v_n = self.policy.value_norm(b["value"][..., None])[..., 0]
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        data = dict(actor=flat(b["actor"]), critic=flat(b["critic"]), action=flat(b["action"]),
                    logp=flat(b["logp"]), adv=flat(adv), ret=flat(ret_n), old_v=flat(old_v_n))
        n = data["adv"].shape[0]
        mb = n // cfg.minibatches
        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0, clip_frac=0.0)
        count = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for k in range(cfg.minibatches):
                i = perm[k * mb:(k + 1) * mb]
                a = data["adv"][i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                dist = self.policy.distribution(data["actor"][i])
                logp = dist.log_prob(data["action"][i]).sum(-1)
                log_ratio = logp - data["logp"][i]
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v = self.policy.value(data["critic"][i])
                v_clip = data["old_v"][i] + (v - data["old_v"][i]).clamp(-cfg.clip, cfg.clip)
                vl = torch.max((v - data["ret"][i]) ** 2, (v_clip - data["ret"][i]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy_coef * ent
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    if kl > 2.0 * cfg.desired_kl:
                        self.lr = max(cfg.lr_min, self.lr / 1.5)
                    elif kl < 0.5 * cfg.desired_kl:
                        self.lr = min(cfg.lr_max, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["policy_loss"] += pg.item()
                stats["value_loss"] += vl.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl.item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                count += 1
        self.t = 0
        out = {k: v / count for k, v in stats.items()}
        out["lr"] = self.lr
        out["action_std"] = self.policy.log_std.clamp(min=self.policy.min_log_std).exp().mean().item()
        return out

    def state_dict(self):
        return {"policy": self.policy.state_dict(), "opt": self.opt.state_dict(), "lr": self.lr}

    def load_state_dict(self, sd, load_optimizer=True):
        self.policy.load_state_dict(sd["policy"])
        if load_optimizer:
            self.opt.load_state_dict(sd["opt"])
            self.lr = sd["lr"]
