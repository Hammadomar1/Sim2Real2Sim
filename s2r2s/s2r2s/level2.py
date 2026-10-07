"""Level 2: two blocks, the T and the 40 mm box, each pushed to its own goal pose.

    python -m s2r2s.level2                              # the planner: two pushes, or three if a block must be parked
    python -m s2r2s.level2 --no-park                    # only scenes that two pushes can do
    python -m s2r2s.level2 --order tee                  # always the T first (no planner)
    python -m s2r2s.level2 --episodes 4 --video videos/level2.mp4
    python -m s2r2s.level2 --view                       # watch it in the MuJoCo viewer

Each block starts in the work band and gets its own goal by the Milestone 1 rules (up to 12 cm away, any
rotation). Two learned skills do the pushing, each a clutter policy: push one block to a target while the other
stays where it is (``--tee``: the T, keeping the box; ``--box``: the box, keeping the T). Every push must satisfy
the skills' own scene rule: the block that stays put is at least ``clutter_min_gap`` (15 mm) from the straight
path of the one being pushed.

A scene places the blocks at least that gap apart at the start and at the goal. A plan is a list of pushes. Two
pushes (one block, then the other) work for 83 % of random scenes; the planner picks the order with more room.
Otherwise it looks for a temporary spot: push block A there, push B to its goal, then push A from the spot to its
goal (three pushes, each within the skills' 12 cm and 15 mm rules). That rescues few scenes (1 % of all): in the
narrow band the paths usually cross whichever block goes first, with no spot within reach. Scenes without a plan
(16 %) are left out (``scene_stats``).

Physics runs in one world with both blocks as free bodies. Each skill sees that world through its own twin
environment, exactly as on the real arm (``hardware/runner.py``): measured joints and camera estimates in, joint
targets out. The twin's controller guards the block that must stay put, so the rod guard follows the push. The
next push starts once the current block has stayed within 8 mm / 8 deg of its target for 1 s (by the camera),
or after 20 s.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .clutter import ClutterEnv
from .env import EnvConfig, PushEnv, wrap
from .evaluate import TEST_SEED
from .objects import OBJECTS
from .tasks import make_env
from .train import load_policy

TEE, BOX = 0, 1                       # which block is being pushed
PUSH_STEPS = 400                      # 20 s per push at most
SETTLE_STEPS = 20                     # 1 s within 8 mm / 8 deg ends a push


class Level2Env(ClutterEnv):
    """The world: the T is the env's object, the box its clutter block; both have goals and a push plan.

    Joint targets come from outside (``external_q``, set by the skills each step); rewards are not used.
    ``parking``: also accept scenes that need a block parked first (three pushes).
    """

    def __init__(self, cfg: EnvConfig, parking: bool = False):
        n = cfg.num_envs
        self.parking = parking
        self.box_goal = np.zeros((n, 3))
        self.plan_who = np.full((n, 3), -1, dtype=np.int64)    # block pushed by each step of the plan
        self.plan_goal = np.zeros((n, 3, 3))                 # target pose of each push
        self.plan_len = np.zeros(n, dtype=np.int64)
        self.plan_clearance = np.zeros(n)                    # the plan's smallest clearance to the block that stays
        self.external_q = None
        super().__init__(cfg)
        self.external_q = self.qarm.copy()

    @property
    def order(self):
        return self.plan_who[:, 0]

    # ------------------------------------------------------------- scenes and plans
    def _sweep(self, moving, start, goal, fixed, steps: int = 33):
        """Smallest clearance between a block moved straight from ``start`` to ``goal`` and the other at ``fixed``."""
        n = len(start)
        f = np.linspace(0, 1, steps)[:, None, None]
        turn = wrap(goal[:, 2] - start[:, 2])
        mid = np.concatenate([start[None, :, :2] + f * (goal[None, :, :2] - start[None, :, :2]),
                              start[None, :, 2:] + f * turn[None, :, None]], -1).reshape(-1, 3)
        other = np.tile(fixed, (steps, 1))
        zeros = np.zeros(n * steps, dtype=np.int64)
        gap = self._gap_to_object(other, mid, zeros) if moving == TEE else self._gap_to_object(mid, other, zeros)
        return gap.reshape(steps, n).min(0)

    def order_clearances(self, t_start, t_goal, b_start, b_goal):
        """For each order (T first, box first), the smallest clearance its two pushes leave to the block that stays."""
        tee_first = np.minimum(self._sweep(TEE, t_start, t_goal, b_start), self._sweep(BOX, b_start, b_goal, t_goal))
        box_first = np.minimum(self._sweep(BOX, b_start, b_goal, t_start), self._sweep(TEE, t_start, t_goal, b_goal))
        return np.stack([tee_first, box_first], -1)

    def park_plans(self, t_start, t_goal, b_start, b_goal, k: int = 24):
        """Best three-push plan per scene: park block A at a spot, push B to its goal, then A to its goal.

        Returns (clearance (n,), parked block (n,), spot (n, 3)); clearance -1 when no candidate spot works.
        """
        t = self.cfg.task
        n = len(t_start)
        best = np.full(n, -1.0)
        who = np.zeros(n, dtype=np.int64)
        spot = np.zeros((n, 3))
        for parked in (TEE, BOX):
            a_start, a_goal, b0, b1 = (t_start, t_goal, b_start, b_goal) if parked == TEE else (b_start, b_goal, t_start, t_goal)
            rows = np.repeat(np.arange(n), k)
            cand = np.c_[self._sample_region(n * k, t.obj_r, t.obj_az), self.rng.uniform(-math.pi, math.pi, n * k)]
            ok = (np.linalg.norm(cand[:, :2] - a_start[rows, :2], axis=-1) <= t.goal_shift[1]) & \
                 (np.linalg.norm(a_goal[rows, :2] - cand[:, :2], axis=-1) <= t.goal_shift[1])
            c = np.full(n * k, -1.0)
            if ok.any():
                r = rows[ok]
                c1 = self._sweep(parked, a_start[r], cand[ok], b0[r])                 # A to the spot, B at its start
                c2 = self._sweep(1 - parked, b0[r], b1[r], cand[ok])                  # B to its goal, A parked
                c3 = self._sweep(parked, cand[ok], a_goal[r], b1[r])                  # A to its goal, B placed
                c[ok] = np.minimum(np.minimum(c1, c2), c3)
            c = c.reshape(n, k)
            j = c.argmax(1)
            better = c[np.arange(n), j] > best
            best = np.where(better, c[np.arange(n), j], best)
            who = np.where(better, parked, who)
            spot = np.where(better[:, None], cand.reshape(n, k, 3)[np.arange(n), j], spot)
        return np.where(best > t.clutter_min_gap, best, -1.0), who, spot

    def _sample_box(self, k, difficulty):
        """Box start and goal candidates by the Milestone 1 rules."""
        t = self.cfg.task
        start = np.c_[self._sample_region(k, t.obj_r, t.obj_az), self.rng.uniform(-math.pi, math.pi, k)]
        shift = t.goal_shift[0] + difficulty * (t.goal_shift[1] - t.goal_shift[0])
        turn = t.goal_turn[0] + difficulty * (t.goal_turn[1] - t.goal_turn[0])
        ang = self.rng.uniform(0, 2 * math.pi, k)
        dist = shift * np.sqrt(self.rng.random(k))
        goal = np.c_[start[:, :2] + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1),
                     wrap(start[:, 2] + self.rng.uniform(-1, 1, k) * turn)]
        return start, goal

    def _scene_candidates(self, t_start, t_goal, difficulty, m: int = 16):
        """A box start and goal for each T scene: the first of ``m`` draws that is a valid two-block scene
        (box goal in the band; the blocks at least clutter_min_gap apart at the start and at the goal).
        Returns (found (k,), box start, box goal)."""
        t = self.cfg.task
        k = len(t_start)
        rows = np.repeat(np.arange(k), m)
        bs, bg = self._sample_box(k * m, difficulty[rows])
        zeros = np.zeros(k * m, dtype=np.int64)
        ok = self._in_region(bg[:, :2], t.obj_r, t.obj_az)
        ok &= self._gap_to_object(bs, t_start[rows], zeros) > t.clutter_min_gap
        ok &= self._gap_to_object(bg, t_goal[rows], zeros) > t.clutter_min_gap
        ok = ok.reshape(k, m)
        pick = np.arange(k) * m + ok.argmax(1)
        return ok.any(1), bs[pick], bg[pick]

    def plan(self, t_start, t_goal, b_start, b_goal):
        """Push plans for scenes: two pushes in the order with more room, else (with parking) three, else none.

        Returns (length (k,): 0 = no plan, who (k, 3), targets (k, 3, 3), clearance (k,))."""
        t = self.cfg.task
        k = len(t_start)
        length, who, goal, clear = np.zeros(k, dtype=np.int64), np.full((k, 3), -1), np.zeros((k, 3, 3)), np.zeros(k)
        c = self.order_clearances(t_start, t_goal, b_start, b_goal)
        two = (c > t.clutter_min_gap).any(1)
        first = np.where(c[:, 0] >= c[:, 1], TEE, BOX)
        goals = np.stack([t_goal, b_goal], 1)                                     # (k, 2, 3): T, box
        r = np.arange(k)
        length[two], clear[two] = 2, c[two].max(1)
        who[two, 0], who[two, 1] = first[two], 1 - first[two]
        goal[two, 0], goal[two, 1] = goals[r, first][two], goals[r, 1 - first][two]
        if self.parking and (~two).any():
            idx = np.flatnonzero(~two)
            pc, parked, spot = self.park_plans(t_start[idx], t_goal[idx], b_start[idx], b_goal[idx])
            good, p = idx[pc > 0], parked[pc > 0]
            length[good], clear[good] = 3, pc[pc > 0]
            who[good] = np.c_[p, 1 - p, p]
            goal[good, 0] = spot[pc > 0]
            goal[good, 1] = goals[good, 1 - p]
            goal[good, 2] = goals[good, p]
        return length, who, goal, clear

    def _sample_scene(self, ids):
        """Random two-block scenes, redrawn until a plan exists (two pushes, or three with parking)."""
        t = self.cfg.task
        n = len(ids)
        start, start_yaw = np.zeros((n, 2)), np.zeros(n)
        b_start, b_goal = np.zeros((n, 3)), np.zeros((n, 3))
        todo = np.arange(n)
        for _ in range(100):
            if len(todo) == 0:
                break
            start[todo], start_yaw[todo] = PushEnv._sample_scene(self, ids[todo])     # the T and its goal
            t_start, t_goal = np.c_[start[todo], start_yaw[todo]], self.goal[ids[todo]]
            found, bs, bg = self._scene_candidates(t_start, t_goal, self.difficulty_ep[ids[todo]])
            sub = np.flatnonzero(found)
            length, who, goal, clear = self.plan(t_start[sub], t_goal[sub], bs[sub], bg[sub])
            ok = sub[length > 0]
            w = ids[todo[ok]]
            b_start[todo[ok]], b_goal[todo[ok]] = bs[ok], bg[ok]
            self.plan_len[w], self.plan_who[w] = length[length > 0], who[length > 0]
            self.plan_goal[w], self.plan_clearance[w] = goal[length > 0], clear[length > 0]
            todo = np.delete(todo, ok)
        if len(todo):
            raise RuntimeError(f"no two-block scene with a plan for {len(todo)} worlds")
        self.clutter_home[ids] = b_start                 # where ClutterEnv._place puts the box
        self.box_goal[ids] = b_goal
        self.sweep_gap[ids] = self.plan_clearance[ids]
        self.in_way[ids] = self.plan_clearance[ids] < t.clutter_near
        return start, start_yaw

    # ------------------------------------------------------------- control and bookkeeping
    def command(self, action):
        """Joint targets come from the skills (``external_q``); the action only feeds the smoothness bookkeeping."""
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0), self.external_q

    def _reward_extra(self):
        return 0.0

    def box_errors(self, ids=None):
        """Box position (m) and symmetry-aware yaw (rad) errors to its goal."""
        ids = np.arange(self.n) if ids is None else ids
        pose, goal = self.clutter_pose()[ids], self.box_goal[ids]
        return np.linalg.norm(pose[:, :2] - goal[:, :2], axis=-1), self._clutter_yaw_err(pose[:, 2], goal[:, 2])

    def _episode_extras(self, ids):
        pos, yaw = self.box_errors(ids)
        box = self.clutter_pose()[ids]
        return {"box_pos_err": pos, "box_yaw_err": yaw,
                "box_outside": ~self._in_region(box[:, :2], self.cfg.task.fail_r, self.cfg.task.fail_az),
                "tee_final": self.object_pose()[ids], "box_final": box}

    def _observe(self, subset=None):
        # The world is never observed by a policy (the skills observe their twins); keep resets and steps cheap.
        k = self.n if subset is None else len(subset)
        return {"actor": np.zeros((k, 1), np.float32), "critic": np.zeros((k, 1), np.float32)}


class Skill:
    """A clutter policy acting through its own twin environment (same layout as its training environment).

    ``layout``: the checkpoint whose environment defines the inputs (a clutter policy). ``policy``: the
    weights to run, by default the same checkpoint; a policy trained without the block inputs gets zero
    weights for them (blind to the other block).
    """

    def __init__(self, layout: str, n: int, device: str, policy: str | None = None):
        _, cfg, ck = load_policy(layout, device)
        self.policy, _, _ = load_policy(policy or layout, device, dims=ck["dims"])
        cfg = copy.deepcopy(cfg)
        cfg.num_envs, cfg.num_threads, cfg.seed = n, 1, 0
        cfg.rand.enabled = False                 # measurements come from the world (camera model, joints)
        self.twin: ClutterEnv = make_env(cfg)
        self.device = device

    def start(self, ids, qarm, pose, block, goal, tool_z):
        """Begin this skill's push in worlds ``ids``: the block at ``block`` must stay there."""
        tw = self.twin
        tip, _ = tw.kin.forward(qarm)
        tw.goal[ids], tw.tool_z_cmd[ids] = goal, tool_z
        tw.cmd_xy[ids], tw.q_cmd[ids], tw.prev_action[ids] = tip[:, :2], qarm, 0.0
        tw.obs_pose[ids] = tw.prev_obs_pose[ids] = pose
        tw.pose_hist[ids] = pose[:, None, :]
        tw.clutter_obs[ids] = tw.clutter_home[ids] = block
        tw.clutter_hist[ids] = block[:, None, :]
        tw.step_count[ids] = 0

    @torch.no_grad()
    def act(self, qarm, pose, block):
        """Joint targets for every world, from measured joints and camera estimates (object, block)."""
        tw = self.twin
        tw.qarm = qarm.copy()
        tw.tip, _ = tw.kin.forward(qarm)
        tw.prev_obs_pose, tw.obs_pose = tw.obs_pose.copy(), pose.copy()
        tw.clutter_obs = block.copy()
        x = torch.as_tensor(tw._observe()["actor"], device=self.device)
        a = self.policy.act_deterministic(x).clamp(-1, 1).cpu().numpy()
        a, q = tw.command(a)
        tw.prev_action = a
        return q


def _yaw_err(yaw, goal, symmetry):
    period = 2 * math.pi / symmetry if symmetry > 0 else 2 * math.pi
    return np.abs((yaw - goal + period / 2) % period - period / 2) if symmetry != 0 else np.zeros_like(yaw)


def run_level2(world_cfg: EnvConfig, tee: Skill, box: Skill, episodes: int, seed: int, order: str = "auto",
               parking: bool = False, frames=None) -> dict:
    """One two-block episode in each of ``episodes`` worlds, following each world's push plan.

    ``order``: auto (the planner's plans), or tee / box: two pushes in that fixed order (no parking).
    ``frames(world, push, active, plan_len, over)``: optional callback after every step (videos, viewer).
    """
    cfg = copy.deepcopy(world_cfg)
    cfg.num_envs, cfg.seed = episodes, seed
    cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
    world = Level2Env(cfg, parking=parking and order == "auto")
    n, t = world.n, cfg.task
    if order != "auto":                                  # fixed order, whatever the scene allows
        first = TEE if order == "tee" else BOX
        world.plan_who[:] = [first, 1 - first, -1]
        goals = np.stack([world.goal, world.box_goal], 1)
        world.plan_goal[:, 0], world.plan_goal[:, 1] = goals[:, first], goals[:, 1 - first]
        world.plan_len[:] = 2
    sym = {TEE: OBJECTS[t.objects[0]].symmetry, BOX: OBJECTS[t.clutter].symmetry}
    skills = {TEE: tee, BOX: box}
    # Scene facts, kept before a finished world is reset (inside step) to a new scene.
    plan_who, plan_goal, plan_len = world.plan_who.copy(), world.plan_goal.copy(), world.plan_len.copy()
    tee0, box0 = world.object_pose().copy(), world.clutter_pose().copy()
    clear = world.order_clearances(tee0, world.goal, box0, world.box_goal)
    push = np.zeros(n, dtype=np.int64)
    settled = np.zeros(n, dtype=np.int64)
    push_start = np.zeros(n, dtype=np.int64)
    pushes_done = np.zeros(n, dtype=np.int64)            # pushes that ended settled at their target
    over = np.zeros(n, dtype=bool)                       # the last push has ended
    finish_step = np.full(n, -1)                         # when the last push settled
    kept_at_start = np.zeros((n, 3))
    kept_moved = np.zeros(n)                             # largest move of a block that had to stay (m)
    leaning = np.zeros(n)                                # steps the rod guard had to deflect the command

    def begin(ids):
        """Start push ``push[ids]`` with its skill; the other block must stay where it is now."""
        who = plan_who[ids, push[ids]]
        for w in (TEE, BOX):
            sub = ids[who == w]
            if len(sub) == 0:
                continue
            pose, block = (world.obs_pose, world.clutter_obs) if w == TEE else (world.clutter_obs, world.obs_pose)
            skills[w].start(sub, world.qarm[sub], pose[sub], block[sub], plan_goal[sub, push[sub]], world.tool_z_cmd[sub])
        kept_at_start[ids] = np.where((who == TEE)[:, None], world.clutter_pose()[ids], world.object_pose()[ids])
        push_start[ids] = world.step_count[ids]

    begin(np.arange(n))
    done_once = np.zeros(n, dtype=bool)
    rows = [None] * n
    for _ in range(world.max_steps):
        active = plan_who[np.arange(n), push]
        q_tee = tee.act(world.qarm, world.obs_pose, world.clutter_obs)
        q_box = box.act(world.qarm, world.clutter_obs, world.obs_pose)
        world.external_q = np.where((active == TEE)[:, None], q_tee, q_box)
        leaning += np.where(active == TEE, tee.twin.cmd_deflection, box.twin.cmd_deflection) > 1e-4
        _, _, _, info = world.step(np.zeros((n, 2)))
        if "episode" in info:
            ids = info["terminal_ids"]
            for j, i in enumerate(ids):
                if done_once[i]:
                    continue
                ep = {k: np.asarray(v)[j] for k, v in info["episode"].items()}
                tee_ok = ep["pos_err"] <= t.success_pos and ep["yaw_err"] <= t.success_yaw
                box_ok = ep["box_pos_err"] <= t.success_pos and ep["box_yaw_err"] <= t.success_yaw
                failed = bool(ep["failed"] or ep["box_outside"])
                kept_final = ep["box_final"] if active[i] == TEE else ep["tee_final"]
                moved = max(kept_moved[i], float(np.linalg.norm(kept_final[:2] - kept_at_start[i, :2])))
                first = int(plan_who[i, 0])
                rows[i] = dict(success=bool(tee_ok and box_ok and not failed), tee_ok=bool(tee_ok), box_ok=bool(box_ok),
                               failed=failed, pushes=int(plan_len[i]), order="tee" if first == TEE else "box",
                               pushes_done=int(pushes_done[i]), kept_moved_mm=moved * 1e3, guard_s=float(leaning[i] * t.control_dt),
                               order_valid=bool(plan_len[i] == 3 or clear[i, first] > t.clutter_min_gap),
                               tee_pos_mm=float(ep["pos_err"] * 1e3), tee_yaw_deg=float(np.degrees(ep["yaw_err"])),
                               box_pos_mm=float(ep["box_pos_err"] * 1e3), box_yaw_deg=float(np.degrees(ep["box_yaw_err"])),
                               time_s=float(finish_step[i] * t.control_dt) if finish_step[i] >= 0 else float("nan"))
                done_once[i] = True
            if done_once.all():
                break
        # Sequencer: once the pushed block has settled at its target (by the camera), start the next push.
        est = np.where((active == TEE)[:, None], world.obs_pose, world.clutter_obs)
        goal = plan_goal[np.arange(n), push]
        pos = np.linalg.norm(est[:, :2] - goal[:, :2], axis=-1)
        yaw = np.where(active == TEE, _yaw_err(est[:, 2], goal[:, 2], sym[TEE]), _yaw_err(est[:, 2], goal[:, 2], sym[BOX]))
        settled = np.where((pos < 0.008) & (yaw < math.radians(8)), settled + 1, 0)
        finished = ~done_once & ~over & ((settled == SETTLE_STEPS) | (world.step_count - push_start == PUSH_STEPS))
        if finished.any():
            ids = np.flatnonzero(finished)
            ok = settled[ids] >= SETTLE_STEPS
            pushes_done[ids] += ok
            kept_now = np.where((active[ids] == TEE)[:, None], world.clutter_pose()[ids], world.object_pose()[ids])
            kept_moved[ids] = np.maximum(kept_moved[ids], np.linalg.norm(kept_now[:, :2] - kept_at_start[ids, :2], axis=-1))
            last = push[ids] == plan_len[ids] - 1
            over[ids[last]] = True                       # the last skill keeps holding until the episode ends
            finish_step[ids[last & ok]] = world.step_count[ids[last & ok]]
            nxt = ids[~last]
            if len(nxt):
                push[nxt] += 1
                settled[nxt] = 0
                begin(nxt)
        if frames is not None:
            frames(world, push, plan_who[np.arange(n), push], plan_len, over)
    return summarize_level2(rows)


def summarize_level2(rows) -> dict:
    col = lambda k: np.array([r[k] for r in rows])
    ok, pushes = col("success"), col("pushes")
    two, three = pushes == 2, pushes == 3
    rate = lambda m: float(ok[m].mean()) if m.any() else float("nan")
    out = {"episodes": len(rows), "success": float(ok.mean()), "tee_at_goal": float(col("tee_ok").mean()),
           "box_at_goal": float(col("box_ok").mean()), "failure": float(col("failed").mean()),
           "all_pushes_settled": float((col("pushes_done") == pushes).mean()),
           "kept_block_disturbed": float((col("kept_moved_mm") > 10).mean()),
           "kept_block_moved_mm_median": float(np.median(col("kept_moved_mm"))),
           "time_s_median": float(np.nanmedian(col("time_s"))) if np.isfinite(col("time_s")).any() else float("nan"),
           "three_push_share": float(three.mean()), "success_two_pushes": rate(two), "success_three_pushes": rate(three),
           "tee_first_share": float((col("order")[two] == "tee").mean()) if two.any() else float("nan"),
           "success_tee_first": rate(two & (col("order") == "tee")), "success_box_first": rate(two & (col("order") == "box")),
           "order_valid": float(col("order_valid").mean()),
           "tee_err_median": (float(np.median(col("tee_pos_mm"))), float(np.median(col("tee_yaw_deg")))),
           "box_err_median": (float(np.median(col("box_pos_mm"))), float(np.median(col("box_yaw_deg"))))}
    out["per_episode"] = rows
    return out


def scene_stats(world_cfg: EnvConfig, k: int = 400000, seed: int = 1) -> dict:
    """What random two-block scenes need: two pushes in some order, three (park a block first), or neither.

    Both blocks get a start and a goal by the Milestone 1 rules; scenes where a goal leaves the band, or where
    the blocks start or end closer than clutter_min_gap, are not counted.
    """
    cfg = copy.deepcopy(world_cfg)
    cfg.num_envs, cfg.seed = 4, seed
    world = Level2Env(cfg)
    t = cfg.task
    world.rng = np.random.default_rng(seed)
    t_start, t_goal = world._sample_box(k, np.ones(k))           # the same start and goal rules for the T
    b_start, b_goal = world._sample_box(k, np.ones(k))
    zeros = np.zeros(k, dtype=np.int64)
    ok = (world._in_region(t_goal[:, :2], t.obj_r, t.obj_az) & world._in_region(b_goal[:, :2], t.obj_r, t.obj_az)
          & (world._gap_to_object(b_start, t_start, zeros) > t.clutter_min_gap)
          & (world._gap_to_object(b_goal, t_goal, zeros) > t.clutter_min_gap))
    world.parking = True
    length = world.plan(t_start[ok], t_goal[ok], b_start[ok], b_goal[ok])[0]
    return {"scenes": int(ok.sum()), "two_pushes": float((length == 2).mean()),
            "three_pushes_only": float((length == 3).mean()), "no_plan": float((length == 0).mean())}


def _video(path, world_cfg, tee, box, episodes, seed, order, parking, size=480):
    import cv2
    import imageio.v2 as imageio
    import mujoco
    from .scene import set_marker
    from .visual import SceneMirror
    grid = int(math.ceil(math.sqrt(episodes)))
    state = {"mirrors": None, "renderers": None, "frames": [], "done": np.zeros(episodes, dtype=bool)}

    def frame(world, push, active, plan_len, over):
        if state["mirrors"] is None:
            state["mirrors"] = [SceneMirror(world, i, show_estimate=False) for i in range(episodes)]
            for m in state["mirrors"]:                   # the box orange, its goal a light orange outline
                home = m.model.body("clutter_home").id
                for g in range(m.model.ngeom):
                    if m.model.geom(g).name.startswith("clutter_"):
                        m.model.geom_rgba[g] = [0.95, 0.55, 0.15, 1.0]
                    elif m.model.geom_bodyid[g] == home:
                        m.model.geom_rgba[g] = [0.95, 0.65, 0.3, 0.5]
            state["renderers"] = [mujoco.Renderer(m.model, size, size) for m in state["mirrors"]]
        state["done"] |= world.step_count[:episodes] == 0       # finished (and reset to a new scene): dim it
        canvas = []
        for i, (m, r) in enumerate(zip(state["mirrors"], state["renderers"])):
            m.sync()
            set_marker(m.model, m.data, "clutter_home", world.box_goal[i])
            mujoco.mj_forward(m.model, m.data)
            r.update_scene(m.data, camera="front")
            img = np.ascontiguousarray(r.render())
            if state["done"][i]:
                img = (img * 0.35).astype(np.uint8)
            tp = world.object_pose()[i]
            te = np.linalg.norm(tp[:2] - world.goal[i, :2]) * 1e3
            ty = math.degrees(_yaw_err(tp[2], world.goal[i, 2], 1))
            bpos, byaw = (v[0] for v in world.box_errors(np.array([i])))
            ok = te <= 10 and ty <= 10 and bpos <= 0.01 and byaw <= math.radians(10)
            parked = plan_len[i] == 3 and push[i] == 0
            what = ("park the " if parked else "") + ("T" if active[i] == TEE else "box")
            cv2.rectangle(img, (0, 0), (img.shape[1], 50), (25, 25, 25), -1)
            cv2.putText(img, f"t {world.step_count[i] * world.cfg.task.control_dt:4.1f}s  push {push[i] + 1}/{plan_len[i]}: {what}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"T {te:4.1f} mm {ty:4.1f} deg  box {bpos * 1e3:4.1f} mm {math.degrees(byaw):4.1f} deg"
                        f"{'  BOTH AT GOAL' if ok else ''}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            canvas.append(img)
        while len(canvas) < grid * grid:
            canvas.append(np.zeros_like(canvas[0]))
        state["frames"].append(np.concatenate([np.concatenate(canvas[k * grid:(k + 1) * grid], 1) for k in range(grid)], 0))

    result = run_level2(world_cfg, tee, box, episodes, seed, order, parking, frames=frame)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, state["frames"], fps=round(1 / world_cfg.task.control_dt), quality=7, macro_block_size=8)
    print(f"wrote {out} ({len(state['frames'])} frames)")
    return result


def _view(world_cfg, tee_args, box_args, seed, park):
    """Watch scenes one after another in the MuJoCo viewer, in real time (close the window to stop)."""
    import time
    import mujoco
    import mujoco.viewer
    from .scene import set_marker
    from .visual import SceneMirror
    tee, box = Skill(*tee_args[:1], 1, "cpu", tee_args[1]), Skill(*box_args[:1], 1, "cpu", box_args[1])
    state = {"viewer": None, "mirror": None, "t": time.time(), "after": 0}

    class NextScene(Exception):
        pass

    def frame(world, push, active, plan_len, over):
        if state["mirror"] is None:
            state["mirror"] = m = SceneMirror(world, 0, show_estimate=False)
            home = m.model.body("clutter_home").id
            for g in range(m.model.ngeom):
                if m.model.geom(g).name.startswith("clutter_"):
                    m.model.geom_rgba[g] = [0.95, 0.55, 0.15, 1.0]
                elif m.model.geom_bodyid[g] == home:
                    m.model.geom_rgba[g] = [0.95, 0.65, 0.3, 0.5]
            state["viewer"] = v = mujoco.viewer.launch_passive(m.model, m.data)
            v.cam.lookat[:] = [0.19, 0.0, 0.02]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.62, 200.0, -42.0
        m, v = state["mirror"], state["viewer"]
        if not v.is_running():
            raise KeyboardInterrupt
        m.env = world                             # every scene has the same layout: reuse the window
        m.sync()
        set_marker(m.model, m.data, "clutter_home", world.box_goal[0])
        mujoco.mj_forward(m.model, m.data)
        tp = world.object_pose()[0]
        te, ty = np.linalg.norm(tp[:2] - world.goal[0, :2]) * 1e3, math.degrees(_yaw_err(tp[2], world.goal[0, 2], 1))
        bpos, byaw = (v_[0] for v_ in world.box_errors(np.array([0])))
        what = ("park the " if plan_len[0] == 3 and push[0] == 0 else "") + ("T" if active[0] == TEE else "box")
        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     f"scene {seed - first_seed + 1}, push {push[0] + 1}/{plan_len[0]}: {what}",
                     f"T {te:.1f} mm {ty:.1f} deg   box {bpos * 1e3:.1f} mm {math.degrees(byaw):.1f} deg"))
        v.sync()
        time.sleep(max(0.0, world.cfg.task.control_dt - (time.time() - state["t"])))
        state["t"] = time.time()
        state["after"] = state["after"] + 1 if over[0] else 0
        if state["after"] > 40:                   # 2 s after the last push: next scene
            raise NextScene

    first_seed = seed
    try:
        while True:
            try:
                run_level2(world_cfg, tee, box, 1, seed, "auto", park, frames=frame)
            except NextScene:
                pass
            state["after"] = 0
            seed += 1
    except KeyboardInterrupt:
        pass
    finally:
        if state["viewer"] is not None:
            state["viewer"].close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tee", default="runs/clutter_v7/best.pt", help="skill: push the T, keep the box in place")
    p.add_argument("--box", default="runs/clutter_box_v2/best.pt", help="skill: push the box, keep the T in place")
    p.add_argument("--tee-policy", default="", help="other weights in the T skill's layout, e.g. runs/tee_v1/best.pt "
                                                   "(blind to the box)")
    p.add_argument("--box-policy", default="", help="other weights in the box skill's layout, e.g. runs/box_v1/best.pt")
    p.add_argument("--no-park", action="store_true", help="only scenes that two pushes can do (no parking plans)")
    p.add_argument("--order", choices=["auto", "tee", "box"], default="auto",
                   help="auto: the planner picks; tee / box: always two pushes in that order")
    p.add_argument("--no-guard", action="store_true", help="switch the rod guard off in both skills")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--video", default="", help="render the episodes (use a few, e.g. --episodes 4) to this MP4")
    p.add_argument("--view", action="store_true", help="watch scenes one after another in the MuJoCo viewer")
    p.add_argument("--stats", action="store_true", help="also report what random scenes need (two pushes, three, none)")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.episodes
    if args.view:
        _, world_cfg, _ = load_policy(args.tee, "cpu")
        park = not args.no_park
        world_cfg.task.episode_seconds = 60.0 if park else 40.0
        return _view(world_cfg, (args.tee, args.tee_policy or None), (args.box, args.box_policy or None), args.seed, park)
    tee = Skill(args.tee, n, device, args.tee_policy or None)
    box = Skill(args.box, n, device, args.box_policy or None)
    if args.no_guard:
        tee.twin.cfg.task.rod_guard = box.twin.cfg.task.rod_guard = 0.0
    _, world_cfg, _ = load_policy(args.tee, "cpu")           # camera model and randomisation of the T skill
    park = not args.no_park and args.order == "auto"
    world_cfg.task.episode_seconds = 60.0 if park else 40.0              # 20 s per push at most
    run = _video if args.video else (lambda *a: run_level2(*a[1:]))
    r = run(args.video, world_cfg, tee, box, n, args.seed, args.order, park)
    print(f"\nLevel 2: {n} two-block scenes (seed {args.seed}), {'with' if park else 'without'} parking, order "
          f"{args.order}, rod guard {'off' if args.no_guard else 'on'} | T skill {args.tee_policy or args.tee} | box skill "
          f"{args.box_policy or args.box}")
    rows = [("success", "both blocks at their goals", "{:.1%}"), ("tee_at_goal", "  T at its goal", "{:.1%}"),
            ("box_at_goal", "  box at its goal", "{:.1%}"), ("all_pushes_settled", "every push settled at its target", "{:.1%}"),
            ("kept_block_disturbed", "a block that had to stay moved >10 mm", "{:.1%}"),
            ("failure", "failures (a block left the band)", "{:.1%}"), ("time_s_median", "time to finish, median (s)", "{:.1f}"),
            ("three_push_share", "scenes with a parking push", "{:.1%}"),
            ("success_two_pushes", "success with two pushes", "{:.1%}"),
            ("success_three_pushes", "success with three pushes", "{:.1%}"),
            ("tee_first_share", "two pushes: T first", "{:.1%}"),
            ("success_tee_first", "  success when the T goes first", "{:.1%}"),
            ("success_box_first", "  success when the box goes first", "{:.1%}"),
            ("order_valid", "plans that keep the 15 mm rule", "{:.1%}")]
    for key, label, fmt in rows:
        print(f"  {label:42s} {fmt.format(r[key])}")
    print(f"  {'final error median, T / box':42s} {r['tee_err_median'][0]:.1f} mm {r['tee_err_median'][1]:.1f} deg / "
          f"{r['box_err_median'][0]:.1f} mm {r['box_err_median'][1]:.1f} deg")
    if args.stats:
        s = scene_stats(world_cfg)
        r["scene_stats"] = s
        print(f"  random scenes ({s['scenes']}): two pushes {s['two_pushes']:.1%}, three pushes only (parking) "
              f"{s['three_pushes_only']:.1%}, no plan {s['no_plan']:.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), **r}, indent=1))
        print(f"per-episode results: {args.out}")


if __name__ == "__main__":
    main()
