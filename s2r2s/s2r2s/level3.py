"""Level 3: connect the blocks. Put the T at its goal and nest the box into one of the T's inner corners.

    python -m s2r2s.level3                              # 1000 held-out scenes
    python -m s2r2s.level3 --view                       # watch it in the MuJoCo viewer
    python -m s2r2s.level3 --episodes 4 --video videos/level3.mp4
    python -m s2r2s.level3 --dock-policy runs/clutter_box_v2/best.pt   # dock with the box skill (never trained to)
    python -m s2r2s.level3 --record-dock-starts runs/dock_v5/start_bank.npz   # docking starts, for training the dock skill

The T starts in the work band with a goal (Milestone 1 rules), the box anywhere in the band at least 15 mm from
it. The assembly: the box nested in the corner where the T's bar overhangs its stem, touching both. Done means
the T within 10 mm / 10 deg of its goal and the box within 3 mm / 5 deg of the corner pose of the T's actual
final pose (so the pieces touch).

Plan (``Level3Env``): the planner of Level 2 brings the T to its goal and the box to a pre-dock spot 25 mm out
from the chosen corner along its diagonal (18 mm from both faces, so both pushes keep the skills' 15 mm rule),
in the order with more room (or with a parking push); then the dock skill (``dock.py``, ``runs/dock_v4``) pushes
the box into the corner without moving the T. A corner is chosen where the docked box and the pre-dock spot fit
the band and the rod can get behind the box.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from .dock import compose, corner_offset
from .env import EnvConfig, PushEnv
from .evaluate import TEST_SEED
from .level2 import BOX, DOCK, TEE, Level2Env, Skill, _yaw_err, execute, summarize_level2, video, view
from .train import load_policy

PREDOCK = 0.025                       # pre-dock spot: this far out from the corner along its diagonal (m)


class Level3Env(Level2Env):
    """Two-block world whose plans end with docking the box into a corner of the T."""

    def __init__(self, cfg: EnvConfig, parking: bool = False):
        self.side = np.ones(cfg.num_envs)
        self.dock_target = np.full((cfg.num_envs, 3), np.nan)   # fixed when docking starts
        super().__init__(cfg, parking)
        self.stop_when_done = True          # the dock skill trained on 10 s episodes: stop once the box is seated

    def docked(self, tee_pose, side):
        """The box's pose nested in the corner on ``side`` of a T at ``tee_pose``."""
        return compose(tee_pose, corner_offset(self.cfg.task.objects[0], self.cfg.task.clutter, side))

    def predock(self, tee_pose, side):
        a = tee_pose[:, 2] + np.arctan2(-1.0, side)                       # out of the corner, along its diagonal
        d = self.docked(tee_pose, side)
        return np.c_[d[:, :2] + PREDOCK * np.stack([np.cos(a), np.sin(a)], -1), d[:, 2]], np.stack([np.cos(a), np.sin(a)], -1)

    def _fits(self, tee_goal, side):
        """The docked box and the pre-dock spot fit the band, and the rod can get behind the box there."""
        t = self.cfg.task
        dock = self.docked(tee_goal, side)
        pre, out = self.predock(tee_goal, side)
        rod = pre[:, :2] + (0.028 + self.cfg.scene.pusher_radius + 0.012) * out
        return (self._in_region(dock[:, :2], (t.obj_r[0] - 0.015, t.obj_r[1] + 0.015), t.obj_az + 0.1)
                & self._in_region(pre[:, :2], (t.obj_r[0] - 0.01, t.obj_r[1] + 0.01), t.obj_az + 0.05)
                & self._in_region(rod, (t.tool_r[0] + 0.01, t.tool_r[1] - 0.01), t.tool_az - 0.1))

    def _sample_scene(self, ids):
        t = self.cfg.task
        n = len(ids)
        start, start_yaw = np.zeros((n, 2)), np.zeros(n)
        b_start, pre, side = np.zeros((n, 3)), np.zeros((n, 3)), np.ones(n)
        todo = np.arange(n)
        for _ in range(100):
            if len(todo) == 0:
                break
            start[todo], start_yaw[todo] = PushEnv._sample_scene(self, ids[todo])     # the T and its goal
            t_start, t_goal = np.c_[start[todo], start_yaw[todo]], self.goal[ids[todo]]
            k = len(todo)
            sd = np.where(self.rng.random(k) < 0.5, 1.0, -1.0)
            sd = np.where(self._fits(t_goal, sd), sd, -sd)                # the other corner if this one does not fit
            fits = self._fits(t_goal, sd)
            # The box's start: anywhere in the band, at least the clutter gap from the T's start.
            m = 16
            rows = np.repeat(np.arange(k), m)
            bs = np.c_[self._sample_region(k * m, t.obj_r, t.obj_az), self.rng.uniform(-math.pi, math.pi, k * m)]
            ok = (self._gap_to_object(bs, t_start[rows], np.zeros(k * m, dtype=np.int64)) > t.clutter_min_gap).reshape(k, m)
            found = fits & ok.any(1)
            bs = bs[np.arange(k) * m + ok.argmax(1)]
            p = self.predock(t_goal, sd)[0]
            sub = np.flatnonzero(found)
            length, who, goal, clear = self.plan(t_start[sub], t_goal[sub], bs[sub], p[sub])
            good = sub[length > 0]
            w = ids[todo[good]]
            b_start[todo[good]], pre[todo[good]], side[todo[good]] = bs[good], p[good], sd[good]
            L, W, G = length[length > 0], who[length > 0], goal[length > 0]
            W[np.arange(len(L)), L] = DOCK                                # then dock the box into the corner
            G[np.arange(len(L)), L] = self.docked(t_goal[good], sd[good])
            self.plan_len[w], self.plan_who[w], self.plan_goal[w] = L + 1, W, G
            self.plan_clearance[w] = clear[length > 0]
            todo = np.delete(todo, good)
        if len(todo):
            raise RuntimeError(f"no assembly scene with a plan for {len(todo)} worlds")
        self.clutter_home[ids], self.box_goal[ids], self.side[ids] = b_start, pre, side
        self.dock_target[ids] = np.nan
        self.sweep_gap[ids] = self.plan_clearance[ids]
        self.in_way[ids] = self.plan_clearance[ids] < t.clutter_near
        return start, start_yaw

    # ------------------------------------------------------------- hooks for the plan executor
    def push_target(self, ids, k):
        target = self.plan_goal[ids, k].copy()
        dock = ids[self.plan_who[ids, k] == DOCK]
        # The corner of the T where the camera sees it when docking starts. Fixed from then on: a target that
        # follows the T lets the skill chase it, shoving both blocks along once the box presses on the T.
        new = dock[np.isnan(self.dock_target[dock, 0])]
        self.dock_target[new] = self.docked(self.obs_pose[new], self.side[new])
        target[self.plan_who[ids, k] == DOCK] = self.dock_target[dock]
        return target

    def push_steps(self, who):
        # The dock skill trained on 10 s episodes: give it that long, not the 20 s of a full push.
        return round(10.0 / self.cfg.task.control_dt) if who == DOCK else super().push_steps(who)

    def settle_tol(self, who):
        t = self.cfg.task
        return (t.dock_pos_tol, t.dock_yaw_tol) if who == DOCK else super().settle_tol(who)

    def dock_errors(self, ids=None):
        """The box's position (m) and yaw (rad) errors to the corner of the T where it is now."""
        ids = np.arange(self.n) if ids is None else ids
        box, target = self.clutter_pose()[ids], self.docked(self.object_pose()[ids], self.side[ids])
        return np.linalg.norm(box[:, :2] - target[:, :2], axis=-1), self._clutter_yaw_err(box[:, 2], target[:, 2])

    def _episode_extras(self, ids):
        pos, yaw = self.dock_errors(ids)
        zeros = np.zeros(len(ids), dtype=np.int64)
        gap = self._gap_to_object(self.clutter_pose()[ids], self.object_pose()[ids], zeros)
        return {**super()._episode_extras(ids), "dock_pos_err": pos, "dock_yaw_err": yaw, "gap": gap}

    def episode_row(self, ep):
        t = self.cfg.task
        tee_ok = ep["pos_err"] <= t.success_pos and ep["yaw_err"] <= t.success_yaw
        docked = ep["dock_pos_err"] <= t.dock_pos_tol and ep["dock_yaw_err"] <= t.dock_yaw_tol
        failed = bool(ep["failed"] or ep["box_outside"])
        return dict(success=bool(tee_ok and docked and not failed), tee_ok=bool(tee_ok), box_ok=bool(docked), failed=failed,
                    tee_pos_mm=float(ep["pos_err"] * 1e3), tee_yaw_deg=float(np.degrees(ep["yaw_err"])),
                    box_pos_mm=float(ep["dock_pos_err"] * 1e3), box_yaw_deg=float(np.degrees(ep["dock_yaw_err"])),
                    gap_mm=float(ep["gap"] * 1e3))

    def box_marker(self, i):
        return self.docked(self.goal[i:i + 1], self.side[i:i + 1])[0]      # where the box ends: the T's goal corner

    def status(self, i, push, who, plan_len):
        tp = self.object_pose()[i]
        te = np.linalg.norm(tp[:2] - self.goal[i, :2]) * 1e3
        ty = math.degrees(_yaw_err(tp[2], self.goal[i, 2], 1))
        dpos, dyaw = (v[0] for v in self.dock_errors(np.array([i])))
        ok = te <= 10 and ty <= 10 and dpos <= self.cfg.task.dock_pos_tol and dyaw <= self.cfg.task.dock_yaw_tol
        return (f"push {push + 1}/{plan_len}: {self.push_label(i, push, who, plan_len)}",
                f"T {te:4.1f} mm {ty:4.1f} deg  box to corner {dpos * 1e3:4.1f} mm {math.degrees(dyaw):4.1f} deg"
                f"{'  ASSEMBLED' if ok else ''}")


def run_level3(world_cfg: EnvConfig, tee: Skill, box: Skill, dock: Skill, episodes: int, seed: int,
               parking: bool = True, frames=None) -> dict:
    cfg = copy.deepcopy(world_cfg)
    cfg.num_envs, cfg.seed = episodes, seed
    cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
    world = Level3Env(cfg, parking=parking)
    rows = execute(world, {TEE: tee, BOX: box, DOCK: dock}, frames)
    out = summarize_level2(rows)
    col = lambda k: np.array([r[k] for r in rows])
    out.update(docked=float(col("box_ok").mean()), gap_mm_median=float(np.median(col("gap_mm"))),
               dock_err_median=(float(np.median(col("box_pos_mm"))), float(np.median(col("box_yaw_deg")))))
    return out


def record_dock_starts(world_cfg: EnvConfig, tee: Skill, box: Skill, dock: Skill, episodes: int, seeds, path):
    """Where docking starts in Level 3 (the T, the box, the rod and the corner) on ``seeds``. On training seeds it
    is the dock skill's start bank (``task.dock_start_bank``), so the skill trains on the states the box skill
    leaves behind; on the test seed it is only for evaluating docking alone."""
    bank = {"tee": [], "box": [], "rod": [], "side": []}
    for seed in seeds:
        cfg = copy.deepcopy(world_cfg)
        cfg.num_envs, cfg.seed = episodes, seed
        cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
        world = Level3Env(cfg, parking=True)
        seen = np.zeros(episodes, dtype=bool)

        def frames(w, push, active, plan_len, over):
            new = np.flatnonzero((active == DOCK) & ~seen & (w.step_count > 0))
            for key, v in (("tee", w.object_pose()), ("box", w.clutter_pose()), ("rod", w.tip[:, :2]), ("side", w.side)):
                bank[key].append(v[new].copy())
            seen[new] = True

        execute(world, {TEE: tee, BOX: box, DOCK: dock}, frames)
        print(f"seed {seed}: {seen.sum()} docking starts")
    out = {k: np.concatenate(v) for k, v in bank.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path} ({len(out['side'])} docking starts)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tee", default="runs/clutter_v7/best.pt", help="skill: push the T, keep the box in place")
    p.add_argument("--box", default="runs/clutter_box_v2/best.pt", help="skill: push the box, keep the T in place")
    p.add_argument("--dock", default="runs/dock_v4/best.pt", help="skill: push the box into the T's corner")
    p.add_argument("--dock-policy", default="", help="other weights for the dock skill, e.g. runs/clutter_box_v2/best.pt")
    p.add_argument("--no-park", action="store_true", help="only scenes that two pushes (then docking) can do")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--video", default="", help="render the episodes (use a few, e.g. --episodes 4) to this MP4")
    p.add_argument("--view", action="store_true", help="watch scenes one after another in the MuJoCo viewer")
    p.add_argument("--out", default="")
    p.add_argument("--record-dock-starts", default="", metavar="NPZ",
                   help="record where docking starts (--episodes scenes per seed) to this file, for the dock skill")
    p.add_argument("--record-seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="seeds for --record-dock-starts")
    args = p.parse_args(argv)
    n, device = (1, "cpu") if args.view else (args.episodes, "cuda" if torch.cuda.is_available() else "cpu")
    tee, box = Skill(args.tee, n, device), Skill(args.box, n, device)
    dock = Skill(args.dock, n, device, args.dock_policy or None)
    _, world_cfg, _ = load_policy(args.tee, "cpu")           # camera model and randomisation of the T skill
    park = not args.no_park
    world_cfg.task.episode_seconds = 80.0 if park else 60.0              # 20 s per push at most
    dt = world_cfg.task.control_dt
    run = lambda k, seed, frames: run_level3(world_cfg, tee, box, dock, k, seed, park, frames)
    if args.record_dock_starts:
        return record_dock_starts(world_cfg, tee, box, dock, n, args.record_seeds, args.record_dock_starts)
    if args.view:
        return view(lambda scene, frames: run(1, args.seed + scene, frames), dt)
    r = video(args.video, n, lambda frames: run(n, args.seed, frames), dt) if args.video else run(n, args.seed, None)
    print(f"\nLevel 3: {n} assembly scenes (seed {args.seed}), {'with' if park else 'without'} parking | T skill "
          f"{args.tee} | box skill {args.box} | dock skill {args.dock_policy or args.dock}")
    rows = [("success", "assembled (T at its goal, box in its corner)", "{:.1%}"), ("tee_at_goal", "  T at its goal", "{:.1%}"),
            ("docked", "  box nested in the corner (3 mm, 5 deg)", "{:.1%}"),
            ("all_pushes_settled", "every push settled at its target", "{:.1%}"),
            ("kept_block_disturbed", "a block that had to stay moved >10 mm", "{:.1%}"),
            ("failure", "failures (a block left the band)", "{:.1%}"), ("time_s_median", "time to finish, median (s)", "{:.1f}"),
            ("three_push_share", "scenes with a parking push", "{:.1%}"), ("tee_first_share", "T placed first", "{:.1%}"),
            ("success_tee_first", "  assembled when the T goes first", "{:.1%}"),
            ("success_box_first", "  assembled when the box goes first", "{:.1%}"),
            ("gap_mm_median", "gap between the pieces at the end, median (mm)", "{:.1f}")]
    for key, label, fmt in rows:
        print(f"  {label:48s} {fmt.format(r[key])}")
    print(f"  {'final error median: T / box to its corner':48s} {r['tee_err_median'][0]:.1f} mm {r['tee_err_median'][1]:.1f} deg / "
          f"{r['dock_err_median'][0]:.1f} mm {r['dock_err_median'][1]:.1f} deg")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), **r}, indent=1))
        print(f"per-episode results: {args.out}")


if __name__ == "__main__":
    main()
