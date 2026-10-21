"""Levels 2 and 3 on the robot: two blocks, a push plan made from the camera's first view, a skill per push.

    python -m s2r2s.hardware.runner --level 2 --robot sim --episodes 50                 # rehearsal, held-out scenes
    python -m s2r2s.hardware.runner --level 3 --robot sim --camera render --episodes 4  # through the C920 pipeline
    python -m s2r2s.hardware.runner --level 2 --robot so101 --port COM5 --map calibration/joint_map.json \\
        --goal 0.20 0.05 90 --box-goal 0.20 -0.06 0                                    # T goal, box goal
    python -m s2r2s.hardware.runner --level 3 --robot so101 --port COM5 --map calibration/joint_map.json \\
        --goal 0.20 0.0 0                                                              # the box nests in the T's corner

What ``level2.execute`` does in simulation, through the hardware path (``LevelRunner``). At the start the camera
finds both blocks (the arm moved out of its way first if it hides one) and Level 2's planner (``Level2Env.plan``)
orders the pushes from those poses: two pushes in the order with more room, or three when a block must be parked
first. Level 3 plans the T to its goal and the box to the pre-dock spot of a corner, then docking; of the T's two
corners that fit the band it takes the one whose plan has fewer pushes, then more room (``Level3Env.corner_plans``;
the simulator draws one at random). Each push runs its skill through that skill's own twin: the T skill
(``--tee``) keeps the box in place, the box and dock skills (``--box``, ``--dock``) keep the T. The next push starts
once the pushed block has stayed within 8 mm / 8 deg of its target for 1 s by the camera (docking: 3 mm / 5 deg), or
after 20 s (docking: 10 s, with its target fixed at the corner where the camera sees the T when docking starts).
The trial ends when the last push has settled (Level 3: also when docking's 10 s are up, where the simulator holds
the arm still), or after 60 s (Level 2) / 80 s (Level 3), the simulator's episode lengths. Safety limits, the servo
ramp, lost-object recovery and the clear start are ``PolicyRunner``'s, for the block the current push moves.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np

from ..level2 import BOX, DOCK, SETTLE_STEPS, TEE, Level2Env, _yaw_err
from ..level3 import Level3Env
from ..objects import OBJECTS
from ..train import load_policy
from .joint_map import JointMap
from .poses import CameraPoseSource, SimPoseSource
from .robots import SimRobot, SO101Robot
from .runner import PROJECT, PolicyRunner, lift_clear, load_skill, move_to, runner_options, save_trial

SKILLS = {TEE: "tee", BOX: "box", DOCK: "dock"}           # as the --tee / --box / --dock checkpoints
EPISODE_SECONDS = {2: 60.0, 3: 80.0}                     # level2 / level3 main, with parking: 20 s per push at most


class LevelRunner(PolicyRunner):
    """``PolicyRunner`` with one twin per skill, swapped at every push of a plan made from the camera's first view.

    ``skills``: checkpoints {TEE: push the T, keep the box; BOX: push the box, keep the T; DOCK (Level 3): push the
    box into the T's corner}. A push runs its skill through that skill's twin, as ``level2.Skill`` does in
    simulation: the block it pushes is the twin's object, the other block the one its rod guard keeps clear of, which
    must stay where the camera saw it when the push began. The pose source swaps the blocks' roles to match
    (``swap``). Each tick's log entry records the push (``push``: index in the plan, -1 before it; ``skill``);
    ``pose`` and ``block`` are as that skill sees them: the block it pushes and the one it keeps.
    """

    def __init__(self, level: int, skills: dict, robot, poses, jmap: JointMap, device: str = "cpu",
                 lightbox: bool = False, **kw):
        if level not in (2, 3) or TEE not in skills or BOX not in skills or (level == 3 and DOCK not in skills):
            raise ValueError("Level 2 needs the T and box skills, Level 3 also the dock skill")
        super().__init__(skills[TEE], robot, poses, jmap, device=device, lightbox=lightbox, **kw)
        self.level = level
        self.skills = {TEE: (self.policy, self.cfg, self.twin)}
        for w in (BOX, DOCK)[:level - 1]:
            self.skills[w] = load_skill(skills[w], device, lightbox)
        tee, box = self.cfg.task.objects[0], self.cfg.task.clutter
        for w, (_, c, _) in self.skills.items():
            moved, kept = (tee, box) if w == TEE else (box, tee)
            if c.task.objects[0] != moved or c.task.clutter != kept or (w == DOCK) != c.task.dock:
                raise ValueError(f"the {SKILLS[w]} skill must {'dock' if w == DOCK else 'push'} the {moved} and keep "
                                 f"the {kept}: {skills[w]} pushes the {c.task.objects[0]} (block {c.task.clutter})")
        cfg = copy.deepcopy(self.cfg)                # the planner: the T skill's task (the T and the box), one world
        cfg.task.difficulty, cfg.task.easy_fraction = 1.0, 0.0
        self.planner = (Level3Env if level == 3 else Level2Env)(cfg, parking=True)
        self.planner.park_candidates = 1000      # one scene at a time: many more parking spots than the batched 24
        self.episode_seconds = EPISODE_SECONDS[level]
        self.who, self.push, self.plan, self.pushes, self.done = TEE, -1, [], [], False

    def _use(self, who):
        """Make skill ``who`` the runner's policy and twin; the camera reports the block it pushes as the object."""
        self.who = who
        self.policy, self.cfg, self.twin = self.skills[who]
        self.poses.swap = who != TEE

    def estimates(self):
        """The newest camera estimates of the T and the box (each held while unseen)."""
        pose, block = self.twin.obs_pose[0].copy(), self.twin.clutter_obs[0].copy()
        return (pose, block) if self.who == TEE else (block, pose)

    def start_level(self, tee_goal, box_goal=None, **kw):
        """Begin a trial: find both blocks, plan the pushes from where the camera sees them, begin the first.

        ``box_goal``: Level 2 only; Level 3 nests the box in a corner of the T at ``tee_goal``. RuntimeError when the
        blocks stay unseen (``PolicyRunner.start``) or no plan keeps the skills' 15 mm rule.
        """
        P, t = self.planner, self.planner.cfg.task
        self.tee_goal = np.asarray(tee_goal, float)
        goals = [self.tee_goal] if self.level == 3 else [self.tee_goal, np.asarray(box_goal, float)]
        if not P._in_region(np.array([g[:2] for g in goals]), t.obj_r, t.obj_az).all():
            raise RuntimeError(f"a goal is outside the work band (r {t.obj_r[0] * 100:.1f}-{t.obj_r[1] * 100:.1f} cm, "
                               f"within {math.degrees(t.obj_az):.0f} deg of straight ahead)")
        self._use(TEE)
        self.push, self.plan, self.pushes, self.done = -1, [], [], False
        self.start(self.tee_goal, **kw)                  # the T skill finds both blocks (a clear start if hidden)
        tee, box = self.estimates()
        if self.level == 2:
            self.side, self.box_goal = 0.0, goals[1]
            length, who, goal, clear = P.plan(tee[None], self.tee_goal[None], box[None], self.box_goal[None])
        else:
            side, length, who, goal, clear = P.corner_plans(tee[None], self.tee_goal[None], box[None])
            self.side = float(side[0])
            self.box_goal = P.docked(self.tee_goal[None], side)[0]       # where the box ends: the corner at the goal
        if length[0] == 0:
            gap = lambda a, b: P._gap_to_object(a[None], b[None], np.zeros(1, dtype=np.int64))[0] * 1e3
            why = f"the blocks are {gap(box, tee):.1f} mm apart"
            if self.level == 2:
                best = P.order_clearances(tee[None], self.tee_goal[None], box[None], self.box_goal[None]).max() * 1e3
                why += f", {gap(self.box_goal, self.tee_goal):.1f} mm at their goals; the better order leaves {best:.1f} mm"
            raise RuntimeError(f"no push plan keeps the skills' {t.clutter_min_gap * 1e3:.0f} mm rule ({why}; no "
                               f"parking spot{' or corner' if self.level == 3 else ''} helps): move a block")
        self.plan = [(int(w), g.copy()) for w, g in zip(who[0, :length[0]], goal[0, :length[0]])]
        self.plan_clearance = float(clear[0])
        self.t_first = self._clock()
        self._begin(0, tee, box)

    def _begin(self, k, tee, box):
        """Start push ``k`` with its skill, from the newest estimates of the T and the box."""
        who, target = self.plan[k]
        if who == DOCK:          # the corner where the camera sees the T now, fixed from here on (Level3Env.push_target)
            target = self.planner.docked(tee[None], np.array([self.side]))[0]
        if (who == TEE) != (self.who == TEE):             # the blocks swap roles, and with them when each was last seen
            self.last_seen, self.block_seen = self.block_seen, self.last_seen
        self._use(who)
        pose, block = (tee, box) if who == TEE else (box, tee)
        self.twin.set_goal(target)
        q = self.jmap.to_sim(self.robot.read_joints())
        self.twin.sync_measurements(q[None], pose[None], first=True, block=block[None])
        self.push, self.settled, self.push_t0 = k, 0, self._clock()
        self.pushes.append(dict(skill=SKILLS[who], target=target.tolist(), start=pose.tolist(), kept=block.tolist(),
                                tick=len(self.log)))

    def _end_push(self, settled: bool):
        p = self.pushes[-1]
        kept = self.twin.clutter_obs[0]
        p.update(settled=settled, seconds=self._clock() - self.push_t0, end_tick=len(self.log),
                 kept_moved_mm=float(np.linalg.norm(kept[:2] - np.array(p["kept"])[:2]) * 1e3))

    def tick(self) -> str:
        """One control period of the current push, then the sequencer: once the pushed block has stayed within the
        settle tolerance of its target for 1 s by the camera (every tick with a pose at most ``fresh_age`` old), or the
        push's time is up, the next push begins (only while the policy has the rod: a recovery restarts the settle
        clock). Returns ``PolicyRunner.tick``'s mode, or "done" once the plan is over (stop ticking then: the servos
        hold their last target)."""
        if self.done:
            return "done"
        mode = super().tick()
        if mode != "policy":
            self.settled = 0
            return mode
        tw, who = self.twin, self.who
        est, target = tw.obs_pose[0], tw.goal[0]
        pos = float(np.linalg.norm(est[:2] - target[:2]))
        yaw = float(_yaw_err(est[2], target[2], OBJECTS[self.cfg.task.objects[0]].symmetry))
        tol_pos, tol_yaw = self.planner.settle_tol(who)
        # Settled by the camera: a held estimate counts only while it is fresh (the arm may hide a block that moves).
        fresh = self._clock() - self.last_seen <= self.fresh_age
        self.settled = self.settled + 1 if fresh and pos < tol_pos and yaw < tol_yaw else 0
        settled = self.settled >= SETTLE_STEPS
        out_of_time = self._clock() - self.push_t0 >= self.planner.push_steps(who) * self.dt - 1e-9
        last = self.push == len(self.plan) - 1
        if not (settled or out_of_time):
            return mode
        if "settled" not in self.pushes[-1]:
            self._end_push(settled)
        if last and not settled and self.level == 2:
            # Out of time, as the simulator counts it, but Level 2's last skill keeps going until the block settles or
            # the trial's time is up (level2.execute).
            return mode
        if last:
            self.done, self.t_done = True, self._clock()
            return "done"
        self._begin(self.push + 1, *self.estimates())
        return mode

    def _log_extra(self) -> dict:
        return dict(push=self.push, skill=SKILLS[self.who])

    def push_names(self) -> list[str]:
        moves = len(self.plan) - (self.level == 3)
        return ["dock the box" if w == DOCK else ("park the " if moves == 3 and k == 0 else "") + ("T" if w == TEE else "box")
                for k, (w, _) in enumerate(self.plan)]

    def outcome(self, tee, box) -> dict:
        """The task's result for final poses of the T and the box (the truth in simulation, the camera on the robot),
        judged as ``Level2Env.episode_row`` / ``Level3Env.episode_row`` do."""
        P, t = self.planner, self.planner.cfg.task
        tee, box = np.asarray(tee, float), np.asarray(box, float)
        tee_pos = float(np.linalg.norm(tee[:2] - self.tee_goal[:2]))
        tee_yaw = float(_yaw_err(tee[2], self.tee_goal[2], OBJECTS[t.objects[0]].symmetry))
        target = self.box_goal if self.level == 2 else P.docked(tee[None], np.array([self.side]))[0]
        box_pos = float(np.linalg.norm(box[:2] - target[:2]))
        box_yaw = float(P._clutter_yaw_err(box[2:], target[2:])[0])
        box_tol = (t.success_pos, t.success_yaw) if self.level == 2 else (t.dock_pos_tol, t.dock_yaw_tol)
        tee_ok = tee_pos <= t.success_pos and tee_yaw <= t.success_yaw
        box_ok = box_pos <= box_tol[0] and box_yaw <= box_tol[1]
        failed = not P._in_region(np.stack([tee[:2], box[:2]]), t.fail_r, t.fail_az).all()
        out = dict(success=bool(tee_ok and box_ok and not failed), tee_ok=bool(tee_ok), box_ok=bool(box_ok),
                   failed=bool(failed), tee_pos_mm=tee_pos * 1e3, tee_yaw_deg=math.degrees(tee_yaw),
                   box_pos_mm=box_pos * 1e3, box_yaw_deg=math.degrees(box_yaw))
        if self.level == 3:
            out["gap_mm"] = float(P._gap_to_object(box[None], tee[None], np.zeros(1, dtype=np.int64))[0] * 1e3)
        return out

    def summary(self) -> dict:
        """The trial's plan and pushes, for the trial log (a push still running counts as not settled)."""
        if self.plan and "settled" not in self.pushes[-1]:
            self._end_push(False)
        done = [p for p in self.pushes if p.get("settled")]
        return dict(level=self.level, side=self.side, tee_goal=self.tee_goal.tolist(),
                    box_goal=self.box_goal.tolist() if self.plan else None,
                    plan=[dict(skill=SKILLS[w], name=n, target=g.tolist()) for (w, g), n in zip(self.plan, self.push_names())],
                    plan_clearance_mm=self.plan_clearance * 1e3 if self.plan else None, pushes=self.pushes,
                    pushes_done=len(done), all_settled=bool(self.plan) and len(done) == len(self.plan),
                    kept_moved_mm=max((p["kept_moved_mm"] for p in self.pushes), default=0.0),
                    time_s=self.t_done - self.t_first if self.done and self.pushes[-1]["settled"] else float("nan"),
                    start_shift=self.start_shift, recoveries=self.events)


def _skills(args) -> dict:
    return {TEE: args.tee, BOX: args.box, DOCK: args.dock}


def dry_run(args):
    """Two-block scenes on SimRobot through the hardware path: the scenes ``python -m s2r2s.level2`` / ``level3``
    draw for the same --seed and --episodes (held out by default), with the simulator's plan beside the runner's."""
    _, cfg, _ = load_policy(args.tee, "cpu")      # the scenes: the T skill's task, as level2 / level3 draw them
    cfg.scene.lightbox = cfg.scene.lightbox or args.lightbox
    scenes_cfg = copy.deepcopy(cfg)
    scenes_cfg.num_envs, scenes_cfg.seed, scenes_cfg.num_threads = args.episodes, args.seed, 4
    scenes_cfg.task.difficulty, scenes_cfg.task.easy_fraction = 1.0, 0.0
    scenes_cfg.task.episode_seconds = EPISODE_SECONDS[args.level]
    scenes = (Level3Env if args.level == 3 else Level2Env)(scenes_cfg, parking=True)
    hidden = JointMap.load(args.hidden_map) if args.hidden_map else JointMap()
    jmap = JointMap.load(args.map) if args.map else JointMap()
    robot = SimRobot(cfg.scene, cfg.task.objects[0], hidden=hidden, encoder_noise_deg=0.05)
    runner = LevelRunner(args.level, _skills(args), robot, None, jmap, realtime=False, **runner_options(args))
    camera = None
    if args.camera == "render":                  # the C920 pipeline on rendered frames (slow: ~0.25 s per tick)
        from .camera_sim import RenderedPoseSource
        camera = RenderedPoseSource(robot, cfg.scene, cfg.task.objects[0], latency_steps=2, dt=runner.dt, seed=args.seed,
                                    mount=args.mount)
    ticks = round(runner.episode_seconds / runner.dt)
    rows, not_started, dropped = [], [], []
    for i in range(args.episodes):
        robot.place(scenes.tip[i, :2], scenes.object_pose()[i])
        robot.place_block(scenes.clutter_pose()[i])
        if camera is not None:
            camera.reset()
            runner.poses = camera
        else:
            runner.poses = SimPoseSource(robot, latency_steps=1, seed=args.seed + i)
        try:
            runner.start_level(scenes.goal[i], scenes.box_goal[i] if args.level == 2 else None)
        except RuntimeError as error:
            not_started.append(i)
            print(f"episode {i + 1:3d}: not started ({error})", flush=True)
            continue
        # How far the block a push must keep in place truly moves during it (the simulator's kept_moved_mm).
        kept = lambda k: robot.block_pose() if runner.plan[k][0] == TEE else robot.object_pose()
        push, home, moved, mode = 0, kept(0), 0.0, "policy"
        for _ in range(ticks):
            mode = runner.tick()
            if runner.push != push:
                moved = max(moved, float(np.linalg.norm(kept(push)[:2] - home[:2])))
                push, home = runner.push, kept(runner.push)
            if mode in ("done", "lost"):
                break
        moved = max(moved, float(np.linalg.norm(kept(push)[:2] - home[:2])))
        row = runner.outcome(robot.object_pose(), robot.block_pose())
        s = runner.summary()
        sim_plan = [int(w) for w in scenes.plan_who[i, :scenes.plan_len[i]]]
        same = sim_plan == [w for w, _ in runner.plan] and (args.level == 2 or bool(runner.side == scenes.side[i]))
        row.update(pushes=len(runner.plan), pushes_done=s["pushes_done"], all_settled=s["all_settled"],
                   kept_moved_true_mm=moved * 1e3, time_s=s["time_s"], same_plan=same, gave_up=mode == "lost",
                   recoveries=len(runner.events), start_shift=runner.start_shift)
        rows.append(row)
        if args.log_dir:
            save_trial(Path(args.log_dir) / f"level{args.level}_{i:03d}", runner, dict(
                skills={SKILLS[w]: str(c) for w, c in _skills(args).items() if w in runner.skills},
                start=scenes.object_pose()[i].tolist(), block_start=scenes.clutter_pose()[i].tolist(),
                tool_start=scenes.tip[i, :2].tolist(), final=robot.object_pose().tolist(),
                block_final=robot.block_pose().tolist(), simulator_plan=[SKILLS[w] for w in sim_plan], **{**row, **s}))
        if camera is not None:
            frames = max(camera.counts["frames"], 1)
            dropped.append([camera.counts["dropped"].get(n, 0) / frames for n in (camera.object_name, camera.block_name)])
        print(f"episode {i + 1:3d}: {'success' if row['success'] else 'miss   '} {', '.join(runner.push_names()):34s} "
              f"settled {s['pushes_done']}/{len(runner.plan)}  T {row['tee_pos_mm']:5.1f} mm {row['tee_yaw_deg']:5.1f} deg  "
              f"box{' to corner' if args.level == 3 else ''} {row['box_pos_mm']:5.1f} mm {row['box_yaw_deg']:5.1f} deg"
              + ("" if same else "  (simulator's plan: " + ", ".join(SKILLS[w] for w in sim_plan) + ")")
              + (f"  started {runner.start_shift * 1e3:.0f} mm away" if runner.start_shift else "")
              + (f"  lost {len(runner.events)}x" if runner.events else "") + ("  GAVE UP" if mode == "lost" else ""),
              flush=True)
    if not_started:
        print(f"\n{len(not_started)} of {args.episodes} episodes not started (episodes {[i + 1 for i in not_started]})")
    if not rows:
        return
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    time_s = col("time_s")
    print(f"\nLevel {args.level}: {len(rows)} scenes (seed {args.seed}) through the hardware path"
          + (f", rendered C920 ({args.mount} mount; {np.mean(dropped, 0)[0]:.1%} of frames without the T, "
             f"{np.mean(dropped, 0)[1]:.1%} without the box)" if dropped else ""))
    for label, value in (("success", col("success").mean()), ("  T at its goal", col("tee_ok").mean()),
                         ("  box " + ("at its goal" if args.level == 2 else "nested in the corner"), col("box_ok").mean()),
                         ("every push settled at its target", col("all_settled").mean()),
                         ("a block that had to stay moved >10 mm", (col("kept_moved_true_mm") > 10).mean()),
                         ("failures (a block left the band)", col("failed").mean()),
                         ("plans as the simulator's", col("same_plan").mean()),
                         ("episodes with a lost-object recovery", (col("recoveries") > 0).mean()),
                         ("gave up (object lost)", col("gave_up").mean())):
        print(f"  {label:48s} {value:.1%}")
    print(f"  {'time to finish, median (s)':48s} "
          + (f"{np.nanmedian(time_s):.1f}" if np.isfinite(time_s).any() else "-"))


def real_run(args):
    jmap = JointMap.load(args.map)
    robot = SO101Robot(args.port, max_step_deg=args.max_step_deg)
    runner = LevelRunner(args.level, _skills(args), robot, None, jmap, **runner_options(args))
    try:      # the camera masks the arm out using the skills' scene (lightbox, rod)
        poses = runner.poses = CameraPoseSource(scene=runner.cfg.scene, object_name=runner.cfg.task.objects[0],
                                                block_name=runner.cfg.task.clutter)
    except FileNotFoundError as error:
        robot.close()
        raise SystemExit(f"{error}\nSee HARDWARE.md section 4. Until then, rehearse with --robot sim.")
    kin = runner.twin.kin
    deg = lambda g: np.array([g[0], g[1], math.radians(g[2])])
    goal, box_goal = deg(args.goal), deg(args.box_goal) if args.level == 2 else None
    mode = "not started"
    try:
        move_to(robot, jmap, kin, (*args.tool_start, 0.03))                    # above the start point
        move_to(robot, jmap, kin, (*args.tool_start, runner.cfg.task.tool_height), seconds=1.0)
        runner.start_level(goal, box_goal)
        print(f"plan: {', '.join(runner.push_names())}" + (f" (corner {runner.side:+.0f})" if args.level == 3 else ""))
        if runner.start_shift:
            print(f"the arm hid a block at the start: started {runner.start_shift * 1e3:.0f} mm from --tool-start")
        push = -1
        for _ in range(round(runner.episode_seconds / runner.dt)):
            mode = runner.tick()
            if runner.push != push:
                push = runner.push
                print(f"push {push + 1}/{len(runner.plan)}: {runner.push_names()[push]}")
            if mode == "done":
                break
            if mode == "lost":
                print(f"object not seen again within {args.recover_seconds:.0f} s: trial ended")
                break
    except KeyboardInterrupt:
        print("stopped by user")
    except RuntimeError as error:
        print(f"trial not started: {error}")
    finally:
        lift_clear(robot, jmap, runner)
        poses.swap = False
        poses.set_arm(jmap.to_sim(robot.read_joints()))
        time.sleep(0.5)                                        # a frame with the rod lifted
        tee, _ = poses.read()
        box, _ = poses.read_block()
        meta = dict(skills={SKILLS[w]: str(c) for w, c in _skills(args).items() if w in runner.skills},
                    final=None if tee is None else tee.tolist(), block_final=None if box is None else box.tolist(),
                    gave_up=mode == "lost", **(runner.summary() if runner.plan else {}))
        if runner.plan and tee is not None and box is not None:
            meta.update(runner.outcome(tee, box))
            print(f"{'success' if meta['success'] else 'miss'}: T {meta['tee_pos_mm']:.1f} mm {meta['tee_yaw_deg']:.1f} deg, "
                  f"box {meta['box_pos_mm']:.1f} mm {meta['box_yaw_deg']:.1f} deg, {meta['pushes_done']}/{len(runner.plan)} "
                  f"pushes settled (by the camera)")
        meta["camera"] = poses.summary()
        save_trial(Path(args.log_dir or PROJECT / "trials") / time.strftime(f"level{args.level}_%Y%m%d_%H%M%S"), runner, meta)
        print(f"camera: {meta['camera']}")
        poses.close()
        robot.close()
