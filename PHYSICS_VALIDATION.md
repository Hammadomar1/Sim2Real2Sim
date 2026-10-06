# Contact stability checkpoint

2026-10-03 — **Passed the defined contact regression suite. Not yet ready for long RL training.**

## What changed

The original scene used 30 Newton solver iterations with only 10 line-search iterations. A resting-block isolation test showed that the small line-search budget could generate spontaneous motion under firmer contacts. Raising Newton iterations to 100 while keeping line search at 10 did not fix it; keeping 30 Newton iterations and raising line search to 50 did. Evidence is retained in `artifacts/drop_100_10.txt` and `artifacts/drop_30_50.txt`.

The scene and GPU environment now both use 100 Newton iterations and 50 line-search iterations. Task contacts explicitly use `solref="0.008 1"` and `solimp="0.99 0.999 0.001 0.5 2"` on the table, block, pusher and gate, avoiding unintended mixing with softer defaults. The physics timestep remains 2 ms, control remains 20 Hz, and the block remains a free 3D body. Friction, mass, actuator force limits and puzzle dimensions were not adjusted in this correction.

MuJoCo documents contact-parameter mixing and the relationship between contact time constants and timestep in its [modeling reference](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters). These numerical settings are validated simulation choices, not measured hardware parameters.

## Executed verification

The suite executes the environment's actual controller without rewards, learning or object repositioning during a trial. Six scenes (five nearby pushing starts and one blocked-against-gate stress scene) are run at 2 ms, 1 ms and 0.5 ms. Each finer-timestep run receives the same recorded 20 Hz action sequence as its 2 ms reference. Separate 30-second resting and random-action trials are included. Standard MuJoCo and MuJoCo Warp are compared for both the canonical push and the blocked-gate trial: **22 trials total**.

| Check | Latest measured result | Threshold/status |
|---|---:|---|
| Maximum penetration across suite | 0.819 mm or less | Pass: at most 1 mm |
| Maximum final position difference across timesteps | 0.594 mm | Pass: at most 2 mm |
| Maximum final yaw difference across timesteps | 1.528 degrees | Pass: at most 2 degrees |
| Native vs Warp, canonical push | 0.134 mm / 0.920 degrees | Pass: 2 mm / 2 degrees |
| Native vs Warp, blocked gate | 0.513 mm / 1.307 degrees | Pass: 2 mm / 2 degrees |
| Initial five-second tool hold | About 0.103 mm | Pass: at most 0.5 mm |
| Command speed / acceleration | 40 mm/s / 200 mm/s² | Pass |
| Fixed-base motion / joint-limit violations | None detected | Pass |
| Finite states / CPU numerical warnings | Finite / none | Pass |

Exact results, source hashes, thresholds, collision pairs and measurement limits are in `artifacts/physics_validation.json`. GPU penetration is checked at control frames through standard-MuJoCo snapshots; CPU penetration is inspected every physics step. Snapshot warning counters do not measure GPU solver diagnostics. Small GPU trajectory variations occurred across reruns and stayed within the recorded thresholds.

Command limits are not actual-motion limits: sampled actual tool speed reached approximately 52.6 mm/s. Physical motion smoothness still needs evaluation. The blocked-gate stress test includes pusher/table contact; passing its numerical checks does not make that motion a valid task solution. These finite regression scenes do not establish stability for every possible RL action or randomized physics setting.

## Review and reproduce

```powershell
# Re-run contact checks, including GPU comparison:
.\scripts\validate-physics.ps1

# Watch the corrected recorded controller trial:
.\scripts\view-audit.ps1

# For comparison, watch the original unstable-contact trial:
.\scripts\view-audit.ps1 -Trial initial
```

The GUI labels recorded playback explicitly. P pauses/resumes; R returns to the beginning and pauses. `artifacts/contact_trial_final.png` is a visually checked final frame. The trial shows pushing/rotation, not completion of the puzzle.

## Remaining prerequisites for long training

1. Demonstrate a physically valid full push-turn-park path within the workspace.
2. Validate randomized resets and success detection, including shortcuts and invalid collisions.
3. Test a short PPO run and complete checkpoint/resume behavior.
4. Repeat the throughput benchmark with the corrected solver settings. The old benchmark does not describe this configuration.

`preflight.py` prevents long training before those required feasibility/test records are available. It rejects missing, failed or stale contact evidence. A deliberately bounded two-update, at-most-32-world, at-most-six-minute fresh PPO smoke test can be allowed after current contact and native/Warp evidence passes; none was run during this checkpoint. The preflight rejection paths were unit-tested. No trained policy or full-task success result is claimed.
