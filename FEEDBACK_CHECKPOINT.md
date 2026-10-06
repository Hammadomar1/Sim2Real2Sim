# Object-pose feedback and shorter path

2026-10-04 — **Both tested executions complete within 30 seconds. RL training remains off.**

## Implementation

`scripts/feedback_path.py` adds a diagnostic controller that recalculates the pushing contact point and direction from the current block pose at every 20 Hz control update. Tangential correction keeps the tool near the selected part of the block as it moves and rotates. After each push, the planner selects the next contact using the measured state and short candidate physics rollouts.

The planner now scores improvement per unit of simulated time, allows sustained pushes instead of many short straight-line pokes, uses a reachable intermediate alignment position at x = 165 mm, and removes fixed inter-push pauses. The existing speed/acceleration filtering, robot actuation, friction, mass, timestep family and contact settings remain unchanged.

This is an **offline model-based diagnostic**, using exact simulated object state and repeated candidate rollouts. Planning computation is not real-time, and simulation is paused while candidates are evaluated. Reported times are simulated execution times. The controller is separate from the RL environment's action-to-joint controller; it is not a learned policy or a hardware controller.

## Measured verification

The same feedback algorithm was executed separately at 2 ms and 1 ms. It chose different pushes in response to each run's object state. Each recorded command sequence was then independently re-simulated from one reset at its corresponding timestep; no intermediate poses were loaded during verification.

| Measure | 2 ms | 1 ms |
|---|---:|---:|
| First completed parking hold of one second | **26.564 s** | **25.072 s** |
| Full verification recording, including extra settling | 29.050 s | 29.800 s |
| Final position error | **1.415 mm** | **1.743 mm** |
| Final yaw error | **3.863 degrees** | **2.624 degrees** |
| Final stationary hold | 3.486 s | 3.426 s |
| Maximum contact penetration | 0.797 mm or less | 0.793 mm or less |
| Entire block crossed gate | Yes | Yes |
| Within 30-second task target | Yes | Yes |

The previous successful fixed-command recording lasted 72.65 seconds. The new 2 ms verified recording is about **60% shorter**, including the same two-second verification extension. First-success timing includes the required one-second stationary hold; the controller may continue refining the pose afterward.

Both paths meet the independent joint-limit, fixed-base, uprightness, workspace and gate-crossing checks. The block contacts the gate during these paths and briefly moves back toward the opening during the final turn; it ends entirely beyond the gate. Gate passage is not constrained to be monotonic.

Contact checks retain their existing 0.05 mm numerical tolerance for otherwise disallowed pairs. The 2 ms trajectory has a shallow pusher/gate contact with maximum penetration **0.0498 mm**, just below that threshold; the 1 ms trajectory has no such contact. This threshold should be included in the upcoming collision-rule review; the results are not evidence of strict zero-touch wall avoidance.

## Scope of the result

- The diagnostic layout remains gate x = 215 mm, parking x = 270 mm, y = 0, yaw = 90 degrees. It is the previously disclosed 15 mm layout shift. The training defaults have not yet been changed.
- These are two deterministic, nominal executions from one starting scene. They do not establish 95% success, randomization robustness, GPU full-path parity, or final motion-smoothness acceptance.
- Each execution uses pose feedback and its configured timestep in candidate simulations. This is not a demonstration that a single fixed action sequence transfers across timesteps or unknown physics.
- The earlier failed 1 ms open-loop replay remains recorded in `FULL_PATH_CHECKPOINT.md` and its original artifacts.

## GUI and commands

```powershell
# View the independently verified feedback demonstration:
.\scripts\view-audit.ps1 -Trial feedback

# View the independently verified 1 ms execution:
.\scripts\view-audit.ps1 -Trial feedback1ms

# Recheck the saved command sequences without rerunning the search:
.\scripts\feedback-path.ps1 -VerifyOnly
.\scripts\feedback-path.ps1 -Timestep 0.001 -VerifyOnly

# Recompute the feedback search and then independently verify it:
.\scripts\feedback-path.ps1
.\scripts\feedback-path.ps1 -Timestep 0.001
```

The GUI is recorded playback, labeled with no trained policy. **P** pauses/resumes; **R** restarts and pauses.

Evidence: `artifacts/feedback_validation_2ms.json`, `artifacts/feedback_validation_1ms.json`, the corresponding `.npz` recordings, `artifacts/feedback_path_2ms.json` and `feedback_path_1ms.json` decision logs, `artifacts/feedback_comparison.png`, and `artifacts/feedback_manifest.json` with file hashes. Source files and environment/controller hashes are recorded for reproducibility.

## Next checkpoint

Integrate the demonstrated layout into the training configuration and validate randomized resets, success detection, invalid shortcuts and collision rules. Then verify the full sequence on the GPU backend and the short PPO/checkpoint-resume workflow before allowing long training. Training remains blocked by preflight checks.
