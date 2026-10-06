# Integrated layout and task rules — 2026-10-04

The demonstrated layout is now the environment default: gate x = 215 mm, parking center x = 270 mm. Stage 5 starts and parks near 90 degrees. No RL has run at this checkpoint.

## Executed validation

| Check | Result |
|---|---|
| Random geometric resets, all five stages | 500 native + 5,120 GPU resets; no initial overlap |
| Reset state | Velocities, actions, passage history, failure flags and hold counters cleared; goal markers match task goals |
| GPU subset reset | Untouched worlds retain simulation, task and rule state |
| State save/load | Rule history round trip passes; this is not a PPO resume test |
| One-second zero-action settling | 20 native + 128 GPU episodes; finite observations, no premature termination |
| Automated tests | 36 passed, including 35 rule tests and the training preflight test |
| Refreshed contact regression | 22 cases pass on the integrated layout, including 2/1/0.5 ms checks and native/GPU push and blocked-gate comparisons |
| Demonstrated 2 ms actions through actual environment step | Success at 26.60 s; 1.415 mm / 3.863 degrees at termination |
| Demonstrated 1 ms actions through actual environment step | Success at 25.10 s; 5.791 mm / 3.616 degrees at termination |

The replay stops on the environment's first successful termination, after one stationary second. Earlier feedback reports continued recording after completion, so their final errors differ. Both replays have zero failure flags and valid complete gate passage. These are two nominal diagnostic executions, not a learned policy or a randomized task success rate.

## Rules now enforced

The monitor checks all 16 corners of the compound block at every physics step. Each corner must cross the gate plane inside the opening, and the entire block must clear the barrier. Historical passage alone cannot satisfy parking while part of the block is back in the gate. Teleports, routes around the barrier, tipping/lifting, workspace escape, joint-limit violations, excessive penetration and invalid robot contacts latch episode failure.

Parking requires position error at most 10 mm, yaw error at most 10 degrees, linear speed at most 5 mm/s and angular speed at most 5 degrees/s for one uninterrupted second. Reward shaping is separate from this detector. Terminal records expose failure categories; automatic reset preserves the terminal record and clears the next episode's rule state.

Allowed contacts are block–table, block–pusher and block–gate. Other contacts fail beyond **50 micrometers penetration**; any penetration above **1 mm** fails. This numerical tolerance is explicit: the previous 2 ms diagnostic reached about 49.7 micrometers of pusher–gate penetration. It has very little margin to that rule and is not a strict zero-contact demonstration.

CPU and GPU rule kernels agree on completed crossings, stationary holds and adversarial fixtures. A corner exactly on the gate plane can be classified one sample apart due to float32/float64 rounding; an explicit regression test verifies that it resolves without false failure or a different final passage result.

## Reproduce and inspect

```powershell
.\scripts\validate-layout.ps1
.\scripts\view.ps1
.\scripts\view-audit.ps1 -Trial feedback
```

The scene GUI starts paused; P toggles physics, R resets. The feedback GUI replays recorded diagnostic states; it is not a trained policy or live re-simulation. Evidence: `artifacts/layout_validation.json`, `artifacts/task_rules_tests.xml`, `artifacts/physics_validation.json`. Reports record implementation source hashes, including the new rule monitor. The refreshed native/GPU final-pose differences are 0.352 mm / 1.561 degrees for pushing and 1.302 mm / 0.738 degrees for blocked-gate contact. These are contact regressions, not a full-path GPU comparison.

## Remaining before long training

These checks cover geometric reset validity and sampled center reachability, not complete path feasibility for every randomized start/goal. Physics/observation robustness ranges have not been validated. Full-path native/GPU dynamics comparison and a short PPO/checkpoint-resume test remain separate checkpoints. Training remains gated; no training-readiness approval artifact was created.
