# Full-path diagnostic checkpoint

2026-10-04. **One continuously simulated full path succeeds at 2 ms. The setup is not yet ready for long RL training.**

## What is demonstrated

The fixed-base SO-101 approaches the L block, turns it to fit the gate, pushes the entire block through the opening, turns it toward the parking pose, and holds it nearly stationary within the specified pose tolerance. The demonstration uses the existing bounded planar controller and physical MuJoCo contacts. It is a model-based diagnostic, not an RL policy.

A separate verifier re-executes every command from a single reset. It does not load intermediate planned poses or teleport the object. It checks all 16 box corners for gate passage at every physics step, uprightness and height, joint limits, fixed-base behavior, collision pairs, penetration, and the final stationary hold. The GUI displays states recorded by that successful verification.

| Measure | Successful 2 ms verification |
|---|---:|
| Entire block through gate | Yes, at 35.184 s |
| Final position error | 2.814 mm |
| Final orientation error | 2.900 degrees |
| Final stationary hold | 2.172 s |
| Total sequence including final hold | 72.650 s |
| Maximum contact penetration | 0.801 mm |
| Invalid contacts / tipping / shortcuts | None detected |
| Command speed / acceleration | At most 40 mm/s / 200 mm/s² |

This exceeds the proposed 30-second episode limit. No inference about 95% policy success or smoothness acceptance follows from this single diagnostic.

## Layout and robustness limits

The demonstration uses a **diagnostic layout variant**: gate x = 215 mm and parking x = 270 mm, each 15 mm farther from the base than the current training defaults. Goal y = 0 and yaw = 90 degrees. The block starts at (150 mm, 0), yaw = 90 degrees. The shift supplies additional room for pre-gate tool approaches. The standard training scene has not been changed to this variant.

The same open-loop commands were also executed at 1 ms. They did **not** complete the task: the final position error was 103.730 mm and yaw error 99.170 degrees; the block did not fully cross the gate. Contact penetration remained below 0.721 mm, with no detected invalid contacts or tipping. Thus the successful 2 ms sequence is not a robust controller. Small pose differences can accumulate and cause later planned contacts to miss. The shorter contact regression suite remains useful evidence about its tested scenes, but does not establish full-sequence robustness.

The earlier saved planner trajectory was rejected when continuous replay missed the orientation tolerance. The final correction was planned from a state reconstructed by replaying the full command prefix. The successful result reported above is the independent replay, not the saved planner poses.

## See and reproduce

```powershell
# Open the successful recorded demonstration, looping at real-time speed:
.\scripts\view-audit.ps1 -Trial full

# Re-simulate all commands and independently check the result:
.\scripts\validate-path.ps1

# Reproduce the known failed smaller-timestep task check:
.\scripts\validate-path.ps1 -Timestep 0.001
```

In the GUI, **P** pauses/resumes and **R** restarts and pauses. Drag to change the camera view. Playback is explicitly labeled as recorded, with no trained policy.

Evidence:

- `artifacts/full_path_validation.json` — successful 2 ms checks.
- `artifacts/full_path_validation.npz` — continuously simulated state replay and commands.
- `artifacts/full_path_validation_1ms.json` — failed task check at 1 ms.
- `artifacts/planned_path.npz` — input command sequence used by the verifier.
- `artifacts/full_path_manifest.json` — hashes of the delivered evidence and scripts.
- `artifacts/full_path_final.png` — verified final rendered frame.
- `artifacts/full_path_comparison.png` — measured trajectories and errors for both timesteps.

## Next checkpoint

Make the diagnostic sequence use object-pose feedback when selecting and executing pushes, and reduce unnecessary travel and pauses. Then retest full completion at both timesteps, select the training layout, and resolve the 30-second episode budget. Reset/success-detector tests and PPO checkpoint/resume validation still follow. Long RL training remains blocked; it was not started.
