# Complete gripper push–turn–park — 2026-10-04

**The canonical gripper-only sequence is physically valid and completes within the 30-second limit.** Independent replay and the production environment's success/failure rules both pass. No RL training was used or started.

## What changed

The diagnostic continues from the verified gate passage. It changes contact to turn the block toward the parking orientation, then pushes it forward again. Part of the block temporarily returns toward the opening during the turn; the audit checks every corner's crossing history and requires the whole block to be clear at parking. Gate passage is not required to be monotonic.

The first finishing attempt left the block about 13 mm off-center. A search over final contact positions, push directions and durations found a final push that preserves the target angle and settles within tolerance. The candidate must pass the stop-and-settle checks, not just visit the target while moving.

The scene geometry, fixed base, jaw geometry, actuator settings, 1 ms timestep, 20 Hz control, collision thresholds and 30-second episode limit were not changed for this checkpoint. The gripper remains closed and contact alone moves the block.

## Verified results

| Measurement | Result | Requirement |
|---|---:|---:|
| Independent completion, including stationary hold | 27.246 s | At most 30 s |
| Production environment success event | 27.25 s | At most 30 s |
| Final position error | 9.316 mm | At most 10 mm |
| Final yaw error | 7.049 degrees | At most 10 degrees |
| Final uninterrupted stationary hold | 3.704 s | At least 1 s |
| Maximum contact penetration | 0.655 mm | At most 1 mm |
| Maximum otherwise-disallowed contact penetration | 0.031 mm | At most 0.05 mm |
| Command speed / acceleration | 40 mm/s / 200 mm/s² | Existing limits |
| Production environment failure flags | Zero | Zero |

The independent replay runs from the original reset and original start pose, through the approach, first turn, gate passage, second turn and parking. All 16 corners pass through the opening. There is no lifting, tipping, shortcut, base motion, joint-limit violation or invalid collision under the existing checks. The saved action sequence lasts 27.95 s; the audit adds two seconds of zero commands and records 29.95 s to verify continued settling. Completion time is earlier than recording duration.

The production replay uses actual `env.step()` with its unchanged 30-second limit. It reports success, no failure and no timeout. It stops on success at the next 20 Hz control boundary, explaining the 27.246 versus 27.25 s timestamps.

## Evidence limits

This is **one nominal diagnostic demonstration**, replayed through two checking paths. It is not two independent tasks, a trained policy, or a success-rate estimate. The position margin is only about 0.68 mm, so this result does not establish robustness to different starts or physics parameters. Small jaw–gate contact below the existing 0.05 mm tolerance remains; strict zero contact is not claimed.

Full-path GPU replay and randomized task success have not been verified. Command limits are verified, but satisfying them does not by itself prove that measured arm motion is sufficiently smooth for the final RL acceptance tests. The existing contact/reset/rule suites remain valid because their production source files were not changed.

## Inspect and reproduce

```powershell
.\scripts\view-audit.ps1 -Trial full
.\scripts\park-gripper.ps1             # independently replay and verify
.\scripts\park-gripper.ps1 -Search     # redo finishing search from frozen prefix, then verify
```

The GUI shows the independently re-simulated trajectory as a recording. P pauses/resumes; R returns to the start. It is not a policy running live.

`artifacts/gripper_parking_prefix.npz` preserves the earlier continuous approach/turn; the final search re-simulates its first 492 commands (24.6 s), never inserting a terminal object pose. Search candidate rollouts restore state internally, but the delivered trajectory and verification runs execute continuously from a single initial reset.

Primary evidence:

- `artifacts/gripper_complete_path_audit.json/.npz`: independent geometric/contact/hold audit and GUI trajectory.
- `artifacts/gripper_complete_env_validation.json`: production environment success record.
- `artifacts/gripper_complete_path.json/.npz`: selected diagnostic commands and final search decision.
- `artifacts/gripper_complete_manifest.json`: hashes linking source, inputs and verification artifacts.

Next checkpoint before RL: replay the complete sequence in MuJoCo Warp and assess sensitivity, especially the small final position margin. Training remains a separate task.
