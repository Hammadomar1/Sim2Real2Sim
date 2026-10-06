# Gripper gate repositioning — 2026-10-04

**Passed for the canonical starting scene in native MuJoCo.** The closed gripper now changes contact safely, pushes the entire block through the original gate, withdraws and lets the block settle. This checkpoint does not include final rotation and parking or RL training.

## What fixed the jam

The old attempt kept pushing inside the L-shaped notch until the gripper was too close to the barrier to withdraw sideways. The new diagnostic changes contact earlier, after the first 7 seconds of a continuously simulated approach. It moves backward, withdraws toward the open side, moves around the block and pushes its rear face. A final backward withdrawal is checked together with stopping and settling. Evaluating a moving endpoint alone had missed a later jaw–gate touch.

The search uses actual simulated contact rollouts and object-pose feedback. It is an offline diagnostic, not a learned policy or a claim of real-time planning. Candidate simulations restore saved states during search; the delivered trajectory and both verification replays run continuously from one initial reset with no object teleporting.

The robot, gate (215 mm), parking target (270 mm), contact parameters, speed/acceleration limits and collision tolerances are unchanged from the passing contact checkpoint.

## Measured results

| Check | Result |
|---|---:|
| Entire block through gate | 14.376 s |
| All 16 block corners crossed inside opening | Yes |
| Final stationary duration beyond gate | 3.824 s |
| Final minimum block clearance beyond barrier | 1.470 mm |
| Maximum contact penetration | 0.626 mm, below 1 mm limit |
| Maximum otherwise-disallowed contact penetration | 0.031 mm, below 0.05 mm tolerance |
| Invalid contact / tipping / shortcut flags | None |
| Production environment replay | Passed gate; zero failure flags |
| Full parking task | **Not completed** |

Small tolerated jaw–gate contact remains; this is not a zero-contact claim. Final pose is approximately (257.17, -8.44) mm at -27.74 degrees. The parking target is (270, 0) mm at 90 degrees, so a substantial final turn is still needed. No randomized gate-passage success rate or full-path GPU agreement is claimed.

## Inspect and reproduce

```powershell
.\scripts\view-audit.ps1 -Trial gate
.\scripts\gate-reposition.ps1            # independently re-simulate saved actions
.\scripts\gate-reposition.ps1 -Search    # regenerate the diagnostic search, then verify
```

The GUI loops recorded states from the independent audit. P pauses/resumes; R returns to the start. It does not run RL.

Evidence in `artifacts`: `gripper_gate_reposition_audit.json/.npz`, `gripper_gate_env_validation.json`, `gripper_gate_reposition.json/.npz`. `gripper_gate_prefix_source.npz` freezes the earlier approach input; `gripper_gate_escape.json/.npz` preserves the withdrawal/rear-contact stage. Verification reports hash their input trajectory and relevant source files.

Next checkpoint: rotate and park the block from this valid gate-passage state, then verify the entire continuous sequence. RL remains off.
