# Gripper-only checkpoint — 2026-10-04

**Update:** the subsequent [gripper contact checkpoint](GRIPPER_PHYSICS.md) passes the full 22-case suite at 1 ms with three-component jaw contacts. The initial 2/1 ms measurements below are retained as history and are superseded for current physics readiness. The complete gripper-only path is still unsolved.

The cyan attachment is removed. Pushing now uses the original SO-101 fixed and moving jaw collision geometry, including its mesh shapes. The gripper servo holds a closed command of -0.174 rad (approximately -10 degrees); it does not grip the block. The invisible, massless controller reference is on the original fixed fingertip, 10 mm above the table. There is no added physical pushing shape.

The starting fingertip position is (120, -60) mm. Reset audits exposed clearance problems with other starting poses; this pose passes the sampled overlap and settling checks without disabling collision geometry or relaxing collision thresholds. The camera mount remains part of the model.

## Verified on the current model

- A 16-second native MuJoCo diagnostic holds, approaches, pushes and settles using the real jaw mesh. The block moves 37.05 mm at 2 ms and 40.19 mm at 1 ms. No invalid-contact, tipping, excessive-penetration or joint-limit failure flags occur. Maximum penetration is 0.459 mm; initial fingertip hold drift is 0.097 mm.
- All 5,620 sampled geometric resets are overlap-free (500 native, 5,120 GPU). Another 20 native and 128 GPU one-second settling episodes pass. GPU subset resets preserve neighboring worlds.
- The rule tests exercise the updated jaw contact permissions and GPU rule kernels. See `artifacts/gripper_rules_tests.xml` for the current test result.
- The native GUI displays a recorded gripper push. This is a diagnostic, not RL.

The two timestep trajectories differ by about 3.15 mm at the end. Both simple pushes pass safety checks, but this does **not** meet the previous 2 mm trajectory-agreement target. Contact consistency needs further work before declaring the gripper ready for long training.

## What is not established yet

The previous full push–turn–park demonstrations used the retired attachment. They are historical evidence only and cannot validate this changed model. Old recordings are rejected by the current viewer to prevent showing attachment-based motions on a different robot geometry. Training preflight rejects the stale physics report after these source changes.

A complete gripper-only path, full contact regression/native–GPU dynamics agreement, and PPO/resume checks remain. The early approach reward/contact proxy still measures distance around the fixed fingertip; its learning behavior with wider jaw contacts needs review before training. No RL was started.

## Commands

```powershell
.\scripts\view.ps1                         # inspect the current scene
.\scripts\view-audit.ps1 -Trial gripper    # recorded simple push
.\scripts\validate-gripper.ps1            # repeat native push checks
.\scripts\validate-layout.ps1             # rules and randomized resets
```

Evidence: `artifacts/gripper_validation.json`, `artifacts/gripper_reset_validation.json`, `artifacts/gripper_rules_tests.xml`, `artifacts/gripper_push.npz`. Reports include source hashes. P pauses/resumes the viewer; R returns to the beginning.
