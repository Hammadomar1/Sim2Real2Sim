"""Summarize controlled settling and moving-contact isolation experiments."""
import hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from so101_m1.scene import ROOT,ARTIFACTS

def main():
    box=json.loads((ARTIFACTS/'isolation_settling_True.json').read_text())
    plane=json.loads((ARTIFACTS/'isolation_settling_True_plane.json').read_text())
    native=json.loads((ARTIFACTS/'isolation_native_settling.json').read_text())
    rows=['# GPU settling and numerical-instability isolation','',
          '**Checkpoint completed: the parking jitter is isolated to a GPU block/table contact case. Full-path numerical failures are not yet resolved. RL remains off.**','',
          'Production robot geometry, controller, scene physics and acceptance thresholds have not been changed. Plane-table and box-jaw variants exist only in diagnostic processes. No pushing attachment was added.','',
          '## Controlled evidence','',
          'Each GPU settling case uses three worlds with identical initial joint/object positions, zero velocities and solver warmstart, and frozen motor targets. It runs for 10 simulated seconds and records every 1 ms. Metrics below cover the final five seconds. The hold column checks only the existing velocity thresholds (5 mm/s and 5 degrees/s); it is not complete-task success. Initial poses are diagnostic resets, not a continuously executed solution.','',
          '| Pose / surface | Peak linear speed (mm/s) | Peak angular speed (deg/s) | Shortest of the three worlds\' longest speed holds (s) |',
          '|---|---:|---:|---:|']
    for label,r in [('Box: '+r['case'],r) for r in box['results']]+[('Plane: '+r['case'],r) for r in plane['results'] if r['case'].startswith('gpu_terminal')]:
        rows.append(f'| {label} | {1000*max(r["last_5s_max_linear"]):.6f} | {math.degrees(max(r["last_5s_max_angular"])):.6f} | {min(r["longest_velocity_hold_seconds"]):.3f} |')
    rows += ['', 'The start pose and native final parking pose settle cleanly on GPU. The GPU replay\'s final pose jitters with the gripper present **and with the arm away**. Native MuJoCo settles that same pose for five continuous seconds below the speed thresholds, both with and without the arm nearby. This rules out the IK controller and gripper contact as necessary causes of the reproduced resting jitter.','',
             'Changing only the diagnostic table collision shape from a finite box to a plane at the same top height eliminates that jitter in the tested cases. Mass, friction, contact solver settings, timestep and speed thresholds remain unchanged. This implicates the box/table contact calculation for this pose; it does not establish an upstream implementation defect or guarantee every pose is stable. A plane is infinite and must be restricted or accompanied by appropriate finite-workspace failure checks before production adoption.','',
             '## Moving sequence and nonfinite states','',
             'A shared PyTorch/Warp stream did not cure the full replay: the three-world control still included a nonfinite state at 8.5 seconds. Stream ordering alone is therefore not a sufficient fix. NVIDIA documents shared-stream interoperability here: [Warp PyTorch interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html).','',
             'With explicitly identical qpos, qvel, motor targets, warmstart and initial site positions, a separate moving-controller diagnostic first exceeded a 1e-7 qpos difference between worlds at 0.9 seconds. Clearing warmstart every control interval did not remove that divergence. The initial disagreement is tiny and is not itself evidence of broken physics; it grows during this contact-sensitive fixed-command path.','',
             'A diagnostic that disables the detailed jaw collision meshes while retaining the jaw box collisions delayed that difference to 5.1 seconds. All three worlds stayed finite for 30 seconds with zero safety flags in that run. This changes contact geometry and was **not** a full-task success test. It is a candidate for geometry review, not permission to silently remove real gripper surfaces. The baseline moving-controller diagnostic also stayed finite in its recorded run; these tests do not prove that meshes alone cause the intermittent NaNs.','',
             'The moving-controller diagnostic bypasses episode termination to inspect numerical evolution and can continue after safety flags. It must not be used as an accepted physical path or task-success result. Parking-pose controller-only holds likewise start beyond the gate and can set shortcut flags; their purpose is settling isolation.','',
             '## Decision for the next checkpoint','',
             '1. Validate a numerically stable table-contact representation over multiple object orientations, with finite-table boundaries preserved.',
             '2. Inspect jaw mesh/primitive contact overlap and trace the first nonfinite transition with contact forces and solver state; retain physically representative gripper geometry.',
             '3. Repeat complete native/GPU paths and perturbation checks after any production fix. Then improve object-pose feedback and parking margin.','',
             'The resting-jitter reproducer and successful plane comparison are complete. The full-path NaN cause, a production contact fix, and RL readiness remain open. No success tolerance was relaxed.','',
             '## Reproduce one test group at a time','',
             '```powershell',
             '.\\scripts\\isolate-gpu.ps1 -Mode settling',
             '.\\scripts\\isolate-gpu.ps1 -Mode plane',
             '.\\scripts\\isolate-gpu.ps1 -Mode native',
             '.\\scripts\\isolate-gpu.ps1 -Mode controller',
             '.\\scripts\\isolate-gpu.ps1 -Mode cold',
             '.\\scripts\\isolate-gpu.ps1 -Mode jaws',
             '.\\scripts\\isolate-gpu.ps1 -Mode report',
             '.\\scripts\\isolate-gpu.ps1 -Mode view',
             '```','',
             'Raw results and trajectories: `artifacts/isolation_*.json` and `.npz`. The manifest hashes current diagnostic code, input recordings and result files; exploratory reports retain the source hashes from their own run. Production source hashes are unchanged from the prior checkpoint. The GUI is a recorded settling test with the original box table, not a policy or live physics.','',
             '![Millisecond settling comparison](artifacts/gpu_settling_isolation.png)','']
    (ROOT/'GPU_ISOLATION_CHECKPOINT.md').write_text('\n'.join(rows),encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for suffix,label in [('', 'Box table'),('_plane','Plane diagnostic')]:
        data=np.load(ARTIFACTS/f'isolation_gpu_terminal_arm_away_True{suffix}.npz')['states'];t=np.arange(len(data))*.001
        axes[0].plot(t,np.linalg.norm(data[:,0,19:22],axis=-1)*1000,label=label,linewidth=.8)
        axes[1].plot(t,np.linalg.norm(data[:,0,22:25],axis=-1)*180/math.pi,label=label,linewidth=.8)
    for ax in axes:ax.axhline(5,ls='--',color='black',label='Stationary threshold');ax.grid(alpha=.2);ax.legend();ax.set_xlim(5,10)
    axes[0].set(ylabel='Block linear speed (mm/s)',title='Same GPU-terminal pose, arm away, frozen motor targets')
    axes[1].set(ylabel='Block angular speed (degrees/s)',xlabel='Simulated time (s)')
    fig.savefig(ARTIFACTS/'gpu_settling_isolation.png',dpi=150);plt.close(fig)
    frames=np.load(ARTIFACTS/'isolation_gpu_terminal_gripper_present_True.npz')['states'][::50,0,:13]
    z=np.load(ARTIFACTS/'full_replay_warp_single_nominal.npz')
    np.savez(ARTIFACTS/'isolation_settling_view.npz',qpos=frames,dt=.05,mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],end_effector='closed_gripper')
    paths=list((ROOT/'scripts').glob('*isolat*.py'))+[ROOT/'scripts/native_settle_reference.py']+list(ARTIFACTS.glob('isolation_*.json'))+[ARTIFACTS/'gripper_complete_path_audit.npz',ARTIFACTS/'full_replay_warp_single_nominal.npz']+[ROOT/'src/so101_m1'/f for f in ['scene.py','env.py','rules.py']]
    manifest=dict(ready_for_rl=False,resting_jitter_isolated=True,full_path_nonfinite_resolved=False,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='isolation_manifest.json'})
    (ARTIFACTS/'isolation_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote GPU_ISOLATION_CHECKPOINT.md, comparison figure, viewer recording and manifest.')

if __name__=='__main__':main()
