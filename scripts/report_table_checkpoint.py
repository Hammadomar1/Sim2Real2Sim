"""Measured table-candidate validation and moving-contact failure report."""
import hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from so101_m1.scene import ARTIFACTS,ROOT,load_model

def main():
    v=json.loads((ARTIFACTS/'table_candidate_validation.json').read_text())
    names=['moving_trace_plane','moving_trace_plane_repeat','moving_trace_box','moving_trace_plane_resets','moving_trace_box_resets']
    reports={n:json.loads((ARTIFACTS/f'{n}.json').read_text()) for n in names}
    r=reports['moving_trace_plane'];joint=next(e for e in r['first_failures'] if e['flags']&64);collision=next(e for e in r['first_failures'] if e['flags']&16)
    model=load_model();lower=float(model.jnt_range[5,0]);pair=next(c for c in joint['contacts'] if 'moving_jaw_mesh_1' in c['pair'])
    hit=next(c for c in collision['contacts'] if 'gate_south' in c['pair'])
    s=v['settling']
    lines=['# Stable table contact and moving-contact trace checkpoint','',
           '**Table candidate: passed the tested settling and boundary checks. Complete-path readiness: not passed. RL remains off.**','',
           '## Implemented candidate','',
           '`scripts/table_candidate.py` builds `artifacts/table_candidate.xml`. A hidden plane provides the top contact surface at z=0; a non-colliding box preserves the original visible slab (x from -0.21 to 0.49 m, y from -0.30 to 0.30 m). The original gripper meshes and primitives, gate, mass, friction, solver settings and success tolerances remain intact. The candidate is opt-in in the validation/trace tools; it is not the production default.','',
           'The original task rule requires every block corner to remain within x=[0.08,0.32] m and y=[-0.14,0.14] m. That rectangle lies strictly inside the visible table. The physics-step monitor rejects excursions; an infinite plane cannot make an outside pose a valid success. This preserves finite **task boundaries**, not physical falling off the slab: the plane still supports objects outside it during diagnostic continuation. Edge-fall dynamics are outside this candidate\'s validated scope.','',
           '## Validation results','',
           f'- Settling: **{s["passed"]}/{s["cases"]}** cases pass. Three positions, 24 yaw angles at 15-degree intervals, three initial tilt/height combinations. Each runs 3 seconds with frozen motor targets; every 1 ms in the last second is checked against the original 5 mm/s and 5 degrees/s thresholds and physics safety flags.',
           f'- Maximum speed in that final second: **{s["max_last_second_linear"]*1000:.6f} mm/s**, **{math.degrees(s["max_last_second_angular"]):.6f} degrees/s**. All recorded states finite.',
           '- Eight boundary probes: four task-boundary violations and four visible-table-edge violations. **8/8 rejected on GPU and 8/8 in native MuJoCo**, each with outside-workspace flag 4. These are synthetic boundary snapshots, not edge-crossing trajectories.',
           '- This is a table-contact test, not a 216-episode full-task success rate. Stage 3 moves the gate away during settling, so gate contact cannot confound that test.','',
           '## Moving failures, localized at 1 ms','',
           'The trace records qpos, velocity, acceleration, generalized constraint force, motor targets, solver iterations, constraint count, safety flags, stationary-hold counter and actual GPU contact pairs/normal constraint forces. It replays the saved commands with the original controller. It deliberately continues after failure (or resets failed worlds in the reset variant) for diagnosis; continuation is not a valid demonstration.','',
           f'1. **Gripper joint limit at {joint["seconds"]:.3f} s**, world {joint["world"]}: gripper angle {joint["qpos"][5]:.9f} rad; physical lower limit {lower:.9f} rad; allowed rule threshold {lower-.001:.9f} rad. The target is -0.174 rad, close to the hard stop. At this instant the moving-jaw mesh presses block_0 with **{pair["normal_constraint_force_n"]:.2f} N** normal constraint force and **{-pair["distance_m"]*1000:.3f} mm** penetration while the block also contacts the north gate. This identifies a loaded/wedged contact condition, not a reason to relax the joint-limit rule.',
           f'2. **Jaw/gate collision at {collision["seconds"]:.3f} s**, world {collision["world"]}: fixed_jaw_mesh_0 against gate_south, **{-hit["distance_m"]*1000:.3f} mm** penetration and **{hit["normal_constraint_force_n"]:.2f} N** normal constraint force. The invalid-contact tolerance remains 0.050 mm.',
           '3. The box-table comparison also catches moving_jaw_mesh_0 against gate_north around 4.47 s. The repeated plane run reaches a jaw/gate collision rather than a valid parking finish. Different outcomes remain possible from the same nominal command sequence.','',
           'These are simulated constraint forces, not calibrated hardware force measurements. The first two reported events used 3 and 6 solver iterations, respectively, below the configured 100-iteration limit; solver-iteration exhaustion is not shown at those events.','',
           '| Instrumented run | Duration (s) | First failure times (s) | Nonfinite detected | Full-task completion before failure |',
           '|---|---:|---|---|---|']
    for name,report in reports.items():
        times=', '.join(f'w{e["world"]}: {e["seconds"]:.3f}' for e in report['first_failures']) or 'none'
        lines.append(f'| {name} | {report["simulated_seconds"]:.1f} | {times} | {report["first_nonfinite"] is not None} | {report["first_success_seconds"]} |')
    lines+=['','No NaN/Inf recurred in these five instrumented, three-world runs. This does **not** resolve the intermittent failures recorded previously: added instrumentation changes execution details, and the underlying cause has not been established. Resets produced zero immediate change in the unreset worlds\' qpos, qvel, motor targets and warmstart in the measured checks; that rules out direct mutation of those fields in these runs, not every possible reset/derived-state issue.','',
            '## Next checkpoint','',
            'Integrate the validated table candidate with explicit finite-task-boundary documentation, then replace the fragile gate repositioning with object-pose feedback and more clearance. Prevent sustained gate wedging and provide gripper joint-limit margin without removing gripper surfaces or weakening collision rules. Re-run native/GPU full sequences, perturbations and a repeated-reset numerical soak before RL. The intermittent NaN issue remains on that validation gate.','',
            '## Commands and GUI','',
            '```powershell','.\\scripts\\table-checkpoint.ps1 -Mode validate','.\\scripts\\table-checkpoint.ps1 -Mode trace-plane','.\\scripts\\table-checkpoint.ps1 -Mode trace-box','.\\scripts\\table-checkpoint.ps1 -Mode reset-plane','.\\scripts\\table-checkpoint.ps1 -Mode reset-box','.\\scripts\\table-checkpoint.ps1 -Mode report','.\\scripts\\table-checkpoint.ps1 -Mode view','```','',
            'The GUI clip shows the measured approach to the gripper-limit event and stops its recording at the first policy frame containing that failure, then loops. It is a failed diagnostic replay, not live physics or a trained policy. The viewer uses the existing scene for display; the candidate has the same visible tabletop and robot.','',
            'Evidence: `table_candidate_validation.json`, `moving_trace_*.json`, full millisecond state arrays and final contact-history chunks in `moving_trace_*_detail.npz`. Contacts at first failures are included directly in JSON. The detailed tensor field layout is defined by `state_record` in the tracer. Reports retain their run-source hashes; optional reset tracing was added after the initial no-reset runs. `table_checkpoint_manifest.json` hashes the current scripts, inputs and reports.','',
            '![Gate failure trace](artifacts/table_candidate_gate_trace.png)','']
    (ROOT/'TABLE_CONTACT_CHECKPOINT.md').write_text('\n'.join(lines),encoding='utf-8')
    data=np.load(ARTIFACTS/'moving_trace_plane_detail.npz')['states'];t=(np.arange(len(data))+1)*.001;w=joint['world'];mask=(t>=12.8)&(t<=14)
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    ax[0].plot(t[mask],data[mask,w,5],label='Measured gripper angle');ax[0].axhline(lower-.001,ls='--',color='red',label='Failure threshold');ax[0].axhline(-.174,ls=':',color='black',label='Closed target');ax[0].set_ylabel('Gripper angle (rad)')
    ax[1].plot(t[mask],data[mask,w,42],label='Gripper constraint torque');ax[1].set(xlabel='Simulated time (s)',ylabel='Constraint torque (N m)')
    for a in ax:a.axvline(joint['seconds'],color='red',alpha=.5);a.grid(alpha=.25);a.legend()
    fig.suptitle('Candidate table, original gripper: joint-limit failure during gate contact');fig.savefig(ARTIFACTS/'table_candidate_gate_trace.png',dpi=150);plt.close(fig)
    z=np.load(ARTIFACTS/'gripper_complete_path.npz');end=math.ceil(joint['seconds']/.05)*50
    np.savez(ARTIFACTS/'table_candidate_failure_view.npz',qpos=data[49:end:50,w,:13],dt=.05,mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],end_effector='closed_gripper')
    files=[ROOT/'scripts'/n for n in ['table_candidate.py','validate_table_candidate.py','trace_moving_contacts.py','report_table_checkpoint.py']]+[ARTIFACTS/f'{n}.json' for n in names]+[ARTIFACTS/'table_candidate_validation.json',ARTIFACTS/'gripper_complete_path.npz']
    (ARTIFACTS/'table_checkpoint_manifest.json').write_text(json.dumps(dict(table_candidate_pass=s['passed']==s['cases'] and v['boundary_pass'],ready_for_rl=False,nonfinite_root_cause_resolved=False,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
    print('Wrote TABLE_CONTACT_CHECKPOINT.md, plot, GUI failure clip, and manifest.')

if __name__=='__main__':main()
