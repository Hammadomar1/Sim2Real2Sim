"""Build a reviewable summary of measured fixed-command replay outcomes."""
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replay_sensitivity import cases
from so101_m1.scene import ARTIFACTS, ROOT


def main():
    paths = [ARTIFACTS/f'full_replay_native_{c["name"]}.json' for c in cases()]
    paths += [ARTIFACTS/f'full_replay_{name}.json' for name in
              ['native_half_timestep', 'warp_single', 'warp_sensitivity', 'warp_replicas_3']]
    reports = {p.stem: json.loads(p.read_text()) for p in paths}
    for report in reports.values():
        for name, digest in report['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Stale replay evidence: {name}; rerun the suite.')
    native = [reports[f'full_replay_native_{c["name"]}']['results'][0] for c in cases()]
    gpu = reports['full_replay_warp_single']
    batch = reports['full_replay_warp_sensitivity']['results']
    replicas = reports['full_replay_warp_replicas_3']['results']
    half = reports['full_replay_native_half_timestep']['results'][0]
    def outcome(r):
        return 'success' if r['success'] else ('timeout' if r['timeout'] else ', '.join(r['failure_categories']))
    def metric(value, scale=1):
        return 'nonfinite' if value is None else f'{value*scale:.3f}'
    lines = ['# Full-sequence GPU replay and sensitivity checkpoint', '',
             '**Assessment: NOT READY for RL training. Tests completed; full-sequence GPU stability and replay robustness did not pass.**', '',
             'The closed gripper and existing physics/controller/success tolerances were retained. No RL training was run.', '',
             '## What was tested', '',
             'The saved demonstration commands were replayed at 20 Hz for up to 30 seconds with the production controller and physics-step success/collision rules. The diagnostic wrapper retains terminal states for recording, then resets finished worlds while other worlds continue. It does not change success or failure calculations. Joint control uses feedback; the recorded task commands do not adapt to the object pose.', '',
             'Thirteen cases per backend: nominal; initial x/y shifts of +/-1 mm; yaw shifts of +/-1 degree; sliding friction, block mass/inertia, and position actuator gain changes of +/-5%, one factor at a time. Friction scales the sliding coefficient on all geoms. Gain scales position stiffness, retaining damping. Native cases run separately; GPU cases run in one 13-world batch. Mass constants are recomputed for the batch containing mass perturbations. The single GPU and three-replica controls do not request this recomputation. Thus backend and batch/setup effects are not separated by this suite.', '',
             'These are diagnostic cases, not 500 unseen trials, trained-policy evaluation, or a statistical success-rate estimate. Observation noise and control delay are not assessed here.', '',
             '## Results', '',
             '| Case | Native outcome (s) | GPU batch outcome (s) |',
             '|---|---|---|']
    for n, g in zip(native, batch):
        lines.append(f'| {n["case"]["name"]} | {outcome(n)} ({n["seconds"]:.2f}) | {outcome(g)} ({g["seconds"]:.2f}) |')
    single = gpu['results'][0]
    trace = gpu['traces'][0]
    max_sampled_hold = max(t['stationary_hold_seconds'] for t in trace)
    lines += ['', f'Native nominal: {metric(native[0]["position_error_m"],1000)} mm and {metric(native[0]["orientation_error_rad"],180/math.pi)} degrees at {native[0]["seconds"]:.2f} s.', '',
              f'Small perturbations completed: native {sum(bool(r["success"]) for r in native[1:])}/12; GPU batch {sum(bool(r["success"]) for r in batch[1:])}/12.', '',
              f'Single-world GPU nominal: **{outcome(single)}** at {single["seconds"]:.2f} s; final error {metric(single["position_error_m"],1000)} mm / {metric(single["orientation_error_rad"],180/math.pi)} degrees. Gate passed: {bool(single["passed_gate"])}. Failure flags: {single["failure_flags"]}. Largest hold counter observed at the 20 Hz logging points: {max_sampled_hold:.3f} s (required: 1 s). The actual acceptance monitor runs at every physics step. The sampled maximum is not an exact substep maximum.', '',
              f'Native half-timestep control (0.5 ms, same actions): **{outcome(half)}**, {metric(half["position_error_m"],1000)} mm / {metric(half["orientation_error_rad"],180/math.pi)} degrees at {half["seconds"]:.2f} s.', '',
              'Three nominal GPU worlds, same prescribed initial pose, goals and commands:', '']
    for r in replicas:
        lines.append(f'- {r["case"]["name"]}: {outcome(r)} at {r["seconds"]:.2f} s.')
    lines += ['', 'GPU outcomes also varied across exploratory reruns. Only the final source-matched suite is tabulated above; these figures should not be interpreted as deterministic per-case predictions. Nonfinite states occurred in GPU testing and remain an unresolved blocker. Floating-point/contact sensitivity, initialization, batching and solver behavior need controlled isolation; their individual causes have not been established.', '',
              '## Interpretation and next checkpoint', '',
              'The native nominal path remains a valid feasibility demonstration. Its final position margin is under 1 mm, and fixed-command replay is fragile to small perturbations and timestep changes. A feedback policy may recover from trajectory deviations, but that does not excuse numerical failures or unresolved resting-contact behavior.', '',
              'Next: isolate GPU resting-block and identical-world consistency in short controlled tests, then retest the full path with object-pose feedback and more parking margin. Preserve collision rules and the one-second stationary requirement. No training-readiness approval was generated; long training remains blocked by the existing preflight.', '',
              '## Reproduce and view', '', '```powershell',
              '.\\scripts\\replay-sensitivity.ps1 -Mode all',
              '.\\scripts\\replay-sensitivity.ps1 -Mode view',
              '```', '',
              'The GUI displays recorded GPU states in the Windows MuJoCo viewer; it does not resimulate them or run a policy. P pauses/resumes and R restarts paused. The native successful recording remains available with `scripts/view-audit.ps1 -Trial full`.', '',
              'Evidence: `artifacts/full_replay_*.json` and matching NPZ trajectories, `artifacts/full_replay_summary.json`, and the comparison plot below. Reports include input and source hashes. Nonfinite numerical metrics are JSON null; failed trajectories remain diagnostic evidence.', '',
              '![Native and GPU nominal comparison](artifacts/full_replay_comparison.png)', '']
    (ROOT/'GPU_REPLAY_SENSITIVITY.md').write_text('\n'.join(lines),encoding='utf-8')
    summary = dict(checkpoint='full_sequence_gpu_replay_and_sensitivity',ready_for_rl=False,
                   native_nominal=native[0],gpu_single=single,native_perturbations_passed=sum(bool(r['success']) for r in native[1:]),
                   gpu_perturbations_passed=sum(bool(r['success']) for r in batch[1:]),
                   native_half_timestep=half,gpu_replicas=replicas,
                   report_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ARTIFACTS/'full_replay_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name, report in [('Native',reports['full_replay_native_nominal']),('GPU single',gpu)]:
        tr=report['traces'][0];t=np.array([r['seconds'] for r in tr]);xy=np.array([r['xy'] for r in tr])
        axes[0,0].plot(xy[:,0]*1000,xy[:,1]*1000,label=name)
        axes[0,1].plot(t,np.linalg.norm(xy-[.27,0],axis=1)*1000,label=name)
        axes[1,0].plot(t,[r['linear_speed']*1000 for r in tr],label=name)
        axes[1,1].plot(t,[r['stationary_hold_seconds'] for r in tr],label=name)
    axes[0,0].scatter([270],[0],marker='x',color='black',label='Goal')
    axes[0,0].set(xlabel='Block x (mm)',ylabel='Block y (mm)',title='Block origin trajectory')
    axes[0,1].set(xlabel='Time (s)',ylabel='Position error (mm)',title='Parking position error')
    axes[0,1].axhline(10,color='black',ls='--')
    axes[1,0].set(xlabel='Time (s)',ylabel='Linear speed (mm/s)',title='Speed sampled at 20 Hz')
    axes[1,0].axhline(5,color='black',ls='--')
    axes[1,1].set(xlabel='Time (s)',ylabel='Hold counter (s)',title='Physics-step hold counter, sampled at 20 Hz')
    axes[1,1].axhline(1,color='black',ls='--')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Fixed-command replay: native nominal succeeds; GPU nominal times out')
    fig.savefig(ARTIFACTS/'full_replay_comparison.png',dpi=150);plt.close(fig)
    print('Wrote GPU_REPLAY_SENSITIVITY.md and full_replay_summary.json; readiness is FALSE.')


if __name__=='__main__':main()
