"""Record matched stage-2 failures without altering the policy or controller."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from push_filter import FilteredEfficientPushEnv
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy, summarize

OUT = Path('artifacts/push_failure_diagnosis')
CHECKPOINT = Path('runs/push_efficiency/efficient/filtered.pt')

class TraceEnv(FilteredEfficientPushEnv):
    def capture(self):
        xy, yaw, tip = self.state()
        self.jaw_gap()  # initialize original jaw-tip geometry indices
        rot = self.d.xmat[:, self.jaw_ids].reshape(self.num_envs, -1, 3, 3)
        centers = self.d.xpos[:, self.jaw_ids] + (rot @ self.jaw_local[None, :, :, None]).squeeze(-1)
        nearest = (centers[:, :, :2] - xy[:, None, :]).norm(dim=-1).argmin(-1)
        jaw = centers[torch.arange(self.num_envs, device=self.device), nearest]
        return {k: v.detach().cpu().numpy().copy() for k, v in dict(
            qpos=self.d.qpos, xy=xy, yaw=yaw, tip=tip, jaw=jaw,
            gap=self.jaw_gap(), touch=self.rules.touch, action=self.previous_action,
            velocity=self.velocity, target=self.target_xy, hold=self.rules.hold).items()}

    def reset(self, ids=None, *, preserve_terminal=False):
        if preserve_terminal:
            self.before_reset = self.capture()
        return super().reset(ids, preserve_terminal=preserve_terminal)

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    e = TraceEnv(EnvConfig(num_envs=64, stage=2, training=False), alpha=.5)
    policy = load_policy(CHECKPOINT, e)
    e.rng.manual_seed(6600000)
    e.reset()
    goal = e.goal.cpu().numpy().copy()
    scene_hash = hashlib.sha256(torch.cat((e.d.qpos.clone(), e.goal), -1).cpu().numpy().tobytes()).hexdigest()
    mp = e.d.mocap_pos.cpu().numpy().copy()
    mq = e.d.mocap_quat.cpu().numpy().copy()
    history = [e.capture()]
    raw = []
    rows = {}
    for step in range(600):
        e.before_reset = None
        action = policy(e.get_observations())
        raw.append(action.cpu().numpy().copy())
        _, _, _, extras = e.step(action)
        if extras['numerical_failures'].any():
            raise RuntimeError('Numerical failure')
        history.append(e.before_reset if e.before_reset is not None else e.capture())
        for row in e.last_terminal:
            i = int(row['env'])
            if i not in rows:
                rows[i] = dict(row, scenario_index=i, steps=step+1)
        if len(rows) == 64:
            break
    arrays = {k: np.stack([h[k] for h in history]) for k in history[0]}
    np.savez_compressed(OUT/'traces.npz', **arrays, raw_actions=raw, goal=goal,
                        mocap_pos=mp, mocap_quat=mq, dt=.05)
    for i, row in rows.items():
        n = row['steps']
        xy = arrays['xy'][:n+1, i]
        tip = arrays['tip'][:n+1, i, :2]
        jaw = arrays['jaw'][:n+1, i, :2]
        error = np.linalg.norm(xy-goal[i, :2], axis=-1)*1000
        touch = arrays['touch'][1:n+1, i] != 0
        progress = -np.diff(error)
        speed = np.linalg.norm(np.diff(tip, axis=0), axis=-1)*20000
        object_speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1)*20000
        last = min(100, n)
        direction = goal[i, :2]-xy
        offset = jaw-xy
        side = (direction*offset).sum(-1)/(np.linalg.norm(direction, axis=-1)*np.linalg.norm(offset, axis=-1)+1e-12)
        # Positive means the jaw is on the goal-facing side, a geometric clue,
        # not proof that a different-side contact is physically feasible.
        ever = bool(touch.any())
        net = float(error[0]-error[-1])
        stalled = abs(float(error[-last-1]-error[-1])) < .5
        stopped = float(speed[-last:].mean()) < 1.
        category = ('success' if row['success'] else 'no_sampled_contact' if not ever
                    else 'net_wrong_direction' if net < -1
                    else 'stopped_outside_goal' if stalled and stopped
                    else 'moving_without_finishing' if stalled
                    else 'still_progressing_or_oscillating')
        row.update(category=category, initial_error_mm=float(error[0]),
                   final_error_mm=float(error[-1]), minimum_error_mm=float(error.min()),
                   net_progress_mm=net, contact_seconds=float(touch.sum()*.05),
                   first_contact_seconds=float((np.flatnonzero(touch)[0]+1)*.05) if ever else None,
                   last5_tool_speed_mm_s=float(speed[-last:].mean()),
                   last5_object_speed_mm_s=float(object_speed[-last:].mean()),
                   last5_progress_mm=float(error[-last-1]-error[-1]),
                   last5_contact_seconds=float(touch[-last:].sum()*.05),
                   last5_command_speed_mm_s=float(np.linalg.norm(arrays['velocity'][-last:, i],axis=-1).mean()*1000) if n==600 else float(np.linalg.norm(arrays['velocity'][n-last+1:n+1,i],axis=-1).mean()*1000),
                   last5_jaw_goal_side_cosine=float(side[-last:].mean()),
                   last5_jaw_gap_mm=float(arrays['gap'][n-last+1:n+1,i].mean()*1000),
                   goal_facing_side_at_end=bool(side[-last:].mean()>.3),
                   cumulative_toward_mm=float(progress.clip(min=0).sum()),
                   cumulative_away_mm=float((-progress).clip(min=0).sum()))
    ordered = [rows[i] for i in sorted(rows)]
    counts = {k: sum(r['category']==k for r in ordered) for k in sorted({r['category'] for r in ordered})}
    report = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                  seed=6600000, initial_scene_hash=scene_hash, summary=summarize(ordered),
                  categories=counts, episodes=ordered,
                  limitations='20 Hz contact sampling can miss brief contacts. Jaw-side cosine is geometric evidence, not a validated alternative push. Diagnostic thresholds: last 5 s progress <0.5 mm; stopped tool <1 mm/s; net wrong direction >1 mm away.')
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
