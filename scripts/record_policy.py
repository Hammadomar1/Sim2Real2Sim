"""Execute one learned-policy episode on GPU and record it for the Windows GUI."""
import argparse,hashlib,json,io
from pathlib import Path
import numpy as np
import torch
from replay_sensitivity import AuditEnv
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--stage',type=int,required=True,choices=range(1,6));p.add_argument('--seed',type=int,default=3000000);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    checkpoint_bytes=Path(a.checkpoint).read_bytes()
    saved=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
    variant=saved.get('reward_experiment',{}).get('variant')
    base=AuditEnv
    distance_level=None
    if 'distance_training' in saved:
        if a.stage!=2:raise ValueError('Distance-curriculum recordings require --stage 2.')
        from stage2_curriculum import DistanceCurriculumEnv
        base=DistanceCurriculumEnv
        if variant=='efficient_push':
            from efficient_push import EfficientPushEnv
            base=EfficientPushEnv
            if 'control_filter_alpha' in saved:
                from push_filter import FilteredEfficientPushEnv
                base=FilteredEfficientPushEnv
        distance_level=saved.get('policy_evaluation_level',saved['distance_training']['level'])
    elif variant=='corrected':
        from stage2_rewards import Stage2RewardEnv
        base=Stage2RewardEnv
    elif variant=='directional':
        from stage2_directional import DirectionalRewardEnv
        base=DirectionalRewardEnv
    class RecordingEnv(base):
        def reset(self,ids=None,*,preserve_terminal=False):
            if preserve_terminal:return self.get_observations()
            return super().reset(ids,preserve_terminal=preserve_terminal)
    kwargs={} if distance_level is None else {'level':distance_level}
    if 'control_filter_alpha' in saved:kwargs['alpha']=saved['control_filter_alpha']
    e=RecordingEnv(EnvConfig(num_envs=1,stage=a.stage,seed=a.seed,training=False),**kwargs);policy=load_policy(io.BytesIO(checkpoint_bytes),e)
    frames=[e.d.qpos[0].cpu().numpy().copy()];actions=[];applied_actions=[];mp=e.d.mocap_pos[0].cpu().numpy().copy();mq=e.d.mocap_quat[0].cpu().numpy().copy()
    with torch.inference_mode():
        for _ in range(e.max_episode_length):
            action=policy(e.get_observations());_,_,done,extras=e.step(action)
            if extras['numerical_failures'].any():raise RuntimeError('Numerical failure during policy recording')
            frames.append(e.d.qpos[0].cpu().numpy().copy());actions.append(action[0].cpu().numpy().copy())
            applied_actions.append(e.previous_action[0].cpu().numpy().copy())
            if done.any():break
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,qpos=frames,actions=actions,applied_actions=applied_actions,control_filter_alpha=saved.get('control_filter_alpha',1.),mocap_pos=mp,mocap_quat=mq,dt=e.cfg.control_dt,end_effector='closed_gripper')
    report=dict(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),reward_variant=variant or 'original',distance_level=distance_level,stage=a.stage,seed=a.seed,backend='warp',episodes=e.last_terminal,scope='One consecutive deterministic learned-policy rollout; not an aggregate evaluation.')
    if 'control_filter_alpha' in saved:report['control_filter_alpha']=saved['control_filter_alpha']
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
