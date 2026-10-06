"""Matched-scene checkpoint and observation-normalizer diagnosis; no training."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy, summarize
from stage2_curriculum import DistanceCurriculumEnv
from stage2_directional import approach_error

OUT=Path('artifacts/stage2_decline')
RUN=Path('runs/stage2_distance_seed73')

@torch.inference_mode()
def audit(env, actor):
    records=[]; starts=[]
    for batch in range(2):
        env.rng.manual_seed(5300000+batch*64); env.reset()
        starts.append(hashlib.sha256(torch.cat((env.d.qpos.clone(),env.goal),-1).cpu().numpy().tobytes()).hexdigest())
        active=torch.ones(64,device=env.device,dtype=torch.bool)
        metrics={k:torch.zeros(64,device=env.device) for k in ['position_reward','approach_reward','other_reward','contact_steps','action_clipped_steps','action_norm_sum','steps']}
        minimum=torch.full((64,),float('inf'),device=env.device)
        for _ in range(env.max_episode_length):
            xy,yaw,tip=(v.clone() for v in env.state()); goal=env.goal.clone()
            action=actor(env.get_observations())
            before=approach_error(xy,yaw,tip,goal)
            _,reward,done,extras=env.step(action)
            if extras['numerical_failures'].any():raise RuntimeError('Nonfinite physics')
            endxy,endyaw,endtip=env.terminal_geometry if env.terminal_geometry is not None else env.state()
            error=torch.linalg.vector_norm(endxy-goal[:,:2],dim=-1)
            pos=300*(torch.linalg.vector_norm(xy-goal[:,:2],dim=-1)-error)
            approach=30*(before-approach_error(endxy,endyaw,endtip,goal))
            values=dict(position_reward=pos,approach_reward=approach,other_reward=reward-pos-approach,
                contact_steps=(env.contact_distance(endxy,endyaw,endtip)<=.002).float(),
                action_clipped_steps=(action.abs()>1).any(-1).float(),action_norm_sum=action.norm(dim=-1),steps=torch.ones_like(error))
            for key,value in values.items():metrics[key][active]+=value[active]
            minimum[active]=torch.minimum(minimum[active],error[active])
            for row in env.last_terminal:
                i=int(row['env'])
                if not active[i]:continue
                row=copy.deepcopy(row);row['scenario_index']=batch*64+i
                row.update({k:float(v[i]) for k,v in metrics.items()})
                row['minimum_position_error_mm']=float(minimum[i]*1000)
                records.append(row);active[i]=False
            if not active.any():break
    records.sort(key=lambda r:r['scenario_index'])
    return dict(summary=summarize(records),initial_scene_hashes=starts,episodes=records,
        diagnostics={k:float(np.mean([r[k] for r in records])) for k in [*metrics,'minimum_position_error_mm']},
        entered_position_tolerance=sum(r['minimum_position_error_mm']<=10 for r in records))

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    checkpoints={k:torch.load(RUN/f'{v}.pt',map_location='cpu',weights_only=False) for k,v in [('best','best_level0'),('latest','latest')]}
    env=DistanceCurriculumEnv(EnvConfig(num_envs=64,stage=2,training=False),0)
    report={}
    for label,weights,norm in [('best','best','best'),('latest','latest','latest'),('latest_best_norm','latest','best'),('best_latest_norm','best','latest')]:
        actor=load_policy(RUN/('best_level0.pt' if weights=='best' else 'latest.pt'),env)
        state=actor.state_dict()
        for key,value in checkpoints[norm]['algorithm']['actor_state_dict'].items():
            if key.startswith('obs_normalizer.'):state[key]=value.to(env.device)
        actor.load_state_dict(state);actor.eval()
        result=audit(env,actor);result['weight_checkpoint']=weights;result['normalizer_checkpoint']=norm
        (OUT/f'{label}.json').write_text(json.dumps(result,indent=2))
        report[label]={k:v for k,v in result.items() if k!='episodes'}
        print(label,json.dumps(report[label]),flush=True)
    logs=EventAccumulator(str(RUN/'tensorboard'),size_guidance={'scalars':0});logs.Reload()
    report['losses']={tag:[dict(step=e.step,value=e.value) for e in logs.Scalars(tag)] for tag in logs.Tags()['scalars'] if tag.startswith('loss/')}
    report['checkpoints']={k:dict(iteration=v['iteration'],std=v['algorithm']['actor_state_dict']['distribution.std_param'].tolist(),optimizer_lrs=[g['lr'] for g in v['algorithm']['optimizer_state_dict']['param_groups']]) for k,v in checkpoints.items()}
    report['scope']='Read-only paired diagnosis; 128 identical development scenes per variant; no gradient updates.'
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
