"""Test bounded action response gains without changing physics or policy weights."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from compare_stage2_lr import ContactEnv
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy,summarize

class ResponseEnv(ContactEnv):
    def __init__(self,cfg,level=0,gain=1.,alpha=1.):
        self.response_gain=gain
        self.response_alpha=alpha
        super().__init__(cfg,level)
    def step(self,actions):
        if self.response_gain!=1:
            actions=torch.tanh(actions.clamp(-1,1)*self.response_gain)/math.tanh(self.response_gain)
        if self.response_alpha!=1:
            actions=self.previous_action+self.response_alpha*(actions.clamp(-1,1)-self.previous_action)
        return super().step(actions)

@torch.inference_mode()
def evaluate(path,gain,seed=6500000,episodes=64,alpha=1.):
    e=ResponseEnv(EnvConfig(num_envs=64,stage=2,training=False),gain=gain,alpha=alpha);policy=load_policy(path,e)
    rows=[];hashes=[]
    for batch in range(episodes//64):
        e.rng.manual_seed(seed+batch*64);e.reset()
        hashes.append(hashlib.sha256(torch.cat((e.d.qpos.clone(),e.goal),-1).cpu().numpy().tobytes()).hexdigest())
        initial=(e.goal[:,:2]-e.d.qpos[:,6:8]).norm(dim=-1);active=torch.ones(64,device=e.device,dtype=torch.bool)
        gap=torch.full((64,),5,device=e.device);bouts=torch.zeros_like(gap);contact=torch.zeros(64,device=e.device)
        speed=contact.clone();raw=contact.clone();steps=contact.clone()
        for _ in range(600):
            oldtip=e.state()[2].clone();action=policy(e.get_observations())
            _,_,_,extras=e.step(action)
            if extras['numerical_failures'].any():raise RuntimeError('Numerical failure')
            tip=e.terminal_geometry[2] if e.terminal_geometry is not None else e.state()[2]
            touch=e.end_touch!=0;bouts+=((gap>=5)&touch&active).long();gap=torch.where(touch,0,gap+1)
            contact[active]+=touch[active].float()*.05;raw[active]+=action[active].norm(dim=-1)
            speed[active]+=(tip[active,:2]-oldtip[active,:2]).norm(dim=-1)/.05*1000;steps[active]+=1
            for row in e.last_terminal:
                i=int(row['env'])
                if not active[i]:continue
                row=dict(row,scenario_index=batch*64+i,progress_mm=float(initial[i]*1000)-row['position_error_m']*1000,
                    contact_bouts=int(bouts[i]),contact_seconds=float(contact[i]),mean_tool_speed_mm_s=float(speed[i]/steps[i]),
                    mean_action_norm=float(raw[i]/steps[i]),jerk_per_second=row['joint_jerk_integral']/row['seconds'])
                rows.append(row);active[i]=False
            if not active.any():break
    rows.sort(key=lambda r:r['scenario_index'])
    return dict(checkpoint=str(path),gain=gain,response_alpha=alpha,seed=seed,initial_scene_hashes=hashes,summary=summarize(rows),
        metrics={k:float(np.mean([r[k] for r in rows])) for k in ['progress_mm','contact_bouts','contact_seconds','mean_tool_speed_mm_s','mean_action_norm','jerk_per_second']},episodes=rows,
        scope='Untrained controller-response ablation; contact bouts separated by at least 250 ms without sampled jaw contact; 20 Hz sampling.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/stage2_lr_comparison/fixed/final.pt');p.add_argument('--seed',type=int,default=6500000);p.add_argument('--episodes',type=int,default=64);a=p.parse_args()
    torch.set_num_threads(1);out=Path('artifacts/push_efficiency');out.mkdir(parents=True,exist_ok=True)
    for gain in [1.,2.,4.]:
        r=evaluate(a.checkpoint,gain,a.seed,a.episodes);(out/f'gain{int(gain)}.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='episodes'}),flush=True)

if __name__=='__main__':main()
