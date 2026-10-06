"""Regression checks plus reward accounting on the user's stalled trajectory."""
import json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from so101_m1.env import EnvConfig,PushTurnParkEnv
from stage2_rewards import Stage2RewardEnv,reward_terms
from stage2_directional import DirectionalRewardEnv,approach_error

def main():
    torch.set_num_threads(1)
    goal=torch.tensor([[.2,0,0.]])
    def terms(x0,x1,yaw0=0.,yaw1=0.,near0=0.,near1=0.):
        return reward_terms(torch.tensor([[x0,0.]]),torch.tensor([[x1,0.]]),torch.tensor([yaw0]),torch.tensor([yaw1]),torch.tensor([near0]),torch.tensor([near1]),goal)
    idle=terms(.15,.15);toward=terms(.15,.16);away=terms(.15,.14)
    assert idle['old']>0 and idle['new']<0
    assert toward['new']>idle['new']>away['new']
    assert torch.equal(terms(.15,.15,0.,1.)['new'],idle['new'])
    a=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=2,training=False,seed=19))
    b=Stage2RewardEnv(EnvConfig(num_envs=1,backend='native',stage=2,training=False,seed=19))
    for _ in range(5):
        _,r0,d0,_=a.step(torch.zeros(1,2));_,r1,d1,_=b.step(torch.zeros(1,2))
        assert torch.equal(a.d.qpos,b.d.qpos) and torch.equal(d0,d1)
        assert torch.allclose(r1,r0+b.last_reward_terms['delta'].float(),atol=1e-6)
    b.episode_length_buf[:]=b.max_episode_length-1
    previous=b.episode_return.clone();_,reward,done,_=b.step(torch.zeros(1,2))
    assert done.all() and abs(b.last_terminal[0]['return']-float(previous+reward))<1e-5
    for stage in [1,3,4,5]:
        a=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=stage,training=False,seed=19))
        b=Stage2RewardEnv(EnvConfig(num_envs=1,backend='native',stage=stage,training=False,seed=19))
        _,r0,d0,_=a.step(torch.zeros(1,2));_,r1,d1,_=b.step(torch.zeros(1,2))
        assert torch.equal(a.d.qpos,b.d.qpos) and torch.equal(r0,r1) and torch.equal(d0,d1)
    xy=torch.tensor([[.15,0.]]);yaw0=torch.zeros(1)
    behind=approach_error(xy,yaw0,torch.tensor([[.115,0.,.01]]),goal)
    wrong_side=approach_error(xy,yaw0,torch.tensor([[.185,0.,.01]]),goal)
    assert behind<wrong_side and torch.isfinite(approach_error(xy,yaw0,torch.tensor([[.115,0.,.01]]),torch.tensor([[.15,0.,0.]]))).all()
    for stage in [1,2,3,4,5]:
        a=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=stage,training=False,seed=19))
        b=DirectionalRewardEnv(EnvConfig(num_envs=1,backend='native',stage=stage,training=False,seed=19))
        _,r0,d0,_=a.step(torch.zeros(1,2));_,r1,d1,_=b.step(torch.zeros(1,2))
        assert torch.equal(a.d.qpos,b.d.qpos) and torch.equal(d0,d1)
        if stage!=2:assert torch.equal(r0,r1)
    b=DirectionalRewardEnv(EnvConfig(num_envs=1,backend='native',stage=2,training=False))
    b.episode_length_buf[:]=b.max_episode_length-1;previous=b.episode_return.clone()
    _,reward,done,_=b.step(torch.zeros(1,2));assert done.all() and abs(b.last_terminal[0]['return']-float(previous+reward))<1e-5
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',stage=2,training=False));z=np.load('artifacts/seed0/policy_stage2_seed3000000.npz');q=z['qpos'];xy=torch.tensor(q[:,6:8],dtype=torch.float64);tips=[]
    for pose in q:
        e.sim.mj_data.qpos[:]=pose;mujoco.mj_forward(e.model,e.sim.mj_data);tips.append(e.sim.mj_data.site_xpos[e.sid].copy())
    quat=q[:,9:13];yaw=torch.tensor(np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2)),dtype=torch.float64)
    near=e.contact_distance(xy,yaw,torch.tensor(np.array(tips)))
    g=z['mocap_pos'][e.goal_mocap,:2];gq=z['mocap_quat'][e.goal_mocap];gy=math.atan2(2*(gq[0]*gq[3]+gq[1]*gq[2]),1-2*(gq[2]**2+gq[3]**2))
    t=reward_terms(xy[:-1],xy[1:],yaw[:-1],yaw[1:],near[:-1],near[1:],torch.tensor([[g[0],g[1],gy]],dtype=torch.float64))
    out=dict(passed=True,checks=['idle contact loses reward','goal progress outranks idle','moving away is penalized','rotation alone has no stage-2 cost','physics and done flags unchanged','terminal return includes correction'],idle_per_step={k:float(v) for k,v in idle.items()},stalled_trajectory_component_sums={k:float(v.sum()) for k,v in t.items()},scope='Reward subtotal excludes unchanged success/failure bonuses and smoothness penalties; recorded path is evaluated without altering it.')
    Path('artifacts/stage2_reward_audit/reward_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
