"""Bounded GPU search for a valid recovery; diagnostic control, not a policy."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import torch
from diagnose_push_failures import TraceEnv,CHECKPOINT
from push_filter import FilteredEfficientPushEnv
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy
from so101_m1.preflight import require_preflight

OUT=Path('artifacts/push_recovery')

class KeepEnv(FilteredEfficientPushEnv):
    def reset(self,ids=None,*,preserve_terminal=False):
        if preserve_terminal:return self.get_observations()
        return super().reset(ids,preserve_terminal=preserve_terminal)

def copy_worlds(env,state,indices):
    for group in ['sim','task','rules']:
        for k,v in state[group].items():
            getattr(env.d if group=='sim' else env.rules if group=='rules' else env,k)[:]=v[indices].to(env.device)
    if 'distance_curriculum' in state:
        env.distance_levels[:]=state['distance_curriculum']['levels'][indices].to(env.device)
        env.initial_goal_distance[:]=state['distance_curriculum']['initial_goal_distance'][indices].to(env.device)
    if 'efficient_push_state' in state:
        for k,v in state['efficient_push_state'].items():getattr(env,k)[:]=v[indices].to(env.device)
    env.sim.forward()

def jaw(env):
    env.jaw_gap()
    rot=env.d.xmat[:,env.jaw_ids].reshape(env.num_envs,-1,3,3)
    centers=env.d.xpos[:,env.jaw_ids]+(rot@env.jaw_local[None,:,:,None]).squeeze(-1)
    # Use a fixed material jaw-tip sphere, selected at the stalled state.
    return centers[:,0,:2]

@torch.inference_mode()
def main():
    require_preflight();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    statepath=OUT/'stalled_bank.pt'
    if not statepath.exists():
        e=TraceEnv(EnvConfig(num_envs=64,stage=2,training=False),alpha=.5)
        policy=load_policy(CHECKPOINT,e);e.rng.manual_seed(6600000);e.reset()
        prefix=[e.d.qpos[56].cpu().numpy().copy()];actions=[];ended=set()
        for t in range(400):
            a=policy(e.get_observations());e.step(a);actions.append(a[56].cpu().numpy().copy())
            prefix.append(e.d.qpos[56].cpu().numpy().copy())
            ended.update(int(r['env']) for r in e.last_terminal)
        if 56 in ended:raise RuntimeError('Chosen scene ended before intervention')
        torch.save(dict(state=e.state_dict(),prefix=np.array(prefix),actions=np.array(actions),ended=list(ended)),statepath)
    bank=torch.load(statepath,weights_only=False);state=bank['state']
    # Exposed boundary of the union of the two block boxes, counterclockwise.
    vertices=np.array([[-.03,-.0175],[.03,-.0175],[.03,-.0025],[-.015,-.0025],[-.015,.0175],[-.03,.0175]])
    options=list(itertools.product([0],[.65,.75,.85,.95,1.],[.003,.008],[.3,.5,.7,.9,1.],[0,2],[.25,.4,.6]))
    e=KeepEnv(EnvConfig(num_envs=len(options),stage=2,training=False),alpha=.5)
    copy_worlds(e,state,torch.full((len(options),),56,dtype=torch.long))
    xy,yaw,tip=e.state();angle=float(yaw[0]);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=xy[0].cpu().numpy();direction=(e.goal[0,:2]-xy[0]);direction/=direction.norm()
    snapshot=e.snapshot(0);meshpoints=[]
    for gid in range(e.model.ngeom):
        if e.model.geom(gid).name.startswith(('fixed_jaw_mesh','moving_jaw_mesh')):
            mid=e.model.geom_dataid[gid];start=e.model.mesh_vertadr[mid];count=e.model.mesh_vertnum[mid]
            world=e.model.mesh_vert[start:start+count]@snapshot.geom_xmat[gid].reshape(3,3).T+snapshot.geom_xpos[gid]
            meshpoints.extend(world[world[:,2]<.019,:2])
    footprint=np.array(meshpoints)-tip[0,:2].cpu().numpy()
    points=[];normals=[];routes=[];blends=[];detours=[];speeds=[]
    for edge,f,clearance,blend,route,speed in options:
        p=vertices[edge];v=vertices[(edge+1)%6]-p;n=np.array([v[1],-v[0]])/np.linalg.norm(v)
        point=(p+f*v)@rot.T+center;normal=n@rot.T
        projection=footprint@normal;support=footprint[projection<projection.min()+.0003].mean(0)
        contact=point-support
        points.append(contact+normal*.0005);normals.append(normal);blends.append(blend)
        approach=contact+normal*clearance;routes.append(approach)
        detours.append(approach if route==0 else np.array([.110,approach[1]]) if route==1 else np.array([approach[0],-.035]))
        speeds.append(speed)
    points=torch.tensor(np.array(points),device=e.device,dtype=e.dtype)
    routes=torch.tensor(np.array(routes),device=e.device,dtype=e.dtype)
    detours=torch.tensor(np.array(detours),device=e.device,dtype=e.dtype)
    normal=torch.tensor(np.array(normals),device=e.device,dtype=e.dtype)
    blend=torch.tensor(blends,device=e.device,dtype=e.dtype)[:,None]
    speeds=torch.tensor(speeds,device=e.device,dtype=e.dtype)[:,None]
    push=-normal*(1-blend)+direction[None]*blend;push/=push.norm(dim=-1,keepdim=True).clamp(min=.01)
    phase=torch.zeros(e.num_envs,device=e.device,dtype=torch.long)
    finished=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
    frames=[e.d.qpos.clone().cpu().numpy()];commands=[];outcomes={}
    for t in range(200):
        j=e.state()[2][:,:2];target=torch.where((phase==0)[:,None],detours,torch.where((phase==1)[:,None],routes,points))
        delta=target-j;near=delta.norm(dim=-1)<.002
        phase=torch.where(near&(phase<3),phase+1,phase)
        a=delta/.04/.15;a=a/a.norm(dim=-1,keepdim=True).clamp(min=1)*.65
        a=torch.where((phase==3)[:,None],push*speeds,a)
        err=(e.state()[0]-e.goal[:,:2]).norm(dim=-1)
        phase=torch.where(err<.009,torch.full_like(phase,4),phase)
        a=torch.where(((phase==4)|finished)[:,None],torch.zeros_like(a),a)
        _,_,done,extras=e.step(a)
        if extras['numerical_failures'].any():raise RuntimeError('Numerical failure')
        frames.append(e.d.qpos.clone().cpu().numpy());commands.append(a.cpu().numpy())
        for r in e.last_terminal:
            i=int(r['env'])
            if i not in outcomes:outcomes[i]=dict(r,recovery_steps=t+1,option=options[i])
        finished|=done
        if t%40==0:print('search',t,'success',sum(r['success'] for r in outcomes.values()),flush=True)
        if finished.all():break
    successes=[(i,r) for i,r in outcomes.items() if r['success']]
    (OUT/'search.json').write_text(json.dumps(dict(candidates=len(options),successes=len(successes),outcomes=outcomes),indent=2))
    if not successes:raise RuntimeError('No valid recovery found; inspect search before training')
    i,row=min(successes,key=lambda ir:(ir[1]['position_error_m'],ir[1]['seconds']))
    frames=np.array(frames)[:row['recovery_steps']+1,i];commands=np.array(commands)[:row['recovery_steps'],i]
    np.savez(OUT/'recovery.npz',qpos=np.concatenate([bank['prefix'][:-1],frames]),actions=np.concatenate([bank['actions'],commands]),
             recovery_actions=commands,mocap_pos=state['sim']['mocap_pos'][56].cpu().numpy(),mocap_quat=state['sim']['mocap_quat'][56].cpu().numpy(),dt=.05,end_effector='closed_gripper')
    result=dict(selected=i,terminal=row,intervention_seconds=20,source_checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                scope='Original policy prefix, followed by a searched velocity-command recovery executed in GPU physics. No block/robot state edits within the recovery. Not learned recovery.')
    (OUT/'recovery.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
