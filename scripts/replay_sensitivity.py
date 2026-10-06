"""Paired native/Warp replay of fixed demonstration actions; not policy evaluation."""
import argparse,hashlib,json,math,time
import numpy as np
import torch
import mujoco
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT,GOAL_X

class AuditEnv(PushTurnParkEnv):
    def reset(self,ids=None,*,preserve_terminal=False):
        # Preserve the exact terminal state for diagnostics. Production step's
        # success, failure, timeout and controller calculations are unchanged.
        if preserve_terminal:return self.get_observations()
        return super().reset(ids,preserve_terminal=preserve_terminal)

def cases():
    out=[dict(name='nominal')]
    for key,size in [('x',.001),('y',.001),('yaw',math.radians(1)),('friction',.05),('mass',.05),('gain',.05)]:
        for sign in [-1,1]:out.append(dict(name=f'{key}_{"minus" if sign<0 else "plus"}',**{key:sign*size}))
    return out

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def run(backend,settings,stem,dt=.001,force_recompute=False):
    e=AuditEnv(EnvConfig(num_envs=len(settings),backend=backend,training=False,stage=5,timestep=dt))
    source=ARTIFACTS/'gripper_complete_path.npz';z=np.load(source)
    for i,c in enumerate(settings):
        yaw=math.pi/2+c.get('yaw',0)
        e.d.qpos[i,6:13]=torch.tensor([.15+c.get('x',0),c.get('y',0),.0092,math.cos(yaw/2),0,0,math.sin(yaw/2)],device=e.device,dtype=e.dtype)
        e.d.mocap_pos[i]=torch.tensor(z['mocap_pos'],device=e.device,dtype=e.dtype)
        e.d.mocap_quat[i]=torch.tensor(z['mocap_quat'],device=e.device,dtype=e.dtype)
        e.goal[i]=torch.tensor([GOAL_X,0,math.pi/2],device=e.device,dtype=e.dtype)
        friction=1+c.get('friction',0);mass=1+c.get('mass',0);gain=1+c.get('gain',0)
        if backend=='warp':
            e.sim.model.geom_friction[i,:,0]*=friction
            e.sim.model.body_mass[i,e.block]*=mass;e.sim.model.body_inertia[i,e.block]*=mass
            e.sim.model.actuator_gainprm[i,:,0]*=gain
            e.sim.model.actuator_biasprm[i,:,1]*=gain
        else:
            e.model.geom_friction[:,0]*=friction;e.model.body_mass[e.block]*=mass;e.model.body_inertia[e.block]*=mass
            e.model.actuator_gainprm[:,0]*=gain;e.model.actuator_biasprm[:,1]*=gain
    recomputed=force_recompute or any(c.get('mass',0)!=0 for c in settings)
    if backend=='warp' and recomputed:
        from mjlab.managers.event_manager import RecomputeLevel
        e.sim.recompute_constants(RecomputeLevel.set_const)
    elif backend=='native' and recomputed:mujoco.mj_setConst(e.model,mujoco.MjData(e.model))
    e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.arange(e.num_envs,device=e.device))
    results={};traces=[[] for _ in settings];frames=[[] for _ in settings]
    started=time.perf_counter()
    for frame in range(600):
        a=z['actions'][frame] if frame<len(z['actions']) else np.zeros(2)
        _,_,done,_=e.step(torch.tensor(a,device=e.device,dtype=e.dtype)[None].repeat(e.num_envs,1))
        q=e.d.qpos.detach().cpu().numpy();v=e.d.qvel.detach().cpu().numpy();_,yaw,_=e.state()
        for i in range(e.num_envs):
            if i in results:continue
            frames[i].append(q[i].copy())
            traces[i].append(dict(seconds=(frame+1)*.05,xy=q[i,6:8].tolist(),yaw=float(yaw[i]),linear_speed=float(np.linalg.norm(v[i,6:9])),angular_speed=float(np.linalg.norm(v[i,9:12])),stationary_hold_seconds=int(e.rules.hold[i])*dt,failure_flags=int(e.rules.flags[i]),passed_gate=bool(e.passed[i])))
        for terminal in e.last_terminal:
            i=int(terminal['env'])
            if i not in results:
                results[i]=dict(case=settings[i],backend=backend,timestep=dt,**terminal)
                print(json.dumps(clean(results[i]),allow_nan=False),flush=True)
        if len(results)==len(settings):break
        # Retain the first terminal snapshot above, then clear failed worlds so
        # NaNs cannot remain active while other worlds finish their rollouts.
        e.reset(done.nonzero().flatten())
    for i in range(e.num_envs):
        if i not in results:raise RuntimeError('Missing terminal outcome')
        results[i]['final_pose']=[*traces[i][-1]['xy'],traces[i][-1]['yaw']]
        # Terminal frame is retained; simulation beyond that frame is excluded.
        np.savez(ARTIFACTS/f'{stem}_{settings[i]["name"]}.npz',qpos=frames[i],mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],dt=.05,end_effector='closed_gripper')
    out=dict(scope='Fixed-action replay, no replanning or RL policy; single-factor sensitivity, not a success-rate estimate.',backend=backend,constants_recomputed=recomputed,wall_seconds=time.perf_counter()-started,results=[results[i] for i in range(e.num_envs)],traces=traces,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',ROOT/'scripts/replay_sensitivity.py']}
    (ARTIFACTS/f'{stem}.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False));return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['native','warp'],required=True);parser.add_argument('--nominal-only',action='store_true');parser.add_argument('--replicas',type=int,default=0);parser.add_argument('--force-recompute',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);settings=cases()[:1] if args.nominal_only else cases()
    if args.replicas:run('warp',[dict(name=f'nominal_{i}') for i in range(args.replicas)],f'full_replay_warp_replicas_{args.replicas}',force_recompute=args.force_recompute)
    elif args.backend=='warp':run('warp',settings,('full_replay_warp_single_recomputed' if args.force_recompute else 'full_replay_warp_single') if args.nominal_only else 'full_replay_warp_sensitivity',force_recompute=args.force_recompute)
    else:
        for c in settings:run('native',[c],'full_replay_native_'+c['name'])
        if not args.nominal_only:run('native',[dict(name='half_timestep')],'full_replay_native_half_timestep',dt=.0005)
if __name__=='__main__':main()
