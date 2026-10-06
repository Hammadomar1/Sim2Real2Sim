"""Pre-RL contact regression checks. No rewards, policy learning or teleporting during trials."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import mujoco
import warp as wp
from so101_m1.env import PushTurnParkEnv, EnvConfig
from so101_m1.scene import ARTIFACTS, ROOT, PHYSICS_TIMESTEP


def trial(env, offset=(0.,0.,0.), actions=None, kind='push', seed=81):
    env.reset()
    x,y,a=offset
    yaw=math.pi/2+a
    env.d.qpos[0,6:13]=torch.tensor([.15+x,y,.0092,math.cos(yaw/2),0,0,math.sin(yaw/2)],device=env.device,dtype=env.dtype)
    env.d.qvel[:]=0
    env.d.qacc_warmstart[:]=0
    env.sim.forward()
    env.target_xy[:]=env.d.site_xpos[:,env.sid,:2]
    qframes=[]; logs=[]; commands=[]; contacts=set()
    base0=env.snapshot().xpos[env.model.body('base').id].copy()
    max_pen=0.; base_drift=0.; limits=0.; finite=True; warnings=0
    rng=np.random.default_rng(seed)
    waypoint=0; count=0
    initial_tip=env.d.site_xpos[0,env.sid].cpu().numpy().copy()
    waypoint_list=[[.12,.02],[.145,.02],[.175,.02]]
    if kind=='gate': waypoint_list=[[.11,.055],[.145,.055],[.210,.055]]

    def inspect(data):
        nonlocal max_pen,base_drift,limits,finite,warnings
        finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        base_drift=max(base_drift,float(np.linalg.norm(data.xpos[env.model.body('base').id]-base0)))
        limits=max(limits,float(np.maximum(env.model.jnt_range[:6,0]-data.qpos[:6],data.qpos[:6]-env.model.jnt_range[:6,1]).max()))
        warnings=max(warnings,int(sum(data.warning.number)))
        for c in data.contact:
            if c.dist<0:
                max_pen=max(max_pen,-float(c.dist))
                contacts.add(tuple(sorted([env.model.geom(c.geom1).name,env.model.geom(c.geom2).name])))

    # The reference trajectory supplies identical policy-rate actions at every dt.
    for frame in range(len(actions) if actions is not None else (600 if kind in ['random','hold'] else 440)):
        tip=env.d.site_xpos[0,env.sid].cpu().numpy().copy()
        if actions is not None:
            action=actions[frame]
        elif kind=='hold': action=np.zeros(2)
        elif kind=='random': action=rng.uniform(-1,1,2)
        elif frame<100 or waypoint>=len(waypoint_list): action=np.zeros(2)
        else:
            delta=np.array(waypoint_list[waypoint])-tip[:2]
            action=np.clip(delta/.01,-1,1)
            count+=1
            if np.linalg.norm(delta)<.001 or count>=100:
                waypoint+=1;count=0
        commands.append(np.asarray(action).copy())
        env._control(torch.tensor([np.asarray(action).tolist()],device=env.device,dtype=env.dtype))
        if env.control_graph is None:
            for _ in range(env.decimation):
                env.sim.step();inspect(env.sim.mj_data)
            env.sim.forward()
            data=env.sim.mj_data
        else:
            with wp.ScopedDevice(env.device): wp.capture_launch(env.control_graph)
            data=env.snapshot();inspect(data)
        xy,yaw_state,_=env.state()
        qframes.append(data.qpos.copy())
        logs.append(dict(block=xy[0].cpu().tolist(),yaw=float(yaw_state[0]),tip=data.site_xpos[env.sid].tolist(),upright=float(data.xmat[env.block].reshape(3,3)[2,2]),height=float(data.qpos[8]),qvel=data.qvel[:5].tolist(),velocity=env.velocity[0].cpu().tolist(),object_speed=float(np.linalg.norm(data.qvel[6:9])),angular_speed=float(np.linalg.norm(data.qvel[9:12]))))
    velocity=np.array([r['velocity'] for r in logs]); tips=np.array([r['tip'] for r in logs])
    result=dict(backend=env.cfg.backend,timestep=env.cfg.timestep,kind=kind,offset=offset,frames=len(logs),finite=finite,warnings=warnings,max_penetration_m=max_pen,penetration_sampling='every physics step' if env.cfg.backend=='native' else 'every control step',base_drift_m=base_drift,joint_limit_violation_rad=max(0.,limits),min_upright=min(r['upright'] for r in logs),min_height_m=min(r['height'] for r in logs),max_height_m=max(r['height'] for r in logs),hold_error_m=float(np.linalg.norm(tips[99]-initial_tip)) if kind!='random' else None,max_command_speed=float(np.linalg.norm(velocity,axis=1).max()),max_command_acceleration=float(np.linalg.norm(np.diff(velocity,axis=0)/.05,axis=1).max()),max_measured_tool_speed=float(np.linalg.norm(np.diff(tips,axis=0)/.05,axis=1).max()),contacts=sorted(contacts),final=logs[-1])
    return result,np.array(commands),np.array(qframes),logs


def compare(a,b):
    return dict(position_m=float(np.linalg.norm(np.array(a['final']['block'])-b['final']['block'])),angle_deg=abs(math.degrees(math.atan2(math.sin(a['final']['yaw']-b['final']['yaw']),math.cos(a['final']['yaw']-b['final']['yaw'])))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--base-timestep',type=float,default=PHYSICS_TIMESTEP)
    args=parser.parse_args()
    torch.set_num_threads(1)
    cases=[(0.,0.,0.),(-.003,0.,-.08),(.003,0.,.08),(0.,-.003,.04),(0.,.003,-.04),(.005,.055,-math.pi/2)]
    thresholds=dict(max_penetration_m=.001,max_position_difference_m=.002,max_angle_difference_deg=2.,max_hold_error_m=.0005)
    out=dict(thresholds=thresholds,results=[],comparisons=[],training_ready=False,full_path_validated=False)
    references={}
    base=args.base_timestep
    envs={dt:PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5,timestep=dt)) for dt in [base,base/2,base/4]}
    for i,offset in enumerate(cases):
        kind='gate' if i==5 else 'push'
        r,actions,frames,logs=trial(envs[base],offset,kind=kind)
        out['results'].append(dict(case=i,**r))
        references[i]=(r,actions)
        for dt in [base/2,base/4]:
            b,_,_,_=trial(envs[dt],offset,actions,kind=kind)
            out['results'].append(dict(case=i,**b))
            out['comparisons'].append(dict(case=i,a=base,b=dt,**compare(r,b)))
        if i==0:
            np.savez(ARTIFACTS/'contact_validated_trial.npz',qpos=frames,mocap_pos=envs[base].sim.mj_data.mocap_pos.copy(),mocap_quat=envs[base].sim.mj_data.mocap_quat.copy(),dt=.05,end_effector='closed_gripper')
            np.save(ARTIFACTS/'contact_reference_actions.npy',actions)
            (ARTIFACTS/'contact_reference_trace.json').write_text(json.dumps(logs,indent=2))
        print('Completed native case',i,flush=True)
    for kind in ['hold','random']:
        r,_,_,_=trial(envs[base],kind=kind)
        out['results'].append(dict(case=kind,**r))
    if args.gpu:
        gpu=PushTurnParkEnv(EnvConfig(num_envs=1,backend='warp',training=False,stage=5,timestep=base))
        for i in [0,5]:
            reference,commands=references[i]
            r,_,_,_=trial(gpu,cases[i],actions=commands,kind='gate' if i==5 else 'push')
            out['results'].append(dict(case=f'gpu_{i}',**r))
            out['comparisons'].append(dict(case='native_vs_warp' if i==0 else 'gate_native_vs_warp',**compare(reference,r)))
    checks=[]
    for r in out['results']:
        checks.append(r['finite'] and r['warnings']==0 and r['base_drift_m']<1e-10 and r['joint_limit_violation_rad']<.001 and r['max_penetration_m']<=thresholds['max_penetration_m'] and r['min_upright']>math.cos(math.radians(15)) and .007<r['min_height_m'] and r['max_height_m']<.015 and r['max_command_speed']<=.0400001 and r['max_command_acceleration']<=.200001)
        if r['hold_error_m'] is not None: checks.append(r['hold_error_m']<=thresholds['max_hold_error_m'])
    checks.extend(c['position_m']<=thresholds['max_position_difference_m'] and c['angle_deg']<=thresholds['max_angle_difference_deg'] for c in out['comparisons'])
    out['contact_checks_pass']=bool(all(checks))
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'src/so101_m1/rules.py',Path(__file__)]}
    (ARTIFACTS/'physics_validation.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['results','source_sha256']},indent=2),flush=True)
    if not out['contact_checks_pass']: raise SystemExit(2)


if __name__=='__main__':main()
