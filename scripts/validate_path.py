"""Re-simulate diagnostic actions from one reset; independently audit full passage and parking."""
import argparse,hashlib,json,math
import numpy as np
import torch
import mujoco
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,BLOCK_BOXES,ROOT,PHYSICS_TIMESTEP

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='planned_path');parser.add_argument('--timestep',type=float,default=PHYSICS_TIMESTEP);parser.add_argument('--output');parser.add_argument('--gate-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    z=np.load(ARTIFACTS/(args.input+'.npz'))
    if 'end_effector' not in z or str(z['end_effector'])!='closed_gripper':
        raise RuntimeError('This path was recorded with the retired attachment.')
    e=PushTurnParkEnv(EnvConfig(num_envs=1,backend='native',training=False,stage=5,timestep=args.timestep));d=e.sim.mj_data
    d.qpos[6:13]=[.15,0,.0092,math.cos(math.pi/4),0,0,math.sin(math.pi/4)]
    d.mocap_pos[:]=z['mocap_pos'];d.mocap_quat[:]=z['mocap_quat'];e.sim.forward()
    e.target_xy[:]=e.d.site_xpos[:,e.sid,:2]
    gate=d.mocap_pos[e.gate_mocap,0];goal=d.mocap_pos[e.goal_mocap,:2].copy();quat=d.mocap_quat[e.goal_mocap];goal_yaw=2*math.atan2(quat[3],quat[0])
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    local_vertices=np.vstack([np.array(center)+signs*np.array(size) for center,size in BLOCK_BOXES])
    actions=np.vstack([z['actions'],np.zeros((40,2))]);frames=[];trace=[];invalid=set();contacts=set();maxpen=0.;crossings=np.zeros(len(local_vertices),bool);passed=False;hold=0.;maxhold=0.;passed_time=None;maxspeed=0.;maxacc=0.;previous_velocity=np.zeros(2)
    base=d.xpos[e.model.body('base').id].copy();maxbase=0.;maxjoint=0.;minimum_upright=1.
    def vertices():return local_vertices@d.xmat[e.block].reshape(3,3).T+d.qpos[6:9]
    previous_vertices=vertices()
    completion_time=None
    gate_hold=0.
    max_disallowed_contact_penetration=0.
    actual=d
    # Post-step contact inspection must not run mj_forward on the simulation
    # data at a different frequency from production. Inspect a separate copy.
    d=mujoco.MjData(e.model)
    mujoco.mj_copyData(d,e.model,actual)
    for action in actions:
        e._control(torch.tensor([action.tolist()],dtype=torch.float64))
        velocity=e.velocity[0].numpy();maxspeed=max(maxspeed,np.linalg.norm(velocity));maxacc=max(maxacc,np.linalg.norm(velocity-previous_velocity)/.05);previous_velocity=velocity.copy()
        for _ in range(e.decimation):
            e.sim.step()
            mujoco.mj_copyData(d,e.model,actual);mujoco.mj_forward(e.model,d)
            v=vertices()
            for i,(a,b) in enumerate(zip(previous_vertices,v)):
                if a[0]<=gate<b[0]:
                    y=a[1]+(b[1]-a[1])*(gate-a[0])/(b[0]-a[0])
                    if abs(y)>.025:invalid.add('crossing_outside_gate')
                    crossings[i]=True
            previous_vertices=v.copy()
            if v[:,0].min()>gate+.006 and crossings.all():
                if not passed:passed_time=float(d.time)
                passed=True
            upright=float(d.xmat[e.block].reshape(3,3)[2,2]);minimum_upright=min(minimum_upright,upright)
            if upright<math.cos(math.radians(15)) or abs(d.qpos[8]-.009)>.006:invalid.add('tipping_or_lifting')
            if np.max(np.abs(v[:,1]))>.14 or v[:,0].min()<.08 or v[:,0].max()>.32:invalid.add('outside_workspace')
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():invalid.add('nonfinite')
            maxbase=max(maxbase,np.linalg.norm(d.xpos[e.model.body('base').id]-base))
            maxjoint=max(maxjoint,float(np.maximum(e.model.jnt_range[:6,0]-d.qpos[:6],d.qpos[:6]-e.model.jnt_range[:6,1]).max()))
            for c in d.contact:
                if c.dist<0:
                    names={e.model.geom(c.geom1).name,e.model.geom(c.geom2).name};contacts.add(tuple(sorted(names)));maxpen=max(maxpen,-float(c.dist))
                    allowed=any(n.startswith('block_') for n in names) and all(n in {'block_0','block_1','table','gate_north','gate_south'} or n.startswith(('fixed_jaw_','moving_jaw_')) for n in names)
                    if not allowed:max_disallowed_contact_penetration=max(max_disallowed_contact_penetration,-float(c.dist))
                    if not allowed and c.dist<-.00005:invalid.add('invalid_contact:'+'/'.join(sorted(names)))
            yaw=float(e.state()[1][0]);poserr=float(np.linalg.norm(d.qpos[6:8]-goal));yawerr=abs(math.atan2(math.sin(yaw-goal_yaw),math.cos(yaw-goal_yaw)))
            good=passed and v[:,0].min()>gate+.006 and poserr<=.01 and yawerr<=math.radians(10) and np.linalg.norm(d.qvel[6:9])<=.005 and np.linalg.norm(d.qvel[9:12])<=math.radians(5)
            hold=hold+args.timestep if good else 0.;maxhold=max(maxhold,hold)
            gate_good=passed and v[:,0].min()>gate+.006 and upright>.966 and np.linalg.norm(d.qvel[6:9])<=.005 and np.linalg.norm(d.qvel[9:12])<=math.radians(5)
            gate_hold=gate_hold+args.timestep if gate_good else 0.
            if hold>=1. and completion_time is None and not invalid:
                completion_time=float(d.time)
        e.sim.forward()
        frames.append(d.qpos.copy());trace.append(dict(time=float(d.time),position_error_m=poserr,yaw_error_deg=math.degrees(yawerr),passed_gate=passed,hold_seconds=hold))
    if maxpen>.001:invalid.add('penetration_over_1mm')
    if maxbase>1e-10:invalid.add('base_moved')
    if maxjoint>.001:invalid.add('joint_limit')
    if sum(d.warning.number):invalid.add('mujoco_warning')
    valid=not invalid and hold>=1. and passed
    report=dict(physically_valid_complete_path=bool(valid),within_30_seconds=bool(valid and completion_time is not None and completion_time<=30),completion_seconds=completion_time,seconds=float(d.time),gate_x=float(gate),goal_xy=goal.tolist(),goal_yaw_deg=math.degrees(goal_yaw),passed_gate=passed,gate_passage_seconds=passed_time,all_vertices_crossed=bool(crossings.all()),position_error_m=poserr,yaw_error_deg=math.degrees(yawerr),final_stationary_hold_seconds=hold,max_penetration_m=maxpen,minimum_upright=minimum_upright,invalid=sorted(invalid),contacts=sorted(contacts),max_command_speed=maxspeed,max_command_acceleration=maxacc,timestep=args.timestep,input_artifact=args.input+'.npz',input_sha256=hashlib.sha256((ARTIFACTS/(args.input+'.npz')).read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1/scene.py',ROOT/'src/so101_m1/env.py',ROOT/'scripts/validate_path.py']})
    stem=args.output or ('full_path_validation' if args.timestep==.002 else 'full_path_validation_1ms')
    report['invalid_contact_tolerance_m']=.00005
    report['max_disallowed_contact_penetration_m']=max_disallowed_contact_penetration
    report['gate_stationary_hold_seconds']=gate_hold
    report['gate_checkpoint_pass']=bool(not invalid and passed and gate_hold>=1.)
    (ARTIFACTS/(stem+'.json')).write_text(json.dumps(report,indent=2));np.savez(ARTIFACTS/(stem+'.npz'),qpos=frames,mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy(),dt=.05,actions=actions,end_effector='closed_gripper')
    print(json.dumps(report,indent=2))
    if not (report['gate_checkpoint_pass'] if args.gate_only else valid):raise SystemExit(2)

if __name__=='__main__':main()
