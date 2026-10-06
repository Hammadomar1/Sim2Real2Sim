"""Diagnostic GPU shooting controller tracking a physically executed reference.

Candidates are simulated in scratch worlds; only chosen actions reach the
execution world. No reference poses are written to the execution trajectory.
"""
import argparse,json,math,time,hashlib
import numpy as np
import torch
import warp as wp
import mujoco
from table_candidate import enable
enable()
from replay_sensitivity import AuditEnv,clean
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,default=64);p.add_argument('--horizon',type=int,default=12);p.add_argument('--execute',type=int,default=2);p.add_argument('--iterations',type=int,default=2);p.add_argument('--reference',default='gripper_precision_path');p.add_argument('--tag',default='precision');p.add_argument('--resume',action='store_true');args=p.parse_args()
    stem='gripper_mpc_'+args.tag
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/gripper_mpc.py']+[ROOT/'src/so101_m1'/n for n in ['scene.py','env.py','rules.py']]}
    torch.set_num_threads(1);wp.init()
    with wp.ScopedDevice('cuda:0'),torch.cuda.stream(wp.stream_to_torch('cuda:0')):
        real=AuditEnv(EnvConfig(num_envs=1,stage=5,training=False));plan=PushTurnParkEnv(EnvConfig(num_envs=args.worlds,stage=5,training=False))
        z=np.load(ARTIFACTS/(args.reference+'.npz'));actions=np.vstack([z['actions'],np.zeros((80,2))])
        # Re-execute reference natively with the exact production integrator.
        ref=AuditEnv(EnvConfig(num_envs=1,backend='native',stage=5,training=False));references=[]
        for e in [real,ref]:
            e.d.qpos[:,6:]=torch.tensor([.15,0,.0092,math.cos(math.pi/4),0,0,math.sin(math.pi/4)],device=e.device,dtype=e.dtype)
            e.d.mocap_pos[:]=torch.tensor(z['mocap_pos'],device=e.device,dtype=e.dtype);e.d.mocap_quat[:]=torch.tensor(z['mocap_quat'],device=e.device,dtype=e.dtype)
            e.goal[:]=torch.tensor([.27,0,math.pi/2],device=e.device,dtype=e.dtype);e.sim.forward();e.target_xy[:]=e.d.site_xpos[:,e.sid,:2];e.rules.reset(torch.arange(e.num_envs,device=e.device))
        for a in actions:
            ref._control(torch.tensor(a,dtype=ref.dtype)[None])
            for _ in range(ref.decimation):ref.sim.step()
            ref.sim.forward();xy,yaw,tip=ref.state()
            references.append(np.r_[xy[0].numpy(),float(yaw[0]),tip[0].numpy(),ref.d.qpos[0,:5].numpy()])
        refs=torch.tensor(np.array(references),device=real.device,dtype=real.dtype);base=torch.tensor(actions,device=real.device,dtype=real.dtype)
        generator=torch.Generator(device=real.device).manual_seed(8123);frames=[];chosen=[];logs=[];start=time.time();frame=0
        if args.resume:
            checkpoint=torch.load(ARTIFACTS/(stem+'_resume.pt'),weights_only=False,map_location=real.device)
            if checkpoint['finished'] or checkpoint['source_sha256']!=hashes:raise RuntimeError('Checkpoint is finished or source changed; start a new diagnostic.')
            for key in ['worlds','horizon','execute','iterations','reference']:
                if checkpoint['config'][key]!=getattr(args,key):raise RuntimeError('Resume configuration mismatch: '+key)
            real.load_state_dict(checkpoint['state']);generator.set_state(checkpoint['generator'].cpu());frame=checkpoint['frame'];frames=checkpoint['frames'];chosen=checkpoint['chosen'];logs=checkpoint['logs']
        while frame<600:
            saved=real.state_dict();mean=base[frame:frame+args.horizon].clone();std=.35
            for iteration in range(args.iterations):
                for group in ['sim','task','rules']:
                    for key,val in saved[group].items():getattr(plan.d if group=='sim' else plan.rules if group=='rules' else plan,key)[:]=val
                plan.sim.forward()
                noise=torch.randn((args.horizon,args.worlds,2),generator=generator,device=real.device)*std
                noise=.7*noise+.3*noise[0:1];candidates=(mean[:,None]+noise).clamp(-1,1);candidates[:,0]=mean
                score=torch.zeros(args.worlds,device=real.device)
                for k in range(args.horizon):
                    act=candidates[k];plan._control(act);wp.capture_launch(plan.control_graph)
                    xy,yaw,tip=plan.state();target=refs[frame+k]
                    ang=torch.atan2(torch.sin(yaw-target[2]),torch.cos(yaw-target[2]))
                    score+=50000*(xy-target[:2]).square().sum(-1)+5*ang.square()+15000*(tip-target[3:6]).square().sum(-1)+.02*(plan.d.qpos[:,:5]-target[6:]).square().sum(-1)
                    score+=1000*(plan.rules.flags!=0).float()+.03*(act-base[frame+k]).square().sum(-1)
                score=torch.nan_to_num(score,nan=1e9,posinf=1e9);elite=torch.argsort(score)[:max(4,args.worlds//8)]
                best=candidates[:,elite[0]].clone();mean=candidates[:,elite].mean(1);std*=.4
            for a in best[:args.execute]:
                _,_,done,_=real.step(a[None]);frames.append(real.d.qpos[0].cpu().numpy().copy());chosen.append(a.cpu().numpy().copy())
                frame+=1
                if done.any():break
            if frame%20==0 or done.any():
                xy,yaw,_=real.state();row=dict(seconds=frame*.05,xy=xy[0].cpu().tolist(),yaw=float(yaw[0]),reference_error=float(torch.linalg.norm(xy[0]-refs[frame-1,:2])),score=float(score.min()),flags=int(real.rules.flags[0]),wall=time.time()-start);logs.append(row);print(json.dumps(clean(row)),flush=True)
                tmp=ARTIFACTS/(stem+'_resume.tmp');torch.save(dict(state=real.state_dict(),generator=generator.get_state(),frame=frame,frames=frames,chosen=chosen,logs=logs,config=vars(args),source_sha256=hashes,finished=bool(done.any())),tmp);tmp.replace(ARTIFACTS/(stem+'_resume.pt'))
            if done.any():break
        np.savez(ARTIFACTS/(stem+'.npz'),qpos=frames,actions=chosen,mocap_pos=z['mocap_pos'],mocap_quat=z['mocap_quat'],dt=.05,end_effector='closed_gripper')
        report=dict(terminal=real.last_terminal,logs=logs,config=vars(args),source_sha256=hashes,input_sha256=hashlib.sha256((ARTIFACTS/(args.reference+'.npz')).read_bytes()).hexdigest(),scope='Offline receding-horizon diagnostic, not RL; chosen actions executed continuously on GPU.')
        (ARTIFACTS/(stem+'.json')).write_text(json.dumps(clean(report),indent=2,allow_nan=False));print(json.dumps(clean(report['terminal'])),flush=True)

if __name__=='__main__':main()
