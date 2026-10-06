"""Exercise contact, termination and subset resets across the curriculum."""
import argparse,collections,hashlib,json,time
import torch
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.scene import ARTIFACTS,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=128);p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    torch.set_num_threads(1);generator=torch.Generator(device='cuda:0').manual_seed(94831)
    e=PushTurnParkEnv(EnvConfig(num_envs=a.num_envs,training=False));rows=[];start=time.time();passed=True
    for stage,randomized in [(1,False),(2,False),(3,False),(4,False),(5,False),(5,True)]:
        e.cfg.stage=stage;e.cfg.randomized=randomized;e.reset();failures=collections.Counter();episodes=0;transitions=0
        for i in range(a.steps):
            if i%10==0:action=torch.rand((a.num_envs,2),device=e.device,generator=generator)*2-1
            obs,reward,done,extras=e.step(torch.zeros_like(action) if i<40 else action)
            transitions+=a.num_envs;episodes+=len(e.last_terminal)
            for r in e.last_terminal:
                for category in r['failure_categories']:failures[category]+=1
            if extras['numerical_failures'].any() or not torch.isfinite(obs['actor']).all() or not torch.isfinite(reward).all():
                passed=False;torch.save(e.last_numerical_failure or e.state_dict(),ARTIFACTS/'soak_numerical_failure.pt');break
        row=dict(stage=stage,randomized=randomized,transitions=transitions,episodes=episodes,failures=dict(failures),finite=passed);rows.append(row);print(json.dumps(row),flush=True)
        if not passed:break
    out=dict(passed=passed and all(not r['failures'].get('contact_capacity_overflow',0) for r in rows),rows=rows,transitions=sum(r['transitions'] for r in rows),wall_seconds=time.time()-start,scope='Zero actions then correlated random actions with real terminations/resets; failures of random behavior are expected, NaNs and capacity overflows are not.')
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/so101_m1'/n for n in ['scene.py','env.py','rules.py']]+[ROOT/'scripts/numerical_soak.py']}
    (ARTIFACTS/'gripper_numerical_soak.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    if not out['passed']:raise SystemExit(2)

if __name__=='__main__':main()
