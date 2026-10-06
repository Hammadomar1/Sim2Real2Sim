"""Bounded paired reward experiment, preserving all production physics and checkpoints."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from so101_m1.env import EnvConfig,PushTurnParkEnv
from so101_m1.training import config,evaluate_policy,summarize,save_checkpoint
from so101_m1.preflight import require_preflight
from stage2_rewards import Stage2RewardEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','corrected'],required=True);p.add_argument('--updates',type=int,default=100);p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--seed',type=int,default=71);p.add_argument('--checkpoint',default='artifacts/stage2_reward_audit/baseline_latest.pt');p.add_argument('--run-dir',required=True);a=p.parse_args()
    assert 0<a.updates<=150 and a.num_envs<=2048,'Bounded diagnostic only; no overnight experiment.'
    require_preflight();torch.set_num_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists():raise RuntimeError('Use a new run directory to preserve the comparison.')
    cfg=EnvConfig(num_envs=a.num_envs,seed=a.seed,stage=2,training=True);env=Stage2RewardEnv(cfg,a.variant)
    train_cfg=config(a.seed);alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(train_cfg),env.device)
    source=torch.load(a.checkpoint,map_location=env.device,weights_only=False)
    # Warm-start actor and normalization, but reset critic/optimizer for both arms.
    alg.load(source['algorithm'],load_cfg={'actor':True},strict=True)
    with torch.no_grad():alg.get_policy().distribution.std_param.fill_(.15)
    val=PushTurnParkEnv(EnvConfig(num_envs=64,stage=2,training=False,seed=1700001))
    meta=dict(variant=a.variant,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),seed=a.seed,updates=a.updates,num_envs=a.num_envs,initial_std=.15,min_std=.05,critic_optimizer='fresh in both arms',validation_seed=1700001,validation_episodes=64,reward_version='stage2_position_v1',source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('stage2_rewards.py')]})
    (out/'experiment.json').write_text(json.dumps(meta,indent=2));start=time.monotonic();best=(-1.,-float('inf'),-float('inf'));history=[]
    def validate(i):
        rows=evaluate_policy(alg.get_policy(),val,64,1700001);metrics=summarize(rows)
        record=dict(iteration=i,transitions=i*a.num_envs*32,stage=2,**metrics);history.append(record)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        (out/f'episodes_{i:04d}.json').write_text(json.dumps(rows,indent=2));print('VALIDATION '+json.dumps(record),flush=True)
        return (metrics['success_rate'],-metrics['mean_position_error_mm'],-metrics['mean_joint_jerk_integral'])
    def save(name,i,score):
        save_checkpoint(out/name,alg,env,train_cfg,i,i*a.num_envs*32,0,score,time.monotonic()-start)
        # Persist reward identity: these are warm-start experiments, not production resumes.
        state=torch.load(out/name,map_location='cpu',weights_only=False);state['reward_experiment']=meta;torch.save(state,out/name)
    best=validate(0);save('initial.pt',0,best);obs=env.get_observations();alg.train_mode();recent=[]
    for i in range(1,a.updates+1):
        with torch.inference_mode():
            for _ in range(32):
                action=alg.act(obs);obs,r,done,extras=env.step(action)
                if extras['numerical_failures'].any() or not torch.isfinite(r).all():
                    torch.save(env.last_numerical_failure or env.state_dict(),out/'numerical_failure.pt');raise RuntimeError('Numerical failure')
                alg.process_env_step(obs,r,done,extras);recent.extend(env.last_terminal)
            alg.compute_returns(obs)
        alg.update()
        with torch.no_grad():alg.get_policy().distribution.std_param.clamp_(min=.05)
        if i%10==0:
            row=dict(iteration=i,transitions=i*a.num_envs*32,stage=2,elapsed_seconds=time.monotonic()-start,**summarize(recent))
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);recent=[]
        if i%25==0 or i==a.updates:
            score=validate(i)
            if score>best:best=score;save('best_stage2.pt',i,best)
            save('latest.pt',i,best);obs=env.get_observations();alg.train_mode()
    (out/'status.json').write_text(json.dumps(dict(state='stopped',iteration=a.updates,transitions=a.updates*a.num_envs*32,stage=2,run_seconds=time.monotonic()-start,milestone_complete=False),indent=2))

if __name__=='__main__':main()
