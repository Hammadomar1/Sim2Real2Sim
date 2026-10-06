"""Paired, bounded reward correction experiment; no changes to action mapping."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from so101_m1.env import EnvConfig
from so101_m1.training import config,save_checkpoint
from so101_m1.preflight import require_preflight
from efficient_push import EfficientPushEnv
from stage2_curriculum import DistanceCurriculumEnv
from audit_push_speed import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=50);a=p.parse_args()
    if not 1<=a.updates<=50:raise ValueError('At most 50 updates per branch')
    require_preflight();torch.set_num_threads(1)
    source=Path('runs/stage2_distance_seed73/best_level0.pt');saved=torch.load(source,map_location='cpu',weights_only=False)
    out=Path('artifacts/push_efficiency');out.mkdir(parents=True,exist_ok=True)
    for variant in ['control','efficient']:
        folder=Path('runs/push_efficiency')/variant
        if not (folder/'status.json').exists():
            if folder.exists():raise RuntimeError('Incomplete existing run: inspect before restarting')
            folder.mkdir(parents=True)
            random.seed(9420);np.random.seed(9420);torch.manual_seed(9420)
            env=(EfficientPushEnv if variant=='efficient' else DistanceCurriculumEnv)(EnvConfig(num_envs=2048,stage=2,training=True,seed=9420),0)
            cfg=config(9420);cfg['algorithm'].update(schedule='fixed',learning_rate=.0003)
            alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(cfg),env.device)
            alg.load(saved['algorithm'],load_cfg={'actor':True},strict=True)
            with torch.no_grad():alg.actor.distribution.std_param.fill_(.10)
            meta=dict(variant=variant,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),updates=a.updates,
                initialization='same early actor and normalization; fresh critic and optimizer in both branches; std 0.10, floor 0.05',
                source_files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('scripts/efficient_push.py')]})
            (folder/'experiment.json').write_text(json.dumps(meta,indent=2));obs=env.get_observations();alg.train_mode();start=time.monotonic()
            for i in range(1,a.updates+1):
                with torch.inference_mode():
                    for _ in range(32):
                        obs,r,done,extras=env.step(alg.act(obs))
                        if extras['numerical_failures'].any() or not torch.isfinite(r).all() or not torch.isfinite(obs['actor']).all():raise RuntimeError('Numerical failure')
                        alg.process_env_step(obs,r,done,extras)
                    alg.compute_returns(obs)
                losses=alg.update()
                with torch.no_grad():alg.actor.distribution.std_param.clamp_(min=.05)
                with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=i,losses=losses))+'\n')
                if i%10==0 or i==a.updates:
                    print(variant,'update',i,flush=True)
                    target=folder/('final.pt' if i==a.updates else 'latest.pt')
                    save_checkpoint(target,alg,env,cfg,i,i*2048*32,0,(-1,0,0),time.monotonic()-start)
                    state=torch.load(target,map_location='cpu',weights_only=False)
                    state.update(distance_training={'level':0,'streak':0,'complete':False},policy_evaluation_level=0,
                        reward_experiment={'variant':'efficient_push' if variant=='efficient' else 'directional'},efficiency_experiment=meta,source_hash='push-efficiency-experiment')
                    torch.save(state,target)
            (folder/'status.json').write_text(json.dumps(dict(updates=a.updates,seconds=time.monotonic()-start,complete=True)))
        for panel,seed in [('development',6500000),('fresh',6600000)]:
            target=out/f'{variant}_{panel}.json'
            if not target.exists():
                result=evaluate(folder/'final.pt',1,seed,64);result['scope']='Reward-trained policy, original action mapping and success rules; contact sampled at 20 Hz.'
                target.write_text(json.dumps(result,indent=2));print(variant,panel,json.dumps(result['summary']),flush=True)
    result=evaluate(source,1,6600000,64);(out/'early_fresh.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
