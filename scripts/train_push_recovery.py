"""Bounded 25-update PPO recovery experiment, filter active during learning."""
import copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from recovery_env import RecoveryEnv
from so101_m1.env import EnvConfig
from so101_m1.training import config,save_checkpoint
from so101_m1.preflight import require_preflight

def main():
    require_preflight();torch.set_num_threads(1)
    out=Path('runs/push_recovery_seed9440')
    if out.exists():raise RuntimeError('Existing run: inspect it before restarting')
    evidence=json.loads(Path('artifacts/push_recovery/recovery.json').read_text())
    if not evidence['terminal']['success'] or evidence['terminal']['failure_flags']:raise RuntimeError('Recovery feasibility gate failed')
    source=Path('runs/push_efficiency/efficient/filtered.pt');saved=torch.load(source,map_location='cpu',weights_only=False)
    seed=9440;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=RecoveryEnv(EnvConfig(num_envs=2048,stage=2,training=True,seed=seed))
    cfg=config(seed);cfg['algorithm'].update(schedule='fixed',learning_rate=.0001)
    alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(cfg),env.device)
    alg.load(saved['algorithm'],load_cfg={'actor':True,'critic':True},strict=True)
    with torch.no_grad():alg.actor.distribution.std_param.fill_(.15)
    meta=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),updates=25,
              initialization='previous actor/critic and normalization; fresh optimizer; std 0.15, floor 0.05',
              filter_alpha=.5,recovery_fraction_of_stage2=.4,learning_rate=.0001,
              bank_sha256=hashlib.sha256(Path('artifacts/push_recovery/stalled_bank.pt').read_bytes()).hexdigest(),
              training_scenes='57 stalled states from former diagnostic seed 6600000; that panel is now training data, not held-out evaluation',
              source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/recovery_env.py'),Path('scripts/push_filter.py'),Path('scripts/efficient_push.py')]})
    out.mkdir();(out/'experiment.json').write_text(json.dumps(meta,indent=2))
    obs=env.get_observations();alg.train_mode();start=time.monotonic();terminals=[]
    for update in range(1,26):
        with torch.inference_mode():
            for _ in range(32):
                obs,r,done,extras=env.step(alg.act(obs))
                if extras['numerical_failures'].any() or not torch.isfinite(r).all() or not torch.isfinite(obs['actor']).all():raise RuntimeError('Numerical failure')
                alg.process_env_step(obs,r,done,extras);terminals.extend(env.last_terminal)
            alg.compute_returns(obs)
        losses=alg.update()
        with torch.no_grad():alg.actor.distribution.std_param.clamp_(min=.05)
        row=dict(update=update,losses=losses,seconds=time.monotonic()-start,episodes=len(terminals),successes=sum(x['success'] for x in terminals),recovery_successes=sum(x['success'] for x in terminals if x['recovery_reset']),failures=sum(x['failed'] for x in terminals))
        with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if update%5==0:
            target=out/f'update{update:03d}.pt'
            save_checkpoint(target,alg,env,cfg,update,update*2048*32,0,(-1,0,0),time.monotonic()-start)
            state=torch.load(target,map_location='cpu',weights_only=False)
            state.update(distance_training={'level':0,'streak':0,'complete':False},policy_evaluation_level=0,
                         reward_experiment={'variant':'efficient_push'},control_filter_alpha=.5,recovery_experiment=meta,
                         source_hash='push-recovery-experiment',resume_note='Requires RecoveryEnv, matching source hashes and stalled bank; not production resume compatible.')
            torch.save(state,target)
    (out/'status.json').write_text(json.dumps(dict(complete=True,updates=25,seconds=time.monotonic()-start)))
    (out/'episodes.json').write_text(json.dumps(terminals))

if __name__=='__main__':main()
