"""Resumable bounded PPO runs with validation-gated stage-2 push distances."""
import argparse,copy,hashlib,json,random,signal,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from torch.utils.tensorboard import SummaryWriter
from so101_m1.env import EnvConfig
from so101_m1.preflight import require_preflight
from so101_m1.training import evaluate_policy,summarize,save_checkpoint,source_hash
from stage2_curriculum import DistanceCurriculumEnv,Promotion,LABELS

def implementation_hash():
    h=hashlib.sha256(source_hash().encode())
    for name in ['train_distance_curriculum.py','stage2_curriculum.py','stage2_directional.py','stage2_rewards.py']:
        h.update(name.encode());h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/stage2_directional_audit/latest.pt');p.add_argument('--resume');p.add_argument('--run-dir',required=True);p.add_argument('--updates',type=int,default=100);p.add_argument('--hours',type=float,default=.5);p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--seed',type=int,default=73);a=p.parse_args()
    if not 0<a.hours<=.5 or not 0<a.updates<=500:raise ValueError('Use bounded runs: at most 30 minutes and 500 updates per invocation.')
    require_preflight();torch.set_num_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'latest.pt').exists() and not a.resume:raise ValueError('Existing run: specify --resume or use another directory.')
    path=Path(a.resume or a.checkpoint);saved=torch.load(path,map_location='cpu',weights_only=False)
    state=Promotion();iteration=0;transitions=0;elapsed_prior=0.;best={};provenance=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if a.resume:
        if saved.get('curriculum_implementation_hash')!=implementation_hash():raise ValueError('Resume source mismatch; preserve this curriculum version.')
        state=Promotion(**saved['distance_training']);cfg=EnvConfig(**saved['env_config'])
        if cfg.num_envs!=a.num_envs:raise ValueError('Resume must preserve environment count')
        iteration=saved['iteration'];transitions=saved['transitions'];elapsed_prior=saved['elapsed_training_seconds'];best=saved['distance_best_scores'];provenance=saved['initialization']
    else:cfg=EnvConfig(num_envs=a.num_envs,stage=2,training=True,seed=a.seed)
    env=DistanceCurriculumEnv(cfg,state.level);train_cfg=copy.deepcopy(saved['train_config'])
    if not a.resume:train_cfg['seed']=cfg.seed
    alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(train_cfg),env.device)
    alg.load(saved['algorithm'],load_cfg=None,strict=True)
    if a.resume:
        for key,value in saved.get('physics_fields',{}).items():getattr(env.sim.model,key)[:]=value.to(env.device)
        from mjlab.managers.event_manager import RecomputeLevel
        env.sim.recompute_constants(RecomputeLevel.set_const);env.load_state_dict(saved['env_state'])
        random.setstate(saved['random_state']);np.random.set_state(saved['numpy_state']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    val=DistanceCurriculumEnv(EnvConfig(num_envs=64,stage=2,training=False),state.level)
    metadata=dict(initialization=provenance,levels=LABELS,threshold=.9,consecutive_checks=3,episodes_per_check=64,validation_every_updates=25,initialization_mode='Actor, critic and optimizer retained; new goals and fresh environments on first run. Full state restored on resume.',implementation_hash=implementation_hash(),minimum_std=.05)
    (out/'curriculum_config.json').write_text(json.dumps(metadata,indent=2))
    writer=SummaryWriter(str(out/'tensorboard'),purge_step=transitions if a.resume else None)
    stop={'requested':False};signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True));signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True))
    start=time.monotonic();start_iteration=iteration;records=[];failed=False
    def save(name,evaluation_level=None):
        staging=out/'checkpoint_build.pt'
        save_checkpoint(staging,alg,env,train_cfg,iteration,transitions,state.streak,tuple(best.get(state.level,(-1.,-float('inf'),-float('inf')))),elapsed_prior+time.monotonic()-start)
        payload=torch.load(staging,map_location='cpu',weights_only=False)
        payload.update(distance_training=asdict(state),distance_best_scores=best,initialization=provenance,curriculum_implementation_hash=implementation_hash(),reward_experiment={'variant':'directional','reward_version':'stage2_directional_v2'},source_hash='distance-curriculum:'+implementation_hash())
        payload['policy_evaluation_level']=state.level if evaluation_level is None else evaluation_level
        temporary=out/(name+'.tmp');torch.save(payload,temporary);temporary.replace(out/name);staging.unlink()
    def validate():
        val.level=state.level;seed=4100000+state.level*100000+iteration*64
        episodes=evaluate_policy(alg.get_policy(),val,64,seed);metrics=summarize(episodes);level=state.level
        score=(metrics['success_rate'],-metrics['mean_position_error_mm'],-metrics['mean_joint_jerk_integral'])
        improved=score>tuple(best.get(level,(-1.,-float('inf'),-float('inf'))))
        if improved:best[level]=score
        promoted=state.observe(sum(bool(r['success']) for r in episodes),len(episodes))
        row=dict(iteration=iteration,transitions=transitions,distance_level=level,distance_label=LABELS[level],validation_seed=seed,consecutive_passes=3 if promoted else state.streak,promoted=promoted,**metrics)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        (out/f'validation_{iteration:05d}_level{level}.json').write_text(json.dumps(episodes,indent=2))
        for k,v in metrics.items():
            if isinstance(v,(float,int)):writer.add_scalar(f'level_{level}/validation_{k}',v,transitions)
        print('VALIDATION '+json.dumps(row),flush=True)
        if promoted:
            env.level=state.level;env.reset();print('DISTANCE ADVANCED: '+LABELS[state.level],flush=True)
        if improved:save(f'best_level{level}.pt',level)
        save('latest.pt');alg.train_mode();writer.flush()
    save('latest.pt')
    try:
        if not a.resume:validate()
        obs=env.get_observations();alg.train_mode()
        while not stop['requested'] and not state.complete and iteration-start_iteration<a.updates and time.monotonic()-start<a.hours*3600:
            with torch.inference_mode():
                for _ in range(train_cfg['num_steps_per_env']):
                    actions=alg.act(obs);obs,reward,done,extras=env.step(actions)
                    if extras['numerical_failures'].any() or not torch.isfinite(reward).all() or not torch.isfinite(obs['actor']).all():
                        torch.save(env.last_numerical_failure or env.state_dict(),out/'numerical_failure.pt');raise RuntimeError('Numerical failure; stopped')
                    alg.process_env_step(obs,reward,done,extras);records.extend(env.last_terminal)
                alg.compute_returns(obs)
            losses=alg.update()
            with torch.no_grad():alg.get_policy().distribution.std_param.clamp_(min=.05)
            iteration+=1;transitions+=env.num_envs*train_cfg['num_steps_per_env']
            for key,value in losses.items():writer.add_scalar('loss/'+key,float(value),transitions)
            if iteration%10==0:
                stats={str(level):summarize([r for r in records if r['distance_level']==level]) for level in sorted({r['distance_level'] for r in records})}
                row=dict(iteration=iteration,transitions=transitions,distance_level=state.level,by_distance_level=stats)
                with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True);records=[]
            if iteration%25==0:validate();obs=env.get_observations()
    except BaseException as error:
        failed=True;(out/'failure.json').write_text(json.dumps(dict(error=repr(error),iteration=iteration)));raise
    finally:
        if not failed:save('latest.pt')
        writer.close();(out/'status.json').write_text(json.dumps(dict(state='failed' if failed else 'stopped',iteration=iteration,transitions=transitions,distance_curriculum=asdict(state),run_seconds=time.monotonic()-start,milestone_complete=False),indent=2))

if __name__=='__main__':main()
