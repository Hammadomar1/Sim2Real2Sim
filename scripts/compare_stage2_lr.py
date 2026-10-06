"""Bounded 25-update learning-rate comparison with matched contact-aware evaluation."""
import copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy,save_checkpoint,summarize
from so101_m1.preflight import require_preflight
from stage2_curriculum import DistanceCurriculumEnv

SOURCE=Path('runs/stage2_distance_seed73/best_level0.pt')
ROOT=Path('runs/stage2_lr_comparison')
OUT=Path('artifacts/stage2_lr_comparison')
PANELS={'fixed':5300000,'fresh':6400000}

class ContactEnv(DistanceCurriculumEnv):
    """Observe the existing actual jaw/block contact flag without modifying physics."""
    def reset(self,ids=None,*,preserve_terminal=False):
        if preserve_terminal:self.end_touch=self.rules.touch.clone()
        return super().reset(ids,preserve_terminal=preserve_terminal)
    def step(self,action):
        self.end_touch=None
        result=super().step(action)
        if self.end_touch is None:self.end_touch=self.rules.touch.clone()
        return result

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))

def train(mode):
    folder=ROOT/mode
    if (folder/'status.json').exists():
        assert json.loads((folder/'status.json').read_text())['completed_updates']==25
        return folder/'final.pt'
    if folder.exists():raise RuntimeError(f'Incomplete branch exists: {folder}; inspect before retrying')
    folder.mkdir(parents=True)
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    env=ContactEnv(EnvConfig(**saved['env_config']),saved['distance_training']['level'])
    from mjlab.managers.event_manager import RecomputeLevel
    for key,value in saved['physics_fields'].items():getattr(env.sim.model,key)[:]=value.to(env.device)
    env.sim.recompute_constants(RecomputeLevel.set_const);env.load_state_dict(saved['env_state'])
    cfg=copy.deepcopy(saved['train_config'])
    if mode=='fixed':cfg['algorithm'].update(schedule='fixed',learning_rate=.0003)
    alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(cfg),env.device)
    alg.load(saved['algorithm'],load_cfg=None,strict=True)
    if mode=='fixed':
        alg.learning_rate=.0003
        for group in alg.optimizer.param_groups:group['lr']=.0003
    random.setstate(saved['random_state']);np.random.set_state(saved['numpy_state'])
    torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    meta=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),mode=mode,updates=25,
        initial_iteration=saved['iteration'],num_envs=env.num_envs,level=env.level,
        restored='actor, critic, optimizer moments, normalization, physics, environment, curriculum, and random generators',
        initial_state_hash=hashlib.sha256(torch.cat((env.d.qpos.clone(),env.goal),-1).cpu().numpy().tobytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(folder/'experiment.json',meta);obs=env.get_observations();alg.train_mode();start=time.monotonic()
    records=[];history=[]
    for update in range(1,26):
        contact=[]
        with torch.inference_mode():
            for _ in range(32):
                action=alg.act(obs);obs,reward,done,extras=env.step(action)
                if extras['numerical_failures'].any() or not torch.isfinite(obs['actor']).all():raise RuntimeError('Numerical failure; abort')
                contact.append(float((env.end_touch!=0).float().mean()));records.extend(copy.deepcopy(env.last_terminal))
                alg.process_env_step(obs,reward,done,extras)
            alg.compute_returns(obs)
        observations=alg.storage.observations.flatten(0,1).clone()
        old=tuple(v.flatten(0,1).clone() for v in alg.storage.distribution_params)
        actions=alg.storage.actions.flatten(0,1).clone();log_prob=alg.storage.actions_log_prob.flatten().clone()
        lr_before=alg.learning_rate;losses=alg.update()
        with torch.no_grad():alg.get_policy().distribution.std_param.clamp_(min=.05)
        # Distribution construction samples actions: preserve RNG so diagnostics cannot change training.
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        with torch.inference_mode():
            alg.actor(observations,stochastic_output=True)
            kl=float(alg.actor.get_kl_divergence(old,alg.actor.output_distribution_params).mean())
            ratio=(alg.actor.get_output_log_prob(actions)-log_prob).exp()
            clip=float(((ratio<.8)|(ratio>1.2)).float().mean())
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        row=dict(update=update,lr_before=lr_before,lr_after=alg.learning_rate,post_update_kl=kl,clip_fraction=clip,
            sampled_jaw_contact_fraction=float(np.mean(contact)),losses=losses)
        history.append(row)
        with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if update%5==0:
            print(mode,update,json.dumps(row),flush=True)
            target=folder/('final.pt' if update==25 else 'latest.pt')
            save_checkpoint(target,alg,env,cfg,saved['iteration']+update,saved['transitions']+update*env.num_envs*32,0,(-1,0,0),time.monotonic()-start)
            payload=torch.load(target,map_location='cpu',weights_only=False)
            payload.update(distance_training=saved['distance_training'],policy_evaluation_level=env.level,
                reward_experiment=saved['reward_experiment'],lr_experiment=meta,source_hash='lr-comparison:'+meta['script_sha256'])
            torch.save(payload,target)
    dump(folder/'training_episodes.json',records)
    dump(folder/'status.json',dict(completed_updates=25,additional_transitions=25*env.num_envs*32,seconds=time.monotonic()-start))
    return folder/'final.pt'

@torch.inference_mode()
def evaluate(checkpoint,panel,seed):
    env=ContactEnv(EnvConfig(num_envs=64,stage=2,training=False),0);actor=load_policy(checkpoint,env)
    rows=[];hashes=[]
    for batch in range(2):
        env.rng.manual_seed(seed+batch*64);env.reset()
        hashes.append(hashlib.sha256(torch.cat((env.d.qpos.clone(),env.goal),-1).cpu().numpy().tobytes()).hexdigest())
        initial=env.d.qpos[:,6:8].clone();direction=env.goal[:,:2]-initial
        initial_error=direction.norm(dim=-1);direction/=initial_error[:,None]
        active=torch.ones(64,device=env.device,dtype=torch.bool)
        contact=torch.zeros(64,device=env.device);streak=contact.clone();longest=contact.clone()
        first=torch.full((64,),-1.,device=env.device);minimum=initial_error.clone()
        for step in range(600):
            _,_,_,extras=env.step(actor(env.get_observations()))
            if extras['numerical_failures'].any():raise RuntimeError('Numerical failure in evaluation')
            touch=env.end_touch!=0
            contact[active]+=touch[active].float()*.05
            first=torch.where(active&touch&(first<0),(step+1)*.05,first)
            streak=torch.where(active&touch,streak+.05,torch.zeros_like(streak));longest=torch.maximum(longest,streak)
            xy,_,_=env.terminal_geometry if env.terminal_geometry is not None else env.state()
            # Reset changes goals; final errors are provided by independent terminal records.
            for r in env.last_terminal:
                i=int(r['env'])
                if not active[i]:continue
                r=copy.deepcopy(r);r.update(scenario_index=batch*64+i,initial_error_mm=float(initial_error[i]*1000),
                    progress_mm=float(initial_error[i]*1000)-r['position_error_m']*1000,
                    forward_displacement_mm=float(((xy[i]-initial[i])*direction[i]).sum()*1000),
                    jaw_contact_seconds=float(contact[i]),jaw_contact_fraction=float(contact[i])/r['seconds'],
                    longest_contact_seconds=float(longest[i]),first_contact_seconds=float(first[i]),
                    jerk_per_second=r['joint_jerk_integral']/r['seconds'])
                rows.append(r);active[i]=False
            if not active.any():break
    rows.sort(key=lambda r:r['scenario_index']);assert len(rows)==128
    keys=['progress_mm','forward_displacement_mm','jaw_contact_seconds','jaw_contact_fraction','longest_contact_seconds','jerk_per_second']
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),panel=panel,seed=seed,
        initial_scene_hashes=hashes,summary=summarize(rows),metrics={k:float(np.mean([r[k] for r in rows])) for k in keys},
        episodes_with_jaw_contact=sum(r['jaw_contact_seconds']>0 for r in rows),episodes=rows,
        contact_definition='Actual allowed jaw/block contact at <=0 separation, sampled at each 20 Hz control boundary, captured before auto-reset; not exact physics-step contact duration.')

def main():
    torch.set_num_threads(1);require_preflight();OUT.mkdir(parents=True,exist_ok=True)
    checkpoints={'early':SOURCE}
    for mode in ['adaptive','fixed']:checkpoints[mode]=train(mode)
    report={}
    for name,path in checkpoints.items():
        report[name]={}
        for panel,seed in PANELS.items():
            target=OUT/f'{name}_{panel}.json'
            if target.exists():r=json.loads(target.read_text());assert r['checkpoint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            else:r=evaluate(path,panel,seed);dump(target,r)
            report[name][panel]={k:v for k,v in r.items() if k!='episodes'}
            print('EVALUATION',name,panel,json.dumps(report[name][panel]),flush=True)
    for panel in PANELS:assert report['early'][panel]['initial_scene_hashes']==report['adaptive'][panel]['initial_scene_hashes']==report['fixed'][panel]['initial_scene_hashes']
    dump(OUT/'comparison.json',report)

if __name__=='__main__':main()
