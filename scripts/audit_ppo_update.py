"""One disposable PPO update on identical data at two learning-rate settings."""
import copy,json
from pathlib import Path
import torch
from rsl_rl.algorithms import PPO
from so101_m1.env import EnvConfig
from stage2_curriculum import DistanceCurriculumEnv

def main():
    torch.set_num_threads(1);torch.manual_seed(9401)
    saved=torch.load('runs/stage2_distance_seed73/best_level0.pt',map_location='cuda:0',weights_only=False)
    env=DistanceCurriculumEnv(EnvConfig(**saved['env_config']),0)
    env.load_state_dict(saved['env_state'])
    alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(saved['train_config']),env.device)
    alg.load(saved['algorithm'],load_cfg=None,strict=True);alg.train_mode();obs=env.get_observations()
    with torch.inference_mode():
        for _ in range(32):
            obs,r,done,extras=env.step(alg.act(obs))
            if extras['numerical_failures'].any():raise RuntimeError('Nonfinite physics')
            alg.process_env_step(obs,r,done,extras)
        alg.compute_returns(obs)
    data=copy.deepcopy(alg.storage);state=copy.deepcopy(alg.save())
    results={}
    for mode in ['inherited_adaptive','fixed_0.0003']:
        branch_state=copy.deepcopy(state)
        with torch.inference_mode():alg.load(branch_state,load_cfg=None,strict=True)
        alg.storage=copy.deepcopy(data)
        alg.schedule='adaptive' if mode=='inherited_adaptive' else 'fixed'
        if mode=='fixed_0.0003':
            alg.learning_rate=.0003
            for group in alg.optimizer.param_groups:group['lr']=.0003
        rates=[];kls=[];original_kl=alg.actor.get_kl_divergence
        def measured_kl(old,new):
            value=original_kl(old,new);kls.append(float(value.mean()));rates.append(alg.optimizer.param_groups[0]['lr']);return value
        alg.actor.get_kl_divergence=measured_kl
        torch.manual_seed(9402);torch.cuda.manual_seed_all(9402)
        losses=alg.update();alg.actor.get_kl_divergence=original_kl
        with torch.inference_mode():
            observations=data.observations.flatten(0,1)
            alg.actor(observations,stochastic_output=True)
            new=alg.actor.output_distribution_params
            old=tuple(v.flatten(0,1) for v in data.distribution_params)
            kl=original_kl(old,new)
            ratio=(alg.actor.get_output_log_prob(data.actions.flatten(0,1))-data.actions_log_prob.flatten()).exp()
            results[mode]=dict(losses=losses,minibatch_kls=kls,learning_rates=rates,final_lr=alg.learning_rate,
                post_update_mean_kl=float(kl.mean()),post_update_clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()),
                mean_action_shift=float((new[0]-old[0]).norm(dim=-1).mean()))
    results['scope']='One diagnostic update per setting on the same 65536 transitions from the best checkpoint; all updated models discarded. Does not measure subsequent task success.'
    Path('artifacts/stage2_decline/ppo_update.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
