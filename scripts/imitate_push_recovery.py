"""Small demonstration warm start, separately labeled from PPO training."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from recover_stalled_push import KeepEnv,copy_worlds
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy

OUT=Path('artifacts/push_recovery')
SOURCE=Path('runs/push_efficiency/efficient/filtered.pt')

def main():
    torch.set_num_threads(1);torch.manual_seed(9470)
    folder=Path('runs/push_recovery_imitation')
    if folder.exists():raise RuntimeError('Existing imitation run must be inspected before retry')
    bank=torch.load(OUT/'stalled_bank.pt',weights_only=False);z=np.load(OUT/'recovery.npz')
    env=KeepEnv(EnvConfig(num_envs=16,stage=2,training=False),alpha=.5)
    copy_worlds(env,bank['state'],torch.full((16,),56,dtype=torch.long))
    x=[];y=[];terminal={}
    with torch.inference_mode():
        for a in z['recovery_actions']:
            x.append(env.get_observations()['actor'].clone());y.append(torch.tensor(a,device=env.device).expand(16,-1).clone())
            _,_,_,extras=env.step(y[-1])
            if extras['numerical_failures'].any():raise RuntimeError('Numerical failure')
            for r in env.last_terminal:terminal.setdefault(int(r['env']),r)
        for _ in range(20):
            x.append(env.get_observations()['actor'].clone());y.append(torch.zeros(16,2,device=env.device));env.step(y[-1])
            for r in env.last_terminal:terminal.setdefault(int(r['env']),r)
    good=[i for i,r in terminal.items() if r['success']]
    if not good:raise RuntimeError('No successful expert replay')
    expert_x=torch.stack(x)[:,good].flatten(0,1).clone();expert_y=torch.stack(y)[:,good].flatten(0,1).clone()
    # Preserve ordinary approach actions on separate development starts.
    anchor=KeepEnv(EnvConfig(num_envs=64,stage=2,training=False,seed=6500000),alpha=.5)
    base=load_policy(SOURCE,anchor);ax=[];ay=[]
    with torch.inference_mode():
        for _ in range(120):
            obs=anchor.get_observations();action=base(obs);ax.append(obs['actor'].clone());ay.append(action.clone());anchor.step(action)
    anchor_x=torch.cat(ax).clone();anchor_y=torch.cat(ay).clone()
    actor=load_policy(SOURCE,env);actor.eval()  # deterministic mean and frozen observation statistics
    optimizer=torch.optim.Adam(actor.parameters(),lr=.0001)
    folder.mkdir();losses=[]
    for step in range(600):
        ei=torch.randint(len(expert_x),(128,),device=env.device);ai=torch.randint(len(anchor_x),(128,),device=env.device)
        xx=torch.cat([expert_x[ei],anchor_x[ai]]);yy=torch.cat([expert_y[ei],anchor_y[ai]])
        pred=actor(TensorDict({'actor':xx},batch_size=[len(xx)],device=env.device))
        expert_loss=(pred[:128]-yy[:128]).square().mean();anchor_loss=(pred[128:]-yy[128:]).square().mean()
        loss=expert_loss+.3*anchor_loss;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if step%100==0:losses.append(dict(step=step,expert_mse=float(expert_loss.detach()),anchor_mse=float(anchor_loss.detach())))
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    state={k:copy.deepcopy(source[k]) for k in ['format_version','train_config','env_config','distance_training','policy_evaluation_level','reward_experiment','control_filter_alpha']}
    state.update(algorithm={'actor_state_dict':actor.state_dict()},source_hash='recovery-imitation-actor-only',
                 supervised_optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 imitation_experiment=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                     steps=600,seed=9470,successful_expert_copies=len(good),expert_samples=len(expert_x),anchor_samples=len(anchor_x),losses=losses,
                     scope='Behavior cloning warm start from one recovery and baseline-action preservation; not PPO updates or independent demonstrations.'),
                 resume_note='Actor-only warm start. No PPO critic/optimizer/environment resume; initialize fresh PPO state for subsequent reinforcement learning.')
    torch.save(state,folder/'actor.pt');(folder/'experiment.json').write_text(json.dumps(state['imitation_experiment'],indent=2))
    print(json.dumps(state['imitation_experiment'],indent=2),flush=True)

if __name__=='__main__':main()
