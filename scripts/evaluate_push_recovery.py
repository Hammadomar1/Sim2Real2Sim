"""Select on a development panel; evaluate selected policy on a new panel."""
import json
from pathlib import Path
import numpy as np
import torch
from audit_push_speed import evaluate
from recover_stalled_push import KeepEnv,copy_worlds
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy,summarize

OUT=Path('artifacts/push_recovery')
BASE=Path('runs/push_efficiency/efficient/filtered.pt')
RUN=Path('runs/push_recovery_seed9440')

@torch.inference_mode()
def stalled(path):
    bank=torch.load(OUT/'stalled_bank.pt',weights_only=False)
    e=KeepEnv(EnvConfig(num_envs=16,stage=2,training=False),alpha=.5)
    policy=load_policy(path,e);copy_worlds(e,bank['state'],torch.full((16,),56,dtype=torch.long))
    frames=[e.d.qpos[0].cpu().numpy().copy()];actions=[];rows={}
    for t in range(200):
        a=policy(e.get_observations());_,_,done,extras=e.step(a)
        if extras['numerical_failures'].any():raise RuntimeError('Numerical failure')
        if 0 not in rows:
            frames.append(e.d.qpos[0].cpu().numpy().copy());actions.append(a[0].cpu().numpy().copy())
        for row in e.last_terminal:rows.setdefault(int(row['env']),row)
        if len(rows)==16:break
    name='baseline' if path==BASE else path.stem
    np.savez(OUT/f'learned_recovery_{name}.npz',qpos=np.concatenate([bank['prefix'][:-1],frames]),
             actions=np.concatenate([bank['actions'],actions]),mocap_pos=bank['state']['sim']['mocap_pos'][56].cpu().numpy(),
             mocap_quat=bank['state']['sim']['mocap_quat'][56].cpu().numpy(),dt=.05,end_effector='closed_gripper')
    return dict(checkpoint=str(path),summary=summarize(list(rows.values())),episodes=list(rows.values()),
                scope='16 identical copies of training scene 56 at 20 seconds; original remaining 10-second budget. Not independent generalization trials.')

def main():
    torch.set_num_threads(1)
    if not (RUN/'status.json').exists():raise RuntimeError('Training not complete')
    development={}
    for name,path in [('baseline',BASE)]+[(f'update{i:03d}',RUN/f'update{i:03d}.pt') for i in [5,15,25]]:
        target=OUT/f'{name}_development.json'
        if not target.exists():target.write_text(json.dumps(evaluate(path,1,6500000,64,alpha=.5),indent=2))
        result=json.loads(target.read_text());development[name]=result
        print(name,'development',result['summary'],flush=True)
    selected=max([n for n in development if n!='baseline'],key=lambda n:(development[n]['summary']['success_rate'],-development[n]['summary']['mean_position_error_mm']))
    selected_path=RUN/f'{selected}.pt'
    for name,path in [('baseline',BASE),(selected,selected_path)]:
        target=OUT/f'{name}_heldout.json'
        if not target.exists():target.write_text(json.dumps(evaluate(path,1,6700000,64,alpha=.5),indent=2))
        print(name,'heldout',json.loads(target.read_text())['summary'],flush=True)
        target=OUT/f'{name}_stalled.json'
        if not target.exists():target.write_text(json.dumps(stalled(path),indent=2))
        print(name,'stalled',json.loads(target.read_text())['summary'],flush=True)
    (OUT/'selection.json').write_text(json.dumps(dict(selected=selected,checkpoint=str(selected_path),
         criterion='Development success, then lower position error; heldout panel not used for checkpoint selection',
         development_seed=6500000,heldout_seed=6700000,training_bank_seed=6600000),indent=2))

if __name__=='__main__':main()
