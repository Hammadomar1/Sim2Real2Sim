import json
from pathlib import Path
import torch
from efficient_push import EfficientPushEnv
from stage2_curriculum import DistanceCurriculumEnv
from so101_m1.env import EnvConfig

torch.set_num_threads(1);results=[]
for stage in [1,2]:
    cfg=EnvConfig(num_envs=1,stage=stage,training=False,backend='native',seed=13)
    a=EfficientPushEnv(cfg);b=DistanceCurriculumEnv(cfg)
    for i in range(8):
        if i==7:a.episode_length_buf[:]=599;b.episode_length_buf[:]=599
        previous=a.episode_return.clone()
        x=a.step(torch.tensor([[.2,.3]]));y=b.step(torch.tensor([[.2,.3]]))
        assert torch.equal(a.d.qpos,b.d.qpos) and torch.equal(x[2],y[2]) and torch.isfinite(x[1]).all()
        if stage==1:assert torch.equal(x[1],y[1])
        if i==7:assert abs(a.last_terminal[0]['return']-float(previous+x[1]))<1e-5
    state=a.state_dict();a.had_contact[:]=True;a.load_state_dict(state)
    assert torch.equal(a.had_contact,state['efficient_push_state']['had_contact'])
    results.append(dict(stage=stage,physics_done_parity=True,terminal_return=True,state_roundtrip=True))
e=EfficientPushEnv(EnvConfig(num_envs=16,stage=2,training=False),0)
for i in range(4):
    _,reward,_,extras=e.step(torch.zeros(16,2,device=e.device))
    assert torch.isfinite(reward).all() and not extras['numerical_failures'].any()
assert e.jaw_gap().shape==(16,) and torch.isfinite(e.jaw_gap()).all()
bad=EfficientPushEnv(EnvConfig(num_envs=1,stage=2,training=False,backend='native'))
bad.rules.flags[0]=16
_,failure_reward,done,_=bad.step(torch.zeros(1,2))
assert done.all() and bad.last_terminal[0]['failed'] and float(failure_reward[0]) < -19
p=Path('artifacts/push_efficiency/checks.json')
p.write_text(json.dumps(dict(native=results,gpu16_zero_action_smoke=True,idle_reward_mean=float(reward.mean()),failure_reward=float(failure_reward[0]),failure_cost_exceeds_full_idle_time_cost=True),indent=2))
print(p.read_text())
