"""Test curriculum progression, nontrivial goals, subset resets and resume state."""
import json
from pathlib import Path
import torch
from so101_m1.env import EnvConfig
from stage2_curriculum import DistanceCurriculumEnv,Promotion,DISTANCES

def main():
    torch.set_num_threads(1);p=Promotion()
    assert not p.observe(58,64) and not p.observe(58,64)
    assert not p.observe(57,64) and p.streak==0
    for level in range(3):
        assert not p.observe(58,64);assert not p.observe(58,64);assert p.observe(58,64);assert p.level==level+1
    assert not p.observe(64,64);assert not p.observe(64,64);assert not p.observe(64,64);assert p.complete
    results=[]
    for backend,n in [('native',1),('warp',128)]:
        e=DistanceCurriculumEnv(EnvConfig(num_envs=n,stage=2,training=False,backend=backend));records=[]
        for level in range(4):
            e.level=level;e.reset();delta=e.goal[:,:2]-e.d.qpos[:,6:8];d=torch.linalg.vector_norm(delta,dim=-1)
            assert (d>.0149).all(),'Never start inside 10 mm success tolerance'
            if level<3:assert (d>=DISTANCES[level][0]-1e-6).all() and (d<=DISTANCES[level][1]+1e-6).all()
            else:assert (delta[:,0]>=.03-1e-6).all() and (delta[:,0]<=.04+1e-6).all() and (delta[:,1].abs()<=.0125+1e-6).all()
            assert torch.allclose(e.d.mocap_pos[:,e.goal_mocap,:2],e.goal[:,:2])
            records.append(dict(level=level,min_distance_mm=float(d.min()*1000),max_distance_mm=float(d.max()*1000)))
        saved=e.state_dict();e.level=0;e.reset();e.load_state_dict(saved)
        assert e.level==3 and torch.equal(e.goal,saved['task']['goal']) and torch.equal(e.distance_levels,saved['distance_curriculum']['levels'])
        if n>1:
            before=e.d.qpos.clone();oldgoal=e.goal.clone();e.reset(torch.tensor([0],device=e.device));assert torch.equal(e.d.qpos[1:],before[1:]) and torch.equal(e.goal[1:],oldgoal[1:])
            e.cfg.training=True;e.level=3;e.reset();assert (e.distance_levels==3).any() and ((e.distance_levels>=0)&(e.distance_levels<3)).any() and (e.stage==1).any()
        e.cfg.training=False;e.level=0;e.reset()
        for _ in range(25):
            _,_,done,extras=e.step(torch.zeros((n,2),device=e.device))
            assert not done.any() and not extras['numerical_failures'].any(),'Idle action must not pass the easy level'
        results.append(dict(backend=backend,resets=records,zero_action_no_success=True,state_roundtrip=True,subset_reset=True if n>1 else None))
    out=dict(passed=True,promotion_streak_and_reset=True,full_range_restored=True,results=results)
    path=Path('artifacts/stage2_distance');path.mkdir(exist_ok=True);(path/'checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
