"""Evaluate a distance-curriculum checkpoint at its saved or an explicit level."""
import argparse,hashlib,json
from pathlib import Path
import torch
from so101_m1.env import EnvConfig
from so101_m1.training import load_policy,evaluate_policy,summarize
from stage2_curriculum import DistanceCurriculumEnv,LABELS

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--level',type=int,choices=range(4));p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=5300000);p.add_argument('--output',default='artifacts/stage2_distance/evaluation.json');a=p.parse_args()
    torch.set_num_threads(1);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    level=a.level if a.level is not None else saved.get('policy_evaluation_level',saved['distance_training']['level'])
    e=DistanceCurriculumEnv(EnvConfig(num_envs=64,stage=2,training=False,seed=a.seed),level)
    policy=load_policy(a.checkpoint,e);rows=evaluate_policy(policy,e,a.episodes,a.seed)
    report=dict(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),distance_level=level,distance_label=LABELS[level],seed=a.seed,summary=summarize(rows),episodes=rows,scope='Stage-2 distance curriculum only; not full push-turn-park success.')
    path=Path(a.output);path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))

if __name__=='__main__':main()
