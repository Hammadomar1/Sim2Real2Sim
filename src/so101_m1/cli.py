from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .scene import ARTIFACTS, build_scene, inspect_workspace

def main():
    p=argparse.ArgumentParser(description="SO-101 Push-Turn-Park: simulation only")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor")
    s=sub.add_parser("scene"); s.add_argument("--workspace",action="store_true")
    s.add_argument("--image",action="store_true")
    s=sub.add_parser("smoke"); s.add_argument("--num-envs",type=int,default=16); s.add_argument("--steps",type=int,default=100); s.add_argument("--backend",choices=["warp","native"],default="warp")
    s=sub.add_parser("benchmark"); s.add_argument("--sizes",type=int,nargs="+",default=[256,512,1024,2048]); s.add_argument("--steps",type=int,default=100)
    s=sub.add_parser("train"); s.add_argument("--num-envs",type=int,default=512); s.add_argument("--seed",type=int,default=0); s.add_argument("--hours",type=float,default=10); s.add_argument("--max-transitions",type=int,default=100_000_000); s.add_argument("--updates",type=int,default=0); s.add_argument("--stage",type=int,default=1,choices=range(1,6)); s.add_argument("--run-dir",default="runs/seed0"); s.add_argument("--resume"); s.add_argument("--no-smoothness",action="store_true"); s.add_argument("--randomized",action="store_true"); s.add_argument("--validation-interval",type=int,default=50)
    s=sub.add_parser("evaluate"); s.add_argument("checkpoint"); s.add_argument("--episodes",type=int,default=500); s.add_argument("--num-envs",type=int,default=128); s.add_argument("--stage",type=int,default=5); s.add_argument("--seed",type=int,default=2000000); s.add_argument("--randomized",action="store_true"); s.add_argument("--backend",choices=["warp","native"],default="warp"); s.add_argument("--output",default="artifacts/evaluation.json")
    s=sub.add_parser("play"); s.add_argument("--checkpoint"); s.add_argument("--stage",type=int,default=5); s.add_argument("--seed",type=int,default=3000000); s.add_argument("--video"); s.add_argument("--episodes",type=int,default=3); s.add_argument("--viewer",action="store_true")
    s=sub.add_parser("diagnose"); s.add_argument("--kind",choices=["zero","push","parity","feasibility"],default="zero")
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.command=="doctor":
        import importlib.metadata as meta
        import warp as wp
        wp.init()
        report={"packages":{x:meta.version(x) for x in ["mujoco","mujoco-warp","mjlab","warp-lang","torch","rsl-rl-lib"]},"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"vram_bytes":torch.cuda.get_device_properties(0).total_memory}
        a=torch.randn(128,128,device="cuda"); report["cuda_math"]=bool(torch.isfinite(a@a).all())
        ARTIFACTS.mkdir(exist_ok=True); (ARTIFACTS/"environment.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    elif args.command=="scene":
        print(build_scene())
        if args.workspace:
            records=inspect_workspace(); print("Reachable:",sum(r["error_m"]<.001 for r in records),"/",len(records))
        if args.image:
            from .render import scene_image
            print(scene_image())
    elif args.command=="smoke":
        from .env import EnvConfig,PushTurnParkEnv
        env=PushTurnParkEnv(EnvConfig(num_envs=args.num_envs,backend=args.backend))
        for i in range(args.steps):
            action=torch.zeros((env.num_envs,2),device=env.device) if i<args.steps//2 else torch.rand((env.num_envs,2),device=env.device)*2-1
            obs,rew,done,_=env.step(action)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rew).all()
        print(json.dumps({"steps":args.steps,"num_envs":env.num_envs,"finite":True,"observation_dim":obs["actor"].shape[-1],"tip":env.state()[2][0].tolist()}))
    elif args.command=="benchmark":
        import subprocess,sys
        reports=[]
        for n in args.sizes:
            # A fresh process frees all CUDA graphs, contact buffers and models.
            result=subprocess.run([sys.executable,"-m","so101_m1.benchmark",str(n),str(args.steps)],check=False,capture_output=True,text=True)
            (ARTIFACTS/f"benchmark_{n}.log").write_text(result.stdout+result.stderr)
            if result.returncode==0: reports.append(json.loads(result.stdout.split("BENCHMARK_JSON=")[-1]))
            else: reports.append({"num_envs":n,"error":result.stderr[-2000:]})
            print(reports[-1],flush=True)
        usable=[r for r in reports if r.get("headroom",0)>=.20 and r.get("finite",False)]
        out={"runs":reports,"recommended_num_envs":max(usable,key=lambda r:r["transitions_per_second"])["num_envs"] if usable else None}
        (ARTIFACTS/"benchmark.json").write_text(json.dumps(out,indent=2))
    elif args.command=="train":
        from .training import train
        train(args)
    elif args.command=="evaluate":
        from .training import evaluate_checkpoint
        result=evaluate_checkpoint(args)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result["summary"],indent=2))
    elif args.command=="play":
        from .render import play
        play(args)
    elif args.command=="diagnose":
        from .diagnostics import diagnose
        diagnose(args.kind)

if __name__=="__main__": main()
