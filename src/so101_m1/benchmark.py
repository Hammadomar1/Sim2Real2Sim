import sys,json,time
import torch
from .env import EnvConfig,PushTurnParkEnv

def main():
    torch.set_num_threads(1)
    n,steps=map(int,sys.argv[1:3])
    env=PushTurnParkEnv(EnvConfig(num_envs=n,stage=5,training=False))
    a=torch.zeros((n,2),device="cuda")
    for _ in range(10): env.step(a)
    torch.cuda.synchronize(); start=time.perf_counter()
    finite=True
    for _ in range(steps):
        obs,r,_,_=env.step(a)
        finite &= bool(torch.isfinite(obs["actor"]).all() & torch.isfinite(r).all())
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    free,total=torch.cuda.mem_get_info()
    print("BENCHMARK_JSON="+json.dumps({"num_envs":n,"steps":steps,"elapsed_seconds":elapsed,"transitions_per_second":n*steps/elapsed,"headroom":free/total,"finite":finite,"memory_used_gib":(total-free)/2**30}),flush=True)

if __name__=="__main__": main()
