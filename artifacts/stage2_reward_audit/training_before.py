"""RSL-RL PPO with local logs, held-out validation and resumable bounded runs."""
import copy
import hashlib
import json
import math
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from .env import EnvConfig,PushTurnParkEnv
from .scene import ROOT,ARTIFACTS,MENAGERIE_REVISION

def config(seed):
    model={"class_name":"MLPModel","hidden_dims":[256,256,128],"activation":"elu","obs_normalization":True}
    actor={**model,"distribution_cfg":{"class_name":"GaussianDistribution","init_std":.6,"std_type":"scalar"}}
    return {"seed":seed,"num_steps_per_env":32,"obs_groups":{"actor":["actor"],"critic":["critic"]},"actor":actor,"critic":model,
            "multi_gpu":None,"algorithm":{"class_name":"PPO","learning_rate":.0003,"schedule":"adaptive","gamma":.995,"lam":.95,"num_learning_epochs":5,"num_mini_batches":4,"clip_param":.2,"entropy_coef":.005,"value_loss_coef":1.,"desired_kl":.015,"max_grad_norm":1.,"use_clipped_value_loss":True}}

def load_policy(path,env):
    checkpoint=torch.load(path,map_location=env.device,weights_only=False)
    cfg=copy.deepcopy(checkpoint["train_config"])
    actor_cfg=cfg["actor"]; actor_cfg.pop("class_name",None)
    actor=MLPModel(env.get_observations(),cfg["obs_groups"],"actor",2,**actor_cfg).to(env.device)
    actor.load_state_dict(checkpoint["algorithm"]["actor_state_dict"])
    actor.eval()
    return actor

def summarize(records):
    n=len(records)
    if not n: return {"episodes":0,"success_rate":0.}
    successes=sum(bool(r["success"]) for r in records)
    p=successes/n; z=1.959963984540054; den=1+z*z/n
    center=(p+z*z/(2*n))/den; radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {"episodes":n,"success_rate":p,"success_95ci_wilson":[center-radius,center+radius],
            "mean_position_error_mm":float(np.mean([r["position_error_m"] for r in records])*1000),
            "mean_orientation_error_deg":float(np.mean([r["orientation_error_rad"] for r in records])*180/math.pi),
            "mean_seconds":float(np.mean([r["seconds"] for r in records])),
            "mean_joint_jerk_integral":float(np.mean([r["joint_jerk_integral"] for r in records])),
            "failures":sum(bool(r["failed"]) for r in records),"timeouts":sum(bool(r["timeout"]) for r in records)}

@torch.inference_mode()
def evaluate_policy(policy,env,episodes,seed):
    """Collect exactly one episode per world per batch; no short-episode bias."""
    records=[]
    policy.eval()
    while len(records)<episodes:
        env.rng.manual_seed(seed+len(records))
        env.reset()
        first={}
        for _ in range(env.max_episode_length):
            action=policy(env.get_observations())
            env.step(action)
            for row in env.last_terminal:
                idx=int(row["env"])
                if idx not in first: first[idx]=row
            if len(first)==env.num_envs: break
        if len(first)!=env.num_envs:
            raise RuntimeError("Evaluation did not terminate every scheduled episode")
        for idx in sorted(first):
            if len(records)>=episodes: break
            row=first[idx]; row["scenario_index"]=len(records); records.append(row)
    return records

def evaluate_checkpoint(args):
    cfg=EnvConfig(num_envs=args.num_envs if args.backend=="warp" else 1,stage=args.stage,training=False,randomized=args.randomized,seed=args.seed,backend=args.backend)
    env=PushTurnParkEnv(cfg)
    policy=load_policy(args.checkpoint,env)
    records=evaluate_policy(policy,env,args.episodes,args.seed)
    return {"checkpoint":str(args.checkpoint),"checkpoint_sha256":hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),"config":asdict(cfg),"summary":summarize(records),"episodes":records}

def save_checkpoint(path,alg,env,train_cfg,iteration,transitions,passes,best_score,elapsed):
    state={"format_version":1,"algorithm":alg.save(),"train_config":train_cfg,"env_config":asdict(env.cfg),"env_state":env.state_dict(),"iteration":iteration,"transitions":transitions,"validation_passes":passes,"best_score":best_score,"elapsed_training_seconds":elapsed,"random_state":random.getstate(),"numpy_state":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"menagerie_revision":MENAGERIE_REVISION,"source_hash":source_hash()}
    if env.cfg.backend=='warp':
        state['physics_fields']={key:getattr(env.sim.model,key).clone() for key in env.default_fields}
    path=Path(path); tmp=path.with_suffix(".tmp"); torch.save(state,tmp); tmp.replace(path)

def source_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/"src"/"so101_m1").glob("*.py")): h.update(p.name.encode()+p.read_bytes())
    return h.hexdigest()

def train(args):
    if not 0<args.hours<=12: raise ValueError("Run duration must be above zero and at most 12 hours")
    from .preflight import require_preflight
    require_preflight(smoke=0<args.updates<=2 and args.num_envs<=32 and args.hours<=.1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    cfg=EnvConfig(num_envs=args.num_envs,seed=args.seed,stage=args.stage,smoothness=not args.no_smoothness,randomized=args.randomized)
    saved=None
    if args.resume:
        saved=torch.load(args.resume,map_location="cuda:0",weights_only=False)
        if saved.get('source_hash')!=source_hash():
            raise ValueError('Resume refused: source files changed since this checkpoint. Preserve the validated training version.')
        cfg=EnvConfig(**saved["env_config"])
        if cfg.num_envs!=args.num_envs: raise ValueError("Resume must preserve --num-envs for full environment state restoration")
    env=PushTurnParkEnv(cfg)
    train_cfg=saved["train_config"] if saved else config(args.seed)
    alg=PPO.construct_algorithm(env.get_observations(),env,copy.deepcopy(train_cfg),env.device)
    iteration=0; transitions=0; passes=0; best_score=(-1.,-float("inf"),-float("inf")); prior_elapsed=0.
    if saved:
        alg.load(saved["algorithm"], load_cfg=None, strict=True)
        if saved.get('physics_fields'):
            from mjlab.managers.event_manager import RecomputeLevel
            for key,value in saved['physics_fields'].items():getattr(env.sim.model,key)[:]=value
            env.sim.recompute_constants(RecomputeLevel.set_const)
        env.load_state_dict(saved["env_state"])
        iteration=saved["iteration"]; transitions=saved["transitions"]; passes=saved["validation_passes"]; best_score=tuple(saved["best_score"]); prior_elapsed=saved["elapsed_training_seconds"]
        random.setstate(saved["random_state"]); np.random.set_state(saved["numpy_state"]); torch.set_rng_state(saved["torch_rng"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
    (out/"config.json").write_text(json.dumps({"environment":asdict(cfg),"ppo":train_cfg,"menagerie_revision":MENAGERIE_REVISION,"source_hash":source_hash()},indent=2))
    writer=SummaryWriter(str(out/"tensorboard"),purge_step=iteration if saved else None)
    stop={"requested":False}
    def request_stop(*_): stop["requested"]=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    start=time.monotonic(); start_iter=iteration; start_transitions=transitions
    val_env=None; since=[]; obs=env.get_observations(); alg.train_mode()
    failed=False
    if not saved:save_checkpoint(out/"latest.pt",alg,env,train_cfg,iteration,transitions,passes,best_score,0.)
    try:
        while not stop["requested"] and time.monotonic()-start<args.hours*3600 and transitions<args.max_transitions:
            if args.updates and iteration-start_iter>=args.updates: break
            t0=time.monotonic()
            with torch.inference_mode():
                for _ in range(train_cfg["num_steps_per_env"]):
                    actions=alg.act(obs)
                    obs,rewards,dones,extras=env.step(actions)
                    if extras['numerical_failures'].any() or not torch.isfinite(obs["actor"]).all() or not torch.isfinite(rewards).all():
                        torch.save(env.last_numerical_failure or env.state_dict(),out/"numerical_failure.pt")
                        raise RuntimeError("Non-finite environment state; training stopped for diagnosis")
                    alg.process_env_step(obs,rewards,dones,extras)
                    since.extend(env.last_terminal)
                alg.compute_returns(obs)
            losses=alg.update()
            iteration+=1; transitions+=env.num_envs*train_cfg["num_steps_per_env"]
            for name,val in losses.items(): writer.add_scalar("loss/"+name,float(val),transitions)
            writer.add_scalar("train/stage",env.cfg.stage,transitions)
            writer.add_scalar("train/transitions_per_second",env.num_envs*train_cfg["num_steps_per_env"]/(time.monotonic()-t0),transitions)
            if iteration%10==0:
                stats=summarize(since)
                row={"iteration":iteration,"transitions":transitions,"stage":env.cfg.stage,"elapsed_seconds":time.monotonic()-start,**stats}
                with (out/"progress.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
                for k,v in stats.items():
                    if isinstance(v,(float,int)): writer.add_scalar("train/"+k,v,transitions)
                print(json.dumps(row),flush=True); since=[]
            if iteration%args.validation_interval==0:
                if val_env is None: val_env=PushTurnParkEnv(EnvConfig(num_envs=32,training=False,stage=env.cfg.stage,randomized=env.cfg.randomized,seed=1000000+args.seed))
                val_env.cfg.stage=env.cfg.stage; val_env.cfg.randomized=env.cfg.randomized
                records=evaluate_policy(alg.get_policy(),val_env,64,1000000+args.seed*10000+iteration*64)
                metrics=summarize(records)
                with (out/"validation.jsonl").open("a") as f: f.write(json.dumps({"iteration":iteration,"transitions":transitions,"stage":env.cfg.stage,**metrics})+"\n")
                for k,v in metrics.items():
                    if isinstance(v,(float,int)): writer.add_scalar("validation/"+k,v,transitions)
                score=(metrics["success_rate"],-metrics["mean_position_error_mm"],-metrics["mean_joint_jerk_integral"])
                if score>best_score:
                    best_score=score
                    save_checkpoint(out/f"best_stage{env.cfg.stage}.pt",alg,env,train_cfg,iteration,transitions,passes,best_score,prior_elapsed+time.monotonic()-start)
                    if env.cfg.stage==5: save_checkpoint(out/"best.pt",alg,env,train_cfg,iteration,transitions,passes,best_score,prior_elapsed+time.monotonic()-start)
                passes=passes+1 if metrics["success_rate"]>=.90 else 0
                print("VALIDATION "+json.dumps({"stage":env.cfg.stage,"passes":passes,**metrics}),flush=True)
                if passes>=3 and env.cfg.stage<5:
                    env.cfg.stage+=1; passes=0; best_score=(-1.,-float("inf"),-float("inf")); env.reset(); print(f"CURRICULUM advanced to {env.cfg.stage}",flush=True)
                elif passes>=3 and env.cfg.stage==5 and metrics["success_rate"]>=.95:
                    print("Full-task nominal validation gate met; final held-out evaluation is still required.",flush=True)
                    stop["requested"]=True
                obs=env.get_observations(); alg.train_mode()
            if iteration%25==0:
                save_checkpoint(out/"latest.pt",alg,env,train_cfg,iteration,transitions,passes,best_score,prior_elapsed+time.monotonic()-start)
                free,total=torch.cuda.mem_get_info()
                writer.add_scalar('system/vram_used_gib',(total-free)/2**30,transitions)
                writer.flush()
    except BaseException as error:
        failed=True
        (out/'failure.json').write_text(json.dumps({'iteration':iteration,'transitions':transitions,'error':repr(error)}))
        raise
    finally:
        # Iteration boundary snapshots contain no partial PPO rollout.
        if not failed:save_checkpoint(out/"latest.pt",alg,env,train_cfg,iteration,transitions,passes,best_score,prior_elapsed+time.monotonic()-start)
        writer.close()
        (out/"status.json").write_text(json.dumps({"state":"failed" if failed else "stopped","iteration":iteration,"transitions":transitions,"stage":env.cfg.stage,"run_seconds":time.monotonic()-start,"milestone_complete":False},indent=2))
    print(f"Checkpoint saved: {out/'latest.pt'}",flush=True)
