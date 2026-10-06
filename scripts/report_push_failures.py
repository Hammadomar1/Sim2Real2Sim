"""Plot observed failure traces and export the exact batch episodes for the GUI."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from so101_m1.scene import BLOCK_BOXES

out=Path('artifacts/push_failure_diagnosis')
r=json.loads((out/'report.json').read_text())
d=np.load(out/'traces.npz')
fail=[x for x in r['episodes'] if not x['success']]
selected=[]
for category in r['categories']:
    candidates=[x for x in fail if x['category']==category]
    if candidates:
        selected.append(sorted(candidates,key=lambda x:x['net_progress_mm'])[len(candidates)//2])
side=[x for x in fail if x['goal_facing_side_at_end']]
if side:
    candidate=max(side,key=lambda x:x['last5_jaw_goal_side_cosine'])
    if candidate not in selected:selected.append(candidate)
selected=selected[:5]
fig,axes=plt.subplots(len(selected),3,figsize=(15,3.6*len(selected)),squeeze=False)
for row,axs in zip(selected,axes):
    i=row['scenario_index']; n=row['steps'];t=np.arange(n+1)*.05
    xy=d['xy'][:n+1,i]*1000; jaw=d['jaw'][:n+1,i,:2]*1000;goal=d['goal'][i,:2]*1000
    err=np.linalg.norm(xy-goal,axis=-1)
    ax=axs[0];ax.plot(*xy.T,label='Block center',lw=2);ax.plot(*jaw.T,label='Nearest jaw tip',alpha=.65)
    ax.scatter(*xy[0],marker='o',c='black',s=20);ax.scatter(*xy[-1],marker='x',c='red',s=50)
    ax.add_patch(Circle(goal,10,fill=False,color='green',ls='--'));ax.scatter(*goal,marker='+',c='green')
    yaw=d['yaw'][n,i];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    for pos,size in BLOCK_BOXES:
        points=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array(size[:2])+np.array(pos[:2])
        ax.add_patch(Polygon(points@rot.T*1000+xy[-1],color='steelblue',alpha=.25))
    ax.set_aspect('equal');ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.legend(fontsize=7)
    ax.set_title(f"Scene {i}: {row['category'].replace('_',' ')}",fontsize=10)
    ax=axs[1];ax.plot(t,err);ax.axhline(10,c='green',ls='--',label='Position tolerance')
    ax.scatter(t[1:][d['touch'][1:n+1,i]!=0],err[1:][d['touch'][1:n+1,i]!=0],s=9,c='orange',label='Sampled contact')
    ax.set_xlabel('Time (s)');ax.set_ylabel('Goal error (mm)');ax.legend(fontsize=7)
    ax=axs[2];speed=np.linalg.norm(np.diff(d['tip'][:n+1,i,:2],axis=0),axis=-1)*20000
    command=np.linalg.norm(d['velocity'][1:n+1,i],axis=-1)*1000
    ax.plot(t[1:],speed,label='Actual tool speed',alpha=.8);ax.plot(t[1:],command,label='Command speed',alpha=.7)
    ax.set_xlabel('Time (s)');ax.set_ylabel('Speed (mm/s)');ax.legend(fontsize=7)
    np.savez(out/f'scene_{i}.npz',qpos=d['qpos'][:n+1,i],mocap_pos=d['mocap_pos'][i],
             mocap_quat=d['mocap_quat'][i],dt=.05,end_effector='closed_gripper')
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Stage 2 failure diagnosis: unchanged filtered policy, seed 6600000',fontsize=14)
fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'failure_traces.png',dpi=160);fig.savefig(out/'failure_traces.pdf')
(out/'representatives.json').write_text(json.dumps(selected,indent=2))
print(json.dumps({'categories':r['categories'],'representatives':[x['scenario_index'] for x in selected]},indent=2))
