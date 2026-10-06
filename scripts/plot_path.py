from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for stem,label,color in [('full_path_validation','2 ms: completed','tab:blue'),('full_path_validation_1ms','1 ms: failed task','tab:orange')]:
    z=np.load(root/'artifacts'/f'{stem}.npz');q=z['qpos'];t=np.arange(1,len(q)+1)*float(z['dt']);yaw=np.unwrap(np.arctan2(2*(q[:,9]*q[:,12]+q[:,10]*q[:,11]),1-2*(q[:,11]**2+q[:,12]**2)))
    axes[0].plot(q[:,6]*1000,q[:,7]*1000,label=label,color=color)
    axes[1].plot(t,np.degrees(yaw),color=color,label=label)
    axes[2].plot(t,np.linalg.norm(q[:,6:8]-[.27,0],axis=1)*1000,color=color)
axes[0].plot([215,215],[-100,-24],color='black',lw=4);axes[0].plot([215,215],[24,100],color='black',lw=4)
axes[0].scatter([150,270],[0,0],marker='x',color='green',s=60);axes[0].set(xlabel='Block reference x (mm)',ylabel='Block reference y (mm)',title='Measured block path');axes[0].set_aspect('equal');axes[0].legend(fontsize=8)
axes[1].axhspan(80,100,color='green',alpha=.12);axes[1].set(xlabel='Simulation time (s)',ylabel='Yaw (degrees)',title='Parking orientation: 90 degrees')
axes[2].axhline(10,color='green',ls='--');axes[2].set(xlabel='Simulation time (s)',ylabel='Parking position error (mm)',title='Position tolerance: 10 mm')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Diagnostic command replay: one reset, physical contacts, no RL')
fig.savefig(root/'artifacts/full_path_comparison.png',dpi=160)
