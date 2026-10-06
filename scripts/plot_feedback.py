from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for stem,label,color in [('full_path_validation','Previous 2 ms sequence','0.6'),('feedback_validation_2ms','Feedback: 2 ms','tab:blue'),('feedback_validation_1ms','Feedback: 1 ms','tab:orange')]:
    z=np.load(root/'artifacts'/f'{stem}.npz');q=z['qpos'];t=np.arange(1,len(q)+1)*float(z['dt'])
    yaw=np.unwrap(np.arctan2(2*(q[:,9]*q[:,12]+q[:,10]*q[:,11]),1-2*(q[:,11]**2+q[:,12]**2)))
    axes[0].plot(t,np.linalg.norm(q[:,6:8]-[.27,0],axis=1)*1000,color=color,label=label)
    axes[1].plot(t,np.degrees(yaw),color=color)
    axes[2].plot(q[:,6]*1000,q[:,7]*1000,color=color)
axes[0].axhline(10,color='green',ls='--');axes[0].axvline(30,color='black',ls=':');axes[0].set(xlabel='Simulation time (s)',ylabel='Parking position error (mm)',title='Faster completion');axes[0].legend(fontsize=8)
axes[1].axhspan(80,100,color='green',alpha=.12);axes[1].axvline(30,color='black',ls=':');axes[1].set(xlabel='Simulation time (s)',ylabel='Object yaw (degrees)',title='Turn, pass, then park')
axes[2].plot([215,215],[-60,-24],color='black',lw=3);axes[2].plot([215,215],[24,60],color='black',lw=3);axes[2].scatter([270],[0],marker='x',color='green');axes[2].set(xlabel='Block reference x (mm)',ylabel='Block reference y (mm)',title='Measured physical paths');axes[2].set_aspect('equal')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Object-pose feedback: same controller, separate 2 ms and 1 ms executions')
fig.savefig(root/'artifacts/feedback_comparison.png',dpi=160)
