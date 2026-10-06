import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path('artifacts/stage2_decline');r=json.loads((p/'comparison.json').read_text())
names=['best','latest','latest_best_norm','best_latest_norm']
labels=['Early\noriginal scaling','Later\noriginal scaling','Later\nearly scaling','Early\nlater scaling']
fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
s=[r[n]['summary'] for n in names];x=np.arange(4);colors=['#147da0','#c17a18','#7c66ad','#458358']
v=np.array([a['success_rate']*100 for a in s]);ci=np.array([a['success_95ci_wilson'] for a in s])*100
axes[0].bar(x,v,color=colors);axes[0].errorbar(x,v,yerr=np.stack((v-ci[:,0],ci[:,1]-v)),fmt='none',color='black',capsize=4)
axes[0].set(ylabel='Success (%)',title='Same 128 starting scenes',ylim=(0,max(40,float(ci.max())+5)))
for i,a in enumerate(s):axes[0].text(i,v[i]+1,f"{round(a['success_rate']*128)}/128",ha='center',fontsize=9)
axes[1].bar(x,[a['mean_position_error_mm'] for a in s],color=colors);axes[1].axhline(10,color='black',ls='--',lw=1);axes[1].set(title='Final position error',ylabel='Mean error (mm)')
axes[2].bar(x,[a['mean_joint_jerk_integral'] for a in s],color=colors);axes[2].set(title='Joint motion abruptness',ylabel='Mean integrated squared joint jerk\n(lower is smoother)')
for ax in axes:ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Stage-2 diagnosis: checkpoint comparison and input-scaling swaps\n15-20 mm targets; success intervals are Wilson 95%; no additional training')
fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.pdf')
