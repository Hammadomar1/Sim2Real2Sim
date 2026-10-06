"""Summarize the bounded LR comparison without choosing from cherry-picked scenes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path('artifacts/stage2_lr_comparison');r=json.loads((p/'comparison.json').read_text())
names=['early','adaptive','fixed'];colors=['#718096','#c47719','#168497'];labels=['Early checkpoint','Adaptive +25','Fixed 0.0003 +25']
fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for row,panel in enumerate(['fixed','fresh']):
    metrics=[r[n][panel] for n in names];summaries=[a['summary'] for a in metrics]
    success=np.array([a['success_rate']*100 for a in summaries]);ci=np.array([a['success_95ci_wilson'] for a in summaries])*100
    axes[row,0].bar(range(3),success,color=colors)
    axes[row,0].errorbar(range(3),success,yerr=np.stack((success-ci[:,0],ci[:,1]-success)),fmt='none',ecolor='black',capsize=4)
    axes[row,0].set(title=f'{panel.capitalize()} panel: success',ylabel='Success (%)',ylim=(0,100))
    for i,s in enumerate(success):axes[row,0].text(i,s+2,f'{round(s*1.28)}/128',ha='center')
    axes[row,1].bar(range(3),[a['metrics']['progress_mm'] for a in metrics],color=colors);axes[row,1].set(title='Useful pushing progress',ylabel='Mean reduction in goal error (mm)')
    axes[row,2].bar(range(3),[a['metrics']['jerk_per_second'] for a in metrics],color=colors);axes[row,2].set(title='Joint-motion abruptness',ylabel='Integrated squared jerk / episode seconds')
    for ax in axes[row]:ax.set_xticks(range(3),labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Same starting policy, 25 updates per branch, matched scenes within each panel\nActual gripper; 15-20 mm targets; 95% Wilson success intervals')
fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.pdf')

fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
for j,panel in enumerate(['fixed','fresh']):
    values=[r[n][panel] for n in names];x=np.arange(3)+(j-.5)*.32
    axes[0].bar(x,[v['episodes_with_jaw_contact'] for v in values],width=.32,label=panel)
    axes[1].bar(x,[100*v['metrics']['jaw_contact_fraction'] for v in values],width=.32,label=panel)
    axes[2].bar(x,[v['metrics']['longest_contact_seconds'] for v in values],width=.32,label=panel)
for ax,title,ylabel in zip(axes,['Episodes with sampled jaw contact','Time in sampled jaw contact','Longest sampled contact streak'],['Episodes / 128','Mean episode fraction (%)','Mean seconds']):
    ax.set(title=title,ylabel=ylabel);ax.set_xticks(range(3),labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend();fig.suptitle('Actual jaw/block contact flags sampled at 20 Hz; brief contacts between samples can be missed')
fig.savefig(p/'contact.png',dpi=160)

paired={}
for panel in ['fixed','fresh']:
    a=json.loads((p/f'adaptive_{panel}.json').read_text())['episodes'];b=json.loads((p/f'fixed_{panel}.json').read_text())['episodes']
    difference=np.array([y['success']-x['success'] for x,y in zip(a,b)])
    rng=np.random.default_rng(116);boot=difference[rng.integers(0,128,(10000,128))].mean(1)
    paired[panel]=dict(fixed_minus_adaptive_success_pp=float(difference.mean()*100),paired_bootstrap_95ci_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
        adaptive_only=sum(x['success'] and not y['success'] for x,y in zip(a,b)),fixed_only=sum(y['success'] and not x['success'] for x,y in zip(a,b)),both=sum(x['success'] and y['success'] for x,y in zip(a,b)))
(p/'paired.json').write_text(json.dumps(paired,indent=2))

fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for name,color in [('adaptive',colors[1]),('fixed',colors[2])]:
    rows=[json.loads(v) for v in (Path('runs/stage2_lr_comparison')/name/'updates.jsonl').read_text().splitlines()]
    for ax,key,title in zip(axes,['lr_after','post_update_kl','clip_fraction'],['Learning rate after update','Post-update policy KL','Post-update clip fraction']):
        ax.plot([a['update'] for a in rows],[a[key] for a in rows],label=name,color=color);ax.set(title=title,xlabel='Additional PPO updates');ax.grid(alpha=.2)
axes[0].set_yscale('log');axes[0].legend();fig.savefig(p/'ppo_updates.png',dpi=160)
print(json.dumps(paired,indent=2))
