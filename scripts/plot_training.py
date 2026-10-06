"""Plot recorded training metrics without changing the training configuration."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='runs/seed0');a=p.parse_args();root=Path(a.run_dir)
    def read(name):return [json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
    train=read('progress.jsonl');val=read('validation.jsonl');out=Path('artifacts')/root.name;out.mkdir(parents=True,exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid');fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors={1:'#1672b8',2:'#d66517',3:'#369849',4:'#785eb8',5:'#b72456'}
    x=[r['transitions']/1e6 for r in train]
    ax=axes[0,0];ax.plot(x,[100*r['success_rate'] for r in train],color='gray',alpha=.65,label='Training: mixed stages after promotion')
    for stage in sorted({r['stage'] for r in val}):
        rows=[r for r in val if r['stage']==stage];vx=[r['transitions']/1e6 for r in rows]
        ax.plot(vx,[100*r['success_rate'] for r in rows],'o-',color=colors[stage],label=f'Stage {stage} validation (64 episodes)')
        ax.fill_between(vx,[max(0,r['success_95ci_wilson'][0])*100 for r in rows],[r['success_95ci_wilson'][1]*100 for r in rows],color=colors[stage],alpha=.13)
    ax.axhline(90,color='black',ls='--',lw=1,label='Promotion threshold');ax.set(title='Success: contact learned, pushing still pending',ylabel='Success (%)',ylim=(-3,104));ax.legend(fontsize=8)
    ax=axes[0,1]
    for stage in sorted({r['stage'] for r in val}):
        rows=[r for r in val if r['stage']==stage];ax.plot([r['transitions']/1e6 for r in rows],[r['mean_seconds'] for r in rows],'o-',color=colors[stage],label=f'Stage {stage}')
    ax.set(title='Validation episode duration (includes timeouts)',ylabel='Seconds');ax.legend()
    ax=axes[1,0]
    for key,label in [('timeouts','Timeout'),('failures','Rule failure')]:ax.plot(x,[100*r[key]/max(1,r['episodes']) for r in train],label=label)
    ax.set(title='Training outcomes',ylabel='Episodes (%)');ax.legend()
    ax=axes[1,1]
    for stage in sorted({r['stage'] for r in val}):
        rows=[r for r in val if r['stage']==stage];ax.plot([r['transitions']/1e6 for r in rows],[r['mean_position_error_mm'] for r in rows],'o-',color=colors[stage],label=f'Stage {stage}')
    ax.axhline(10,color='black',ls='--',lw=1,label='Stage 2 position tolerance');ax.set(title='Validation block-to-goal error',ylabel='Mean error (mm)');ax.legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Environment transitions (millions)')
    fig.suptitle(f'{root.name}: logged learning progress\nStage 1 = gripper contact; stage 2 = nearby pushing. Shading: validation 95% Wilson intervals.',fontsize=13)
    fig.savefig(out/'training_progress.png',dpi=170);fig.savefig(out/'training_progress.pdf')
    (out/'metrics_snapshot.json').write_text(json.dumps(dict(training=train,validation=val,status=json.loads((root/'status.json').read_text())),indent=2))
    print(out/'training_progress.png')

if __name__=='__main__':main()
