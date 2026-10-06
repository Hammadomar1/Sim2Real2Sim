import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='runs/stage2_distance_seed73');a=p.parse_args();folder=Path(a.run_dir)
    rows=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');colors=['#147da0','#be640e','#38954c','#884da3']
    for level in sorted({r['distance_level'] for r in rows}):
        r=[v for v in rows if v['distance_level']==level];x=[v['transitions']/1e6 for v in r];label=r[0]['distance_label']
        axes[0].plot(x,[100*v['success_rate'] for v in r],'o-',label=label,color=colors[level])
        axes[1].plot(x,[v['mean_position_error_mm'] for v in r],'o-',label=label,color=colors[level])
    axes[0].axhline(90,ls='--',color='black',lw=1,label='Advance: 3 checks at >=90%');axes[0].set(title='Stage-2 validation success',ylabel='Success (%)',ylim=(-2,102));axes[0].legend(fontsize=8)
    axes[1].axhline(10,ls='--',color='black',lw=1,label='Position tolerance');axes[1].set(title='Final position error',ylabel='Mean error (mm)');axes[1].legend(fontsize=8)
    axes[2].step([v['transitions']/1e6 for v in rows],[v['distance_level']+1 for v in rows],where='post');axes[2].set(title='Distance level evaluated',ylabel='Level',yticks=[1,2,3,4],ylim=(.7,4.3))
    for ax in axes:ax.set_xlabel('Additional transitions (millions)');ax.grid(alpha=.25)
    fig.suptitle('Stage 2: shorter pushes first; unchanged 10 mm tolerance and one-second stationary hold\n64 new validation scenes per check; no claim of full-task mastery')
    out=Path('artifacts/stage2_distance');out.mkdir(exist_ok=True);fig.savefig(out/'learning.png',dpi=160);fig.savefig(out/'learning.pdf')
    print(out/'learning.png')

if __name__=='__main__':main()
