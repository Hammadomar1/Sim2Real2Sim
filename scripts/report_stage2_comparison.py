"""Render a paired, stage-specific reward experiment report from real logs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    output=Path('artifacts/stage2_reward_audit');results={}
    for variant in ['baseline','corrected','directional']:
        folder=Path('runs/stage2_'+variant+'_audit')
        if not (folder/'validation.jsonl').exists():continue
        results[variant]=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');colors={'baseline':'#777777','corrected':'#087c91','directional':'#bf5916'}
    for name,rows in results.items():
        x=[r['transitions']/1e6 for r in rows]
        axes[0].plot(x,[100*r['success_rate'] for r in rows],'o-',label=name,color=colors[name])
        axes[1].plot(x,[r['mean_position_error_mm'] for r in rows],'o-',label=name,color=colors[name])
        axes[2].plot(x,[r['failures'] for r in rows],'o-',label=name,color=colors[name])
    for ax,title,ylabel in zip(axes,['Stage 2 validation success','Mean final distance to goal','Invalid episodes'],['Success (%)','Error (mm)','Rule failures / 64']):
        ax.set(title=title,xlabel='Additional transitions (millions)',ylabel=ylabel);ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylim(-2,102);axes[1].axhline(10,ls='--',color='black',lw=1)
    fig.suptitle('Matched stage-2 comparison: same starting actor, seed, scenes and restored exploration\n64 validation episodes; one training seed; development results, not a final test')
    fig.savefig(output/'comparison.png',dpi=170);fig.savefig(output/'comparison.pdf')
    (output/'comparison.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v[-1] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
