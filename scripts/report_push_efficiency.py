import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path('artifacts/push_efficiency')
files=['early_fresh','control_fresh','efficient_fresh'];labels=['Early\npolicy','Old reward\n+50 updates','Corrected reward\n+50 updates']
if (p/'filter0.5_fresh.json').exists():files.append('filter0.5_fresh');labels.append('Corrected\n+ filter')
data=[json.loads((p/(f+'.json')).read_text()) for f in files]
assert data[0]['initial_scene_hashes']==data[1]['initial_scene_hashes']==data[2]['initial_scene_hashes']
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');colors=['#7a8798','#ce8626','#168b91','#7b66b2'][:len(data)]
fields=[('success_rate','Success on fresh scenes','Success (%)'),('mean_position_error_mm','Accuracy','Final error (mm)'),('mean_seconds','Completion or timeout','Episode seconds'),('contact_bouts','Separated contact attempts','Mean sampled contact bouts'),('mean_tool_speed_mm_s','Actual gripper motion','Mean tool speed (mm/s)'),('jerk_per_second','Motion abruptness','Integrated squared joint jerk / seconds')]
for ax,(key,title,ylabel) in zip(axes.flat,fields):
    values=[r['summary'][key] if key in r['summary'] else r['metrics'][key] for r in data]
    if key=='success_rate':values=[100*v for v in values]
    ax.bar(range(len(data)),values,color=colors);ax.set(title=title,ylabel=ylabel);ax.set_xticks(range(len(data)),labels,fontsize=7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    if key=='success_rate':
        ci=np.array([r['summary']['success_95ci_wilson'] for r in data])*100;v=np.array(values)
        ax.errorbar(range(len(data)),v,yerr=np.stack((v-ci[:,0],ci[:,1]-v)),fmt='none',color='black',capsize=4);ax.set_ylim(0,100)
    if key=='mean_position_error_mm':ax.axhline(10,ls='--',color='black',lw=1)
fig.suptitle('Push-efficiency correction: same 64 fresh starts; original 40 mm/s speed limit\nContact attempts are separated by >=250 ms without sampled contact; speed alone is not success')
fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.pdf')
paired={}
for baseline in ['early_fresh','control_fresh']:
    old=json.loads((p/(baseline+'.json')).read_text())['episodes'];new=data[2]['episodes']
    common=[(a,b) for a,b in zip(old,new) if a['success'] and b['success']]
    paired[baseline]={'common_successes':len(common),'old_only':sum(a['success'] and not b['success'] for a,b in zip(old,new)),'new_only':sum(b['success'] and not a['success'] for a,b in zip(old,new))}
    for key in ['seconds','contact_bouts','position_error_m','jerk_per_second']:
        paired[baseline][key]=None if not common else {'old_mean':float(np.mean([a[key] for a,b in common])),'new_mean':float(np.mean([b[key] for a,b in common]))}
(p/'paired_successes.json').write_text(json.dumps(paired,indent=2))
print(json.dumps(paired,indent=2))
