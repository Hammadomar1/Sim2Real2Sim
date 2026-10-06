"""Verify recorded evidence and authorize curriculum training only when it passes."""
import hashlib,json,xml.etree.ElementTree as ET
import torch
from so101_m1.scene import ROOT,ARTIFACTS
from so101_m1.training import source_hash

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):
    data=json.loads((ARTIFACTS/name).read_text())
    for path,expected in data.get('source_sha256',{}).items():
        assert digest(ROOT/path)==expected, 'Stale evidence: '+path
    return data

def main():
    native=read('gripper_precision_independent.json');gpu=read('gripper_mpc_precision.json')
    reset=read('gripper_reset_validation.json');soak=read('gripper_numerical_soak.json')
    table=read('table_candidate_validation.json')
    assert table['boundary_pass'] and table['settling']['finite'] and table['settling']['passed']==table['settling']['cases']
    assert native['input_sha256']==digest(ARTIFACTS/native['input_artifact'])==gpu['input_sha256']
    assert native['physically_valid_complete_path'] and native['within_30_seconds'] and native['position_error_m']<.003 and native['yaw_error_deg']<2
    assert len(gpu['terminal'])==1
    result=gpu['terminal'][0]
    assert result['success'] and not result['failure_flags'] and result['position_error_m']<.003 and abs(result['orientation_error_rad'])<.035
    assert reset['pass'] and sum(r['resets'] for r in reset['resets'])>=5620
    assert soak['passed'] and soak['transitions']>=460800 and len(soak['rows'])==6
    suites=list(ET.parse(ARTIFACTS/'gripper_final_rules_tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites)>=41
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    paths=[ROOT/'runs/gripper_smoke_final/latest.pt',ROOT/'runs/gripper_resume_final/latest.pt']
    a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    assert a['source_hash']==b['source_hash']==source_hash()
    assert (a['iteration'],b['iteration'],a['transitions'],b['transitions'])==(2,4,1024,2048)
    for checkpoint in [a,b]:
        for group in ['actor_state_dict','critic_state_dict']:
            assert all(torch.isfinite(v).all() for v in checkpoint['algorithm'][group].values())
        assert all(k in checkpoint for k in ['env_state','torch_rng','cuda_rng','random_state','numpy_state','validation_passes'])
    opt_a=a['algorithm']['optimizer_state_dict']['state'];opt_b=b['algorithm']['optimizer_state_dict']['state']
    assert opt_a and opt_a.keys()==opt_b.keys()
    assert all(float(opt_b[k]['step'])>float(opt_a[k]['step']) for k in opt_a)
    assert b['algorithm']['actor_state_dict']['obs_normalizer.count']>a['algorithm']['actor_state_dict']['obs_normalizer.count']
    assert a['physics_fields'].keys()==b['physics_fields'].keys()
    assert all(torch.equal(v,b['physics_fields'][k]) for k,v in a['physics_fields'].items())
    assert any(not torch.equal(a['algorithm']['actor_state_dict'][k],v) for k,v in b['algorithm']['actor_state_dict'].items())
    validation=[json.loads(line) for line in (ROOT/'runs/gripper_resume_final/validation.jsonl').read_text().splitlines()]
    assert validation[-1]['iteration']==4 and validation[-1]['episodes']==64
    smoke=dict(passed=True,iterations=[2,4],transitions=[1024,2048],optimizer_resumed=True,validation_episodes=64,source_hash=source_hash(),checkpoints={str(p.relative_to(ROOT)):digest(p) for p in paths})
    (ARTIFACTS/'gripper_ppo_resume_smoke.json').write_text(json.dumps(smoke,indent=2))
    mapping=dict(full_path_feasible='gripper_precision_independent.json',gpu_feedback_path_pass='gripper_mpc_precision.json',reset_tests_pass='gripper_reset_validation.json',success_detector_tests_pass='gripper_final_rules_tests.xml',numerical_soak_pass='gripper_numerical_soak.json',ppo_resume_smoke_pass='gripper_ppo_resume_smoke.json')
    report={key:True for key in mapping}
    report['evidence']={key:dict(path='artifacts/'+name,sha256=digest(ARTIFACTS/name)) for key,name in mapping.items()}
    sources=list((ROOT/'src/so101_m1').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'uv.lock',ROOT/'scripts/finalize_gripper_readiness.py']
    report['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sources}
    report['scope']='Ready to START curriculum RL. Diagnostic controllers are not learned policies. No trained success-rate or robustness claim. Table edges enforced as task termination boundaries.'
    (ARTIFACTS/'training_readiness.json').write_text(json.dumps(report,indent=2))
    from so101_m1.preflight import require_preflight
    require_preflight();print('PASS: gripper environment ready to start curriculum RL; trained milestone not complete.')

if __name__=='__main__':main()
