"""Prevent expensive training while physical validation remains incomplete."""
import hashlib
import json
from .scene import ARTIFACTS, ROOT


def require_preflight(smoke=False):
    report_path=ARTIFACTS/'physics_validation.json'
    if not report_path.exists():
        raise RuntimeError('Training disabled: run scripts/validate_physics.py --gpu first.')
    report=json.loads(report_path.read_text())
    if not report.get('contact_checks_pass'):
        raise RuntimeError('Training disabled: contact stability checks have not passed.')
    for name, expected in report.get('source_sha256',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Training disabled: physics evidence is stale after changing {name}.')
    if not any(c.get('case')=='native_vs_warp' for c in report.get('comparisons',[])):
        raise RuntimeError('Training disabled: native/Warp contact comparison is missing.')
    if smoke:
        return
    readiness_path=ARTIFACTS/'training_readiness.json'
    if not readiness_path.exists():
        raise RuntimeError('Long training disabled: full-path feasibility, reset/success tests, and PPO resume smoke test are still required.')
    readiness=json.loads(readiness_path.read_text())
    required=['full_path_feasible','gpu_feedback_path_pass','reset_tests_pass','success_detector_tests_pass','numerical_soak_pass','ppo_resume_smoke_pass']
    missing=[key for key in required if readiness.get(key) is not True]
    if missing:
        raise RuntimeError('Long training disabled: incomplete checks: '+', '.join(missing))
    for key in required:
        evidence=readiness.get('evidence',{}).get(key,{})
        path=ROOT/evidence.get('path','missing-evidence')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence.get('sha256'):
            raise RuntimeError('Long training disabled: missing or changed evidence for '+key)
    for name,expected in readiness.get('source_sha256',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Long training disabled: readiness evidence is stale after changing '+name)
