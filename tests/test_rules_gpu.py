"""Run the actual CUDA rule kernels against CPU reference fixtures."""
import math
import numpy as np
import pytest
import torch
import warp as wp
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.rules import COLLISION,PENETRATION,SHORTCUT

@pytest.fixture(scope='module')
def worlds():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    return [PushTurnParkEnv(EnvConfig(num_envs=1,backend=b,training=False,stage=5)) for b in ['native','warp']]

def place(e,x,y=0.,yaw=0.,z=.0092,reset=False):
    e.d.qpos[0,:6]=torch.tensor(e.q_initial,device=e.device,dtype=e.dtype)
    e.d.qpos[0,6:13]=torch.tensor([x,y,z,math.cos(yaw/2),0,0,math.sin(yaw/2)],device=e.device,dtype=e.dtype)
    e.d.qvel[:]=0;e.goal[0]=torch.tensor([.27,0.,math.pi/2],device=e.device,dtype=e.dtype)
    e.sim.forward()
    if reset:e.rules.reset(torch.tensor([0],device=e.device))

def tick(e):
    if e.cfg.backend=='native':e.rules.native()
    else:e.rules.launch()

def summary(e):return tuple(int(getattr(e.rules,k)[0]) for k in ['flags','mask','passed','hold'])

def test_cuda_corner_history_matches_cpu(worlds):
    for e in worlds:e.reset();place(e,.15,reset=True)
    # Offset away from exact plane equality: float32/64 can classify equality
    # one sample apart, but must agree on resolved crossings and completion.
    for x in np.arange(.15013,.271,.002):
        for e in worlds:place(e,x);tick(e)
        assert summary(worlds[0])==summary(worlds[1])
    assert summary(worlds[0])[1:3]==(65535,1)

def test_exact_plane_roundoff_resolves_without_false_failure(worlds):
    for e in worlds:e.reset();place(e,.15,reset=True)
    # At exact equality intermediate masks may differ by one sample because
    # the two engines use different precision. Final passage must agree.
    for x in np.arange(.15,.271,.002):
        for e in worlds:place(e,x);tick(e)
    assert summary(worlds[0])==summary(worlds[1])==(0,65535,1,0)

@pytest.mark.parametrize('case',['teleport','lift','around'])
def test_cuda_rejects_shortcuts_and_lifting(worlds,case):
    for e in worlds:
        e.reset();place(e,.15,reset=True)
        if case=='teleport':place(e,.27);tick(e)
        if case=='lift':place(e,.15,z=.03);tick(e)
        if case=='around':
            place(e,.15,y=.09,reset=True)
            for x in np.arange(.15,.271,.002):place(e,x,y=.09);tick(e)
    assert summary(worlds[0])==summary(worlds[1])
    assert summary(worlds[0])[0]!=0

def test_cuda_stationary_hold_matches_cpu(worlds):
    for e in worlds:
        e.reset();place(e,.27,yaw=math.pi/2,reset=True);e.rules.mask[:]=65535;e.rules.passed[:]=1
    for _ in range(500):
        for e in worlds:tick(e)
    assert summary(worlds[0])==summary(worlds[1])==(0,65535,1,500)
    for e in worlds:e.d.qvel[0,6]=.0051;tick(e)
    assert summary(worlds[0])==summary(worlds[1])==(0,65535,1,0)

@pytest.mark.parametrize('a,b,depth,expected',[('fixed_jaw_box3','table',-.000051,COLLISION),('fixed_jaw_box3','gate_north',-.000049,0),('block_0','table',-.0011,PENETRATION),('block_0','fixed_jaw_box3',-.0005,0)])
def test_actual_cuda_contact_kernel(worlds,a,b,depth,expected):
    cpu,gpu=worlds
    for e in worlds:e.reset()
    cpu.rules.inspect_contact(cpu.model.geom(a).id,cpu.model.geom(b).id,depth)
    d=gpu.sim.wp_data
    wp.to_torch(d.nacon)[:]=1;wp.to_torch(d.contact.worldid)[0]=0
    wp.to_torch(d.contact.geom)[0]=torch.tensor([gpu.model.geom(a).id,gpu.model.geom(b).id],device=gpu.device)
    wp.to_torch(d.contact.dist)[0]=depth
    gpu.rules.launch(advance=False)
    assert int(cpu.rules.flags[0])==int(gpu.rules.flags[0])==expected

@pytest.mark.parametrize('other,depth,touch',[('fixed_jaw_box3',-.0001,1),('moving_jaw_mesh_1',-.0001,1),('fixed_jaw_box3',.0001,0),('table',-.0001,0)])
def test_stage_one_requires_actual_jaw_contact(worlds,other,depth,touch):
    cpu,gpu=worlds
    for e in worlds:e.reset()
    cpu.rules.inspect_contact(cpu.model.geom('block_0').id,cpu.model.geom(other).id,depth)
    d=gpu.sim.wp_data
    wp.to_torch(d.nacon)[:]=1;wp.to_torch(d.contact.worldid)[0]=0
    wp.to_torch(d.contact.geom)[0]=torch.tensor([gpu.model.geom('block_0').id,gpu.model.geom(other).id],device=gpu.device)
    wp.to_torch(d.contact.dist)[0]=depth;gpu.rules.launch(advance=False)
    assert int(cpu.rules.touch[0])==int(gpu.rules.touch[0])==touch

