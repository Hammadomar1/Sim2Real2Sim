import os,json,ctypes.util
import mujoco
import numpy as np
import imageio.v2 as imageio
from so101_m1.env import PushTurnParkEnv,EnvConfig
from so101_m1.render import camera
from so101_m1.scene import ARTIFACTS
e=PushTurnParkEnv(EnvConfig(num_envs=1,backend="native",stage=5,training=False))
r=mujoco.Renderer(e.model,height=360,width=640)
d=e.snapshot(); r.update_scene(d,camera=camera()); img=r.render()
print('GL',os.environ.get('MUJOCO_GL'),'ngeom',r.scene.ngeom,'extent',e.model.stat.extent,'tip',e.state()[2],'pixels',img.min(),img.max(),img.mean(),'osmesa',ctypes.util.find_library('OSMesa'),flush=True)
imageio.imwrite(ARTIFACTS/('probe_'+os.environ.get('MUJOCO_GL','default')+'.png'),img)
r.close()
