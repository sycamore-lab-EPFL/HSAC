import numpy as np
from masongraph_envs.masongraph import MasonGraphCover as Env
from simulator.cont3Dsimulator.batchsim import activeUI
import trimesh
import os
import polyscope.imgui as psim

import polyscope as ps
if __name__ == "__main__":
    seed = 0
    rnd = np.random.default_rng(seed)
    print("Start test")
    n_batch = 20
    env = Env(n_batch=n_batch,scales=[1,2],autoreset=False,render_mode='rgb_array')
    env.reset()
    direction = np.zeros((n_batch,3))
    direction[:,2]=-1
    offset = np.zeros((n_batch,3))
    offset[:,0] =100
    offset[:,1]=50
    action1 = {"robot_id":np.zeros(n_batch,dtype=int),
               "block_type":np.zeros(n_batch,dtype=int),
               #"block_face":np.zeros(n_batch,dtype=int),
               "support_block":np.zeros(n_batch,dtype=int),
               #"support_face":np.ones(n_batch,dtype=int),
               "center_offset":offset,
               'rotation_offset':np.zeros((n_batch,6)),
               "direction":direction,
              }
    s,r,t,_,info = env.step(action1)
    direction = rnd.normal(np.zeros((n_batch,3)),1)
    rot =rnd.normal(np.zeros((n_batch,6)),1)
    action2 = {"robot_id":np.zeros(n_batch,dtype=int),
               "block_type":np.ones(n_batch,dtype=int),
               #"block_face":np.zeros(n_batch,dtype=int),
               "support_block":np.ones(n_batch,dtype=int)*4,
               #"support_face":np.ones(n_batch,dtype=int)*2,
               "center_offset":np.zeros((n_batch,3)),
               'rotation_offset':rot,
               "direction":direction,
              }
    env.step(action2)
    direction = rnd.normal(np.zeros((n_batch,3)),1)
    rot =rnd.normal(np.zeros((n_batch,6)),1)
    action = {"robot_id":np.zeros(n_batch,dtype=int),
               "block_type":np.ones(n_batch,dtype=int),
               #"block_face":np.zeros(n_batch,dtype=int),
               "support_block":np.ones(n_batch,dtype=int)*4,
               #"support_face":np.ones(n_batch,dtype=int)*2,
               "center_offset":np.zeros((n_batch,3)),
               'rotation_offset':rot,
               "direction":direction,
              }
    env.step(action)
    #BATCH 0,6 are an example of the limitation of the env
    #BATCH 12,15 and 16 DOES NOT WORK
    activeUI(env.sim)
    env.sim.render()
    ps.show()