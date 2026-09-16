import numpy as np
from masongraph_envs import ColumnMasonGraph as Env
from simulator.cont3Dsimulator.batchsim import activeUI
from simulator.cont3Dsimulator.objectives.cover import corners_encode
import trimesh
import os
import polyscope.imgui as psim
import warp as wp
wp.set_device("cuda")
import polyscope as ps
if __name__ == "__main__":
    seed = 0
    rnd = np.random.default_rng(seed)
    print("Start test")
    n_batch = 1
    env = Env(n_batch=n_batch,min_dist=10,max_dist=10.1,scales=[np.pi/10],autoreset=False,
              max_blocks=30,friction_coef=0.5,start_dist = 60,tol=1e-1 ,max_action_substeps=100,reset_on_fall=False,continuous_reward=True,rep='bframegraph')
    env.reset(seed=seed)
    activeUI(env.sim)
    offset = np.zeros((n_batch,3))
    #env.sim.start_dist = 2
    noise = 0.05
    x = 1
    for i in range(20):
        env.render()
        #ps.show()
        print(f"Step {i}")
        rot= noise*rnd.normal(np.zeros((n_batch)),1)
        pos = noise*rnd.normal(np.zeros((n_batch,3)),1)
        action2 = { "block_type":np.zeros(n_batch,dtype=int),
                "support_block":np.ones(n_batch,dtype=int)*env.sim.n_block_t[0]-1,
                "center_offset":pos+[0,1.2,0],
                'rotation_angle':rot,
                }
        state,reward,terminated,truncated,info=env.step(action2)
        if info['falling'].any():
            x = -x
        if (terminated|truncated).any():
            print("Terminated")
            break
        #print(env.sim.physics_solver[0].invM.shape)
    ps.show()
