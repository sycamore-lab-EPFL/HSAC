import numpy as np
from masongraph_envs.straight_masongraph import StraightMasonGraphCover as Env
from simulator.cont3Dsimulator.batchsim import activeUI
from simulator.cont3Dsimulator.objectives.cover import corners_encode
import trimesh
import os
import polyscope.imgui as psim
import warp as wp
wp.set_device("cpu")
import polyscope as ps
if __name__ == "__main__":
    seed = 0
    rnd = np.random.default_rng(seed)
    print("Start test")
    n_batch = 1
    env = Env(n_batch=n_batch,min_area=[3,5],max_area=[3.1,5.1],scales=[1],autoreset=False,limit_height=50,
              max_blocks=30,friction_coef=0.5,max_dist = 40,tol=1e-2,max_action_substeps=100,reset_on_fall=False,use_batchsolver=True,
              continuous_reward=False,rep='naive_jagged')
    env.reset(seed=seed)
    activeUI(env.sim)
    direction = np.zeros((n_batch,3))
    direction[:,2]=-1
    offset = np.zeros((n_batch,3))
    offset[:,0] =1
    offset[:,1]=0.5
    #env.sim.start_dist = 2
    noise = 0.15
    x = 1
    for i in range(20):
        env.render()
        #ps.show()
        print(f"Step {i}")
        #rot= noise*rnd.normal(np.zeros((n_batch,2)),1)
        rot = np.array([[1,0.1]]*n_batch)
        action2 = {"robot_id":np.full(n_batch,i%env.n_robots,dtype=int),
                "block_type":np.zeros(n_batch,dtype=int),
                "support_block":np.ones(n_batch,dtype=int)*env.sim.n_block_t[0]-1,
                "center_offset":np.tile([0.1,0,0],(n_batch,1)),
                'rotation_offset':rot,
                "direction":direction,
                }
        state,reward,terminated,truncated,info=env.step(action2)
        if info['falling'].any():
            x = -x
        if (terminated|truncated).all():
            print("Terminated")
            break
        #print(env.sim.physics_solver[0].invM.shape)

    env.sim.render()
    ps.show()
