import numpy as np
from masongraph_envs.masongraph import MasonGraphCover as Env
from simulator.cont3Dsimulator.batchsim import activeUI
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
    env = Env(n_batch=n_batch,max_area=[0.11,0.11],scales=[1],autoreset=False,max_blocks=56,friction_coef=0.5,n_robots=1,max_dist = 40,tol=1e-3,max_action_substeps=100,physics_on=False)
    env.reset(seed=seed)
    activeUI(env.sim)
    direction = np.zeros((n_batch,3))
    direction[:,2]=-1
    offset = np.zeros((n_batch,3))
    offset[:,0] =1
    offset[:,1]=0.5
    action1 = {"robot_id":np.zeros(n_batch,dtype=int),
               "block_type":np.zeros(n_batch,dtype=int),
               #"block_face":np.zeros(n_batch,dtype=int),
               "support_block":np.zeros(n_batch,dtype=int),
               #"support_face":np.ones(n_batch,dtype=int),
               "center_offset":offset,
               'rotation_offset':np.zeros((n_batch,6)),
               "direction":direction,
              }
    env.step(action1)
    #env.sim.start_dist = 2
    noise = 0.15
    for i in range(50):
        env.render()
        #ps.show()
        if i == 16:
            pass
        print(f"Step {i}")
        #rot =rnd.normal(np.zeros((n_batch,6)),1)
        rot = np.tile(np.array([1,0,0,0,1,0],dtype=float),(n_batch,1))
        rot += noise*i*np.array([[0,1,0,0,0,0]])
        rot+= noise*rnd.normal(np.zeros((n_batch,6)),1)
        action2 = {"robot_id":np.full(n_batch,i%env.n_robots,dtype=int),
                "block_type":np.zeros(n_batch,dtype=int),
                #"block_face":np.zeros(n_batch,dtype=int),
                "support_block":np.ones(n_batch,dtype=int)*env.sim.n_block_t[0]-1,
                #"support_face":np.ones(n_batch,dtype=int)*2,
                "center_offset":np.zeros((n_batch,3)),#+np.array([[0,1,0]])*i/100,
                'rotation_offset':rot,
                "direction":direction,
                }
        state,reward,terminated,truncated,info=env.step(action2)
        #if terminated:
            #print("Terminated")
            #break
        #print(env.sim.physics_solver[0].invM.shape)

    env.sim.render()
    ps.show()