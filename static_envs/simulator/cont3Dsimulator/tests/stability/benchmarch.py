import masongraph_envs
import numpy as np
import copy
import torch
import torch_geometric
import time
def dome_bench(env):
    from simulator.cont3Dsimulator.utils import geometry
    env.touch_target = False
    state, info = env.reset()
    s = env.max_area[0]-0.6
    radius = s/np.sqrt(2)
    direction = np.zeros((env.n_batch,3))
    direction[:,2]=-1
    offset = np.zeros((env.n_batch,3))
    offset[:,0] =1
    offset[:,1]=0.5
    done = False
    # might need to change the 21 magic number
    next_sup = np.ones(env.n_batch,dtype=int)*21
    next_offset = np.array([[-1.5,0,0]]*env.n_batch)
    firstblockid = env.sim.n_block_t[0]
    start_angle = 0
    trajs = []
    rewards = []
    n_blocks_layer = []
    support_block = []
    while not done:
        n_blocks = max(np.floor(2*np.pi*radius / 3.2).astype(int),1)
        ori = np.linspace(start_angle,2*np.pi+start_angle,n_blocks,endpoint=False)
        n_blocks_layer.append(n_blocks)
        for i in range(n_blocks):
            print(f"Step {i}")
            action_abs = {"robot_id":np.full(env.n_batch,i%env.n_robots,dtype=int),
                    "block_type":np.zeros(env.n_batch,dtype=int),
                    "support_block": next_sup,
                    "center_offset":next_offset,
                    'rotation_offset':np.tile(np.array([[np.cos(ori[i]),np.sin(ori[i])]]),(env.n_batch,1)),
                    "direction":direction,
                    }
            if env.local_ref:
                action = copy.deepcopy(action_abs)
                nontouch_ref = env.sim.referencial[0,action['support_block'][0]]
                action['rotation_offset'] = (nontouch_ref.T@geometry.rotmat_from_2D(action['rotation_offset']))[:,:2,0]
                action['center_offset'][:,:2] = np.tile((nontouch_ref.T@action['center_offset'][0])[None,:2],(env.n_batch,1) )
            else:
                action = action_abs
            nstate,reward,terminated,truncated,info=env.step(action)
            rewards.append(reward)
            if not terminated.all():
                actual_t = env.sim.action_t[0,env.sim.n_actions[0]-1,1]['contacts'][0]['partIDA']
                ref = env.sim.referencial[0,actual_t]
                actual_offset_pos = (ref.T@(env.sim.assembly_sequence[0,env.sim.n_block_t[0]-1].center_mass-env.sim.assembly_sequence[0,actual_t].center_mass))[None,:2]
                actual_offset_rot = (ref.T@geometry.rotmat_from_2D(action_abs['rotation_offset'])[0])[None,:2,0]
            else:
                actual_t = action['support_block'][0]
                actual_offset_pos = action['center_offset'][:,:2]
                actual_offset_rot = action['rotation_offset']
            support_block.append(actual_t)
            
            if i < n_blocks -1:
                next_sup = np.ones(env.n_batch,dtype=int)*env.sim.n_block_t[0]-1
                next_offset = np.tile(np.array([[-radius*(np.sin(ori[i+1])-np.sin(ori[i])),radius*(np.cos(ori[i+1])-np.cos(ori[i])),0]]),(env.n_batch,1))
            else:
                starting_block = 0#np.random.randint(n_blocks)
                next_sup = (np.ones(env.n_batch,dtype=int)*firstblockid)+starting_block
                old_radius = radius
                radius = radius - 0.3
                if radius < 1:
                    start_angle = ori[0]
                    radius = 0
                else:
                    start_angle = (ori[(starting_block+1)%n_blocks]+ori[starting_block])/2
                firstblockid = env.sim.n_block_t[0]
                next_offset = np.tile(np.array([[-radius*np.sin(start_angle)+old_radius*np.sin(ori[starting_block]),
                                        radius*np.cos(start_angle)-old_radius*np.cos(ori[starting_block]),0]]),(env.n_batch,1))
            done = terminated.any() or info['falling'].any()
if __name__ == "__main__":
    from simulator.cont3Dsimulator.batchsim import BatchSimulator, activeUI
    import polyscope as ps
    env = masongraph_envs.StraightMasonGraphCover(n_batch=16,max_blocks=80,use_batchsolver=True,physics_on=True,tol=1e-2,ground_gen='around',autoreset=False)
    #activeUI(env.sim)
    t0 = time.perf_counter()
    dome_bench(env)
    t1 = time.perf_counter()
    print(f"Time taken: {t1-t0} seconds")
    #ps.show()