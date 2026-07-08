import numpy as np
from hsac.buffer import ReplayBufferGNN


import torch
import torch_geometric
import wandb
import copy
from .helper import floatType, intType, device
from torch.cuda import Stream
import torch.nn.functional as F
class PrefilledReplayBufferGNN(ReplayBufferGNN):
    def __init__(self, batch_size,her,
                 storing_device='cpu',
                 max_loss=100,
                 max_blocks=30,
                 n_actions_per_block=1,
                 max_size=1000000,
                 prioritized = False,
                 max_td_error = 1e6,
                 n_samples = None,
                 replace = True,
                 alpha=0.6,
                 anneal_steps=20000,
                 rep='noigraph',
                 prefill_weight=10,
                 gamma=0.99):
        super().__init__(batch_size,her,storing_device,max_loss,max_blocks,n_actions_per_block,max_size,
                        prioritized,max_td_error,n_samples,replace,alpha,anneal_steps)
        self.prefill_weight = prefill_weight
        self.prefilled = ReplayBufferGNN(batch_size,her,storing_device,max_loss,max_blocks,n_actions_per_block,max_size,
                        prioritized,max_td_error,n_samples,replace,alpha,anneal_steps)
        import masongraph_envs # import here to initialize static variables with the config first
        prefill_env = masongraph_envs.StraightMasonGraphCover(n_batch=1,
                                                render_mode=None,
                                                scales=[1],
                                                max_area=[5.000001,5.000001],
                                                ground_width=2.5,
                                                min_area=[5,5],
                                                max_blocks=max_blocks,
                                                friction_coef=0.5,
                                                tol=1e-2,
                                                rep=rep,
                                                reset_on_fall=False,
                                                additional_contact_reward=0.0,
                                                touch_target=False,
                                                autoreset=False,
                                                terminal_on_invalid=True)
        
        self.dome_fill(prefill_env,gamma=gamma,augment=wandb.config.prefilled_augment)
    @property
    def all_states(self):
        return super().all_states + self.prefilled.all_states*self.prefill_weight
    @property
    def next_states(self):
        return super().next_states + self.prefilled.next_states*self.prefill_weight
    def dome_fill(self,env,gamma=0,augment=10):
        from simulator.cont3Dsimulator.utils import geometry
        env.touch_target = False
        state, info = env.reset()
        s = env.max_area[0]
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
                        'rotation_offset':np.array([[np.cos(ori[i]),np.sin(ori[i])]]),
                        "direction":direction,
                        }
                if env.local_ref:
                    action = copy.deepcopy(action_abs)
                    nontouch_ref = env.sim.referencial[0,action['support_block'][0]]
                    action['rotation_offset'] = (nontouch_ref.T@geometry.rotmat_from_2D(action['rotation_offset']))[:,:2,0]
                    action['center_offset'][:,:2] = (nontouch_ref.T@action['center_offset'][0])[None,:2]
                else:
                    action = action_abs
                nstate,reward,terminated,truncated,info=env.step(action)
                rewards.append(reward)
                if not terminated:
                    actual_t = env.sim.action_t[0,env.sim.n_actions[0]-1,1]['contacts'][0]['partIDA']
                    ref = env.sim.referencial[0,actual_t]
                    actual_offset_pos = (ref.T@(env.sim.assembly_sequence[0,env.sim.n_block_t[0]-1].center_mass-env.sim.assembly_sequence[0,actual_t].center_mass))[None,:2]
                    actual_offset_rot = (ref.T@geometry.rotmat_from_2D(action_abs['rotation_offset'])[0])[None,:2,0]
                else:
                    actual_t = action['support_block'][0]
                    actual_offset_pos = action['center_offset'][:,:2]
                    actual_offset_rot = action['rotation_offset']
                support_block.append(actual_t)
                actiontup = (torch.tensor(actual_t*env.n_robots + action['robot_id'],device=self.storing_device,dtype=intType),
                            torch.cat([torch.tensor((actual_offset_pos-env.action_mean()[:2])/env.action_range()[:2],device=self.storing_device,dtype=floatType),
                                       torch.tensor((actual_offset_rot-env.action_mean()[2:])/env.action_range()[2:],device=self.storing_device,dtype=floatType)],dim=-1))
                self.prefilled.add_inputs(state.to(self.storing_device),actiontup,torch.zeros((env.n_batch,1),dtype=floatType,device=self.storing_device))
                state = nstate
                self.prefilled.add_results(reward,terminated,truncated,action['rotation_offset'])
                if i < n_blocks -1:
                    next_sup = np.ones(env.n_batch,dtype=int)*env.sim.n_block_t[0]-1
                    next_offset = np.array([[-radius*(np.sin(ori[i+1])-np.sin(ori[i])),radius*(np.cos(ori[i+1])-np.cos(ori[i])),0]])
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
                    next_offset = np.array([[-radius*np.sin(start_angle)+old_radius*np.sin(ori[starting_block]),
                                            radius*np.cos(start_angle)-old_radius*np.cos(ori[starting_block]),0]])
                done = terminated.any() or info['falling'].any()
        self.prefilled.add_inputs(state,actiontup,torch.zeros((env.n_batch,1),dtype=floatType,device=self.storing_device))
        self.prefilled.add_results(reward,torch.zeros(1,device=self.storing_device,dtype=bool),torch.ones(1,device=self.storing_device,dtype=bool),action['rotation_offset'])
        #self.prefilled.update_info(np.arange(len(rewards)-1),'value',torch.tensor(v[2:],dtype=floatType))
        self.prefilled.add_traj()
        len_init = len(self.prefilled.all_states)
        for i in range(augment):
            env.touch_target = True
            state, info = env.reset()
            done = False
            idx = 0
            offsets = np.array([sum(n_blocks_layer[:k]) for k in range(len(n_blocks_layer))])
            idx_state = np.concatenate([offsets[k]+np.random.permutation(n_blocks_layer[k]) for k in range(len(n_blocks_layer))])
            idx_ori = np.argsort(idx_state)
            old_block = np.arange(len(idx_state))

            idx_block = idx_ori
            
            old_support_block_oldidx = np.array(support_block)
            old_support_block_newidx = old_support_block_oldidx[idx_state]
            idx_support = np.concatenate([old_support_block_newidx[:n_blocks_layer[0]],
                                         idx_ori[old_support_block_newidx[n_blocks_layer[0]:]-24]+24])
            
            print(f"Augmentation {i}")
            rewards = []
            while not done:
                if idx >= len_init:
                    self.prefilled.add_traj()
                    break
                actiond = torch.tensor(idx_support[idx],device=device,dtype=int).unsqueeze(0)
                #if idx >= n_blocks_layer[0]:
                #    actionc = self.prefilled.all_states[idx]['action_continuous'].x.to(device)
                #else:
                actionc = self.prefilled.all_states[idx_state[idx]]['action_continuous'].x.to(self.storing_device)
                actionc += torch.normal(0.0,wandb.config.exploration_noise_init/10,actionc.shape,device=self.storing_device)
                self.prefilled.add_inputs(state,(actiond,actionc),torch.zeros((env.n_batch,1),dtype=floatType,device=self.storing_device))
                state,reward,terminated,truncated,info=env.step((actiond,
                                                                actionc))
                self.prefilled.add_results(reward,terminated,truncated)
                rewards.append(reward)
                idx += 1
                done = terminated.any() or truncated.any()
        self.prefilled.add_traj()
    def atomic_fill(self,env):
        pass