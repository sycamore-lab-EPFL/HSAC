import copy
import os
from typing import Literal
import gymnasium as gym
from gymnasium import spaces

try:
    from masongraph_envs.tools.HER import hindsightexperiencereplay
except ImportError:
    hindsightexperiencereplay = None
import shapely
from simulator.cont3Dsimulator.batchsim import BatchSimulator, activeUI
import polyscope as ps
import numpy as np
try:
    import hex_envs.tools.render_pygame as render  # optional legacy pygame renderer
except ImportError:
    render = None
import matplotlib.pyplot as plt
from simulator.cont3Dsimulator.state_rep.bframegraph import BFrameGraphConstructor
from simulator.cont3Dsimulator.utils import geometry
from . import StraightMasonGraphCover
import trimesh
from .tools import converter
import warp as wp
DATAPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator", "cont3Dsimulator", "data")
NOBLOCK = -1
device = 'cuda'
class AtomicStraightMasonGraphCover(StraightMasonGraphCover):
    metadata = {"render_modes": ["human", "agent_input","rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 n_batch = 1,
                 max_blocks = 50,
                 min_area = [0.1,0.1],
                 max_offset = [10,10],
                 block_list: list[trimesh.Trimesh]|Literal['brick','kapla']='brick',
                 scales: list[float] = [1],
                 friction_coef:float=0.5,
                 glue_strength:float=0.0,
                 discount_factor = 0.1,
                 ground_gen: Literal['around','2sides','multibricks']='multibricks',
                 ground_width = 2.5,
                 autoreset:bool=True,
                 physics_on: bool = True,
                 n_frames_per_action: int = 4,
                 max_action_substeps: int = 50,
                 max_dist = 30,
                 use_her: bool = False,
                 touch_target: bool = True,
                 tol = 1e-5,
                 rep = 'noigraph',
                 reset_on_fall = True,
                 terminal_on_invalid = True,
                 additional_contact_reward = 0
                 ):
        self.super().__init__(render_mode,n_batch,max_blocks,min_area,max_offset,block_list,scales, friction_coef,glue_strength,discount_factor,ground_gen,
                              ground_width,autoreset,physics_on,n_frames_per_action,max_action_substeps,max_dist,rep,tol,additional_contact_reward)
        self.max_offset = np.array([max_offset[0]/2, max_offset[1]/2])
    def reset(self, seed=None, options=None,reset_batches=None,debug=False):
        # We need the following line to seed self.np_random
        #super().reset(seed=seed)
        #checked_struct = self.sim.new_struct
        #self.setup.checked_struct 
        if reset_batches is None:
            reset_batches = np.ones(self.n_batch,dtype=bool)
        self.sim.reset(reset_batches)
        #self.sim.checked_struct = checked_struct
        options = options or {'ground_gen':self.ground_gen,
                              'min_area':self.min_area,
                              "max_area":self.max_area,
                              }
        batch_to_reset = np.nonzero(reset_batches)[0]
        if len(batch_to_reset) == 0:
            return self._get_observation(), {}
        if options['ground_gen']=='multibricks':
            #width = self.np_random.uniform(options['min_area'][1],options['max_area'][1],(batch_to_reset.shape[0],))
            #depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0],(batch_to_reset.shape[0],))
            #with wp.ScopedTimer("reset",color='black',use_nvtx=True):
            #TODO: make the reset adaptive so that not all batches are the same
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
                
            
            self.sim.add_ground([self.block_type[0] for _ in batch_to_reset],pos = np.array([[0,0,0] for _ in batch_to_reset]), env_ids=batch_to_reset)
            
        elif options['ground_gen']=='manual':
            for new_ground in options['ground_blocks']:
                self.sim.add_ground([new_ground]*batch_to_reset.shape[0], env_ids=batch_to_reset)
            depth = [options['cover_area'][0]]*batch_to_reset.shape[0]
            width = [options['cover_area'][1]]*batch_to_reset.shape[0]
        else:
            raise NotImplementedError
        
        if options['ground_gen']=='manual':
            raise NotImplementedError
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        else:
            w = self.block_type[0].vertices[:,1].max()-self.block_type[0].vertices[:,1].min() - self.sim.tol*2
            d = self.block_type[0].vertices[:,0].max()-self.block_type[0].vertices[:,0].min() - self.sim.tol*2
            areas = [shapely.Polygon([[ -d/2, -w/2],
                                [  d/2, -w/2],
                                [  d/2,  w/2],
                                [ -d/2,  w/2]]) for i in range(batch_to_reset.shape[0])]
            offsets = [self.np_random.uniform(-self.max_offset,self.max_offset) for i in range(batch_to_reset.shape[0])]
            angles = [self.np_random.uniform(0,360) for i in range(batch_to_reset.shape[0])]
            areas = [shapely.affinity.rotate(areas[i], angle=angles[i], origin='centroid') for i in range(batch_to_reset.shape[0])]
            areas = [shapely.affinity.translate]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)

        if options.get('obstacle_gen') is not None:
            raise NotImplementedError

        observation = self._get_observation()
        info = {'center':offsets,'rot':angles}

        if self.render_mode is not None and self.sim.batchi in batch_to_reset:
            info['image'] = self.render()
    
        return observation, info
    def step(self,action,render=False,normalized=True):
        info={}
        if isinstance(action,tuple):
            action = converter.tuple2dict_flat(action,1,len(self.block_type),self.max_blocks,vertical=True,normalized=normalized,action_boundary=self.action_metadata()['boundary'])
            info['rotationvec'] =  geometry.rotmat_from_2D(action['rotation_offset'])[:,:2].reshape(-1,2)
        to_test = self.sim.prepare_action()
        to_truncate = ~to_test.copy()
        valid = np.ones(self.n_batch,dtype=bool)
        env_ids = np.nonzero(to_test)[0]
        rewards = np.zeros(self.n_batch,dtype=float)
        if env_ids.shape[0] > 0:
            valid_a,rewards[env_ids],info['action'] = self.sim.put_block_rel_noface(self.block_type[action['block_type']],
                                                                                    action['support_block'],
                                                                                    action['center_offset'],
                                                                                    action['rotation_offset'],
                                                                                    action['direction'],
                                                                                    env_ids=env_ids,touch_target=self.touch_target,pivot=False,hold=False,slide=False,local_ref=self.local_ref)
            valid[env_ids]=valid_a
            env_ids = env_ids[valid_a]
        #Dont hold the block and check stability
        info['falling'] = np.zeros((self.n_batch,),dtype=bool)
        if self.sim.physics_on:
            falling = self.sim.simulate_physics(env_ids)
            info['falling'][env_ids] = falling.copy()
            self.sim.remove_last_block(env_ids[falling])
            #[self.sim.physics_model[b].fix_part(self.sim.n_block_t[b]-1) for b in np.nonzero(falling)[0]]
            valid[env_ids] = ~falling
            rewards[env_ids[falling]] = -1
        if (rewards <-0.1).any():
            pass
        self.sim.pass_time(np.nonzero(~valid | to_truncate)[0])
        if render:
            info['image'] = self.render()
        info['success'] = self.sim.check_success()
        if info['success'].any():
            print("Success in batch",np.nonzero(info['success'])[0])
        info['areas'] = [self.sim.to_cover[i].area for i in range(self.n_batch)]
        terminated = (self.sim.n_block_t >= self.max_blocks-1) | info['success'] | (info['falling'] * (self.reset_on_fall)) | (~valid * (self.terminal_on_invalid))
        if self.autoreset:
            observation,_ = self.reset(reset_batches=terminated)
        else:
            observation,_ = self.reset(reset_batches=to_truncate)
        #print(valid)
        print(rewards.mean())
        
        return observation, rewards, terminated, to_truncate, info
    def init_render(self):
        self.sim.reset(np.ones(self.n_batch,dtype=bool))
        activeUI(self.sim)
        self.sim.render()
    def render(self,cam_pos=[10,0,10]):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self.sim.render()
            ps.frame_tick()
            return None
        if self.render_mode == "rgb_array":
            frames = []
            ps.look_at(cam_pos, [0,0,0], fly_to=False)
            self.sim.tstep =  self.sim.n_actions[self.sim.batchi]- self.sim.n_block_init[self.sim.batchi]-1
            nsteps = self.sim.n_actions_steps[self.sim.batchi,self.sim.n_actions[self.sim.batchi]-1]
            if nsteps >0:
                if self.sim.action_t[self.sim.batchi,self.sim.n_actions[self.sim.batchi]-1,nsteps-1]['valid'] and not self.sim.falling[self.sim.batchi,self.sim.n_actions[self.sim.batchi]]:
                    nf = self.n_frames_per_action
                else:
                    nf = self.n_frames_per_action *5
                for i in np.linspace(0,1,nf):
                    self.sim.t = i
                    self.sim.render()
                    frames.append(ps.screenshot_to_buffer())
                return np.stack(frames)
            else:
                return np.zeros((0,ps.screenshot_to_buffer().shape[0], ps.screenshot_to_buffer().shape[1],4),dtype=np.uint8)
    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.nan
    def HER(self,actions,mode=None):
        raise NotImplementedError
        return hindsightexperiencereplay(self,actions,mode)