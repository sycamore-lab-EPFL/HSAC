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
import matplotlib.pyplot as plt
from simulator.cont3Dsimulator.utils import geometry
import trimesh
from .tools import converter
DATAPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator", "cont3Dsimulator", "data")
NOBLOCK = -1

class MasonGraphCover():
    metadata = {"render_modes": ["human", "agent_input","rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 n_batch = 1,
                 n_robots=2,
                 max_blocks = 50,
                 min_area = [0.1,0.1],
                 max_area = [10,10],
                 block_list: list[trimesh.Trimesh]|Literal['brick']='brick',
                 scales: list[float] = [1],
                 friction_coef:float=0.5,
                 discount_factor = 0.1,
                 ground_gen: Literal['around','2sides']='around',
                 ground_width = 3,
                 autoreset:bool=True,
                 touch_target:bool=True,
                 vertical=False,
                 physics_on:bool=False,
                 n_frames_per_action: int = 4,
                 max_action_substeps: int = 50,
                 max_dist = 40,
                 use_her: bool = False,
                 tol = 1e-5
                 ):
        self.use_her = use_her
        self.np_random = np.random.default_rng(seed=None)
        self.n_batch = n_batch
        self.vertical = vertical
        self.touch_target = touch_target
        self.gamma = 1-discount_factor
        self.friction_coef = friction_coef 
        self.n_robots = n_robots 
        self.is_holding = np.ones((n_batch,self.n_robots),dtype=int)*NOBLOCK
        self.ground_width = ground_width
        if block_list == 'brick':
            obj = DATAPATH+'/standard_brick/base_brick.obj'
            trimesh_obj = trimesh.load(obj, force = 'mesh')
            centroid = trimesh_obj.centroid
            face_centers = trimesh_obj.triangles.mean(axis=1)
            outward = face_centers - centroid
            dot = np.sum(trimesh_obj.face_normals * outward, axis=1)
            assert np.all(dot > 0), "Some normals point inward!"
            assert trimesh_obj.is_watertight
            self.block_type = np.array([trimesh_obj.copy().apply_scale(1/s/100) for s in scales])
        elif type(block_list) ==str:
            assert False, f"unkwown preset {block_list}"
        else:
            self.block_type = np.array(block_list)
        self.max_blocks = max_blocks
        self.max_area = max_area
        self.min_area = min_area
        self.ground_gen = ground_gen
        self.sim = BatchSimulator(n_batch=n_batch,
                                  n_block_max=max_blocks,
                                  max_action_steps=max_action_substeps,
                                  max_dist = max_dist,
                                  physics_on=physics_on,
                                  mu=self.friction_coef,
                                  tol=tol)
        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Render mode {render_mode} is unknown "
        self.render_mode = render_mode
        if self.render_mode is not None:
            activeUI(self.sim)
            self.render_size = ps.screenshot_to_buffer().shape[:2]
            self.n_frames_per_action = n_frames_per_action
        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.autoreset = autoreset
    def _get_observation(self):
        #note that the robots are ordered from next to act to just acted
        state = self.sim.current_graph()
         #        "turn":self.sim.turn}
        return state
    def state_metadata(self):
        return self.sim.graph.metadata
    def action_metadata(self):
        if self.vertical:
            return {'discrete':[self.n_robots,
                                len(self.block_type)],
                    'continuous':9,
                    'boundary':np.array([[-1,1],
                                         [-1,1],
                                         [-1e-2,1e-2],
                                         [-0.2,0.2],[-0.2,0.2],
                                         [-0.2,0.2],[-0.2,0.2],
                                         [-0.2,0.2],[-0.2,0.2]])}
        else:
            return {'discrete':[self.n_robots,
                                len(self.block_type)],
                    'continuous':12,
                    'boundary':np.array([[-500,500],
                                         [-500,500],
                                         [-500,500],
                                         [-1,1],[-1,1],[-1,1],
                                         [-1,1],[-1,1],
                                         [-1,1],[-1,1],
                                         [-1,1],[-1,1]])}
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
        width = self.np_random.uniform(options['min_area'][0],options['max_area'][0],(batch_to_reset.shape[0],))
        depth = self.np_random.uniform(options['min_area'][1],options['max_area'][1],(batch_to_reset.shape[0],))
        
        
        
        if options['ground_gen']=='around':
            ground_1 = [trimesh.creation.box(bounds=[[-width[i]/2-self.ground_width, -depth[i]/2+self.sim.tol/4,-1],
                                                    [-width[i]/2-self.sim.tol/4, depth[i]/2+self.ground_width,-0.05]]) for i in range(batch_to_reset.shape[0])]
            
            ground_2 = [trimesh.creation.box(bounds=[[-width[i]/2+self.sim.tol/4, depth[i]/2+self.sim.tol/4,-1],
                                                    [width[i]/2+self.ground_width, depth[i]/2+self.ground_width,-0.05]]) for i in range(batch_to_reset.shape[0])]
            
            ground_3 = [trimesh.creation.box(bounds=[[width[i]/2+self.sim.tol/4, -depth[i]/2-self.ground_width,-1],
                                                    [width[i]/2+self.ground_width, depth[i]/2-self.sim.tol/4,-0.05]]) for i in range(batch_to_reset.shape[0])]
            ground_4 = [trimesh.creation.box(bounds=[[-width[i]/2-self.ground_width, -depth[i]/2-self.ground_width,-1],
                                                    [width[i]/2-self.sim.tol/4, -depth[i]/2-self.sim.tol/4,-0.05]]) for i in range(batch_to_reset.shape[0])]
            r = self.sim.add_ground(ground_1,env_ids=batch_to_reset)
            r =self.sim.add_ground(ground_2,env_ids=batch_to_reset)
            r =self.sim.add_ground(ground_3,env_ids=batch_to_reset)
            r =self.sim.add_ground(ground_4,env_ids=batch_to_reset)

        elif options['ground_gen']=='manual_grounds':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        if debug:
            areas = [shapely.Polygon([[-width[i]/2-self.ground_width, -depth[i]/2],
                                    [-width[i]/2, -depth[i]/2],
                                    [-width[i]/2, depth[i]/2+self.ground_width],
                                    [-width[i]/2-self.ground_width, depth[i]/2+self.ground_width]]) for i in range(batch_to_reset.shape[0]//2)]
            areas += [shapely.Polygon([[-width[i]/2, depth[i]/2],
                                    [width[i]/2+self.ground_width, depth[i]/2],
                                    [width[i]/2+self.ground_width, depth[i]/2+self.ground_width],
                                    [-width[i]/2, depth[i]/2+self.ground_width]]) for i in range(batch_to_reset.shape[0]//2,batch_to_reset.shape[0])]
        else:
            areas = [shapely.Polygon([[ -width[i]/2, -depth[i]/2],
                                [  width[i]/2, -depth[i]/2],
                                [  width[i]/2,  depth[i]/2],
                                [ -width[i]/2,  depth[i]/2]]) for i in range(batch_to_reset.shape[0])]
        self.sim.add_cover_area(areas,env_ids=batch_to_reset)

        if options.get('obstacle_gen') is not None:
            raise NotImplementedError

        observation = self._get_observation()
        info = {'width':width,'depth':depth}

        if self.render_mode is not None:
            info['image'] = self.render()
    
        return observation, info
    def step(self,action,render=False):
        info={}
        #info['intermediate_frames'] = np.expand_dims(self._render_frame(),0)
        if isinstance(action,tuple):
            action = converter.tuple2dict(action,self.n_robots,len(self.block_type),self.max_blocks,vertical=self.vertical)
            info['rotationvec'] =  geometry.rotmat_from_6D(action['rotation_offset'])[:,:2].reshape(-1,6)
        valid = self.sim.prepare_action()
        leaving_batch = self.is_holding[np.arange(self.n_batch),action['robot_id']]!=NOBLOCK
        
        valid[leaving_batch] = self.sim.leave_block(leaving_batch,self.is_holding[leaving_batch,action['robot_id'][leaving_batch]])
        #As this cannot be corrected by any other action, we reset the environment
        info['falling'] = ~valid.copy()
        to_reset = ~valid.copy()
        env_ids = np.nonzero(valid)[0]
        rewards = np.zeros(self.n_batch,dtype=float)
        if env_ids.shape[0] > 0:
            valid_a,rewards[env_ids],info['action'] = self.sim.put_block_rel_noface(self.block_type[action['block_type']],
                                                action['support_block'],
                                                action['center_offset'],
                                                action['rotation_offset'],
                                                action['direction'],
                                                env_ids=env_ids,touch_target=self.touch_target)
            valid[env_ids]=valid_a
        rewards[~valid] = -1
        self.sim.pass_time(np.nonzero(~valid)[0])
        """valid,rewards = self.sim.put_block_rel(self.block_type[action['block_type']],
                                               action['support_block'],
                                               action['block_face'],
                                               action['support_face'],
                                               action['center_offset'],
                                               action['rotation_offset'],
                                               action['direction'])"""
        self.is_holding[valid,action['robot_id'][valid]]=self.sim.n_block_t[valid]-1
        if render:
            info['image'] = self.render()
        #observation = self._get_observation()
        info['success'] = self.sim.check_success()
        if info['success'].any():
            print("Success in batch",np.nonzero(info['success'])[0])
        #terminated = np.logical_not(valid) | info['success']
        terminated = (self.sim.n_actions >= self.sim.n_block_max-2) | info['success'] | to_reset
        if self.autoreset:
            observation,_ = self.reset(reset_batches=terminated)
        else:
            observation = self._get_observation()
        #print(valid)
        print(rewards.mean())
        
        return observation, rewards, terminated, False, info
    def init_render(self):
        self.sim.reset(np.ones(self.n_batch,dtype=bool))
        activeUI(self.sim)
        self.sim.render()
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self.sim.render()
            ps.frame_tick()
            return None
        if self.render_mode == "rgb_array":
            frames = []
            ps.look_at([0,10,10], [0,0,0], fly_to=False)
            self.sim.tstep =  self.sim.n_actions[self.sim.batchi]- self.sim.n_block_init[self.sim.batchi]-1
            nsteps = self.sim.n_actions_steps[self.sim.batchi,self.sim.n_actions[self.sim.batchi]-1]
            if nsteps >0:
                if self.sim.action_t[self.sim.batchi,self.sim.n_actions[self.sim.batchi]-1,nsteps-1]['valid']:
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