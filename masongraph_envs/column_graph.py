import copy
import os
from typing import Literal, Union
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
import trimesh
from .tools import converter
import warp as wp
DATAPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator", "cont3Dsimulator", "data")
NOBLOCK = -1
#device = 'cuda'
class ColumnMasonGraph():
    metadata = {"render_modes": ["human", "agent_input","rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 n_batch = 1,
                 max_blocks = 50,
                 min_dist = 0.1,
                 max_dist = 10,
                 area_width = 0.5,
                 block_list: list[trimesh.Trimesh]|Literal['trapezoid','kapla']='trapezoid',
                 scales: list[float] = [10],
                 friction_coef:float=0.5,
                 glue_strength:float=0.0,
                 discount_factor = 0.1,
                 ground_gen: Literal['aligned']='aligned',
                 ground_angle: Union[float,None] = None,
                 autoreset:bool=True,
                 physics_on: bool = True,
                 n_frames_per_action: int = 4,
                 max_action_substeps: int = 50,
                 start_dist = 50,
                 use_her: bool = False,
                 touch_target: bool = True,
                 tol = 1e-2,
                 rep = 'bframegraph',
                 her: bool = False,
                 reset_on_fall = True,
                 terminal_on_invalid = True,
                 additional_contact_reward = 0,
                 continuous_reward = False,
                 max_falling_speed = 1
                 ):
        self.use_her = use_her
        self.touch_target = touch_target
        self.continuous_reward = continuous_reward
        self.max_falling_speed = max_falling_speed

        self.terminal_on_invalid = terminal_on_invalid
        self.reset_on_fall = reset_on_fall
        self.np_random = np.random.default_rng(seed=None)
        self.n_batch = n_batch
        self.perpendicular = True
        self.gamma = 1-discount_factor
        self.friction_coef = friction_coef 
        self.n_robots = 1 
        self.is_holding = np.ones((n_batch,self.n_robots),dtype=int)*NOBLOCK
        self.width = area_width
        self.block_type = []
        if block_list == 'trapezoid':
            d_top = np.tan(np.radians(scales))*1.2
            for d in d_top:
                corners = np.array([[0, 0, 0.8],[d,1.2,0.8],[2.4,0,0.8],[2.4-d,1.2,0.8],
                                    [2.4,0,0], [2.4-d,1.2,0], [0,0,0],[d,1.2,0]])
                
                trimesh_obj = trimesh.Trimesh(vertices=corners,
                                           faces=np.array([[2,8,1],[8,7,1],[7,5,1],[5,3,1],
                                                           [6,5,7],[8,6,7],[4,6,8],[2,4,8],
                                                           [4,3,5],[6,4,5],[2,1,3], [4,2,3]])-1)

                centroid = trimesh_obj.centroid
                trimesh_obj.apply_translation(-centroid)
                face_centers = trimesh_obj.triangles.mean(axis=1)
                outward = face_centers
                dot = np.sum(trimesh_obj.face_normals * outward, axis=1)
                assert np.all(dot > 0), "Some normals point inward!"
                assert trimesh_obj.is_watertight
                self.block_type.append(trimesh_obj)
            if ground_angle is None:
                self.ground_angle = np.radians(scales[0])
            else:
                self.ground_angle = np.radians(ground_angle)
            d_topg = np.tan(self.ground_angle)*1.2
            corners = np.array([[0, 0, 0.8],[d_topg,1.2,0.8],[2.4,0,0.8],[2.4-d_topg,1.2,0.8],
                                    [2.4,0,0], [2.4-d_topg,1.2,0], [0,0,0],[d_topg,1.2,0]])
                
            trimesh_obj = trimesh.Trimesh(vertices=corners,
                                           faces=np.array([[2,8,1],[8,7,1],[7,5,1],[5,3,1],
                                                           [6,5,7],[8,6,7],[4,6,8],[2,4,8],
                                                           [4,3,5],[6,4,5],[2,1,3], [4,2,3]])-1)
            self.ground_block = trimesh_obj
            self.block_type = np.array(self.block_type)
        elif type(block_list) ==str:
            assert False, f"unkwown preset {block_list}"
        else:
            self.block_type = np.array(block_list)
            self.ground_block = self.block_type[0]
        self.max_blocks = max_blocks
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.ground_gen = ground_gen
        self.sim = BatchSimulator(n_batch=n_batch,
                                  n_block_max=max_blocks,
                                  max_action_steps=max_action_substeps,
                                  max_dist = start_dist,
                                  physics_on=physics_on,
                                  mu=self.friction_coef,
                                  glue_strength=glue_strength,
                                  autoleave=True,
                                  autoleave_post=True,
                                  batchsolver=False,
                                  rep = rep,
                                  tol=tol,
                                  additional_contact_reward=additional_contact_reward,
                                  her=her,
                                  action_node='last',
                                  action_metadata=self.action_metadata()
                                  )
        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Render mode {render_mode} is unknown "
        self.render_mode = render_mode
        if self.render_mode is not None:
            activeUI(self.sim)
            self.render_size = ps.screenshot_to_buffer().shape[:2]
            self.n_frames_per_action = n_frames_per_action
        
        self.local_ref = isinstance(self.sim.graph,BFrameGraphConstructor)
        self.autoreset = autoreset
        if self.autoreset and self.terminal_on_invalid:
            print("Warning: autoreset and terminal_on_invalid are both on. The environment will reset often.")
    def _get_observation(self):
        #note that the robots are ordered from next to act to just acted
        state = self.sim.current_graph()#.to(device)
         #        "turn":self.sim.turn}
        
        return state
    def state_metadata(self):
        return self.sim.graph.metadata
    def action_metadata(self):
        if self.perpendicular:
            return {'discrete':[len(self.block_type),1],
                    'continuous':4,
                    'boundary':np.array([[-1.2,1.2],
                                         [-1.2,1.2],
                                         [-2,2],
                                         [-2,2]])}
        else:
            raise NotImplementedError
            return {'discrete':[len(self.block_type)],
                    'continuous':12,
                    'boundary':np.array([[-50,50],
                                         [-50,50],
                                         [-50,50],
                                         [-1,1],[-1,1],[-1,1],
                                         [-1,1],[-1,1]])}
    def action_mean(self):
        return self.action_metadata()['boundary'].mean(-1)
    def action_range(self):
        return (self.action_metadata()['boundary'][:,1] - self.action_metadata()['boundary'][:,0])/2
    def reset(self, seed=None, options=None,reset_batches=None,debug=False):
        # We need the following line to seed self.np_random
        #super().reset(seed=seed)
        #checked_struct = self.sim.new_struct
        #self.setup.checked_struct 
        info = {}
        if reset_batches is None:
            reset_batches = np.ones(self.n_batch,dtype=bool)
        self.sim.reset(reset_batches)
        #self.sim.checked_struct = checked_struct
        options = options or {'ground_gen':self.ground_gen,
                              'min_dist':self.min_dist,
                              "max_dist":self.max_dist,
                              }
        batch_to_reset = np.nonzero(reset_batches)[0]
        if len(batch_to_reset) == 0:
            return self._get_observation(), {}
        if options['ground_gen']=='aligned':
            #TODO: make the reset adaptive so that not all batches are the same
            width = self.np_random.uniform(options['min_dist'],options['max_dist'])
            #the camera looks along the x axis
            depth = self.width
            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2

            ground_brick = copy.deepcopy(self.ground_block)
            ground_brick.apply_translation(-ground_brick.centroid).apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))#.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,0,1]))
            self.sim.referencial[batch_to_reset,0] = trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0])[None,:3,:3]
            self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[0,0,0] for _ in batch_to_reset]), env_ids=batch_to_reset)
            info['width']=width
            info['depth']=depth
            areas = [shapely.Polygon([[ -depth/2, 0.4],
                                [  depth/2, 0.4 ],
                                [  depth/2,  width+0.4],
                                [ -depth/2,  width+0.4]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='manual':
            for block,pos in zip(options['ground_blocks'],options['ground_pos']):
                self.sim.add_ground(block,pos, env_ids=batch_to_reset)
            areas = [options['cover_area'][i] for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        else:
            raise NotImplementedError(f"Ground gen {options['ground_gen']} not implemented")
                    

        if options.get('obstacle_gen') is not None:
            raise NotImplementedError

        observation = self._get_observation()
        

        if self.render_mode is not None and self.sim.batchi in batch_to_reset:
            info['image'] = self.render()
    
        return observation, info
    def step(self,action,render=False,normalized=True):
        supface = 0
        bface = 9
        info={}
        if isinstance(action,tuple):
            action = converter.tuple2dict_col(action,len(self.block_type),self.max_blocks,perpendicular=True,normalized=normalized,action_boundary=self.action_metadata()['boundary'])
            #info['rotationvec'] =  geometry.rotmat_from_2D(action['rotation_offset'])[:,:2].reshape(-1,2)
        to_test = self.sim.prepare_action()
        to_truncate = ~to_test.copy()

        valid = np.ones(self.n_batch,dtype=bool)
        
        env_ids = np.nonzero(to_test)[0]
        rewards = np.zeros(self.n_batch,dtype=float)
        if env_ids.shape[0] > 0:
            if self.perpendicular:
                action['direction'] = [-self.sim.assembly_sequence[i,self.sim.n_block_t[i]-1].face_normals[supface] for i in env_ids]
            valid_a,rewards[env_ids] = self.sim.put_block_rel(self.block_type[action['block_type']],
                                                                            self.sim.n_block_t[env_ids]-1, #support block
                                                                            np.full(env_ids.shape[0],bface), #face of block
                                                                            np.full(env_ids.shape[0],supface), #face of support
                                                                            action['center_offset'],
                                                                            action['rotation_angle'],
                                                                            action['direction'],)
            valid[env_ids]=valid_a
            env_ids = env_ids[valid_a]
        arched = np.zeros((self.n_batch,),dtype=bool)
        arched[env_ids] = np.array([np.any(self.sim.assembly_sequence[b,self.sim.n_block_t[b]-1].vertices[:,2]<0) for b in env_ids])
        self.sim.is_ground[arched,self.sim.n_block_t[arched]-1] = True
        info['falling'] = np.zeros((self.n_batch,),dtype=bool)
        stable = self.sim.leave_block(valid & (self.sim.n_block_t>2),self.sim.n_block_t[valid]-2)
        
        
        falling = ~(stable)   
        info['falling'][valid & (self.sim.n_block_t>2)] =falling

        #stable_end = self.sim.leave_block(valid & arched & (self.sim.n_block_t>2),self.sim.n_block_t[arched]-2)
        #info['falling'][valid & arched & (self.sim.n_block_t>2)] = ~stable_end
        #valid[valid & arched & (self.sim.n_block_t>2)] = stable_end
        #Dont hold the block and check stability
        
        
        if self.continuous_reward and falling.any():
            rewards[info['falling']] = - np.clip(np.square(self.sim.falling_speed[info['falling'],self.sim.n_actions[info['falling']]]/self.max_falling_speed).max(-1),0,1)
        else:
            rewards[info['falling']] = -1
        if (rewards <-0.1).any():
            pass
        self.sim.pass_time(np.nonzero(~valid | to_truncate)[0])
        if render:
            info['image'] = self.render()
        info['success'] = self.sim.check_success()
        if info['success'].any():
            print("Success in batch",np.nonzero(info['success'])[0])
        info['areas'] = [self.sim.to_cover[i].area for i in range(self.n_batch)]
        terminated = (self.sim.n_block_t >= self.max_blocks-1) | info['success'] | (info['falling'] * (self.reset_on_fall)) | (~valid * (self.terminal_on_invalid)) | arched
        if self.autoreset:
            observation,_ = self.reset(reset_batches=terminated)
        else:
            observation = self._get_observation()
        #print(valid)
        print(rewards.mean())
        return observation, rewards, terminated, to_truncate, info
    def init_render(self):
        self.sim.reset(np.ones(self.n_batch,dtype=bool))
        activeUI(self.sim)
        self.sim.render()
    def render(self,cam_pos=[15,5,10]):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self.sim.render()
            ps.frame_tick()
            return None
        if self.render_mode == "rgb_array":
            frames = []
            ps.look_at(cam_pos, [0,10,5], fly_to=False)
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