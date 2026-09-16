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
import trimesh
from .tools import converter
import warp as wp
DATAPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator", "cont3Dsimulator", "data")
NOBLOCK = -1
#device = 'cuda'
class StraightMasonGraphCover():
    metadata = {"render_modes": ["human", "agent_input","rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None,
                 n_batch = 1,
                 max_blocks = 50,
                 min_area = [5,5],
                 max_area = [5.1,5.1],
                 block_list: list[trimesh.Trimesh]|Literal['brick','kapla']='brick',
                 scales: list[float] = [1],
                 friction_coef:float=0.5,
                 glue_strength:float=0.0,
                 discount_factor = 0.1,
                 ground_gen: Literal['around','arch','minimalist','hybrid']='around',
                 ground_width = 2.5,
                 autoreset:bool=True,
                 physics_on: bool = True,
                 n_frames_per_action: int = 4,
                 max_action_substeps: int = 50,
                 max_dist = 30,
                 limit_height = 20,
                 use_her: bool = False,
                 touch_target: bool = True,
                 tol = 1e-5,
                 rep = 'bframegraph',
                 embedding_dim=16,
                 her: bool = False,
                 reset_on_fall = True,
                 terminal_on_invalid = True,
                 additional_contact_reward = 0,
                 continuous_reward = False,
                 use_batchsolver = False,
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
        self.vertical = True
        self.gamma = 1-discount_factor
        self.friction_coef = friction_coef 
        self.n_robots = 1 
        self.is_holding = np.ones((n_batch,self.n_robots),dtype=int)*NOBLOCK
        self.ground_width = ground_width
        if block_list == 'brick':
            obj = DATAPATH+'/standard_brick/base_brick_flat.obj'
            trimesh_obj = trimesh.load_mesh(obj)
            centroid = trimesh_obj.centroid
            face_centers = trimesh_obj.triangles.mean(axis=1)
            outward = face_centers - centroid
            dot = np.sum(trimesh_obj.face_normals * outward, axis=1)
            assert np.all(dot > 0), "Some normals point inward!"
            assert trimesh_obj.is_watertight
            self.block_type = np.array([trimesh_obj.copy().apply_scale((1*s/100,1/100,1/100)) for s in scales])
        elif block_list == 'kapla':
            obj = DATAPATH+'/kapla/base_brick_flat.obj'
            trimesh_obj = trimesh.load_mesh(obj)
            centroid = trimesh_obj.centroid
            face_centers = trimesh_obj.triangles.mean(axis=1)
            outward = face_centers - centroid
            dot = np.sum(trimesh_obj.face_normals * outward, axis=1)
            assert np.all(dot > 0), "Some normals point inward!"
            assert trimesh_obj.is_watertight
            self.block_type = np.array([trimesh_obj.copy().apply_scale((1*s/100,1/100,1/100)) for s in scales])
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
                                  limit_height=limit_height,
                                  physics_on=physics_on,
                                  mu=self.friction_coef,
                                  glue_strength=glue_strength,
                                  autoleave=True,
                                  autoleave_post=True,
                                  batchsolver=use_batchsolver,
                                  rep = rep,
                                  tol=tol,
                                  additional_contact_reward=additional_contact_reward,
                                  her=her,
                                  action_node='all',
                                  embedding_dim=embedding_dim,
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
        if self.vertical:
            return {'discrete':[len(self.block_type),self.max_blocks],
                    'continuous':4,
                    'boundary':np.array([[-2.7,2.7],
                                         [-2.7,2.7],
                                         [-2,2],
                                         [-2,2]])}
        else:
            return {'discrete':[len(self.block_type),self.max_blocks],
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
                              'min_area':self.min_area,
                              "max_area":self.max_area,
                              }
        batch_to_reset = np.nonzero(reset_batches)[0]
        if len(batch_to_reset) == 0:
            return self._get_observation(), {}
        if options['ground_gen']=='around':
            #TODO: make the reset adaptive so that not all batches are the same
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
            n_bricks_w = np.floor((width+self.ground_width*2)/(brick_w)).astype(int)
            n_bricks_d = np.floor((depth)/(brick_d)).astype(int)
            n_bricks_gw = np.floor((self.ground_width)/(brick_w)).astype(int)
            n_bricks_gd = np.floor((self.ground_width)/(brick_d)).astype(int)
            for xw in np.linspace(-self.ground_width-(width-brick_w)/2,self.ground_width+(width-brick_w)/2,n_bricks_w):
                for yd in np.linspace(-brick_d/2,self.ground_width+brick_d/2,n_bricks_gd):
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[-depth/2+yd,
                                                                                                xw, 
                                                                                                0] for _ in batch_to_reset]), env_ids=batch_to_reset)
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[depth/2-yd,
                                                                                                xw,
                                                                                                0] for _ in batch_to_reset]),env_ids=batch_to_reset)
            for yd in np.linspace(-depth/2+brick_d/2,depth/2-brick_d/2,n_bricks_d):
                for xw in np.linspace(-brick_w/2,-self.ground_width+brick_w/2,n_bricks_gw):
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[yd,
                                                                                                -width/2+xw,
                                                                                                0] for _ in batch_to_reset]),env_ids=batch_to_reset)
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[yd,
                                                                                                width/2-xw,
                                                                                                0] for _ in batch_to_reset]),env_ids=batch_to_reset)
            info['width']=width
            info['depth']=depth
            areas = [shapely.Polygon([[ -depth/2, -width/2],
                                [  depth/2, -width/2],
                                [  depth/2,  width/2],
                                [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='arch':
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
                
            n_bricks_w = np.floor((width+self.ground_width*2)/(brick_w)).astype(int)
            n_bricks_d = np.floor((depth)/(brick_d)).astype(int)
            n_bricks_gw = np.floor((self.ground_width)/(brick_w)).astype(int)
            n_bricks_gd = np.floor((self.ground_width)/(brick_d)).astype(int)
            center_pos = []

            for yd in np.linspace(-depth/2+brick_d/2,depth/2-brick_d/2,n_bricks_d):
                for xw in np.linspace(-brick_w/2,-self.ground_width+brick_w/2,n_bricks_gw):
                    center_pos.append(np.array([[yd,-width/2+xw,0] for _ in batch_to_reset]))
                    center_pos.append(np.array([[yd,width/2-xw,0] for _ in batch_to_reset]))

            grounds = [[ground_brick for _ in batch_to_reset ] for _ in range(len(center_pos))]
            center_pos = np.stack(center_pos,axis=0)
            for block,pos in zip(grounds,center_pos):
                self.sim.add_ground(block,pos, env_ids=batch_to_reset)

            areas = [shapely.Polygon([[ -depth/2, -width/2],
                                [  depth/2, -width/2],
                                [  depth/2,  width/2],
                                [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='minimalist':
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
                
            n_bricks_w = np.floor((width+self.ground_width)/(brick_w)/1.5).astype(int)
            n_bricks_d = np.floor((depth)/(brick_d)).astype(int)
            n_bricks_gw = np.floor((self.ground_width)/(brick_d)).astype(int)
            n_bricks_gd = np.floor((self.ground_width)/(brick_d)).astype(int)
            center_pos = []
            for yw in np.linspace(-self.ground_width/2-(width-brick_w)/2,self.ground_width/2+(width-brick_w)/2,n_bricks_w):
                for xd in np.linspace(-brick_d/2,self.ground_width+brick_d/2,n_bricks_gd):
                    center_pos.append(np.array([[-depth/2+xd,yw, 0] for _ in batch_to_reset]))
                    center_pos.append(np.array([[depth/2-xd,yw, 0] for _ in batch_to_reset]))

            for xd in np.linspace(-depth/2+brick_d/2,depth/2-brick_d/2,n_bricks_d):
                for yw in np.linspace(-brick_w/2,-self.ground_width+brick_w/2,n_bricks_gw):
                    center_pos.append(np.array([[xd,-width/2+yw,0] for _ in batch_to_reset]))
                    center_pos.append(np.array([[xd,width/2-yw,0] for _ in batch_to_reset]))

            cover_area = [shapely.Polygon([[ -depth/2, -width/2],
                                            [  depth/2, -width/2],
                                            [  depth/2,  width/2],
                                            [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            center_pos = np.stack(center_pos,axis=0)
            grounds = [[ground_brick for _ in batch_to_reset ] for _ in range(center_pos.shape[0])]
            areas = [shapely.Polygon([[ -depth/2, -width/2],
                                [  depth/2, -width/2],
                                [  depth/2,  width/2],
                                [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            for block,pos in zip(grounds,center_pos):
                self.sim.add_ground(block,pos, env_ids=batch_to_reset)
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='cantilever':
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
                
            n_bricks_w = (self.ground_width/(brick_w)).astype(int)
            n_bricks_d = np.floor(self.ground_width/(brick_d)).astype(int)
            center_pos = []
            for yw in np.linspace(-self.ground_width/2+brick_w/2,self.ground_width/2-brick_w/2,n_bricks_w):
                for xd in np.linspace(-self.ground_width/2+brick_d/2,self.ground_width/2-brick_d/2,n_bricks_d):
                    center_pos.append(np.array([[xd,yw, 0] for _ in batch_to_reset]))
            center_pos = np.stack(center_pos,axis=0)
            grounds = [[ground_brick for _ in batch_to_reset ] for _ in range(center_pos.shape[0])]
            areas = [shapely.Polygon([[ -depth/2, -width/2],
                                [  depth/2, -width/2],
                                [  depth/2,  width/2],
                                [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            for block,pos in zip(grounds,center_pos):
                self.sim.add_ground(block,pos, env_ids=batch_to_reset)
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='hybrid':
            #TODO: make the reset adaptive so that not all batches are the same
            width = self.np_random.uniform(options['min_area'][1],options['max_area'][1])
            depth = self.np_random.uniform(options['min_area'][0],options['max_area'][0])

            brick_w = self.block_type[0].vertices[:,1].max()+self.sim.tol*2
            brick_d = self.block_type[0].vertices[:,0].max()+self.sim.tol*2
            ground_brick = copy.deepcopy(self.block_type[0])
            ground_brick.apply_translation(-ground_brick.centroid)
            n_bricks_w = np.floor((width+self.ground_width*2)/(brick_w)).astype(int)
            n_bricks_d = np.floor((depth)/(brick_d)).astype(int)
            n_bricks_gw = np.floor((self.ground_width)/(brick_w)).astype(int)
            n_bricks_gd = np.floor((self.ground_width)/(brick_d)).astype(int)
            for xw in np.linspace(-self.ground_width-(width-brick_w)/2,self.ground_width+(width-brick_w)/2,n_bricks_w):
                for yd in np.linspace(-brick_d/2,self.ground_width+brick_d/2,n_bricks_gd):
                    if np.abs(xw) > self.ground_width - brick_w:
                        self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[-depth/2+yd,
                                                                                                    xw, 
                                                                                                    0] for _ in batch_to_reset]), env_ids=batch_to_reset)
                        self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[depth/2-yd,
                                                                                                    xw,
                                                                                                    0] for _ in batch_to_reset]),env_ids=batch_to_reset)
            for yd in np.linspace(-depth/2+brick_d/2,depth/2-brick_d/2,n_bricks_d):
                for xw in np.linspace(-brick_w/2,-self.ground_width+brick_w/2,n_bricks_gw):
                    
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[yd,
                                                                                                -width/2+xw,
                                                                                                0] for _ in batch_to_reset]),env_ids=batch_to_reset)
                    self.sim.add_ground([ground_brick for _ in batch_to_reset],pos = np.array([[yd,
                                                                                                width/2-xw,
                                                                                                0] for _ in batch_to_reset]),env_ids=batch_to_reset)
            info['width']=width
            info['depth']=depth
            main_areas = [shapely.Polygon([[ -depth/2, -width/2],
                                [  depth/2, -width/2],
                                [  depth/2,  width/2],
                                [ -depth/2,  width/2]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(main_areas,env_ids=batch_to_reset)
            front_areas = [shapely.Polygon([[ depth/2, -self.ground_width/2],
                                [  depth/2+self.ground_width, -self.ground_width/2],
                                [  depth/2+self.ground_width,  self.ground_width/2],
                                [  depth/2,  self.ground_width/2]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(front_areas,env_ids=batch_to_reset)
            back_areas = [shapely.Polygon([[ -depth/2, -self.ground_width/2],
                                [  -depth/2-self.ground_width, -self.ground_width/2],
                                [  -depth/2-self.ground_width,  self.ground_width/2],
                                [  -depth/2,  self.ground_width/2]]) for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(back_areas,env_ids=batch_to_reset)
        elif options['ground_gen']=='manual':
            for block,pos in zip(options['ground_blocks'],options['ground_pos']):
                self.sim.add_ground(block,pos, env_ids=batch_to_reset)
            areas = [options['cover_area'][i] for i in range(batch_to_reset.shape[0])]
            self.sim.add_cover_area(areas,env_ids=batch_to_reset)
        else:
            raise NotImplementedError
                    

        if options.get('obstacle_gen') is not None:
            raise NotImplementedError

        observation = self._get_observation()
        

        if self.render_mode is not None and self.sim.batchi in batch_to_reset:
            info['image'] = self.render()
    
        return observation, info
    def step(self,action,render=False,normalized=True):
        info={'equivalent_actions':[[] for _ in range(self.n_batch)]}
        
        if isinstance(action,tuple):
            tuple_action = action
            action = converter.tuple2dict_flat(action,1,len(self.block_type),self.max_blocks,vertical=True,normalized=normalized,action_boundary=self.action_metadata()['boundary'])
            info['rotationvec'] =  geometry.rotmat_from_2D(action['rotation_offset'])[:,:2].reshape(-1,2)
        else:
            tuple_action = None
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
            eq_action_dict = self.sim.equivalent_actions([info['action'][i]['new_contacts'] for i in env_ids])
            for i,b in enumerate(env_ids):
                info['equivalent_actions'][i] = []
                if valid_a[i]:
                    for newa in eq_action_dict[i]:
                        newa.update({'block_type':action['block_type'][i],'direction':action['direction'][i], 'robot_id':0})
                        info['equivalent_actions'][i].append(converter.dict2tuple_flat(newa,1,len(self.block_type),self.max_blocks,vertical=True,normalized=normalized,action_boundary=self.action_metadata()['boundary']))
                else:
                    info['equivalent_actions'][i].append((tuple_action[0][i,None].detach().cpu().numpy(),tuple_action[1][i,None].detach().cpu().numpy()))
            valid[env_ids]=valid_a
            env_ids = env_ids[valid_a]
        
        
        print("blocks placed")
        #Dont hold the block and check stability
        info['falling'] = np.zeros((self.n_batch,),dtype=bool)
        if self.sim.physics_on:
            falling = self.sim.simulate_physics(env_ids)
            info['falling'][env_ids] = falling.copy()
            self.sim.remove_last_block(env_ids[falling])
            #[self.sim.physics_model[b].fix_part(self.sim.n_block_t[b]-1) for b in np.nonzero(falling)[0]]
            valid[env_ids] = ~falling
            if self.continuous_reward and falling.any():
                rewards[env_ids[falling]] = - np.clip(np.square(self.sim.falling_speed[env_ids[falling],self.sim.n_actions[env_ids[falling]]]/self.max_falling_speed).max(-1),0,1)
            else:
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
            observation = self._get_observation()
        #print(valid)
        print(f"Step completed, rewards:{ rewards}")
        
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