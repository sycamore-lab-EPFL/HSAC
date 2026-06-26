import copy
import os
from typing import List, Literal, Union
import warnings
import numpy as np
from shapely import MultiPoint, Polygon
import shapely
from scipy.spatial.transform import Rotation as R

from .state_rep.ftgraph import FTGraphConstructor
from .render import render_objectives
from .render.render_action import render_action
from .render.render_forces import render_forces
from .render.render_structure import render_struct
from .objectives import cover
import torch
import trimesh
import polyscope as ps
import polyscope.imgui as psim
from .placement.placement import put_block_abs,put_block_from, put_block_pivot,put_block_slide
from .placement.placement_warp import put_block_abs_ker
from .utils import geometry
import warp as wp
class WarpSimulator():
    def __init__(self,n_batch=4,n_block_max=16,mu=0.5,max_action_steps=10,tol=1e-3,max_dist=4000):
        self.tol = tol
        self.name = 'sim'
        self.mu = mu
        self.batchsize = n_batch
        self.n_block_max = n_block_max
        self.max_action_steps = max_action_steps
        #default force and torque
        self.default_f = np.array([0,0,1])
        self.default_t = np.array([1,1,1])
        self.max_dist = max_dist
        self.start_dist = max_dist/2
        #self.solver = stability.GUROBISolver()

        #self.collision = [trimesh.collision.CollisionManager() for i in range(n_batch)]
        
        self.reset(np.ones(n_batch,dtype=bool))

        #rendering
        self.batchi = 0
        self.t = 0.01
        self.tstep = n_block_max
        self.use_forces =True

        #state representation

        self.graph = FTGraphConstructor()

        #objectives
        self.to_cover = [Polygon([]) for i in range(n_batch)]
        self.cover_dir = np.tile([0,0,1],(n_batch,1))
    def set_current_state_as_init(self):
        self.n_block_init = self.n_block_t.copy()
    def put_block_abs(self,block,pos,env_ids:Union[int,List[int]]=None):
        valid = np.zeros(self.batchsize,dtype=bool)
        rewards = np.zeros(self.batchsize)
        wp.launch(put_block_abs_ker,dim=(self.batchsize,self.n_block_t.max(), self.n_contact_id),inputs=())
        wp.launch(compute_objectives_ker,dim=(self.batchsize,self.size_obj_max(), self.n_contact_id),inputs=())
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        for i,b in enumerate(env_ids):
            print(f"{i=}")
            if i ==3:
                pass
            action, = put_block_abs(block[i],self.assembly_sequence[b,:self.n_block_t[b]],pos[i],self.tol)
            self.action_t[b,self.n_actions[b],self.n_actions_steps[b,self.n_actions[b]]]=action
            self.n_actions_steps[b,self.n_actions[b]]+=1
            self.n_actions[b]+=1
            valid[b] = action['valid']
            rewards[b] = self.compute_objectives(b,[action])
            if action['valid']:
                self.contacts[b]+=action['contacts']
                self.n_contact_t[b,self.n_actions[b]-1]=len(action['contacts'])+self.n_contact_t[b,self.n_actions[b]-2]
                self.assembly_sequence[b,self.n_block_t[b]]=action['block']
                self.is_held[b,self.n_actions[b]:,self.n_block_t[b]]=True
                self.n_block_t[b]+=1
            else:
                self.n_contact_t[b,self.n_actions[b]-1]=self.n_contact_t[b,self.n_actions[b]-2]
            self.n_block_anyt[b,self.n_actions[b]]=self.n_block_t[b]
        return valid,rewards
    
    def put_block_from(self,
                      block:List[trimesh.Trimesh],
                      pos:List[np.ndarray],
                      force:Union[None,List[np.ndarray]]=None,
                      grip_location:Union[None,List[np.ndarray]]=None,
                      slide=False,
                      pivot=False,
                      env_ids:Union[int,List[int]]=None,
                      ):
        valid = np.zeros(self.batchsize,dtype=bool)
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        rewards = np.zeros(self.batchsize)
        ##warp code
        

        ## end warp code
        for i,b in enumerate(env_ids):
            actions = put_block_abs(block[i],self.assembly_sequence[b,:self.n_block_t[b]],pos[i],self.tol)
            while actions[-1]['valid'] and len(actions)<self.max_action_steps:
                if slide:
                    action = put_block_slide(actions[-1],
                                             self.assembly_sequence[b,:self.n_block_t[b]],
                                             force[i],
                                             self.mu,
                                             max_range=self.max_dist,
                                             max_steps=self.max_action_steps-len(actions),
                                             tol=self.tol,
                                             d=actions[-2] if len(actions)>1 else None)
                    if len(action)==0 and not pivot:
                        break
                    actions+=action
                else:
                    action = put_block_from(actions[-1],
                                            self.assembly_sequence[b,:self.n_block_t[b]],
                                            force[i],
                                            max_range=self.max_dist,tol=self.tol,slide_over=False)
                    actions+=action
                if pivot and len(actions)<self.max_action_steps:
                    action_pivot = put_block_pivot(actions[-1],self.assembly_sequence[b,:self.n_block_t[b]],force[i],
                                                   grip_location[i] if grip_location is not None else np.zeros(3),
                                                   tol=self.tol)
                    if len(action_pivot)==0 or action_pivot[-1]['angle']<1e-5:
                        if len(action_pivot)>0 and action_pivot[-1]['contacts'][0]['points'].shape[0]<=4:
                            put_block_pivot(actions[-1],self.assembly_sequence[b,:self.n_block_t[b]],force[i],
                                                   grip_location[i],
                                                   tol=self.tol)
                        break
                    

                    actions+=action_pivot
            
            self.action_t[b,self.n_actions[b],self.n_actions_steps[b,self.n_actions[b]]:self.n_actions_steps[b,self.n_actions[b]]+len(actions)]=actions
            self.n_actions_steps[b,self.n_actions[b]]+=len(actions)
            self.n_actions[b]+=1
            valid[b] = actions[-1]['valid']
            
            
            rewards[b] = self.compute_objectives(b,actions)
            if actions[-1]['valid']:
                self.contacts[b]+=actions[-1]['contacts']
                self.n_contact_t[b,self.n_actions[b]-1]=len(actions[-1]['contacts'])+self.n_contact_t[b,self.n_actions[b]-2]
                self.assembly_sequence[b,self.n_block_t[b]]=actions[-1]['block']
                self.is_held[b,self.n_actions[b]:,self.n_block_t[b]]=True
                self.n_block_t[b]+=1
            else:
                self.n_contact_t[b,self.n_actions[b]-1]=self.n_contact_t[b,self.n_actions[b]-2]
            self.n_block_anyt[b,self.n_actions[b]]=self.n_block_t[b]
        return valid,rewards
    def put_block_rel_noface(self,
                      block:List[trimesh.Trimesh],
                      support:List[int],
                      contact_offset:List[np.ndarray],
                      rotation_offset:List[np.ndarray],
                      direction:List[np.ndarray],
                      grip_location=None,env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        valid_ind = np.ones(env_ids.shape[0],dtype=bool)
        newbs = []
        pos = []
        
        if grip_location is None:
            grip_location = np.zeros((len(block),3))
        rotation_matrix = geometry.rotmat_from_6D(rotation_offset)
        for i,b in enumerate(env_ids):
            if b == 12:
                pass
            if self.n_block_t[b]<=support[i] or self.n_actions[b]==self.n_block_max-1:
                #should be prevented by the action space
                valid_ind[i]=False
                self.n_actions[b]+=1
                continue

            newb = block[i].copy()
            # Apply rotation to the mesh
            newb.apply_transform(np.vstack([np.hstack([rotation_matrix[i], np.zeros((3, 1))]), [0, 0, 0, 1]]))
            newbs.append(newb)
            posi = self.assembly_sequence[b,support[i]].center_mass.copy()
            posi += contact_offset[i]
            posi -= direction[i]*self.start_dist/ np.linalg.norm(direction[i])
            pos.append(posi)
        valid,rewards = self.put_block_from(newbs,pos,direction,grip_location,slide = True,pivot=True,env_ids=env_ids[valid_ind])
        rewards[~valid_ind] = -1
        return valid,rewards
    def put_block_rel(self,
                      block:List[trimesh.Trimesh],
                      support:List[int],
                      face_block:List[int],
                      face_support:List[int],
                      contact_offset:List[np.ndarray],
                      rotation_offset:List[np.ndarray],
                      direction:List[np.ndarray],
                      grip_location=None):
        newbs = []
        pos = []
        valid = np.ones(self.batchsize,dtype=bool)
        if grip_location is None:
            grip_location = np.zeros((len(block),3))
        for i in range(self.batchsize):
            newb = block[i].copy()

            normal_i = newb.face_normals[face_block[i]]  

            # Define the target normal (x-axis)
            if self.n_block_t[i]>support[i] and face_support[i] < self.assembly_sequence[i,support[i]].face_normals.shape[0]:
                target_normal = -self.assembly_sequence[i,support[i]].face_normals[face_support[i]]  
            else:
                valid[i]=False
                continue
            # Compute rotation axis and angle
            rotation_axis = np.cross(normal_i, target_normal)
            axis_n = np.linalg.norm(rotation_axis,axis=-1)
            if axis_n>1e-5:
                rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize
            else:
                rotation_axis = np.array([1,0,0])
            angle = np.arccos(np.clip(np.dot(normal_i, target_normal), -1.0, 1.0))  # Angle between the vectors
            # Create rotation matrix
            rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()
            rotation_matrix = R.from_rotvec(rotation_offset[i] * target_normal).as_matrix()@rotation_matrix
            # Apply rotation to the mesh
            newb.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
            newbs.append(newb)
            
            posi = np.mean(self.assembly_sequence[i,support[i]].vertices[self.assembly_sequence[i,support[i]].faces[face_support[i]]],0)
            posi += contact_offset[i]
            posi -= direction[i]*self.max_dist*0.5
            pos.append(posi)
        valid,rewards = self.put_block_from(newbs,pos,direction,grip_location,slide = True,pivot=True,env_ids=np.arange(self.batchsize)[valid])
        return valid,rewards
    def compute_objectives(self,batchid,actions,debug=False):
        if debug:
            if actions[-1]['valid']:
                reward = actions[-1]['block'].vertices[:,2].max()
            else:
                reward = -1
            return reward

        if actions[-1]['valid']:
            reward,self.to_cover[batchid] = cover.cover_mesh(actions[-1]['block'],self.to_cover[batchid])
        else:
            reward = -1
        return reward
    def add_ground(self,blocks:List[trimesh.Trimesh],env_ids:Union[int,List[int]]=None):
        valid,reward = self.put_block_abs(blocks,[block.center_mass for block in blocks],env_ids=env_ids)
        self.is_ground[valid,self.n_block_t[valid]-1]=True
        self.n_block_init[valid]+=1
    def hold_block(self,bid:List[int]):
        self.is_held[:,self.n_actions:,bid]=True
    def leave_block(self,batch_boolar,bid:np.array):
        #TODO: slightly modify the position of the block as well ("drop the block"); if not too hard
        batchid, = np.nonzero(batch_boolar)

        for i,b in enumerate(batchid):
            self.is_held[b,self.n_actions[b]+1:,bid[i]]=False
        valid = self.check_stability()
        return valid
    def check_stability(self):
        return np.ones(self.batchsize,dtype=bool)
    def add_fill_area(self,area:List[trimesh.Trimesh]):
        raise NotImplementedError
    def add_reach_block(self,bid:List[int]):
        raise NotImplementedError
    def add_avoid_area(self,area:List[trimesh.Trimesh]):
        raise NotImplementedError
    def add_cover_area(self,area:List[Polygon],env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        for i,b in enumerate(env_ids):
            self.to_cover[b]=self.to_cover[b].union(area[i])
    def render(self):
        self.batchi = np.clip(self.batchi, 0, self.batchsize-1)
        self.tstep =  np.clip(self.tstep, 0, self.n_actions[self.batchi]-self.n_block_init[self.batchi])
        n_blocks_id = self.n_block_anyt[self.batchi,self.tstep+self.n_block_init[self.batchi]]
        n_actions_id = self.tstep+self.n_block_init[self.batchi]
        ps.remove_all_structures()
        ps.remove_all_groups()
        # blocks
        assembly_group = render_struct(self.name,self.assembly_sequence[self.batchi, :n_blocks_id],self.is_ground[self.batchi],self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],tol=self.tol)
        objectives_group = render_objectives.render_cover(self.to_cover[self.batchi])
        if self.use_forces:
            try:
                contacts = self.contacts[self.batchi][:self.n_contact_t[self.batchi,n_actions_id-1]]
            except:
                pass
            else:
                forces_group = render_forces(contacts,normals=None,frictions=None,tol=self.tol)
        if self.t>0:
            try:
                action = self.action_t[self.batchi,n_actions_id,:self.n_actions_steps[self.batchi,n_actions_id]]
            except:
                pass
            else:
                action_group = render_action(action,self.t,structure=self.assembly_sequence[self.batchi,:n_blocks_id],name='action')

    def render_callback(self):
        change1, self.batchi = psim.SliderInt("batch", self.batchi, 0, self.batchsize-1)
        change2, self.tstep = psim.SliderInt("istep", self.tstep, 0, self.n_actions[self.batchi]-self.n_block_init[self.batchi])
        change3, self.t = psim.SliderFloat("time", self.t, 0, 1)
        change_forces, self.use_forces = psim.Checkbox("Show forces", self.use_forces) 

        change = change1 or change2 or change_forces or change3
       
        if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
            self.batchi = self.batchi - 1
            change = True

        if psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
            self.batchi = self.batchi + 1
            change = True
        if change:
            self.render()
        
        
    def current_graph(self):
        return self.graph.compute_full_graph_batched(self.assembly_sequence,self.contacts,self.is_ground,self.is_held[np.arange(self.batchsize),self.n_actions,:],self.to_cover)
        #return self.graph.current_graph(self.assembly_sequence,self.contacts,self.is_ground,self.is_held[np.arange(self.batchsize),self.n_actions,:],self.to_cover)
    def reset(self,batches):
        if all(batches):
            self.assembly_sequence = wp.zeros((self.batchsize,self.n_block_max),dtype=object)
            self.is_ground = np.zeros((self.batchsize,self.n_block_max),dtype=bool)
            self.is_held = np.zeros((self.batchsize,self.n_block_max,self.n_block_max),dtype=bool)
            self.assembly_sequence[:]=None
            self.n_block_t = np.zeros(self.batchsize,dtype=int)
            self.n_block_anyt = np.zeros((self.batchsize,self.n_block_max),dtype=int)
            self.n_block_init = np.zeros(self.batchsize,dtype=int)
            self.contacts = [[] for i in range(self.batchsize)]
            self.n_contact_t = np.zeros((self.batchsize,self.n_block_max),dtype=int)
            #action arrays
            self.n_actions = np.zeros((self.batchsize),dtype=int)
            self.action_t = np.zeros((self.batchsize,self.n_block_max,self.max_action_steps),dtype=object)
            self.action_t[:]=None
            self.n_actions_steps = np.zeros((self.batchsize,self.n_block_max),dtype = int)
            self.to_cover = [Polygon([]) for i in range(self.batchsize)]
            self.cover_dir = np.tile([0,0,1],(self.batchsize,1))
        else:
            self.assembly_sequence[batches] = None
            self.is_ground[batches] = False
            self.is_held[batches] = False
            self.n_block_t[batches] = 0
            self.n_block_init[batches] = 0
            batches_id = np.nonzero(batches)[0]
            for i in batches_id:
                self.contacts[i] = []
                self.to_cover[i] = Polygon([])
            self.cover_dir[batches] = np.tile([0,0,1],(batches.sum(),1))
            self.n_contact_t[batches] = np.zeros((self.n_block_max),dtype=int)
            #action arrays
            self.n_actions[batches] = 0
            self.action_t[batches] = None
            self.n_actions_steps[batches] = 0
    def check_success(self,success_criteria:Literal['cover'] = 'cover'):
        if success_criteria == 'cover':
            return np.array([self.to_cover[i].area < self.tol for i in range(self.batchsize)])
        else:
            warnings.warn(f"Unknown success criteria {success_criteria}, returning False")
            return np.zeros(self.batchsize,dtype=bool)
def activeUI(sim):
    try:
        ps.set_allow_headless_backends(True)
    except:
        pass
    ps.init()
    ps.set_SSAA_factor(2)
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode('shadow_only')
    ps.set_user_callback(sim.render_callback)
    ps.frame_tick()
