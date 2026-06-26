import copy
import os
from typing import List, Literal, Union
import warnings
import numpy as np
from shapely import MultiPoint, Polygon
import shapely
from scipy.spatial.transform import Rotation as R

from .state_rep.naive_jagged import NaiveJaggedConstructor
from .state_rep.naive_fss import NaiveSetConstructor
from simulator.cont3Dsimulator.physics.physics_model import FIXED, FREE
from simulator.cont3Dsimulator.physics.RBE import RBEModelDiag, BatchRBEModel
from simulator.cont3Dsimulator.physics.physics_solver import GUROBISolver
from simulator.cont3Dsimulator.render import render_graph


from .state_rep.ftbframegraph import FTBFrameGraphConstructor
from .state_rep.nograph import NOGraphConstructor
from .state_rep.noigraph import NOIGraphConstructor
from .state_rep.bframegraph import BFrameGraphConstructor
from .state_rep.ftgraph import FTGraphConstructor
from .state_rep.bframegraph_noobj import BFrameNoObjGraphConstructor
from .state_rep.bframegraph_local import  BFrameGraphConstructorLocal
from .render import render_objectives
from .render.render_action import render_action, render_fall
from .render.render_forces import render_forces
from .render.render_structure import render_struct
from .objectives import cover
import torch
import trimesh
import polyscope as ps
import polyscope.imgui as psim
from .placement.placement import put_block_abs, put_block_abs_batched,put_block_from, put_block_from_batched, put_block_pivot,put_block_slide
from .utils import geometry
import warp as wp
class BatchSimulator():
    def __init__(self,n_batch=4,n_block_max=16,mu=0.5,glue_strength = 0,max_action_steps=10,tol=1e-3,max_dist=40,
                physics_on=False,autoleave=False,autoleave_post = False,batchsolver=True,
                rep='bframegraph',additional_contact_reward=0,her=False,freeze_level=None,limit_height=10,embedding_dim=16,
                action_node:Literal['none','last','all']='none',action_metadata=None,use_mortar=False,mortar_thickness=0.1):
        self.tol = tol
        self.angle_tol = 1e-2
        self.identical_tol = None # automatically set to tol/10 if None
        self.autoleave_post = autoleave_post
        self.name = "sim"
        self.mu = mu
        self.use_mortar = use_mortar
        self.is_mortar_piece = np.zeros((n_batch,n_block_max),dtype=bool)
        self.mortar_thickness = mortar_thickness
        self.batchsize = n_batch
        self.n_block_max = n_block_max
        self.max_action_steps = max_action_steps
        self.additional_contact_reward = additional_contact_reward
        self.autoleave = autoleave
        #default force and torque
        self.default_f = np.array([0,0,1])
        self.default_t = np.array([1,1,1])
        self.max_dist = max_dist
        self.start_dist = max_dist/2
        self.building_area =np.array(([max_dist/2,max_dist/2,limit_height],[-max_dist/2,-max_dist/2,-2]))
        if freeze_level is not None:
            self.freeze_level = freeze_level
            self.freeze = True
        else:
            self.freeze_level = -np.inf
            self.freeze = False
        self.physics_on = physics_on
        if self.physics_on:
            if batchsolver:
                self.batchsolver = True
                #self.physics_model = BatchStabilityModelFlat(n_batch= n_batch,rho=1,name=f"model_batch",n_block_max=n_block_max,glue_strength=glue_strength,lb=tol)
                self.physics_model = BatchRBEModel(n_batch= n_batch,rho=1,nt=4,name=f"model_batch",n_block_max=n_block_max,qp_solver=GUROBISolver(),glue_strength=glue_strength,mu=mu,lb=tol)
            else:
                self.batchsolver = False
                self.physics_model = [RBEModelDiag(rho=1,nt=4,name=f"model_{i}",n_block_max=n_block_max,qp_solver=GUROBISolver(),glue_strength=glue_strength,mu=mu,lb=tol) for i in range(n_batch)]
        self.falling = np.zeros((n_batch,n_block_max+1),dtype=bool)
        self.marked = np.zeros((n_batch,n_block_max+1),dtype=bool)
        self.falling_speed = np.zeros((n_batch,n_block_max+1,n_block_max*6))
        #self.collision = [trimesh.collision.CollisionManager() for i in range(n_batch)]
        
        self.reset(np.ones(n_batch,dtype=bool))

        #rendering
        self.batchi = 0
        self.t = 0.01
        self.tstep = n_block_max
        self.use_forces =True
        self.graph_mode = False
        self.expscale = -1.0
        #state representation
        self.reptype = rep
        if self.reptype == 'ftgraph':
            self.graph = FTGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'nograph':
            self.graph = NOGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'noigraph':
            self.graph = NOIGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'bframegraph':
            self.graph = BFrameGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'ftbframegraph':
            self.graph = FTBFrameGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'bframegraph_local':
            self.graph = BFrameGraphConstructorLocal(radius=2.7,keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'bframegraph_noobj':
            self.graph = BFrameNoObjGraphConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata)
        elif self.reptype == 'naive_fss':
            self.graph = NaiveSetConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata,
                                             embedding_dim=16)
        elif self.reptype == 'naive_jagged':
            self.graph = NaiveJaggedConstructor(keep_covered_areas=her,action_node=action_node,action_metadata=action_metadata,embedding_dim=embedding_dim)
        else:
            raise NotImplementedError
        #objectives
        self.to_cover = np.array([Polygon([]) for i in range(n_batch)])
        self.covered = [[] for i in range(n_batch)]
        self.last_covered = np.array([Polygon([]) for i in range(n_batch)])
        self.cover_dir = np.tile([0,0,1],(n_batch,1))
        self.reptype = rep
    def set_current_state_as_init(self):
        self.n_block_init = self.n_block_t.copy()
    def remove_last_block(self,env_ids:Union[int,List[int]]=None):
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        valid = np.zeros(env_ids.shape[0],dtype=bool)
        for i,b in enumerate(env_ids):
            # A mortar placement appends the real block followed by its mortar
            # pieces, so the parts to drop are the contiguous run of mortar pieces
            # sitting on top of the stack plus the real block beneath them. With
            # mortar disabled this is just the single last block.
            n_remove = 1
            while (self.n_block_t[b]-n_remove > 0 and
                   self.is_mortar_piece[b,self.n_block_t[b]-n_remove]):
                n_remove += 1
            for _ in range(n_remove):
                self.n_block_t[b]-=1
                if self.physics_on:
                    if not self.batchsolver:
                        self.physics_model[b].remove_part(self.n_block_t[b])
                    else:
                        self.physics_model.remove_part(b, self.n_block_t[b])
                self.is_mortar_piece[b,self.n_block_t[b]]=False
            self.n_contact_t[b,self.n_actions[b]]=self.n_contact_t[b,self.n_actions[b]-1]
            self.contacts[b] = self.contacts[b][:self.n_contact_t[b,self.n_actions[b]]]
            self.n_block_anyt[b,self.n_actions[b]]=self.n_block_t[b]
            self.to_cover[b] = self.last_covered[b].union(self.to_cover[b])
            self.covered[b].pop(-1)
    def put_block_abs(self,block,pos,env_ids:Union[None,List[int]]=None,update_obj = True, rotmat=None,check_contact=True):
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        if rotmat is not None:
            self.referencial[env_ids,self.n_block_t[env_ids]] = rotmat
        rewards = np.zeros(env_ids.shape[0])
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        #with wp.ScopedTimer("putblock",use_nvtx=True):
        actions = put_block_abs_batched(block,None,self.n_block_t,pos,self.tol,identical_tol=self.identical_tol,
                                        old_blocks_ids=self.assembly_sequence_wpid,old_blocks_wps=self.assembly_sequence_wp,
                                        env_ids=env_ids,check_contact=check_contact,use_mortar=self.use_mortar, mortar_thickness=self.mortar_thickness)

        self.action_t[env_ids,self.n_actions[env_ids]-1,self.n_actions_steps[env_ids,self.n_actions[env_ids]]]=actions
        self.n_actions_steps[env_ids,self.n_actions[env_ids]-1]+=1
        valid = np.array([action['valid'] for action in actions])
        if update_obj:
            rewards = self.compute_objectives(env_ids,actions)
        if self.use_mortar and check_contact:
            # Parts are added in increasing id order: the new block first
            # (id new_block_id), then its mortar blocks (new_block_id + 1 + m).
            # Route each contact to the add_part call of its higher-id part so
            # both referenced parts already exist when it is wired into the
            # physics model: a direct block-block contact attaches to the new
            # block, while an A<->mortar / mortar<->block contact attaches to the
            # mortar that sits between the two blocks.
            new_block_ids = [self.n_block_t[b] for b in env_ids]
            block_contacts = [[c for c in actions[i]['contacts']
                               if max(c['partIDA'], c['partIDB']) == new_block_ids[i]]
                              for i in range(len(env_ids))]
            self.add_to_assembly([action['block'] for action in actions], block_contacts, env_ids)
            for i, b in enumerate(env_ids):
                for m, mb in enumerate(actions[i]['mortar_blocks']):
                    mid = new_block_ids[i] + 1 + m
                    mortar_contacts = [c for c in actions[i]['contacts']
                                       if max(c['partIDA'], c['partIDB']) == mid]
                    self.add_to_assembly([mb], [mortar_contacts], [b], is_mortar=True)
        else:
            self.add_to_assembly([action['block'] for action in actions],
                                 [action['contacts'] for action in actions], env_ids)
        return valid,rewards
    
    def put_block_from(self,
                      block:List[trimesh.Trimesh],
                      pos:List[np.ndarray],
                      force:Union[None,List[np.ndarray]]=None,
                      init_rot = None,
                      grip_location:Union[None,List[np.ndarray]]=None,
                      slide=False,
                      pivot=False,
                      env_ids:Union[int,List[int]]=None,
                      touch_target=False,
                      hold = True,
                      target = -1,
                      ):
        
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        valid = np.zeros(env_ids.shape[0],dtype=bool)
        rewards = np.zeros(env_ids.shape[0])
        if init_rot is not None:
            if init_rot.shape[1]==6:
                rotation_matrix = geometry.rotmat_from_6D(init_rot)
            elif init_rot.shape[1]==2:
                rotation_matrix = geometry.rotmat_from_2D(init_rot)
            else:
                raise NotImplementedError
            rotblocks = []
            for i,b in enumerate(env_ids):
                newb = block[i].copy()
                # Apply rotation to the mesh
                newb.apply_transform(np.vstack([np.hstack([rotation_matrix[i], np.zeros((3, 1))]), [0, 0, 0, 1]]))
                rotblocks.append(newb)
        else:
            rotblocks = block
        actions = [put_block_abs_batched(rotblocks,None,self.n_block_t,pos,self.tol,
                                         old_blocks_ids=self.assembly_sequence_wpid,
                                         old_blocks_wps=self.assembly_sequence_wp,
                                         identical_tol=self.identical_tol,
                                         angle_tol=self.angle_tol,env_ids=env_ids,use_mortar=self.use_mortar, mortar_thickness=self.mortar_thickness)]
        if not slide and not pivot:
            action = put_block_from_batched(actions[-1],
                                            self.assembly_sequence_wp,#[np.arange(self.batchsize),self.n_block_t-1,None],
                                            self.assembly_sequence_wpid,#[np.arange(self.batchsize),self.n_block_t-1,None],
                                            self.n_block_t,#np.ones(self.batchsize,dtype=int),
                                            force,
                                            max_range=self.max_dist,tol=self.tol,
                                            identical_tol = self.identical_tol,
                                            angle_tol=self.angle_tol,
                                            use_mortar=self.use_mortar, mortar_thickness=self.mortar_thickness,
                                            env_ids=env_ids)
                                            
            actions.append(action)
            for i,b in enumerate(env_ids):
                actions[-1][b]['valid'] = actions[-1][b]['valid'] and len(actions[-1][b]['contacts'])>0
                self.action_t[b,self.n_actions[b]-1,self.n_actions_steps[b,self.n_actions[b]-1]:self.n_actions_steps[b,self.n_actions[b]-1]+len(actions)]=[action[i] for action in (actions)]
                self.n_actions_steps[b,self.n_actions[b]-1]+=len(actions)
                actions[-1][b]['valid'] = self.in_building_area([actions[-1][b]['block']]) and actions[-1][b]['valid']
                if touch_target and actions[-1][b]['valid']:
                    actions[-1][b]['valid'] &= np.any(np.vstack([c['partIDA'] for c in actions[1][b]['contacts']])==target[i])
                valid[i] = actions[-1][b]['valid']
            #with wp.ScopedTimer("compute obj",use_nvtx=True,color='orange'):
            print("Computing objectives...")
            rewards = self.compute_objectives(env_ids,actions[-1],additional_contact_reward=self.additional_contact_reward)
            
           
            if self.use_mortar:
                # Parts are added in increasing id order: the new block first
                # (id new_block_id), then its mortar blocks (new_block_id + 1 + m).
                # Route each contact to the add_part call of its higher-id part so
                # both referenced parts already exist when it is wired into the
                # physics model.
                new_block_ids = {b: self.n_block_t[b] for b in env_ids[valid]}
                block_contacts = [[c for c in actions[-1][b]['contacts']
                                   if max(c['partIDA'], c['partIDB']) == new_block_ids[b]]
                                  for b in env_ids[valid]]
                self.add_to_assembly([actions[-1][b]['block'] for b in env_ids[valid]],
                                     block_contacts, env_ids[valid], hold=hold)
                for b in env_ids[valid]:
                    for m, mb in enumerate(actions[-1][b]['mortar_blocks']):
                        mid = new_block_ids[b] + 1 + m
                        mortar_contacts = [c for c in actions[-1][b]['contacts']
                                           if max(c['partIDA'], c['partIDB']) == mid]
                        self.add_to_assembly([mb], [mortar_contacts], [b], is_mortar=True)
            else:
                self.add_to_assembly([actions[-1][b]['block'] for b in env_ids[valid]],
                                     [actions[-1][b]['contacts'] for b in env_ids[valid]],
                                     env_ids[valid],hold=hold)
            return valid,rewards,actions
        raise NotImplementedError("Only slide or pivot placement is supported")
        for i,b in enumerate(env_ids):
            actions = actionsi[i]
            while actions[-1]['valid'] and len(actions)<self.max_action_steps:
                if slide:
                    action = put_block_slide(actions[-1],
                                             self.assembly_sequence[b,:self.n_block_t[b]],
                                             force[i],
                                             self.mu,
                                             max_range=self.max_dist,
                                             max_steps=self.max_action_steps-len(actions),
                                             tol=self.tol,
                                             identical_tol = self.identical_tol,
                                             angle_tol=self.angle_tol,
                                             d=actions[-2] if len(actions)>1 else None)
                    if len(action)==0 and not pivot:
                        break
                    if self.n_actions_steps[b,self.n_actions[b]-1]+len(actions)<self.max_action_steps:
                        actions+=action
                    else:
                        actions[-1]['valid'] = False
                        break
                    if touch_target and len(actions)>1 and len(actions[1]['contacts'])>0:
                        actions[-1]['valid'] &= np.any(np.vstack([c['partIDA'] for c in actions[1]['contacts']])==target[i])
                        
                else:
                    action = put_block_from(actions[-1],
                                            self.assembly_sequence[b,:self.n_block_t[b]],
                                            force[i],
                                            max_range=self.max_dist,tol=self.tol,
                                            identical_tol = self.identical_tol,
                                            angle_tol=self.angle_tol)
                                            
                    actions+=action
                    if not pivot:
                        break
                if pivot and len(actions)<self.max_action_steps:
                    action_pivot = put_block_pivot(actions[-1],self.assembly_sequence[b,:self.n_block_t[b]],force[i],
                                                   grip_location[i] if grip_location is not None else np.zeros(3),
                                                   tol=self.tol,angle_tol = self.angle_tol,mu=self.mu)
                    if len(action_pivot)==0 or action_pivot[-1]['angle']<1e-6:
                        break
                    actions+=action_pivot
                    if len(actions)==self.max_action_steps:
                        actions[-1]['valid'] = False
                        break
            actions[-1]['valid'] = actions[-1]['valid'] and len(actions[-1]['contacts'])>0
            self.action_t[b,self.n_actions[b]-1,self.n_actions_steps[b,self.n_actions[b]-1]:self.n_actions_steps[b,self.n_actions[b]-1]+len(actions)]=actions
            self.n_actions_steps[b,self.n_actions[b]-1]+=len(actions)
            actions[-1]['valid'] = self.in_building_area([actions[-1]['block']]) and actions[-1]['valid']
            valid[i] = actions[-1]['valid']
            rewards[i] = self.compute_objectives([b],[actions],additional_contact_reward=self.additional_contact_reward)
            if actions[-1]['valid']:
                self.add_to_assembly([actions[-1]['block']],[actions[-1]['contacts']],[b],hold=hold)
            
        return valid,rewards
    def put_block_rel_noface(self,
                      block:List[trimesh.Trimesh],
                      support:List[int],
                      contact_offset:List[np.ndarray],
                      rotation_offset:List[np.ndarray],
                      direction:List[np.ndarray],
                      pivot=True,
                      grip_location=None,
                      env_ids=None,
                      touch_target=False,
                      hold=True,
                      slide=True,
                      local_ref=None):
        if local_ref is None:
            local_ref = isinstance(self.graph,BFrameGraphConstructor) or isinstance(self.graph,FTBFrameGraphConstructor)
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        valid_ind = np.ones(env_ids.shape[0],dtype=bool)
        newbs = []
        pos = []
        info = np.array([{'new_contacts':[]} for _ in range(env_ids.shape[0])])
        support = np.array(support)
        if grip_location is None:
            grip_location = np.zeros((len(block),3))
        if rotation_offset.shape[1]==6:
            rotation_matrix = geometry.rotmat_from_6D(rotation_offset)
        elif rotation_offset.shape[1]==2:
            rotation_matrix = geometry.rotmat_from_2D(rotation_offset)
        elif rotation_offset.shape[1]==3 and rotation_offset.shape[2]==3:
            rotation_matrix = rotation_offset
        else:
            raise NotImplementedError
        for i,b in enumerate(env_ids):
            info[i]['rotationvec'] = rotation_offset[i]
            if support[i]>=self.n_block_t[b] or self.n_actions[b]==self.n_block_max:
                #should be prevented by the action space
                print('Invalid support block')
                valid_ind[i]=False                
                continue
            if local_ref:
                #use the principal axis of inertia of the support block as reference
                ref = self.referencial[b,support[i]]
                offet_abs = ref@contact_offset[i]
                rotation_matrix[i] = rotation_matrix[i]@ref
                self.referencial[b,self.n_block_t[b]] = rotation_matrix[i]
            else:
                offet_abs = contact_offset[i]
                self.referencial[b,self.n_block_t[b]] = rotation_matrix[i]
            newb = block[i].copy()
            # Apply rotation to the mesh
            newb.apply_transform(np.vstack([np.hstack([rotation_matrix[i], np.zeros((3, 1))]), [0, 0, 0, 1]]))
            newbs.append(newb)
            posi = self.assembly_sequence[b,support[i]].center_mass.copy()
            posi += offet_abs
            posi -= direction[i]*self.start_dist/ np.linalg.norm(direction[i])
            pos.append(posi)
        rewards= np.zeros(env_ids.shape[0])
        valid_a,rewards[valid_ind],action = self.put_block_from(newbs,pos,direction[valid_ind],
                                                         grip_location=grip_location[valid_ind],
                                                         slide = slide,
                                                         pivot=pivot,
                                                         env_ids=env_ids[valid_ind],
                                                         touch_target=touch_target,
                                                         target = support[env_ids[valid_ind]],hold=hold)
        #ideally, we can correct the invalid actions there
        #for i,b in enumerate(env_ids[valid_ind]):
        #    info[b]['new_contacts'] = action[-1][i]['contacts']
        valid_ind[valid_ind] = valid_a
        rewards[~valid_ind] = -1
        for i,b in enumerate(env_ids[valid_ind]):
            info[b]['new_contacts'] = action[-1][b]['contacts']
        return valid_ind,rewards,info
    def add_to_assembly(self,block:List[trimesh.Trimesh],contacts:List[dict],env_ids:List[int],hold=True,is_mortar=False):
            for i,b in enumerate(env_ids):
                self.contacts[b]+=contacts[i]

                # contacts[b] is the running cumulative list, so its length is the
                # cumulative contact count for this action. Using len() (rather than
                # len(contacts[i]) + previous) stays correct when add_to_assembly is
                # called several times per action (e.g. the block plus each mortar piece).
                self.n_contact_t[b,self.n_actions[b]]=len(self.contacts[b])
                #assert block[i].is_volume
                self.assembly_sequence[b,self.n_block_t[b]]=block[i]
                #with wp.ScopedTimer("add to list",color='cyan',use_nvtx=True):    
                self.assembly_sequence_wp[b,self.n_block_t[b]] = wp.Mesh(wp.array(block[i].vertices,dtype=wp.vec3),
                                                                            wp.array(block[i].faces.flatten(),dtype=wp.int32))
                self.assembly_sequence_wpid[b,self.n_block_t[b]] = self.assembly_sequence_wp[b,self.n_block_t[b]].id
                self.is_held[b,self.n_actions[b]:,self.n_block_t[b]]=hold
                self.is_mortar_piece[b,self.n_block_t[b]]=is_mortar
                if self.physics_on:
                    if not self.batchsolver:
                        self.physics_model[b].add_part(block[i],self.n_block_t[b],contacts=contacts[i],state=FIXED if hold else FREE)
                self.n_block_t[b]+=1
                self.n_block_anyt[b,self.n_actions[b]]=self.n_block_t[b]
            if self.physics_on:
                if self.batchsolver:
                    self.physics_model.add_part(env_ids,block,self.n_block_t[env_ids]-1, contacts=contacts,state=FIXED if hold else FREE)
            
    def pass_time(self,env_ids,steps=1):
        for i,b in enumerate(env_ids):
            self.n_contact_t[b,self.n_actions[b]]=self.n_contact_t[b,self.n_actions[b]-1]
            self.n_block_anyt[b,self.n_actions[b]]=self.n_block_t[b]
    def prepare_action(self,env_ids:Union[int,List[int]]=None):
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        valid = self.n_actions[env_ids] < self.n_block_max-1
        self.n_actions[env_ids[valid]]+=1
        return valid
    def put_block_rel(self,
                      block:List[trimesh.Trimesh],
                      support:List[int],
                      face_block:List[int],
                      face_support:List[int],
                      contact_offset:List[np.ndarray],
                      rotation_offset:List[np.ndarray],
                      direction:List[np.ndarray],
                      grip_location=None,
                      env_ids:Union[int,List[int]]=None,
                      target=None,
                      local_ref=None):
        if local_ref is None:
            local_ref = isinstance(self.graph,BFrameGraphConstructor) or isinstance(self.graph,FTBFrameGraphConstructor)
        if env_ids is None:
            env_ids = np.arange(self.batchsize)
        newbs = []
        pos = []
        valid = np.ones(self.batchsize,dtype=bool)
        if grip_location is None:
            grip_location = np.zeros((len(block),3))
        for i,b in enumerate(env_ids):
            corners_support = self.assembly_sequence[b,support[i]].vertices[self.assembly_sequence[b,support[i]].faces[face_support[i]]]
            
            longest_sides = np.argmax(np.linalg.norm(corners_support[None] - corners_support[:,None],axis=-1),axis=None)
            init_corner = min(longest_sides//3,longest_sides%3)
            longest_side_id = (init_corner,(init_corner+1)%3)
            x_axisid = (init_corner,(init_corner+2)%3)
            y_axisid = ((init_corner+1)%3,(init_corner+2)%3)
            midpoint_support = 0.5*(corners_support[longest_side_id[0]]+corners_support[longest_side_id[1]])
            x_axis = corners_support[x_axisid[0]]-corners_support[x_axisid[1]]
            y_axis = corners_support[y_axisid[0]]-corners_support[y_axisid[1]]
            ref_face = np.stack([x_axis/np.linalg.norm(x_axis),y_axis/np.linalg.norm(y_axis),np.cross(x_axis,y_axis)/np.linalg.norm(np.cross(x_axis,y_axis))]).T
            if local_ref:
                #use the principal axis of inertia of the support block as reference
                ref = self.referencial[b,support[i]]
                offset_abs = ref_face@contact_offset[i]
                #offet_abs = ref@contact_offset[i]
                
            else:
                offset_abs = contact_offset[i]
                ref = np.eye(3)
                self.referencial[b,self.n_block_t[b]] = np.eye(3)
            newb = block[i].copy()
            normal_i = ref@newb.face_normals[face_block[i]]
            
            # Define the target normal (x-axis)
            if self.n_block_t[b]>support[i] and face_support[i] < self.assembly_sequence[b,support[i]].face_normals.shape[0]:
                target_normal = -self.assembly_sequence[b,support[i]].face_normals[face_support[i]]  
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
            rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()@ref
            rotation_matrix = R.from_rotvec(rotation_offset[i] * target_normal).as_matrix()@rotation_matrix
            if local_ref:
                self.referencial[b,self.n_block_t[b]] = rotation_matrix
            # Apply rotation to the mesh
            newb.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
            newbs.append(newb)
            corners_block = newb.vertices[newb.faces[face_block[i]]]-newb.center_mass
            longest_sideb = np.argmax(np.linalg.norm(corners_block[None] - corners_block[:,None],axis=-1))
            midpoint_block = 0.5*(corners_block[longest_sideb//3]+corners_block[longest_sideb%3])
            posi = midpoint_support-midpoint_block
            posi += offset_abs
            posi -= direction[i]*self.max_dist*0.5
            pos.append(posi)
        valid,rewards,_ = self.put_block_from(newbs,pos,direction,grip_location=grip_location,slide = False,pivot=False,env_ids=env_ids[valid],target=target)
        return valid,rewards
    def equivalent_actions(self, new_contacts, local_ref=None,rotation2D=True):
        #action parameters:
        """block:List[trimesh.Trimesh],
                      support:List[int],
                      contact_offset:List[np.ndarray],
                      rotation_offset:List[np.ndarray],
                      direction:List[np.ndarray],
                      local_ref=None"""
        # returns a list of all rel_noface actions that ends up with the same contacts
        if local_ref is None:
            local_ref = isinstance(self.graph,BFrameGraphConstructor) or isinstance(self.graph,FTBFrameGraphConstructor)
        
        eq_actions = [[] for c in range(len(new_contacts))]
        for i in range(len(new_contacts)):
            if len(new_contacts[i])==0:
                continue
            for c in new_contacts[i]:
                partAID = c['partIDA']
                partBID = c['partIDB']
                offset =  self.assembly_sequence[c['batchID'], partBID].center_mass - self.assembly_sequence[c['batchID'], partAID].center_mass
                if local_ref:
                    ref = self.referencial[c['batchID'],partAID]
                    offset_3D = ref@offset
                    newori= ref@self.referencial[c['batchID'],partBID]
                else:
                    offset_3D = offset
                    ori = self.referencial[c['batchID'],partBID]
                    newori = self.referencial[c['batchID'],partBID]
                if rotation2D:
                    newori = geometry.rotmat_to_2D(newori[None])
                offset_2D = offset_3D[:2]
                eq_actions[i].append({"support_block":partAID,"center_offset":offset_2D[None],"rotation_offset":newori})
                
        return eq_actions
    def compute_objectives(self,batchid,actions,debug=False,additional_contact_reward=0):
        rewards = np.zeros(len(batchid))
        for i,b in enumerate(batchid):
            if actions[i]['valid']:
                rewards[i],self.to_cover[b],self.last_covered[b] = cover.cover_mesh(actions[i]['block'],self.to_cover[b])
                self.covered[b].append(self.last_covered[b])
                rewards[i] += additional_contact_reward*(len(actions[i]['contacts'])-1)
            else:
                rewards[i] = -1
        return rewards
    def add_ground(self,blocks:List[trimesh.Trimesh],pos=None,env_ids:Union[int,List[int]]=None):
        #with wp.ScopedTimer('add_ground',color='red',use_nvtx=True):
            valid = self.prepare_action(env_ids)
            if pos is None:
                pos = [block.center_mass for block in blocks]
            valid[valid],reward = self.put_block_abs(blocks,pos,env_ids=env_ids[valid],update_obj=False,check_contact=False)
            assert np.all(valid), "Invalid ground placement"

            self.is_ground[env_ids[valid],self.n_block_t[env_ids[valid]]-1]=True
            self.n_block_init[env_ids[valid]]+=1
            if self.physics_on:
                if not self.batchsolver:
                #with wp.ScopedTimer("Fixed part",color='yellow',use_nvtx=True):
                    for i,b in enumerate(env_ids[valid]):
                        self.physics_model[b].fix_part(self.n_block_t[b]-1)
                else:
                    self.physics_model.fix_part(env_ids[valid],self.n_block_t[env_ids[valid]]-1)
    def hold_block(self,bid:List[int]):
        self.is_held[:,self.n_actions:,bid]=True
    def leave_block(self,batch_boolar,bid:np.array):
        #TODO: slightly modify the position of the block as well ("drop the block"); if not too hard
        batchid, = np.nonzero(batch_boolar)
        valid = np.zeros(batchid.shape[0],dtype=bool)
        for i,b in enumerate(batchid):
            self.is_held[b,self.n_actions[b]:,bid[i]]=False
        if self.physics_on:
            if not self.batchsolver:
                for i,b in enumerate(batchid):
                    self.physics_model[b].leave_part(bid[i])
                fn,ft,valid = self.check_stability(batchid)
                for i,b in enumerate(batchid):
                    if valid[i]:
                        continue
                    #self.is_held[b,self.n_actions[b]+1:,bid[i]]=True
                    self.falling_speed[b, self.n_actions[b]-1,:self.n_block_t[b]*6] = self.physics_model[b].velocity(self.physics_model[b].part_states, fn[i], ft[i]).flatten()
                    self.physics_model[b].fix_part(bid[i])
                    self.falling[b, self.n_actions[b]-1] = True
            else:
                self.physics_model.leave_part(batchid, bid)
                _,_,valid = self.physics_model.simulate(batchid)
                for i,b in enumerate(batchid):
                    if not valid[i]:
                        self.physics_model.fix_part(np.array([b]), np.array([bid[i]]))
                        self.falling[b, self.n_actions[b]-1] = True
        else:
            valid = np.ones(batchid.shape[0],dtype=bool)
        return valid
    def simulate_physics(self,env_ids:List[int]):
        falling = np.zeros(len(env_ids),dtype=bool)
        if self.batchsolver:
            xn,xt,stable =self.physics_model.simulate(env_ids)
            falling[:] = ~stable
        else:
            for i,b in enumerate(env_ids):
                xn,xt,stable = self.physics_model[b].simulate()
                self.falling[b, self.n_actions[b]] = not stable
                falling[i] = not stable
                if not stable:
                    self.falling_speed[b, self.n_actions[b],:self.n_block_t[b]*6] = self.physics_model[b].velocity(self.physics_model[b].part_states,xn, xt).flatten()
        return falling
    def check_stability(self,batchid:List[int]):
        xn = [None]*len(batchid)
        xt = [None]*len(batchid)
        flags = np.ones(len(batchid),dtype=bool)
        assert self.physics_on, "Physics model is not enabled"
        for i,b in enumerate(batchid):
            xn[i],xt[i], flags[i] = self.physics_model[b].simulate()
        return xn,xt,flags
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
        """ps.register_surface_mesh("ground", 
                             np.array([[-100, -100, -0.1],
                                       [100, -100, -0.1],
                                       [100, 100, -0.1],
                                       [-100, 100, -0.1]]), 
                             np.array([[0, 1, 2],
                                       [0, 2, 3]]), 
                             color=(0.05, 0.05, 0.05))"""
        # blocks
        if self.physics_on:
            if self.autoleave:
                if self.autoleave_post:
                    if self.falling[self.batchi,n_actions_id] and self.t >0.5:
                        assembly_group = render_fall(self.assembly_sequence[self.batchi,:n_blocks_id+1],
                                                    self.is_ground[self.batchi,:n_blocks_id+1],
                                                    self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+1)],
                                                    self.falling_speed[self.batchi,n_actions_id,:n_blocks_id*6+6],
                                                    2*(self.t-0.5),
                                                    tol=self.tol,
                                                    speed_scale=np.power(10,-self.expscale))
                    else:
                        assembly_group = render_struct(self.name,self.assembly_sequence[self.batchi, :n_blocks_id],
                                                       self.is_ground[self.batchi],
                                                       (self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id)]|
                                                            (self.falling[self.batchi,n_actions_id-1]*self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id-1)])),
                                                       tol=self.tol,marked=self.marked[self.batchi,:n_blocks_id],is_mortar=self.is_mortar_piece[self.batchi,:n_blocks_id])
                        if self.t>0:
                            try:
                                action = self.action_t[self.batchi,n_actions_id,:self.n_actions_steps[self.batchi,n_actions_id]]
                            except:
                                pass
                            else:
                                action_group = render_action(action,2*self.t if self.falling[self.batchi,n_actions_id] else self.t,structure=self.assembly_sequence[self.batchi,:n_blocks_id],
                                                            name='action',tol=self.tol,mu=self.mu)

                else:
                    if self.falling[self.batchi,n_actions_id+1] and self.t >0.5:
                        assembly_group = render_fall(self.assembly_sequence[self.batchi,:n_blocks_id+1],
                        self.is_ground[self.batchi,:n_blocks_id+1],
                        self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],
                        self.falling_speed[self.batchi,n_actions_id+1,:n_blocks_id*6+6],
                        2*(self.t-0.5),
                        tol=self.tol,
                        speed_scale=np.power(10,-self.expscale))
                    else:
                        assembly_group = render_struct(self.name,self.assembly_sequence[self.batchi, :n_blocks_id],
                                                       self.is_ground[self.batchi],
                                                       self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],
                                                       tol=self.tol,marked=self.marked[self.batchi,:n_blocks_id],is_mortar=self.is_mortar_piece[self.batchi,:n_blocks_id])
                        try:
                            action = self.action_t[self.batchi,n_actions_id,:self.n_actions_steps[self.batchi,n_actions_id]]
                        except:
                            pass
                        else:
                            action_group = render_action(action,2*self.t,structure=self.assembly_sequence[self.batchi,:n_blocks_id],
                                                        name='action',tol=self.tol,mu=self.mu)

            else:
                if self.falling[self.batchi,n_actions_id+1] and self.t >0.1:
                    assembly_group = render_fall(self.assembly_sequence[self.batchi,:n_blocks_id+1],
                    self.is_ground[self.batchi,:n_blocks_id+1],
                    self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],
                    self.falling_speed[self.batchi,n_actions_id+1,:n_blocks_id*6+6],
                    self.t,
                    tol=self.tol,
                    speed_scale=np.power(10,-self.expscale))
                else:
                    assembly_group = render_struct(self.name,self.assembly_sequence[self.batchi, :n_blocks_id],
                                                   self.is_ground[self.batchi],
                                                   self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],
                                                   tol=self.tol,marked=self.marked[self.batchi,:n_blocks_id],is_mortar=self.is_mortar_piece[self.batchi,:n_blocks_id])
                    try:
                        action = self.action_t[self.batchi,n_actions_id,:self.n_actions_steps[self.batchi,n_actions_id]]
                    except:
                        pass
                    else:
                        action_group = render_action(action,self.t,structure=self.assembly_sequence[self.batchi,:n_blocks_id],
                                                    name='action',tol=self.tol,mu=self.mu)
        else:
            assembly_group = render_struct(self.name,self.assembly_sequence[self.batchi, :n_blocks_id],
                                           self.is_ground[self.batchi],
                                           self.is_held[self.batchi,min(self.n_block_max-1,n_actions_id+int(self.t>0))],
                                           tol=self.tol,marked=self.marked[self.batchi,:n_blocks_id],is_mortar=self.is_mortar_piece[self.batchi,:n_blocks_id])
            try:
                action = self.action_t[self.batchi,n_actions_id,:self.n_actions_steps[self.batchi,n_actions_id]]
            except:
                pass
            else:
                action_group = render_action(action,self.t,structure=self.assembly_sequence[self.batchi,:n_blocks_id],
                                            name='action',tol=self.tol,mu=self.mu)                
        objectives_group = render_objectives.render_cover(self.to_cover[self.batchi])
        if self.use_forces:
            try:
                contacts = self.contacts[self.batchi][:self.n_contact_t[self.batchi,n_actions_id]]
            except:
                pass
            else:
                forces_group = render_forces(contacts,normals=None,frictions=None,tol=self.tol)
        if self.graph_mode:
            for block in assembly_group.get_child_structure_names():
                try:
                    ps.get_surface_mesh(block).set_transparency(0.2)
                except:
                    ps.get_curve_network(block).set_enabled(False)
            try:
                objectives_group.set_enabled(False)
            except:
                pass
            graph = self.current_graph().index_select([self.batchi])[0]
            graph_group = render_graph.render_graph(graph)
        else:
            #the following does not work on windows
            try:
                for block in assembly_group.get_child_structure_names():
                    try:
                        ps.get_surface_mesh(block).set_transparency(1)
                    except:
                        ps.get_curve_network(block).set_enabled(True)
                try:
                    objectives_group.set_enabled(True)
                except:
                    pass
            except:
                pass
    def render_callback(self):
        change1, self.batchi = psim.SliderInt("batch", self.batchi, 0, self.batchsize-1)
        change2, self.tstep = psim.SliderInt("istep", self.tstep, 0, self.n_actions[self.batchi]-self.n_block_init[self.batchi])
        change3, self.t = psim.SliderFloat("time", self.t, 0, 1)
        _, self.expscale = psim.SliderFloat("scale", self.expscale, -5, -1)
        change_forces, self.use_forces = psim.Checkbox("Show forces", self.use_forces) 
        change_graph,self.graph_mode = psim.Checkbox("Graph mode", self.graph_mode)
        change = change1 or change2 or change_forces or change3 or change_graph
       
        if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
            self.batchi = self.batchi - 1
            change = True

        if psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
            self.batchi = self.batchi + 1
            change = True
        if psim.IsKeyPressed(psim.ImGuiKey_UpArrow):
            self.t = self.t + np.power(10,self.expscale)
            change = True
        if psim.IsKeyPressed(psim.ImGuiKey_DownArrow):
            self.t = self.t - np.power(10,self.expscale)
            change = True
        if change:
            self.render()
        
        
    def current_graph(self):
        #with wp.ScopedTimer("graph",color='purple',use_nvtx=True):
            return self.graph.compute_full_graph_batched(self.n_block_t,
                                                         self.assembly_sequence,
                                                         self.contacts,
                                                         self.is_ground,
                                                         self.is_held[np.arange(self.batchsize),self.n_actions,:],
                                                         self.to_cover,
                                                         covered = self.covered,
                                                         referencial=self.referencial)
        #return self.graph.current_graph(self.assembly_sequence,self.contacts,self.is_ground,self.is_held[np.arange(self.batchsize),self.n_actions,:],self.to_cover)
    def reset(self,batches):
        if all(batches):
            self.assembly_sequence = np.zeros((self.batchsize,self.n_block_max),dtype=object)
            self.is_ground = np.zeros((self.batchsize,self.n_block_max),dtype=bool)
            self.is_held = np.zeros((self.batchsize,self.n_block_max,self.n_block_max),dtype=bool)
            self.assembly_sequence[:]=None
            self.assembly_sequence_wp = np.zeros((self.batchsize,self.n_block_max),dtype=object)
            self.assembly_sequence_wpid = np.zeros((self.batchsize,self.n_block_max),dtype=np.uint64)
            self.referencial = np.tile(np.eye(3),(self.batchsize,self.n_block_max,1,1))
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
            self.to_cover = np.array([Polygon([]) for i in range(self.batchsize)])
            self.covered = [[] for i in range(self.batchsize)]
            self.cover_dir = np.tile([0,0,1],(self.batchsize,1))
            if self.physics_on:
                if self.batchsolver:
                    self.physics_model.reset(np.arange(self.batchsize))
                else:
                    [model.reset() for model in self.physics_model]
            self.falling = np.zeros((self.batchsize,self.n_block_max+1),dtype=bool)
            self.falling_speed = np.zeros((self.batchsize,self.n_block_max+1,self.n_block_max*6))
        
        else:
            self.assembly_sequence[batches] = None
            self.assembly_sequence_wp[batches,:] = None
            self.assembly_sequence_wpid[batches,:] = np.zeros((batches.sum(),self.n_block_max),dtype=np.uint64)
            self.referencial[batches,:] = np.eye(3)[None,None]
            self.is_ground[batches] = False
            self.is_held[batches] = False
            self.n_block_t[batches] = 0
            self.n_block_init[batches] = 0
            batches_id = np.nonzero(batches)[0]
            for i in batches_id:
                self.contacts[i] = []
                self.to_cover[i] = Polygon([])
                self.covered[i] = []
                if self.physics_on and not self.batchsolver:
                    self.physics_model[i].reset()
            if self.physics_on and self.batchsolver:
                self.physics_model.reset(batches_id)
            self.cover_dir[batches] = np.tile([0,0,1],(batches.sum(),1))
            self.n_contact_t[batches] = np.zeros((self.n_block_max),dtype=int)
            #action arrays
            self.n_actions[batches] = 0
            self.action_t[batches] = None
            self.n_actions_steps[batches] = 0
            if self.physics_on:
                self.falling[batches] = False
                self.falling_speed[batches] = 0
            
    def check_success(self,success_criteria:Literal['cover'] = 'cover'):
        if success_criteria == 'cover':
            covered = np.array([self.to_cover[i].area < self.tol for i in range(self.batchsize)])
            #no blocks should be held
            held = np.array([np.any(~self.is_ground[i,:self.n_block_t[i]] & self.is_held[i,self.n_actions[i]:,:self.n_block_t[i]]) for i in range(self.batchsize)])
            return covered & (~held)
        else:
            warnings.warn(f"Unknown success criteria {success_criteria}, returning False")
            return np.zeros(self.batchsize,dtype=bool)
    def in_building_area(self,new_blocks):
        valid = np.ones(len(new_blocks),dtype=bool)
        for i,new_block in enumerate(new_blocks):
            valid[i] = np.logical_and(np.all(self.building_area[None,0,:] >= new_block.vertices),
                                      np.all(self.building_area[None,1,:] <= new_block.vertices)) 
        return valid 
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
