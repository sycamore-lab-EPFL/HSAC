import copy
from typing import List
import numpy as np 
import warnings
from shapely import MultiPoint, Polygon
import shapely
from simulator.cont3Dsimulator.utils.geometry import closest_point_in_hull, isinhull, make_hull, project_vectors,plot_hull
from ..physics import single_block
import torch
import trimesh
import matplotlib.pyplot as plt
#from .placement_warp import check_contact_batch_wrapper, translate_toA_wrapper
from .placement_warp_static import init_kernels
from .placement_mortar_warp_static import init_kernels as init_kernels_mortar
import warp as wp
import wandb
if wandb.run is not None:
    translate_toAll_wrapper, check_contact_batch_wrapper = init_kernels(wandb.config.n_batch,wandb.config.max_blocks,8,36)
    check_contact_mortar_batch_wrapper = init_kernels_mortar(wandb.config.n_batch,wandb.config.max_blocks,8,36)
else:
    print("Initializing warp with default values: 16,200,8,36")
    translate_toAll_wrapper, check_contact_batch_wrapper = init_kernels(16,200,8,36)
    check_contact_mortar_batch_wrapper = init_kernels_mortar(16,200,8,36)
def put_block_abs_batched(new_block:List[trimesh.Trimesh],
                          old_blocks:List[List[trimesh.Trimesh]],
                          n_old_blocks:np.ndarray,
                          coords:np.ndarray,
                          tol=1e-3,
                          identical_tol = 1e-3,
                          angle_tol=1e-4,
                          old_blocks_ids=None,
                          old_blocks_wps=None,
                          env_ids=None,
                          check_contact=True,
                          use_mortar=False,
                          mortar_thickness=0.1
                          ):
    new_blocks = []
    assert old_blocks is None, 'Deprecated, use old_blocks_ids and old_blocks_wps'
    if env_ids is None:
        env_ids = np.arange(len(new_block))
    for i,b in enumerate(env_ids):
        if new_block is None:
            return []
        partB=new_block[i]#copy.deepcopy(new_block[i])
        partB.apply_translation(coords[i]-partB.center_mass)
        new_blocks.append(partB)
        contacts = []
        contacts_degen = []
        valid = True #The method always return a valid placement
    if check_contact:
        if use_mortar:
            contacts,contacts_degen,mortar_blocks = check_contact_mortar_batch_wrapper(old_blocks_wps,old_blocks_ids, new_blocks,n_old_blocks,
                                                                         tol=tol,identical_tol=identical_tol,angle_tol=angle_tol,env_ids=env_ids,mortar_thickness=mortar_thickness)
            action_descs = []
            for i,partB in enumerate(new_blocks):
                action_descs.append({'contacts':contacts[i],
                                     'mortar_blocks':mortar_blocks[i],
                            'contacts_degen':contacts_degen[i],
                            'dir':np.zeros(3),
                            'dist':0,
                            'cm':coords[i],
                            'block':partB.copy(),
                            'valid':valid,
                            'force':np.zeros(3),
                            'tot_dist':0,
                            'angle':0,
                            'axis':np.zeros(3),
                            'pivot':np.zeros(3)})
        else:
            contacts,contacts_degen = check_contact_batch_wrapper(old_blocks_wps,old_blocks_ids, new_blocks,n_old_blocks,
                                                                tol=tol,identical_tol=identical_tol,angle_tol=angle_tol,env_ids=env_ids)
            action_descs = []
            for i,partB in enumerate(new_blocks):
                action_descs.append({'contacts':contacts[i],
                            'contacts_degen':contacts_degen[i],
                            'dir':np.zeros(3),
                            'dist':0,
                            'cm':coords[i],
                            'block':partB.copy(),
                            'valid':valid,
                            'force':np.zeros(3),
                            'tot_dist':0,
                            'angle':0,
                            'axis':np.zeros(3),
                            'pivot':np.zeros(3)})
    else:
        action_descs = []
        for i,partB in enumerate(new_blocks):
                action_descs.append({'contacts':[],
                                     'mortar_blocks':[],
                            'contacts_degen':[],
                            'dir':np.zeros(3),
                            'dist':0,
                            'cm':coords[i],
                            'block':partB.copy(),
                            'valid':valid,
                            'force':np.zeros(3),
                            'tot_dist':0,
                            'angle':0,
                            'axis':np.zeros(3),
                            'pivot':np.zeros(3)})
    return action_descs
def put_block_abs(new_block:trimesh.Trimesh,
                  old_blocks:List[trimesh.Trimesh],
                  coords:np.ndarray,
                  tol=1e-3,
                  identical_tol = 1e-3,
                  angle_tol=1e-4,
                  old_blocks_wp=None,
                  old_blocks_id=None,
                  max_blocks=100):
    if new_block is None:
        return []
    partB=copy.deepcopy(new_block)
    partB.apply_translation(coords-partB.center_mass)
    if old_blocks_id is None:
        old_blocks_wp = [wp.Mesh(wp.array(partA.vertices,dtype=wp.vec3), wp.array(partA.faces.flatten(),dtype=wp.int32)) for partA in old_blocks]
        old_blocks_id = [partA.id for partA in old_blocks_wp]
    #TODO filter the other blocks maybe            
    """area = trimesh.convex.convex_hull(np.vstack([convex.vertices, convex.apply_translation(force[i]*(max_range-tot_dist)/np.linalg.norm(force[i])).vertices]))
    col, list,data = self.collision[i].in_collision_single(area,return_names=True,return_data=True)"""
    contacts = []
    contacts_degen = []
    valid = True
    
    contacts,contacts_degen = check_contact_batch_wrapper(np.array([old_blocks_wp]),np.array([old_blocks_id]), [partB],np.array([len(old_blocks_id)]),tol=tol,identical_tol=identical_tol,angle_tol=angle_tol)
    action_desc = {'contacts':contacts[0],
                   'contacts_degen':contacts_degen[0],
                   'dir':np.zeros(3),
                   'dist':0,
                   'cm':coords,
                   'block':partB.copy(),
                   'valid':valid,
                   'force':np.zeros(3),
                   'tot_dist':0,
                   'angle':0,
                   'axis':np.zeros(3),
                   'pivot':np.zeros(3)}
    return [action_desc]
def put_block_from_batched(last_action:List[dict],
                   old_blocks_wp:np.ndarray,
                   old_blocks_ids:np.ndarray,
                   n_old_blocks:np.ndarray,
                   direction,
                   max_range=5,
                   tol=1e-5,
                   identical_tol = 1e-3,
                   angle_tol=1e-4,
                   use_mortar=False,
                   mortar_thickness=0.1,
                   env_ids=None):
    if env_ids is None:
        env_ids = np.arange(len(last_action))
    partBs=[lab['block'].copy() for lab in last_action]
    cms = [partB.center_mass for partB in partBs]
    actions = []
    dire = direction/np.linalg.norm(direction,axis=-1)[:,None]

    mindist,mindistover = translate_toAll_wrapper(old_blocks_ids,n_old_blocks,partBs,direction,tol=tol,max_dist=max_range,env_ids=env_ids)
    #mindist[mindistover > mindist] = np.min(np.stack([mindistover[mindistover > mindist],mindist[mindistover > mindist]+3*tol],axis=0),axis=0)

    #mindist[mindistover > mindist] = mindistover[mindistover > mindist]

    new_pos = [cm+dire[i]*(mindist[i]) for i,cm in enumerate(cms)]
    action = put_block_abs_batched(partBs,None,n_old_blocks,new_pos,old_blocks_wps=old_blocks_wp,old_blocks_ids=old_blocks_ids,
                                    tol=tol,identical_tol=identical_tol,angle_tol=angle_tol,env_ids=env_ids,use_mortar=use_mortar,mortar_thickness=mortar_thickness)
    for i,b in enumerate(env_ids):
        action[i]['dir']=dire[i]
        action[i]['dist'] = mindist[i]
        action[i]['tot_dist'] = mindist[i]
        action[i]['force'] = direction[i]
    print("placement ok")
    return action
def put_block_from(last_action,
                   old_blocks:List[trimesh.Trimesh],
                   direction,
                   max_range=5,
                   tol=1e-5,
                   identical_tol = 1e-3,
                   angle_tol=1e-4,
                   oob=None):
    
    partB=last_action['block'].copy()
    cm = partB.center_mass
    actions = []
    dire = direction/np.linalg.norm(direction)
    #TODO filter the other blocks maybe          
    if oob is not None:
        pass
    """area = trimesh.convex.convex_hull(np.vstack([convex.vertices, convex.apply_translation(force[i]*(max_range-tot_dist)/np.linalg.norm(force[i])).vertices]))
    col, list,data = self.collision[i].in_collision_single(area,return_names=True,return_data=True)"""
    #assume collision
    """ if not col:
        valid=False"""
    min_dist = max_range
    min_dist_over = max_range
    for partIDA,partA in enumerate(old_blocks):
        mindistAB,mindistoverAB = translate_toA_wrapper(partA,partB,direction,tol=tol,max_dist=max_range) # type: ignore
        if mindistAB < min_dist:
            min_dist = mindistAB
        if mindistoverAB < min_dist_over:
            min_dist_over = mindistoverAB
    if min_dist_over > min_dist:
        min_dist = min(min_dist_over,min_dist+3*tol)
        #min_dist = min(min_dist_over,min_dist)

    new_pos = cm+dire*min_dist
    action, = put_block_abs(partB,old_blocks,new_pos,tol=tol,identical_tol=identical_tol,angle_tol=angle_tol)
    action['dir']=dire
    action['dist'] = min_dist
    action['tot_dist'] = min_dist
    action['force'] = direction
    actions.append(action)
    return actions
def put_block_slide(last_action,
                    old_blocks:List[trimesh.Trimesh],
                    force_init,mu,max_range=5,max_steps = 100,tol=1e-3,angle_tol=1e-4,identical_tol=None,d=None):
    if identical_tol is None:
        identical_tol = tol/1000
    valid = last_action['valid']
    
    tot_dist  = last_action['tot_dist']
    actions = []
    if tot_dist>=max_range:
        return []
    n_steps = 0
    force = force_init.copy()
    f_i = force_init.copy()
    offset = np.zeros(3)
    while last_action['valid'] and np.linalg.norm(force)>tol and tot_dist<max_range and n_steps<max_steps:
        n_steps += 1
        if last_action['contacts'] is None or len(last_action['contacts'])==0:
            force = force_init.copy()
        else:
            force,valid = single_block.slide_direction(np.vstack([item['points'] for item in last_action['contacts']]),
                                                 np.vstack([np.tile(item['normal'],(item['points'].shape[0],1))
                                                            for item in last_action['contacts']]),
                                                        mu,
                                                        force_init,
                                                        n_tangents=4,tol=angle_tol)
            if not valid:
                actions += put_block_abs(last_action['block'],old_blocks,last_action['cm'],tol=tol,identical_tol=identical_tol,angle_tol=angle_tol)
                actions[-1]['valid'] = False
                return actions
        if np.linalg.norm(force)>tol:
            
            action_i = put_block_from(last_action,old_blocks,force,tol=tol,max_range=max_range,identical_tol = identical_tol,angle_tol = angle_tol)
            last_action = action_i[-1]

            

            last_action['tot_dist']=tot_dist+last_action['tot_dist']
            tot_dist = last_action['tot_dist']
            
            last_action['valid']= last_action['valid'] and tot_dist<max_range
            last_action['force']= force
            actions.append(last_action)
    if len(actions) >0 and (n_steps>=max_steps) or not valid:
        actions = actions[:max_steps]
        actions[-1]['valid']=False

    return actions
def put_block_pivot(last_action,old_blocks,force,grip_loc,tol=1e-5,angle_tol=1e-2,mu=0.5):
    if len(last_action['contacts'])==0 or not last_action['valid']:
        return []
    contact_pointsA = np.vstack([item['pointsA'] for item in last_action['contacts']])
    contact_pointsB = np.vstack([item['pointsB'] for item in last_action['contacts']])
    contact_points =  np.vstack([item['points'] for item in last_action['contacts']])
    contact_normals = np.vstack([np.tile(item['normal'],(item['points'].shape[0],1)) for item in last_action['contacts']])
    dist_tol = np.linalg.norm(contact_pointsB-contact_pointsA,axis=-1)
    diff_tol = dist_tol - np.min(dist_tol)
    #to_consider =  (diff_tol<tol/4) # | (dist_tol<tol/2) 
    to_consider = np.ones_like(diff_tol, dtype=bool)
    contact_points = contact_points[to_consider]
    contact_pointsA = contact_pointsA[to_consider]
    contact_pointsB = contact_pointsB[to_consider]
    contact_normals = contact_normals[to_consider]
    turning_block = last_action['block'].copy()
    grip_loc = grip_loc + turning_block.center_mass
    rotation_point,axis,idxpoint = single_block.rotation_direction(contact_points,contact_normals,force,grip_loc,tol=tol,mu=mu*1.1)
    if np.linalg.norm(axis)<tol/100:
        return []
    min_angle = np.pi
    offset_contact = np.linalg.norm(contact_pointsB[idxpoint] - contact_pointsA[idxpoint])
    rotation_pointA = contact_pointsA[idxpoint]
    rotation_pointB = contact_pointsB[idxpoint]
    for partIDA,old_block in enumerate(old_blocks):
        old_blockp = old_block.copy()
        old_blockp.apply_translation(-old_blockp.center_mass)
        min_angleAB,inter_point,dirs = contact_angle_rotate_fc(old_block,turning_block,rotation_pointA,axis,tol=offset_contact*2)
        min_angleBA,inter_point,dirs = contact_angle_rotate_fc(turning_block,old_block,rotation_pointB,-axis,tol=offset_contact*2)
        min_angleee,inter_point,dirs = contact_angle_rotate_ee(old_block,turning_block,rotation_pointB,axis,tol=offset_contact)
        #TODO optimize
        min_angleeeB,inter_point,dirs = contact_angle_rotate_ee(turning_block,old_block,rotation_pointA,-axis,tol=offset_contact)

        min_angle = min(min_angle,min_angleAB,min_angleBA,min_angleee,min_angleeeB)

    rotation = trimesh.transformations.rotation_matrix(min_angle, axis,point = rotation_point)
    turning_block.apply_transform(rotation)

    action, = put_block_abs(turning_block,old_blocks,turning_block.center_mass,tol=tol,angle_tol = angle_tol)
    if min_angle > np.pi-tol:
        action['valid']=False
    action['angle']=min_angle
    action['axis'] = axis
    action['pivot'] = rotation_point
    action['dist'] = min_angle*np.linalg.norm(rotation_point-grip_loc)
    action['tot_dist'] = last_action['tot_dist']+min_angle*np.linalg.norm(rotation_point-grip_loc)
    action['force'] = last_action['force']
    return [action]

def check_contact(partA,partB,tol=1e-5):
    contacts = []
    penetrating_fc, locations_fc, normals_fc = contact_face_corner(partA,partB,tol=tol)
    penetrating_cf, locations_cf, normals_cf = contact_face_corner(partB,partA,tol=tol)
    normals_cf = -normals_cf
    #normals_cfc = [-n for n in normals_cfc]

    penetrating_e,  locations_e,  normals_e,normals_degen =  contact_edges(partA,partB,tol=tol)
    
    locations = np.vstack([locations_fc,locations_cf,locations_e])
    #normals_cf = -normals_cf
    normals = np.vstack([normals_fc,normals_cf,normals_e])
    #normals = normals_cf
    penetrating = penetrating_cf or penetrating_e or penetrating_fc
    #locations = np.vstack([locations_fc])
    #normals = np.vstack([normals_fc])
    rn = np.round(normals,6)
    nu,nidx = np.unique(rn,return_inverse=True,axis = 0)
    for k,n in enumerate(nu):
        contact_points = locations[nidx==k]
        contact_points = np.unique(np.round(contact_points,int(-np.log10(tol)+1)),axis=0)
        if contact_points.shape[0]>3:
            pass
            #contact_points = trimesh.convex.convex_hull(contact_points).vertices
        contacts.append({"points":contact_points,
                        "partIDA":None,  
                        "partIDB":None, 
                        "normal":n})
            
    return penetrating,contacts,normals_degen

def translate_toA(partA,partB,dir,tol=1e-5,max_dist=np.inf,include_parallel=True):
    locationsAB, index_rayAB, index_triAB  = partA.ray.intersects_location(ray_origins=partB.vertices, ray_directions=[dir]*partB.vertices.shape[0])
    if locationsAB.shape[0]>0:
        dists = np.linalg.norm(locationsAB-partB.vertices[index_rayAB],axis=1)

        dists_AB = np.min(dists[dists>tol],initial=max_dist)
    else:
        dists_AB = max_dist
    locationsBA, index_rayBA, index_triBA  = partB.ray.intersects_location(ray_origins=partA.vertices, ray_directions=[-dir]*partA.vertices.shape[0])
    if locationsBA.shape[0]>0:
        dists = np.linalg.norm(locationsBA-partA.vertices[index_rayBA],axis=1)
        dists_BA = np.min(dists[dists>tol],initial=max_dist)
    else:
        dists_BA = max_dist
    dist_e_all = ray_edge(partA,partB,dir,tol=tol,max_dist=max_dist,include_degen=include_parallel)
    dists_e = np.min(dist_e_all[dist_e_all>tol],initial=max_dist)
    min_dist = min(dists_AB,dists_BA,dists_e)
    return min_dist

def contact_angle_rotate_fc(partA,partB,point,axis,tol=1e-5):
    #concider the circle made by each vertice of B
    axis  = axis/np.linalg.norm(axis)
    vertices_b = partB.vertices-point
    R2B = np.square(vertices_b).sum(-1) - np.square(np.einsum("jk,k->j",vertices_b,axis))

    #the intersection will be on the line of interection of face of A and circle made by vertice of B
    parallel,line_point,line_dir = plane_intersection(np.tile(axis,(vertices_b.shape[0],1)),partB.vertices,partA.face_normals,partA.triangles[:,0]+partA.face_normals*tol/2)
    idxB,idxA = np.nonzero(~parallel)
    
    line_dir = line_dir[~parallel]/np.linalg.norm(line_dir[~parallel],axis=-1)[:,None]
    
    point_in_plane =  partB.vertices[idxB]-np.einsum("ik,k->i",partB.vertices[idxB]-point,axis)[:,None]*axis[None]
    d0 = line_point[~parallel]-point
    prod = np.einsum("ik,ik->i",line_dir,d0)
    delta = np.square(prod) - (np.einsum("ik,ik->i",d0,d0)-np.einsum("ik,ik->i",partB.vertices[idxB]-point,partB.vertices[idxB]-point))#R2B[idxB])
    #delta = np.square(prod) - np.eisum()
    #reach
    inreach = delta>=0
    
    #get the list of possible intersection points
    lambda1 = (-prod[inreach] + np.sqrt(delta[inreach]))
    lambda2 = (-prod[inreach] - np.sqrt(delta[inreach]))
    intersection_points = np.vstack([line_point[~parallel][inreach]+lambda1[:,None]*line_dir[inreach],
                                     line_point[~parallel][inreach]+lambda2[:,None]*line_dir[inreach]])
    idxA  = np.tile(idxA[inreach],2)
    idxB = np.tile(idxB[inreach],2)
    #find if the intersection is in the triangle
    
    triangles = partA.triangles[idxA]
    # Get the three vertices of each triangle 
    A0 = triangles[:, 0, :]-intersection_points
    A1 = triangles[:, 1, :]-intersection_points
    A2 = triangles[:, 2, :]-intersection_points
    # sides of the triangles
    A01 = A1 - A0
    A12 = A2 - A1
    A20 = A0 - A2

    # Area vectors
    area_0 = np.cross(A0, A01,axis=-1)
    area_1 = np.cross(A1, A12,axis=-1)
    area_2 = np.cross(A2, A20,axis=-1)
    isin = ((np.einsum("ik,ik->i",area_0,partA.face_normals[idxA]) > -tol) & 
            (np.einsum("ik,ik->i",area_1,partA.face_normals[idxA]) > -tol) &
            (np.einsum("ik,ik->i",area_2,partA.face_normals[idxA]) > -tol)
            )
    # only consider where point projects inside
    if np.any(isin):
        ori = partB.vertices[idxB][isin]
        new_pos = intersection_points[isin]
        dist = np.linalg.norm(new_pos-ori,axis=-1)
        center = np.einsum("ik,k->i",new_pos-point,axis)[:,None]*axis[None]+point
        center_ori = np.einsum("ik,k->i",ori-point,axis)[:,None]*axis[None]+point
        #remove points that are already in contact
        moving = dist>tol/4
        new_pos = new_pos[moving]
        ori = ori[moving]
        center = center[moving]
        #tolangle = np.arcsin(tol/2/np.linalg.norm(ori-center,axis=-1))
        sign = np.clip(np.einsum("ik,k->i",np.cross(ori-center,new_pos-center,axis=-1),axis),-1,1)
        cos = np.clip(np.einsum("ik,ik->i",ori-center,new_pos-center)
                      /np.einsum("ik,ik->i",ori-center,ori-center),-1,1)
        angle = np.arccos(cos)+np.pi*(sign<0)
        
        min_angle = np.min(angle,initial=np.pi)
        
    #return the min angle to rotate
    else:
        min_angle = np.pi
    #min_angle=0
    return min_angle,0,0#)*R2B[idxB,None]
    return min_angle#,new_pos[moving],np.tile(line_dir[inreach],(2,1))[isin]#)*R2B[idxB,None]
def contact_angle_rotate_ee(partA,partB,point,axis,tol=1e-5,tol_angle=1e-5,dmin=None):
    dmin = dmin or tol/4
    #concider the circle made by each vertice of B
    axis  = -axis/np.linalg.norm(axis)
    
    base_static = partA.vertices[partA.edges_unique[:,0]]-point[None]
    edge_static = partA.vertices[partA.edges_unique[:,1]]-partA.vertices[partA.edges_unique[:,0]]
    #tolaxis = np.cross(base_static,axis)
    #tolaxisn = tolaxis/np.linalg.norm(tolaxis,axis=-1)[:,None]
    #add a tolerance to the base edge
    base_moving = partB.vertices[partB.edges_unique[:,0]]-point[None]
    edge_moving = partB.vertices[partB.edges_unique[:,1]]-partB.vertices[partB.edges_unique[:,0]]
    #convert the vectors in cylindrical coordinates
    zA0 = np.einsum("jk,k->j",base_static,axis)[:,None]
    zAe = np.einsum("jk,k->j",edge_static,axis)[:,None]
    zB0 = np.einsum("jk,k->j",base_moving,axis)[None]
    zBe = np.einsum("jk,k->j",edge_moving,axis)[None]
    RB0 = base_moving-zB0[:,:,None]*axis[None,None]
    RA0 = base_static[:,None]-zA0[:,:,None]*axis[None,None]
    #RA0 = np.sqrt(np.square(base_static).sum(-1)[:,None]-np.square(zA0))
    #Approximation that needs to be fixed
    #RBe = np.sqrt(np.square(edge_moving).sum(-1)-np.square(zBe))
    #RAe = np.sqrt(np.square(edge_static).sum(-1)[:,None]-np.square(zAe))
    RBe = edge_moving-zBe[:,:,None]*axis[None,None]
    RAe = edge_static[:,None]-zAe[:,:,None]*axis[None,None]
    #find tA / tB using zA = zB
    #TODO handle the zAe == 0 case
    #use this line if we launch the function with partA and partB switched
    not_parallel =(np.abs(zBe)<=(np.abs(zAe))) & (np.abs(zAe)>tol) #tol+np.abs(zBe)<(np.abs(zAe))
    #use this line if no other check are done
    #not_parallel =np.tile(tol<(np.abs(zAe)),(1,zBe.shape[1])) #tol+np.abs(zBe)<(np.abs(zAe))
    paral = ~not_parallel.flatten()
    deltae = np.zeros((zAe.shape[0],zBe.shape[1]))
    delta0 = np.zeros((zAe.shape[0],zBe.shape[1]))

    deltae[not_parallel] = np.tile(zBe,(zAe.shape[0],1))[not_parallel]/np.tile(zAe,(1,zBe.shape[1]))[not_parallel]
    delta0[not_parallel] = (np.tile(zB0,(zAe.shape[0],1))[not_parallel]-np.tile(zA0,(1,zB0.shape[1]))[not_parallel])/np.tile(zAe,(1,zBe.shape[1]))[not_parallel]
    
    #deltae = zBe/zAe
    #delta0 = (zB0-zA0)/zAe

    RAep = deltae[:,:,None]*RAe
    RA0p = RAe*delta0[:,:,None]+RA0
    #switch A and B
    if paral.any():
        #deltaeB = 0 for all
        #deltaeB = zAe[paral]/zBe
        #delta0B = (zA0[paral]-zB0)/zBe
        pass
    #compute the delta of the radius equality
    diff_cross = np.einsum("ijk,ijk->ij",RAep,RA0p)-np.einsum("ijk,ijk->ij",RB0,RBe)
    diff_0 = np.einsum("ijk,ijk->ij",RA0p,RA0p)-np.einsum("ijk,ijk->ij",RB0,RB0)
    diff_e = np.einsum("ijk,ijk->ij",RAep,RAep)-np.einsum("ijk,ijk->ij",RBe,RBe)
    delta_radius = np.square(diff_cross)-diff_0*diff_e

    valid = (delta_radius > 0) & (np.abs(diff_e)>tol)
    idxA,idxB = np.nonzero(valid)
    tB = np.stack([(-diff_cross[valid]+np.sqrt(delta_radius[valid]))/diff_e[valid],
                   (-diff_cross[valid]-np.sqrt(delta_radius[valid]))/diff_e[valid]])
    
    tA = tB*deltae[valid]+delta0[valid]

    inB = (tB > 0) & (tB<1)
    inA = (tA > 0) & (tA<1)

    isin = inA & inB
    
    if np.any(isin):
        idxparity,idx_v = np.nonzero(isin)
        idxA = idxA[idx_v]
        idxB = idxB[idx_v]

        new_pos = base_static[idxA]+tA[isin,None]*edge_static[idxA]
        ori =     base_moving[idxB]+tB[isin,None]*edge_moving[idxB]

        circle_center = np.einsum("k,ik->i",axis,new_pos)[:,None]*axis

        #new_pos = intersection_points
        dist = np.linalg.norm(new_pos-ori,axis=-1)
        r_new = np.linalg.norm(circle_center-new_pos,axis=-1)
        r_old = np.linalg.norm(circle_center-ori,axis=-1)
        if np.any(np.abs(r_new-r_old)>tol):
            warnings.warn("Both points should be on the same circle")
            #contact_angle_rotate_ee(partA,partB,point,axis,tol=tol)
            r_old[np.abs(r_new-r_old)<tol] = 0
        #remove points that are already in contact
        moving = (r_new > tol/2) #& (dist>tol)
        delta_angle = np.arcsin(np.clip(tol/2/r_new[moving],-1+1e-4,1-1e-4))
        sign = np.clip(np.einsum("ik,k->i",np.cross(ori-circle_center,new_pos-circle_center,axis=-1),axis),-1,1)
        #cos = -np.ones(ori.shape[0])
        cos = np.clip(np.einsum("ik,ik->i",ori-circle_center,new_pos-circle_center)[moving]
                      /np.square(r_old[moving]),-1,1)
        angle = np.arccos(cos)+np.pi*(sign[moving]>0)
        actually_moving = angle-delta_angle>0#dist[moving]*np.cos(tolangle/2)-np.cos(angle/2)*r_new[moving]*2*np.sin(tolangle/2) >= r_new[moving]
        #direction of edge_moving at impact:

        min_angle = np.min(angle[actually_moving]-delta_angle[actually_moving],initial=np.pi)
        new_pos+=point[None]
    #return the min angle to rotate
    else:
        min_angle = np.pi
    return min_angle,0,0#)*R2B[idxB,None]
    return min_angle,new_pos,ori+point[None]-new_pos#)*R2B[idxB,None]

def plane_intersection(normal1,point1,normal2,point2,tol=1e-5):
    point_line = np.zeros((point1.shape[0],point2.shape[0],3))
    dir_line= np.cross(normal1[:,None],normal2[None],axis=-1)
    parallel = np.linalg.norm(dir_line,axis=-1)<tol
    #offset1 = 0 np.einsum("ik,ik->i",normal1,point1)
    offset2 = np.einsum("jk,ijk->ij",normal2,point2[None]-point1[:,None])

    nothorizontal = (np.abs(dir_line[...,2])>tol)
    
    if nothorizontal.any():
        idx1,idx2 = np.nonzero(nothorizontal)
        delta = normal1[:,None,1] * normal2[None,:,0] - normal2[None,:,1] * normal1[:,None,0]
        delta = delta[nothorizontal]
        #the two planes intersect the ground
        point_line[nothorizontal,0] = normal1[idx1,1] * offset2[nothorizontal] / delta
        point_line[nothorizontal,1] = -normal1[idx1,0] * offset2[nothorizontal] / delta
    perp_y  = ~nothorizontal & (np.abs(dir_line[...,1])>tol)
    if perp_y.any():
        idx1,idx2 = np.nonzero(perp_y)
        delta = normal1[:,None,2] * normal2[None,:,0] - normal2[None,:,2] * normal1[:,None,0]
        delta = delta[perp_y]
        #the two planes intersect the xz
        point_line[perp_y,0] = normal1[idx1,2] * offset2[perp_y] / delta
        point_line[perp_y,2] = -normal1[idx1,0] * offset2[perp_y] / delta
    perp_x  =( ~nothorizontal) & (~ perp_y) & (np.abs(dir_line[...,0])>tol)
    if perp_x.any():
        idx1,idx2 = np.nonzero(perp_x)
        delta = normal1[:,None,2] * normal2[None,:,1] - normal2[None,:,2] * normal1[:,None,1]
        delta = delta[perp_x]
        #the two planes intersect the yz
        point_line[perp_x,1] = normal1[idx1,2] * offset2[perp_x] / delta
        point_line[perp_x,2] = -normal1[idx1,1] * offset2[perp_x] / delta
    point_line = point_line+point1[:,None]
    return parallel,point_line,dir_line
def plane_intersection_one2one(normal1,point1,normal2,point2,tol=1e-5):
    assert normal1.shape[0]==normal2.shape[0],'use plane_intersection to compute the intersection of each plane 1 with all plane 2'
    point_line = np.zeros((point1.shape[0],3))
    dir_line= np.cross(normal1,normal2,axis=-1)
    parallel = np.linalg.norm(dir_line,axis=-1)<tol
    #offset1 = 0 np.einsum("ik,ik->i",normal1,point1)
    offset2 = np.einsum("ik,ik->i",normal2,point2-point1)

    nothorizontal = (np.abs(dir_line[:,2])>tol)
    
    if nothorizontal.any():
        delta = normal1[:,1] * normal2[:,0] - normal2[:,1] * normal1[:,0]
        delta = delta[nothorizontal]
        #the two planes intersect the ground
        point_line[nothorizontal,0] = normal1[nothorizontal,1] * offset2[nothorizontal] / delta
        point_line[nothorizontal,1] = -normal1[nothorizontal,0] * offset2[nothorizontal] / delta
    perp_y  = ~nothorizontal & (np.abs(dir_line[...,1])>tol)
    if perp_y.any():
        delta = normal1[:,2] * normal2[:,0] - normal2[:,2] * normal1[:,0]
        delta = delta[perp_y]
        #the two planes intersect the xz
        point_line[perp_y,0] = normal1[perp_y,2] * offset2[perp_y] / delta
        point_line[perp_y,2] = -normal1[perp_y,0] * offset2[perp_y] / delta
    perp_x  =( ~nothorizontal) & (~ perp_y) & (np.abs(dir_line[...,0])>tol)
    if perp_x.any():
        delta = normal1[:,2] * normal2[:,1] - normal2[:,2] * normal1[:,1]
        delta = delta[perp_x]
        #the two planes intersect the yz
        point_line[perp_x,1] = normal1[perp_x,2] * offset2[perp_x] / delta
        point_line[perp_x,2] = -normal1[perp_x,1] * offset2[perp_x] / delta
    point_line = point_line+point1
    return parallel,point_line,dir_line
def ray_edge(partA,partB,direction,tol =1e-5,max_dist = np.inf,include_degen=True):
    d = direction/np.linalg.norm(direction)
    ideA = partA.edges[partA.edges[:,0]<partA.edges[:,1]]
    ideB = partB.edges[partB.edges[:,0]<partB.edges[:,1]]
    edgeAd = partA.vertices[ideA[:,1]]-partA.vertices[ideA[:,0]]

    edgeBd = partB.vertices[ideB[:,1]]-partB.vertices[ideB[:,0]]
    orid = partB.vertices[ideB[:,0]][None]-partA.vertices[ideA[:,0]][:,None]
    #plane = np.cross(edgeBd,d).T
    normals = np.cross(edgeAd[:,None],edgeBd[None],axis=-1)
    nn = np.linalg.norm(normals,axis=-1)
    triple_prod = np.power(nn,2)
    #den = edgeAd@plane
    denp = np.einsum("k,ijk->ij",d,normals)
    #filter out the parallel lines
    #paral = den==0
    paral = np.abs(nn)<tol
    #filter out the lines the sliding motion ie: movement coplanar with the sides. 
    slide = np.abs(denp)<tol
    #except for lines that are in the plane
    into = np.abs(np.einsum("ijk,ijk->ij",orid,normals))<tol
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dist= -np.einsum("ijk,ijk->ij",orid,normals)/denp
        dist[slide]=0
        translation = np.einsum("ij,k->ijk",dist,d)
        new_ori = translation+orid#     
        intersect_plane = np.cross(normals,new_ori,axis=-1)
        tap = np.einsum("jk,ijk->ij",edgeBd, intersect_plane)/triple_prod
        tbp = np.einsum("ik,ijk->ij",edgeAd,intersect_plane)/triple_prod
    if include_degen:
        outof = (tbp>1+tol)|(tap>1+tol)|(tbp<-tol)|(tap<-tol)
    else:
        outof = (tbp>1-tol)|(tap>1-tol)|(tbp<tol)|(tap<tol)
    mask = (slide)|(dist<0)|paral|outof|into
    
    dist[mask] = max_dist
    
    return dist

def contact_face_corner(partA,partB,tol= 1e-5):
    
    points = partB.vertices
    triangles = partA.triangles
    # Get the three vertices of each triangle
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]
    
    # Vectors from vertices to point
    PA = points[None] - A[:,None]
    PB = points[None] - B[:,None]
    PC = points[None] - C[:,None]
    
    # Cross products for normal vectors
    AB = B - A
    AC = C - A
    BC = C - B
    
    # Area vectors
    area_ABC = np.cross(AB, AC,axis=-1)
    area_PAB = np.cross(PA, PB,axis=-1)
    area_PBC = np.cross(PB, PC,axis=-1)
    area_PCA = np.cross(PC, PA,axis=-1)
    
    # Dot products to determine if point is inside prism
    d1 = np.sum(area_PAB * area_ABC[:,None], axis=-1)/np.square(np.linalg.norm(area_ABC, axis=-1)[:,None])
    d2 = np.sum(area_PBC * area_ABC[:,None], axis=-1)/np.square(np.linalg.norm(area_ABC, axis=-1)[:,None])
    d3 = np.sum(area_PCA * area_ABC[:,None], axis=-1)/np.square(np.linalg.norm(area_ABC, axis=-1)[:,None])
    
    inside = (d1 > tol) & (d2 >  tol) & (d3 >  tol)
    
    # Calculate distances
    distances = np.ones((len(triangles),len(points)))*tol*2
    
    # only consider where point projects inside
    if np.any(inside):
        # Normal vectors
        norm = np.linalg.norm(area_ABC, axis=-1)
        # Distance from point to plane
        distances_t = np.abs(np.einsum("ijk,ik->ij",PA,area_ABC)) / norm[:,None]
        distances[inside]=distances_t[inside] 
    
    fidx,vidx = np.nonzero(distances<tol)
    locations = partB.vertices[vidx]
    normals_faces = partA.face_normals[fidx]
    corner_id, faceid = np.nonzero((partB.faces[None]==vidx[:,None,None]).any(2))

    if locations.shape[0]==0:
        return False,locations,normals_faces 
    penetrating =False #(trimesh.proximity.signed_distance(partA,partB.vertices)>tol).any()
    
    return penetrating,locations,normals_faces 
def dist_point_edge(points,start,end):
    # Vectors from vertices to point
    PA = points[None] - start[:,None]
    PB = points[None] - end[:,None]
    BA = end-start
    PA2 = np.square(PA).sum(-1)
    PB2 = np.square(PB).sum(-1)
    dist2 = PA2 - np.square(np.einsum("ijk,ik->ij",PA,BA))
    dist2[PA2<dist2]=PA2[PA2<dist2]
    dist2[PB2<dist2]=PB2[PB2<dist2]
    return np.sqrt(dist2)

def contact_edges(partA,partB,tol=1e-5):
    maskedgeA = np.ones(partA.edges.shape[0],dtype=bool)#partA.edges[:,0]<partA.edges[:,1]#
    maskedgeB = np.ones(partB.edges.shape[0],dtype=bool)#partB.edges[:,0]<partB.edges[:,1]#
    ideA = partA.edges[maskedgeA]
    ideB = partB.edges[maskedgeB]
    #facesAn = partA.face_normals[partA.edges_face[partA.edges[:,0]<partA.edges[:,1]]]
    #facesBn = partB.face_normals[partB.edges_face[partB.edges[:,0]<partB.edges[:,1]]]
    edgeAd = partA.vertices[ideA[:,1]]-partA.vertices[ideA[:,0]]

    edgeBd = partB.vertices[ideB[:,1]]-partB.vertices[ideB[:,0]]
    
    orid = partB.vertices[ideB[:,0]][None]-partA.vertices[ideA[:,0]][:,None]
    edgeAn = edgeAd/np.linalg.norm(edgeAd,axis=-1)[:,None]
    edgeBn = edgeBd/np.linalg.norm(edgeBd,axis=-1)[:,None]
    normals = np.cross(edgeAd[:,None],edgeBd[None],axis=-1)
    nnormals = normals.copy()
    nn = np.linalg.norm(normals,axis=-1)
    normals[nn>tol] /= nn[nn>tol][:,None]
    triple_prod = np.power(nn,2)
    paral = nn<tol
    into0 = np.abs(np.einsum("ijk,ijk->ij",orid,normals))<tol
    towardB = np.einsum("ijk,jk->ij",normals,partB.face_normals[partB.edges_face])<-tol#*np.sum(np.square(normals),axis=-1)
    awayA   = np.einsum("ijk,ik->ij",normals,partA.face_normals[partA.edges_face])>tol#*np.sum(np.square(normals),axis=-1)
    into =  ~ paral & towardB & into0  #& awayA

    intoidxA,intoidxB = np.nonzero(into)
    edgeBf = edgeBd[intoidxB]
    edgeAf = edgeAd[intoidxA]
    normals = normals[into]
    orid = orid[into]
    nn=nn[into]
    triple_prod = triple_prod[into]
    intersect_plane = np.cross(normals,orid,axis=-1)
    tap = np.einsum("fk,fk->f",edgeBf, intersect_plane)/nn
    tbp = np.einsum("fk,fk->f",edgeAf,intersect_plane)/nn
    inside = (tbp<1-tol)&(tap<1-tol)&(tbp>tol)&(tap>tol)
    
    outside = (tbp>1+tol)|(tap>1+tol)|(tbp<-tol)|(tap<-tol)
    mask = inside
    
    #no need to consider the t=1
    #edge_cases0  = (np.abs(tbp-1)<tol) & (~outside)
    #edge_cases1  = (np.abs(tap-1)<tol) & (~outside)
    edge_cases0  = (np.abs(tbp)  <tol) & (~outside)
    edge_cases1  = (np.abs(tap)  <tol) & (~outside)
    edge_cases2 = edge_cases0 & edge_cases1
    edge_cases3 = edge_cases0 & edge_cases1
    edge_cases1[edge_cases2] = False
    edge_cases0[edge_cases2] = False
    #this case is handled by duplicating the edges
    edge_cases0[edge_cases0] = np.abs(tap[edge_cases0]-1) >tol
    edge_cases1[edge_cases1] = np.abs(tbp[edge_cases1]-1) >tol
    #if the faces of A are identical, modify the normals
    _,complement_edges0 = np.nonzero((partA.edges[None] == partA.edges[intoidxA[edge_cases0]][:,None,[1,0]]).all(-1))
    flatA = np.zeros(edge_cases0.shape[0],dtype=bool)
    flatA[edge_cases0] = np.abs(partA.face_normals[partA.edges_face[intoidxA[edge_cases0]]]-partA.face_normals[partA.edges_face[complement_edges0]]).sum(-1)<tol
    normals[flatA] = partA.face_normals[partA.edges_face[intoidxA[flatA]]]
    #nn[flatA] = 1
    #one triangle needs to be coplanar with the normal, or all normals of the faces of A are identical
    edge_cases0[edge_cases0] =np.abs((partA.face_normals[partA.edges_face[intoidxA[edge_cases0]]]-normals[edge_cases0]).sum(-1))<tol
    
    #there need to be an overlap between the triangles
    edge_cases0[edge_cases0] =(normals[edge_cases0]*partA.face_normals[partA.edges_face[intoidxA[edge_cases0]]]).sum(-1) > tol

    #one triangle needs to be coplanar with the normal, or all normals of the faces of B are identical
    _,complement_edges1 = np.nonzero((partB.edges[None] == partB.edges[intoidxB[edge_cases1]][:,None,[1,0]]).all(-1))
    flatB = np.zeros(edge_cases1.shape[0],dtype=bool)
    flatB[edge_cases1] = np.abs(partB.face_normals[partB.edges_face[intoidxB[edge_cases1]]]-partB.face_normals[partB.edges_face[complement_edges1]]).sum(-1)<tol
    normals[flatB] = -partB.face_normals[partB.edges_face[intoidxB[flatB]]]
    #nn[flatB] = 1
    #edge_cases1[edge_cases1] = np.abs((partB.face_normals[partB.edges_face[intoidxB[edge_cases1]]]*nn[edge_cases1][:,None]+normals[edge_cases1]).sum(-1))<tol*nn[edge_cases1]
    edge_cases1[edge_cases1] = np.abs((partB.face_normals[partB.edges_face[intoidxB[edge_cases1]]]+normals[edge_cases1]).sum(-1))<tol
    
    #there need to be an overlap between the triangles
    #edge_cases1[edge_cases1] =((normals[edge_cases1]*partB.face_normals[partB.edges_face[intoidxB[edge_cases1]]]).sum(-1) <- tol*nn[edge_cases1])
    edge_cases1[edge_cases1] =((normals[edge_cases1]*partB.face_normals[partB.edges_face[intoidxB[edge_cases1]]]).sum(-1) <- tol)


    #remaining corner to corner case
    
    edge_cases2[edge_cases2] = np.abs((partA.face_normals[partA.edges_face[intoidxA[edge_cases2]]]*edgeBf[edge_cases2]).sum(-1))<tol*np.linalg.norm(edgeBf[edge_cases2],axis=-1)
    othersideA = intoidxA[edge_cases2]-(intoidxA[edge_cases2]%3>0)+2*(intoidxA[edge_cases2]%3==0)

    #assert (partA.edges_face[intoidxA[edge_cases2]]==partA.edges_face[othersideA]).all()

    """edge_cases2[edge_cases2] = (((normals[edge_cases2]*partA.face_normals[partA.edges_face[intoidxA[edge_cases2]]]).sum(-1)>tol*nn[edge_cases2])
                               &((np.cross(edgeBf[edge_cases2],-edgeAd[othersideA])*partA.face_normals[partA.edges_face[intoidxA[edge_cases2]]]).sum(-1)>
                                  tol*np.linalg.norm(edgeBf[edge_cases2],axis=-1)*np.linalg.norm(edgeAd[othersideA],axis=-1)))"""
    edge_cases2[edge_cases2] = (((normals[edge_cases2]*partA.face_normals[partA.edges_face[intoidxA[edge_cases2]]]).sum(-1)>tol)
                               &((np.cross(edgeBf[edge_cases2],-edgeAd[othersideA])*partA.face_normals[partA.edges_face[intoidxA[edge_cases2]]]).sum(-1)>
                                  tol*np.linalg.norm(edgeBf[edge_cases2],axis=-1)*np.linalg.norm(edgeAd[othersideA],axis=-1)))

    edge_cases3[edge_cases3] = np.abs((partB.face_normals[partB.edges_face[intoidxB[edge_cases3]]]*edgeAf[edge_cases3]).sum(-1))<tol*np.linalg.norm(edgeAf[edge_cases3],axis=-1)
    othersideB = intoidxB[edge_cases3]-(intoidxB[edge_cases3]%3>0)+2*(intoidxB[edge_cases3]%3==0)

    """edge_cases3[edge_cases3] = (((normals[edge_cases3]*partB.face_normals[partB.edges_face[intoidxB[edge_cases3]]]).sum(-1)<tol*nn[edge_cases3])
                               &((np.cross(edgeAf[edge_cases3],-edgeBd[othersideB])*partB.face_normals[partB.edges_face[intoidxB[edge_cases3]]]).sum(-1)> 
                                 tol*np.linalg.norm(edgeAf[edge_cases3], axis=-1)*np.linalg.norm(edgeBd[othersideB], axis=-1)))"""
    edge_cases3[edge_cases3] = (((normals[edge_cases3]*partB.face_normals[partB.edges_face[intoidxB[edge_cases3]]]).sum(-1)<tol)
                               &((np.cross(edgeAf[edge_cases3],-edgeBd[othersideB])*partB.face_normals[partB.edges_face[intoidxB[edge_cases3]]]).sum(-1)> 
                                 tol*np.linalg.norm(edgeAf[edge_cases3], axis=-1)*np.linalg.norm(edgeBd[othersideB], axis=-1)))

    mask[edge_cases0] = True 
    mask[edge_cases1] = True
    mask[edge_cases2] = True
    mask[edge_cases3] = True
    mask[flatA] = True
    mask[flatB] = True
    maskedidxA = intoidxA[mask]
    maskedidxB = intoidxB[mask]

    locations_wfp = partA.vertices[ideA[:,0]][maskedidxA]+orid[mask]+tbp[mask,None]*edgeBd[maskedidxB]


    degenarate_mask = ((np.abs(tap)< tol) |(np.abs(tbp)  <tol)) & (~outside) & ~mask
    normals_degenerate = normals[degenarate_mask]
    #normals_degenerate = normals_degenerate/np.linalg.norm(normals_degenerate,axis=-1)[:,None]
    #idx_degenerateA = intoidxA[degenarate_mask]
    #idx_degenerateB = intoidxB[degenarate_mask]
    #locations_degenerate =partA.vertices[ideA[:,0]][idx_degenerateA]+orid[degenarate_mask]+tbp[degenarate_mask,None]*edgeBd[idx_degenerateB]
    

    normals = normals[mask]
    #normals /= nn[mask][:,None]
    keep = np.ones(locations_wfp.shape[0],dtype=bool)
    
    if locations_wfp.shape[0]==0:
        penetrating = False
    else:
        penetrating = False#(trimesh.proximity.signed_distance(partB,locations_wfp)<-tol).any()
        #other option: check the distance to the center point of the faces of each related triangle
        #switch_signA = trimesh.proximity.signed_distance(partA,locations_wfp+tol*normals)>=trimesh.proximity.signed_distance(partA,locations_wfp-tol*normals)

        #normals[switch_signA]=-normals[switch_signA]
    
    return penetrating,locations_wfp,normals, normals_degenerate