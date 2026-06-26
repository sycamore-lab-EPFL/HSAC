import polyscope as ps 
import numpy as np
import scipy
from simulator.cont3Dsimulator.placement.placement import put_block_pivot, put_block_slide
import trimesh
from .render_forces import render_forces
def render_action(action,t,name='action',debug=False,structure=None,tol=1e-3,mu=0.5):
        try:
            action_mesh = action[0]['block']
        except:
            return
        action_group = ps.create_group(name)
        dists = np.array([a['dist'] for a in action])
        if action[-1]['tot_dist'] == 0:
            breakpoints = [1]
        else:
            breakpoints = np.cumsum(dists)/action[-1]['tot_dist']
        
        seg = np.searchsorted(breakpoints,t)
        seg = min(seg,len(action)-1)
        action_mesh = action[seg-1]['block'].copy()
        if (breakpoints[seg]-breakpoints[seg-1])>0:
            t_in_seg = (t - breakpoints[seg-1])/(breakpoints[seg]-breakpoints[seg-1])
        else:
            t_in_seg = 0
        if action[seg]['angle']==0:
            prev_pos = action[seg-1]['cm']
            next_pos = action[seg]['cm']
            pos_t = t_in_seg*(next_pos-prev_pos)+prev_pos
            action_mesh.apply_translation(pos_t-action_mesh.center_mass)
            ps_cloud = ps.register_point_cloud(f"{name}_cm",
                                points=action_mesh.center_mass[None], color=(0,1,0))
            ps_cloud.add_vector_quantity(f"{name}_direction",( next_pos-prev_pos)[None], color=(0,1,0), vectortype='standard',length=1,enabled=True)
            ps_cloud.add_to_group(name)
        else:
            angle_t = t_in_seg*action[seg]['angle']
            rotpoint = action[seg]['pivot']
            ps_cloud = ps.register_point_cloud(f"{name}_pivot",
                                points=rotpoint[None], color=(0,1,0))
            ps_cloud.add_vector_quantity(f"{name} pivot normal", action[seg]['axis'][None], color=(0,1,0), vectortype='standard',length=1,enabled=True)
            ps_cloud.add_to_group(name)
            rotation = trimesh.transformations.rotation_matrix(angle_t, action[seg]['axis'],point=action[seg]['pivot'])
            action_mesh.apply_transform(rotation)
        color = [1.0, 0.0, 0.0]
        if debug and len(action)>1:
            print("Debug: disregard error")
            last_action = action[seg-1]
            action_test = put_block_slide(last_action,
                                             structure,
                                             action[1]['dir'],
                                             mu,
                                             max_range=2000,
                                             max_steps=10,
                                             tol=tol,
                                             d= None)
            action_pivot0 =  put_block_pivot(action[-1],structure,action[1]['dir'],
                                                   np.zeros(3),
                                                   tol=tol,mu=mu)
            action_pivot = put_block_pivot(last_action,structure,action[1]['dir'],
                                                   np.zeros(3),
                                                   tol=tol,mu=mu)
            if len(action_pivot)>0:
                action_pivot2 = put_block_pivot(action_pivot[0],structure,action[1]['dir'],
                                                   np.zeros(3),
                                                   tol=tol,mu=mu)
            print("Debug: end")
        # faces
        obj = ps.register_surface_mesh(f"new block",action_mesh.vertices, action_mesh.faces, color=color,transparency=0.2)
        obj.add_to_group(name)

        # wireframe
        sharp = action_mesh.face_adjacency_angles > np.radians(10)
        sharp[:]=True
        edges = action_mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"action_partwire",
                                            nodes=action_mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0))
            obj.add_to_group(name)
            obj.set_radius(0.0005, False)
        # render forces
        if seg>0:
            obj = render_forces(action[seg-1]['contacts'], name=name,
                        color=(0.4, 0.1, 0.1),tol=0.01)
            obj.add_to_group(name)
            obj = render_forces(action[seg-1]['contacts_degen'], name=name+"_degen",
                        color=(0.4, 0.4, 0.1),tol=0.01)
            obj.add_to_group(name)
        else:
            obj = render_forces(action[seg]['contacts'], name=name,
                        color=(0.4, 0.1, 0.1),tol=0.01)
            obj.add_to_group(name)
            obj = render_forces(action[seg]['contacts_degen'], name=name+"_degen",
                        color=(0.4, 0.4, 0.1),tol=0.01)
            obj.add_to_group(name)
         #self.render_contact(states)
        # contact.set_enabled(False)
def render_fall(structure,grounds,held,  speed,t,name='falling',tol=1e-5,speed_scale=1.):

    # blocks
    ps.remove_all_structures()
    ps.remove_all_groups()
    assembly_group = ps.create_group(name)
    wire_group = ps.create_group(name + "_wire")
    speed = speed*speed_scale
    for part_id, part_mesh in enumerate(structure):
        if part_mesh is None:
            break
        if grounds[part_id]:
            color = [0.,0, 0]
        elif held[part_id]:
            color = [1,0.7,0.7]
        else:
            color = [1.0, 1.0, 1.0]
        mesh = part_mesh.copy()
        center = mesh.center_mass
        mesh.apply_translation(-center)
        R = scipy.spatial.transform.Rotation.from_rotvec(speed[part_id * 6 + 3 : part_id * 6 +6]*t)
        v = speed[part_id * 6 : part_id * 6 + 3]*t
        T = np.identity(4)
        T[:3, :3] = R.as_matrix()
        T[:3, 3] = v + center
        mesh.apply_transform(T)

        # faces
        obj = ps.register_surface_mesh(f"{name}_part{part_id}", mesh.vertices, mesh.faces,
                                        color=color)
        obj.add_to_group(assembly_group)

        # wireframe
        sharp = mesh.face_adjacency_angles > np.radians(10)
        sharp[:]=True
        edges = mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"{name}_partwire{part_id}",
                                            nodes=mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0))
            obj.add_to_group(wire_group)
            obj.set_radius(0.0005, False)
    assembly_group.set_hide_descendants_from_structure_lists(True)
    wire_group.set_hide_descendants_from_structure_lists(True)
    return assembly_group