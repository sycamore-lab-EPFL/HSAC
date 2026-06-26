import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
import warp as wp
wp.set_device('cpu')
obj_folder_path = './gym/static_envs/simulator/cont3Dsimulator/data/arch'
objs = []
scale = 1
for f in os.listdir(obj_folder_path):
    file_path = os.path.join(obj_folder_path, f)
    if os.path.isfile(file_path) and f[: -4].isdigit() and f[-3: ] == 'obj': 
        trimesh_obj = trimesh.load(file_path, force = 'mesh', )
        partID = int(f[:-4])
        trimesh_obj.apply_scale(scale)
        objs.append(trimesh_obj)
        assert trimesh_obj.is_watertight, "Sanitize the blocks first"
sim = sim3D.BatchSimulator(n_batch=1,mu=0.1,tol=1e-2,max_dist = scale)
sim3D.activeUI(sim)
sim.prepare_action()
sim.put_block_abs([objs[0]],[objs[0].centroid])
#valid = sim.put_block_abs([objs[5]],[objs[0].centroid+[0,-0.3,0.15]])
#print(valid)
for i in range(1,2):
    sim.prepare_action()
    #valid = sim.put_block_rel_noface([objs[i]],[0],[np.array([-0.01,0.02,0])],np.array([[1,0.1,0,0,1,0]]),[-objs[0].face_normals[10]])
    valid = sim.put_block_rel_noface([objs[i]],[0],[scale*np.array([-0.01,0.02,0])],np.array([[1,-0.1,0,0,1,0]]),[-objs[0].face_normals[10]])
    #valid = sim.put_block_rel_noface([objs[i]],[0],[np.array([-0.01,0.0,0])],np.array([[1,-0.1,0,0,1,0]]),[-objs[0].face_normals[10]])
    print(valid)
sim.render()
ps.show()