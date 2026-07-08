import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
import warp as wp
obj_folder_path = './simulator/cont3Dsimulator/data/arch'
objs = []
wp.set_device("cpu")
wp.config.mode = "debug"
for f in os.listdir(obj_folder_path):
    file_path = os.path.join(obj_folder_path, f)
    if os.path.isfile(file_path) and f[: -4].isdigit() and f[-3: ] == 'obj': 
        trimesh_obj = trimesh.load(file_path, force = 'mesh', )
        partID = int(f[:-4])
        objs.append(trimesh_obj)
        assert trimesh_obj.is_watertight, "Sanitize the blocks first"
sim = sim3D.BatchSimulator(n_batch=1,tol=1e-3,max_dist=3,max_action_steps=100,mu=0.5,)
sim3D.activeUI(sim)
sim.prepare_action()
sim.put_block_abs([objs[5].apply_transform(trimesh.transformations.rotation_matrix(-1.7,[1,0,0]))],[objs[0].centroid])
#valid = sim.put_block_abs([objs[5]],[objs[0].centroid+[0,-0.3,0.15]])
#print(valid)
for i in range(1,2):
    sim.prepare_action()
    valid = sim.put_block_from([objs[i]],[objs[i].centroid+[0.,1.1,2]],force = [np.array([0,-1,-2])],slide=True)
    print(valid)
sim.render()
ps.show()