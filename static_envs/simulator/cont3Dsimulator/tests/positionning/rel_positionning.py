import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
obj_folder_path = './gym/static_envs/simulator/cont3Dsimulator/data/arch'
objs = []
for f in os.listdir(obj_folder_path):
    file_path = os.path.join(obj_folder_path, f)
    if os.path.isfile(file_path) and f[: -4].isdigit() and f[-3: ] == 'obj': 
        trimesh_obj = trimesh.load(file_path, force = 'mesh', )
        partID = int(f[:-4])
        objs.append(trimesh_obj)
        assert trimesh_obj.is_watertight, "Sanitize the blocks first"
sim = sim3D.BatchSimulator(n_batch=1,tol=1e-2,mu=0.1,max_dist=2)
sim3D.activeUI(sim)
sim.prepare_action()
sim.put_block_abs([objs[0]],[objs[0].centroid])
sim.prepare_action()
valid = sim.put_block_abs([objs[5]],[objs[0].centroid+[0,-0.3,0.15]])
print(valid)
for i in range(1,2):
    sim.prepare_action()
    valid = sim.put_block_rel([objs[i]],[0],[1],[10],[np.array([0,0,0.0])],[0.5],[np.array([0,-1,-1.2])],grip_location=[np.array([0,0,0])])
    print(valid)
sim.render()
ps.show()