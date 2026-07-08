import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
obj_folder_path = './simulator/cont3Dsimulator/data/arch'
objs = []
for f in os.listdir(obj_folder_path):
    file_path = os.path.join(obj_folder_path, f)
    if os.path.isfile(file_path) and f[: -4].isdigit() and f[-3: ] == 'obj': 
        trimesh_obj = trimesh.load(file_path, force = 'mesh', )
        partID = int(f[:-4])
        assert trimesh_obj.is_watertight, "Sanitize the blocks first"
        trimesh_obj.fix_normals()
        objs.append(trimesh_obj)
        
        
sim = sim3D.BatchSimulator(n_batch=1)
sim3D.activeUI(sim)
sim.add_cover_area([trimesh.Trimesh(vertices=np.array([[1,1],
                                                       [1,0],
                                                       [0,1]]),
                                    faces=np.array([[0,1,2]]))])
sim.add_cover_area([trimesh.Trimesh(vertices=np.array([[1,1],
                                                       [2,0],
                                                       [0,1]]),
                                    faces=np.array([[0,1,2]]))])

valid = sim.put_block_abs([objs[0]],[objs[0].center_mass])
print(valid)

#valid = sim.put_block_abs([objs[1]],[objs[1].center_mass])
print(valid)
sim.render()
ps.show()