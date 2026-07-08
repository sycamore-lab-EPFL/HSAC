import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
import warp as wp
wp.set_device("cpu")
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
        
        
sim = sim3D.BatchSimulator(n_batch=27,tol=1e-2)
sim3D.activeUI(sim)
valid = sim.put_block_abs([objs[0]]*27,[objs[0].center_mass]*27)
print(valid)
#degenerate point
"""valid = sim.put_block_abs([objs[1]]*27,
                          [objs[1].center_mass-objs[1].vertices[i]+objs[0].vertices[j]+[0,0,sim.tol/10]
                           for i in range(0,8) for j in range(6)
                           ],env_ids=None)"""
#close to degenerate point

valid = sim.put_block_abs([objs[1]]*27,
                          [objs[1].center_mass-objs[1].vertices[0]+objs[0].vertices[3]+np.array([i*sim.tol,j*sim.tol,0])
                           for i in range(8) for j in range(6)
                           ],env_ids=None)
"""valid = sim.put_block_abs([objs[1]]*27,
                          [objs[1].center_mass-objs[1].vertices[0]+objs[0].vertices[3]+np.array([sim.tol*e,0,0]) for e in np.linspace(0,5,20)],
                          env_ids=np.arange(20))"""

#penetrating point

"""valid = sim.put_block_abs([objs[1]]*27,
                          [objs[1].center_mass-objs[1].vertices[0]+objs[0].vertices[3]+np.array([i*sim.tol,j*sim.tol*0,0])
                           for i in range(8) for j in range(6)
                           ])"""
"""valid = sim.put_block_abs([objs[1]]*27,
                          [objs[1].center_mass-objs[1].vertices[0]+objs[0].vertices[3]+np.array([sim.tol,0,0])
                           for i in range(8) for j in range(6)
                           ])"""
#valid = sim.put_block_abs([objs[1]],[objs[1].center_mass])
print(valid)
sim.render()
ps.show()