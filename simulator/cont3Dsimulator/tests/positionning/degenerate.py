import numpy as np
import simulator.cont3Dsimulator as sim3D
import trimesh
import os
import polyscope.imgui as psim
import polyscope as ps
import warp as wp
wp.set_device("cpu")
obj = trimesh.load('./simulator/cont3Dsimulator/data/test/simplex.obj', force='mesh')
assert obj.is_watertight, "Sanitize the blocks first"
tol = 1e-1
sim = sim3D.BatchSimulator(n_batch=1,tol=tol,max_dist =2,max_action_steps=2, use_mortar=True, mortar_thickness=2*tol)
sim.mu = 0.5
sim3D.activeUI(sim)
sim.prepare_action()
sim.put_block_abs([obj],[obj.center_mass])
#sim.prepare_action()
#valid = sim.put_block_abs([obj],[obj.center_mass+[0,tol/2,1+tol/2]])

sobj = obj.copy().apply_transform(trimesh.transformations.rotation_matrix(np.pi/2+0.2*sim.use_mortar,[1,0,0],point=obj.vertices[0]))
#sobj = obj.copy().apply_translation([-0,-0.3,1+tol/2])
#sobj = obj.copy().apply_translation([-0.1,-0.3,1+tol/2])
#sobj = obj.copy().apply_translation([0.3,0.3,0.65])

sim.prepare_action()
if sim.use_mortar:
    valid = sim.put_block_abs([sobj],[sobj.center_mass+[0,-tol,0]])
else:
    valid = sim.put_block_abs([sobj],[sobj.center_mass+[0,-tol/2,0]])
#assert (np.abs(np.array([c['pointsA'] for c in sim.contacts[0]])-np.array([[0,0,0]])).sum(-1)<tol).any(), "Contact points are not correct, check the contact detection method"
#assert (np.abs(np.array([c['pointsA'] for c in sim.contacts[0]])-np.array([[1,0,0]])).sum(-1)<tol).any(), "Contact points are not correct, check the contact detection method"
#assert (np.abs(np.array([c['pointsA'] for c in sim.contacts[0]])-np.array([[0,0,1]])).sum(-1)<tol).any(), "Contact points are not correct, check the contact detection method"
ps.show()