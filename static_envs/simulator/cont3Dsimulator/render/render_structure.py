
import polyscope as ps
import numpy as np
import trimesh
def render_struct(name,structure,grounds,held,tol=1e-5,marked=None,is_mortar=None):
    assembly_group = ps.create_group(name)
    for part_id, part_mesh in enumerate(structure):
        if part_mesh is None:
            break
        if grounds[part_id]:
            color = [0.,0, 0]
        elif is_mortar is not None and is_mortar[part_id]:
            color = [0.7,0.7,1]
        elif held[part_id]:
            color = [1,0.7,0.7]
        elif marked is not None and marked[part_id]:
            color = [0.7,1,0.7]
        else:
            color = [1.0, 1.0, 1.0]

        """
        if states[part_id] == 1:
            
        elif states[part_id] == 2:
            color = [0.0, 0.0, 0.0]
        else:
            continue
        """
        # faces
        if (part_mesh.vertices<0).any():
            
            assert part_mesh.is_volume
            copy_mesh = part_mesh.copy()
            copy_mesh = trimesh.boolean.difference([copy_mesh,trimesh.creation.box(bounds=[[-1000,-1000,-1000],[1000,1000,0]])])

            obj = ps.register_surface_mesh(f"{name}_part{part_id}", copy_mesh.vertices, copy_mesh.faces, color=color)

        else:
            copy_mesh = part_mesh.copy()
            obj = ps.register_surface_mesh(f"{name}_part{part_id}", np.clip(part_mesh.vertices,[-np.inf,-np.inf,0],np.inf), part_mesh.faces, color=color)
        obj.add_to_group(assembly_group)

        # wireframe
        sharp = copy_mesh.face_adjacency_angles > np.radians(0.2)
        #sharp[:]=True
        edges = copy_mesh.face_adjacency_edges[sharp]
        if edges is not []:
            obj = ps.register_curve_network(f"{name}_partwire{part_id}",
                                            #nodes= np.clip(part_mesh.vertices,[-np.inf,-np.inf,0],np.inf),
                                            nodes=copy_mesh.vertices,
                                            edges=edges,
                                            color=(0, 0, 0))
            obj.add_to_group(assembly_group)
            obj.set_radius(tol/4, False)
            """
            ide = part_mesh.edges
            edgeBd = part_mesh.vertices[ide[:,1]]-part_mesh.vertices[ide[:,0]]
            edgeBd *=10
            idxs = np.arange(edgeBd.shape[0])

            objc = ps.register_curve_network(f"{self.name}_direction",
                                            nodes=np.vstack([part_mesh.vertices[ide[:,0]],part_mesh.vertices[ide[:,0]]+edgeBd]),
                                            edges=np.hstack([idxs[:,None],idxs[:,None]+edgeBd.shape[0]]),
                                            color=(0, 1, 0))
            objc.set_radius(0.001)
            objc.add_to_group(assembly_group)"""
    #self.render_contact(states)
    # contact.set_enabled(False)
    assembly_group.set_hide_descendants_from_structure_lists(True)
    return assembly_group