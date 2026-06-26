import polyscope as ps
import numpy as np
def render_forces(contacts,normals=None, frictions=None,length_ratio = 0.1, radius_ratio = 1.0,name='forces',tol=1e-5,color=(0.1, 0.4, 0.4)):
    points=np.zeros((0,3))
    normal = np.zeros((0,3))
    
    for contact in contacts:
        for p in contact["points"]:
            points = np.vstack([points,p])
            normal = np.vstack([normal,contact['normal']])
        #for p in contact["pointsB"]:
        #    points = np.vstack([points,p])
            #normal = np.vstack([normal,-contact['normal']])
    ps_cloud = ps.register_point_cloud(f"{name}_forces",
                                points=points, color=color)
    ps_cloud.add_vector_quantity(f"{name} normal", normal, color=color, vectortype='standard',enabled=True,length = length_ratio
                                    )
    return ps_cloud