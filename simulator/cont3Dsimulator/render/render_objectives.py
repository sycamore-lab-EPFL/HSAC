import polyscope as ps 
import numpy as np
import trimesh
import shapely
def render_cover(area,name='cover',show_ray = False):
    obj_group = ps.create_group(name)
    assert not show_ray, "not implemented"
    color = [1.0, 1.0, 0.0]
    # faces
    if isinstance(area,shapely.MultiPolygon):
        for i,p in enumerate(area.geoms):
            mesh_v,mesh_f = trimesh.creation.triangulate_polygon(p)
            obj = ps.register_surface_mesh(f"cover_{i}",mesh_v,  mesh_f, color=color)
            obj.add_to_group(name)
    elif isinstance(area,shapely.Polygon):
        if area.area > 0:
            mesh_v,mesh_f = trimesh.creation.triangulate_polygon(area)
            obj = ps.register_surface_mesh(f"cover",mesh_v,  mesh_f, color=color)
            obj.add_to_group(name)
    else:
        raise NotImplementedError
    obj_group.set_hide_descendants_from_structure_lists(True)
    return obj_group
