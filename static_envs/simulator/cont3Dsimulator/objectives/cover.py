import copy
from typing import List
import numpy as np 
import warnings
from shapely import MultiPoint, Polygon
import shapely
import torch
import trimesh
from trimesh.path.raster import rasterize


def cover_mesh(covering_mesh,shape_to_cover,tol=1e-6,scale=1):
    dire = np.array([0,0,1])
    
    new_vertices = covering_mesh.vertices - np.einsum("ik,k->i",covering_mesh.vertices,dire)[:,None]*dire[None]/np.square(np.linalg.norm(dire))
    # Get the 2D path of the mesh's outer boundary
    new_shape = trimesh.path.polygons.edges_to_polygons(covering_mesh.edges,new_vertices[:,:2])
    hull = trimesh.convex.convex_hull(new_vertices)
    
    mpt = MultiPoint(new_vertices[:,:2])
    # Create Shapely Polygon
    #polygon = Polygon(boundary_vertices)
    # Compute intersection/union
    area_0 = shape_to_cover.area
    shape_covered =shape_to_cover.intersection(mpt.convex_hull)
    if isinstance(shape_covered,shapely.GeometryCollection):
        shape_covered = Polygon()
    new_shape_to_cover = shape_to_cover.difference(mpt.convex_hull)
    area_1 =new_shape_to_cover.area
    reward = (area_0-area_1)/scale
    if not isinstance(new_shape_to_cover,shapely.Polygon) and not isinstance(new_shape_to_cover,shapely.MultiPolygon):
        assert False, f"New shape to cover is not a polygon or multipolygon, but is {type(new_shape_to_cover)}"
    return reward, new_shape_to_cover,shape_covered
def circle_encode(shape_to_cover):
    #TODO find a geometric way of encoding this
    nodes = []
    if isinstance(shape_to_cover,shapely.MultiPolygon):
        for i,p in enumerate(shape_to_cover.geoms):
            #mesh_v,mesh_f = trimesh.creation.triangulate_polygon(p)
            nodes.append([p.centroid.x,p.centroid.y,p.boundary.distance(p.centroid)])
    elif isinstance(shape_to_cover,shapely.Polygon):
        if shape_to_cover.area > 0:
           # mesh_v,mesh_f = trimesh.creation.triangulate_polygon(shape_to_cover)
            nodes.append([shape_to_cover.centroid.x,shape_to_cover.centroid.y,shape_to_cover.boundary.distance(shape_to_cover.centroid)])
    else:
        raise NotImplementedError
    return np.array(nodes)
def corners_encode(shape_to_cover):
    if isinstance(shape_to_cover,shapely.MultiPolygon):
        nodes =np.zeros((0,2))
        edges = np.zeros((0,2),dtype=int)
        for i,p in enumerate(shape_to_cover.geoms):
            if p.area > 0:
                nodes_arr= np.array(p.exterior.coords[0:-1])
                edges_arr = np.vstack([np.repeat(np.arange(len(nodes_arr)),len(nodes_arr)),
                                   np.tile(np.arange(len(nodes_arr)),len(nodes_arr))]).T
                edges = np.vstack([edges,edges_arr + nodes.shape[0]])
                nodes = np.vstack([nodes,nodes_arr])
                
    elif isinstance(shape_to_cover,shapely.Polygon):
        nodes= shape_to_cover.exterior.coords[0:-1]
        edges = np.vstack([np.repeat(np.arange(len(nodes)),len(nodes)),
                           np.tile(np.arange(len(nodes)),len(nodes))]).T
    else:
        raise NotImplementedError
    return np.array(nodes),np.array(edges)
def delaunay_encode(shape_to_cover,tol=1e-2):
    #assert False, "Does not work with non-convex shapes"
    if isinstance(shape_to_cover,shapely.MultiPolygon):
        nodes =np.zeros((0,2))
        edges = np.zeros((0,2),dtype=int)
        edges_tri = np.array([[0,1],[1,2],[2,0],[1,0],[2,1],[0,2]])
        for i,p in enumerate(shape_to_cover.geoms):
            p = shapely.simplify(p, tolerance=tol)
            if p.area > 0:
                valp = shapely.make_valid(p)
                tri = shapely.constrained_delaunay_triangles(valp)
                for t in tri.geoms:
                    nodes_arr= np.array(t.exterior.coords[0:-1])
                    edges_arr = edges_tri + nodes.shape[0]
                    edges = np.vstack([edges,edges_arr])
                    nodes = np.vstack([nodes,nodes_arr])
    elif isinstance(shape_to_cover,shapely.Polygon):
        shape_to_cover = shapely.simplify(shape_to_cover, tolerance=tol)
        nodes= np.zeros((0,2))
        edges = np.zeros((0,2),dtype=int)
        edges_tri = np.array([[0,1],[1,2],[2,0],[1,0],[2,1],[0,2]])
        valp = shapely.make_valid(shape_to_cover)
        tri = shapely.constrained_delaunay_triangles(valp)
        for t in tri.geoms:
            nodes_arr= np.array(t.exterior.coords[0:-1])
            edges_arr = edges_tri + nodes.shape[0]
            edges = np.vstack([edges,edges_arr])
            nodes = np.vstack([nodes,nodes_arr])
    else:
        raise NotImplementedError
    return np.array(nodes),np.array(edges)
def sparse_corner_encode(shape_to_cover,tol=1e-2):
    if isinstance(shape_to_cover,shapely.MultiPolygon):
        nodes =np.zeros((0,2))
        edges = np.zeros((0,2),dtype=int)
        for i,p in enumerate(shape_to_cover.geoms):
            p = shapely.simplify(p, tolerance=tol)
            if p.area > 0:
                nodes_arr= np.array(p.exterior.coords[0:-1])
                edges_arr = np.vstack([np.arange(len(nodes_arr)),
                                       np.arange(1,len(nodes_arr)+1)]).T
                #close the loop
                edges_arr[-1,-1] = 0
                #revert the edges to have both directions
                edges_arr = np.vstack([edges_arr,edges_arr[[1,0],:]])
                edges = np.vstack([edges,edges_arr + nodes.shape[0]])
                nodes = np.vstack([nodes,nodes_arr])
                
    elif isinstance(shape_to_cover,shapely.Polygon):
        shape_to_cover = shapely.simplify(shape_to_cover, tolerance=tol)
        nodes= shape_to_cover.exterior.coords[0:-1]
        edges = np.vstack([np.arange(len(nodes)),
                        np.arange(1,len(nodes)+1)]).T
        if edges.shape[0]==0:
            return np.array(nodes),np.array(edges)
        #close the loop
        edges[-1,-1] = 0
        #revert the edges to have both directions
        edges = np.vstack([edges,edges[[1,0],:]])
    else:
        raise NotImplementedError
    return np.array(nodes),np.array(edges)