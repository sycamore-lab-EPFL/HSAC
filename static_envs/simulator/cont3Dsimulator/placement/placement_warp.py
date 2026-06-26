from typing import List
import trimesh
import numpy as np
import warp as wp
# The init() function prints the directory of the kernel cache which contains the .cpp files
# generated from Warp kernels. You can put breakpoints in these C++ files through Visual Studio Code,
# but it's generally more convenient to use wp.breakpoint(). See the example below.
wp.config.mode = "release"
wp.config.verbose = True
MAX_EDGES = wp.constant(36)  # Maximum number of edges to return from a corner
wp.init()

# Enable kernels to be compiled with debug info and disable optimizations
@wp.kernel
def put_block_abs_ker():
    pass
def put_block_abs_wrapper(new_block:trimesh.Trimesh,
                  old_blocks:List[trimesh.Trimesh],
                  coords:np.ndarray,
                  tol=1e-3,
                  bounding_boxes=None):
    """
    Place a block at the given coordinates.
    """
    return

def put_block_from(last_action,
                   old_blocks:List[trimesh.Trimesh],
                   direction,
                   max_range=5,
                   tol=1e-5,
                   slide_over=True,
                   oob=None):
    pass
def put_block_pivot(last_action,old_blocks,force,grip_loc,tol=1e-5,debug=False):
    pass
def put_block_slide(last_action,
                    old_blocks:List[trimesh.Trimesh],
                    force_init,mu,max_range=5,max_steps = 100,tol=1e-3,d=None):
    pass

def translate_toA_wrapper(partA,partB,dir,tol=1e-5,max_dist=np.inf):
    partAwp = wp.Mesh(wp.array(partA.vertices,dtype=wp.vec3),wp.array(partA.faces.flatten(),dtype=wp.int32))
    partBwp = wp.Mesh(wp.array(partB.vertices,dtype=wp.vec3),wp.array(partB.faces.flatten(),dtype=wp.int32))

    distances_ee = wp.full(value=max_dist,dtype=wp.float32, shape=(partAwp.indices.shape[0],
                                                  partBwp.indices.shape[0]))
    distances_eeover = wp.full(value=max_dist,dtype=wp.float32, shape=(partAwp.indices.shape[0],
                                                  partBwp.indices.shape[0]))
    distances_cf = wp.full(value=max_dist,dtype=wp.float32, shape=(partAwp.points.shape[0]))
    distances_fc = wp.full(value=max_dist,dtype=wp.float32, shape=(partBwp.points.shape[0]))
    t = wp.zeros(dtype=wp.vec3, shape=(2))

    wp.launch(ray_corner_face,dim =partAwp.points.shape[0],inputs=(partAwp.points,
                                                                    partBwp.id,
                                                                     wp.vec3(-dir),
                                                                     wp.float32(tol),
                                                                     wp.float32(max_dist),
                                                                     distances_cf
                                                                     ), outputs=())
    wp.launch(ray_corner_face,dim = partBwp.points.shape[0], inputs=(partBwp.points, partAwp.id,dir,tol,max_dist,distances_fc))

    wp.launch(ray_edge,dim = (partAwp.indices.shape[0], partBwp.indices.shape[0]), inputs=(partAwp.id, partBwp.id,dir,tol,max_dist,True,distances_ee), outputs=())
    wp.launch(ray_edge,dim = (partAwp.indices.shape[0], partBwp.indices.shape[0]), inputs=(partAwp.id, partBwp.id,dir,tol,max_dist,False,distances_eeover), outputs=())
    #dists = np.concatenate([distances_ee.numpy().flatten(), distances_cf.numpy(), distances_fc.numpy()])
    mindists_fc = np.concatenate([distances_cf.numpy(), distances_fc.numpy()]).min()

    min_dist_over = min(distances_eeover.numpy().min()-tol/2,mindists_fc-tol/2)
    min_dist = min(distances_ee.numpy().min()-tol/2,mindists_fc-tol/2)
    #min_dist = np.min(dists)-tol/2

    return min_dist, min_dist_over
@wp.kernel
def ray_corner_face(cornersA:wp.array(dtype=wp.vec3f, ndim=1),# type: ignore
                   partIDB:wp.uint64,
                   direction:wp.vec3,
                   tol:wp.float32,
                   max_dist:wp.float32,
                   distances:wp.array(dtype=wp.float32, ndim=1)# type: ignore
                   ):
    """
    Compute the distance between the points of two meshes along a ray.
    """
    tid = wp.tid()
    d = wp.normalize(direction)
    distances[tid] = ray_corner_face_f(cornersA[tid], partIDB, d, tol, max_dist)
@wp.func
def ray_corner_face_f(cornerA:wp.vec3f,# type: ignore
                   partIDB:wp.uint64,
                   d:wp.vec3,
                   tol:wp.float32,
                   max_dist:wp.float32,
                   )-> wp.float32:
    #Note: The direction should be normalized before calling this function
    query = wp.mesh_query_ray(partIDB,cornerA,d,max_t=max_dist)
    if query.result:
        #side_length_u,side_length_v = wp.mesh_eval_edge_length(partIDB, query.face)
        #rel_tolu = tol/side_length_u
        #rel_tolv = tol/side_length_v
        #if query.t>tol:# and query.u > tol and query.v > tol and query.v + query.u < 1.0-tol:
        return query.t
    return max_dist
@wp.kernel
def ray_edge(partIDA:wp.uint64,
             partIDB:wp.uint64,
             direction:wp.vec3,
             tol:wp.float32,
             max_dist:wp.float32,
             include_degen:wp.bool,
             distances:wp.array(dtype=wp.float32, ndim=2)# type: ignore
             ):
    """
    Compute the distance between the edges of two meshes along a ray.
    """
    edgeidA,edgeidB = wp.tid()
    distances[edgeidA, edgeidB] = ray_edge_f(partIDA, partIDB,edgeidA,edgeidB, direction, tol, max_dist, include_degen)
@wp.func
def ray_edge_f(partIDA:wp.uint64,
             partIDB:wp.uint64,
             edgeidA:wp.int32,
                edgeidB:wp.int32,
             direction:wp.vec3,
             tol:wp.float32,
             max_dist:wp.float32,
             include_degen:wp.bool,
             factor:wp.float32=100.
             )-> wp.float32:
    """
    Compute the distance between the edges of two meshes along a ray.
    """
    d = wp.normalize(-direction)
    p0A,p1A,_,_ = get_edge_vertices(partIDA,edgeidA)
    p0B,p1B,_,_ = get_edge_vertices(partIDB,edgeidB)

    orid = p0B - p0A
    normal = wp.cross(p1A - p0A, p1B - p0B)
    nn = wp.length(normal)
    triple_prod = nn*nn
    denp = wp.dot(d,normal)
    #filter out the parallel lines
    paral = nn<tol*tol
    #filter out the lines the sliding motion ie: movement coplanar with the sides. 
    slide = wp.abs(denp)<tol*tol
    #except for lines that are in the plane
    into = wp.abs(wp.dot(orid,normal))<tol*tol
    
    if into or slide or paral:
        return max_dist        

    dist= wp.dot(orid,normal)/denp
    if dist<tol:
        return max_dist
    else:
        translation = dist*d
        new_ori = -translation+orid#     
        intersect_plane = wp.cross(normal,new_ori)
        tap = wp.dot(p1B-p0B, intersect_plane)/triple_prod
        tbp = wp.dot(p1A-p0A, intersect_plane)/triple_prod
        #wp.printf("tbp: %f, tap: %f, dist: %f\n", tbp, tap, dist)
        if include_degen:
            #fact = 5.
            tol_relB = tol#tol*dist/wp.length(p0B-p1B)*factor
            tol_relA = tol#tol*dist/wp.length(p0A-p1A)*factor
            if (tbp>1.+tol_relB) or (tap>1.+tol_relA) or (tbp<-tol_relB) or (tap<-tol_relA):
                #wp.printf("tap: %f, tbp: %f, dist: %f, tol_relA: %f, tol_relB: %f\n", tap, tbp, dist, tol_relA, tol_relB)
            #if (tbp>1.+tol/wp.length(p0B-p1B)) or (tap>1.+tol/wp.length(p0A-p1A)) or (tbp<-tol/wp.length(p0B-p1B)) or (tap<-tol/wp.length(p0A-p1A)):
                #if not (tap > 1.+fact*tol/wp.length(p0A-p1A) or tbp > 1.+fact*tol/wp.length(p0B-p1B) or tap < -fact*tol/wp.length(p0A-p1A) or tbp < -fact*tol/wp.length(p0B-p1B)):
                    #wp.printf("Warning: edge %d-%d and %u-%u are too far apart: tbp: %f, tap: %f, dist: %f\n", edgeidA, edgeidB, partIDA, partIDB, tbp, tap, dist)
                return max_dist
            
            else:
                return dist
        else:
            if (tbp>1.) or (tap>1.) or (tbp<0.) or (tap<0.):
            #if (tbp>1.-tol_relB) or (tap>1.-tol_relA) or (tbp<tol_relB) or (tap<tol_relA):
            #if (tbp>1.-tol/wp.length(p0B-p1B))or(tap>1.-tol/wp.length(p0A-p1A)) or (tbp<tol/wp.length(p0B-p1B))or(tap/wp.length(p0A-p1A)<tol):
                #wp.printf("Degenerate edge: %f %f; tol_rel: %f %f \n",tap,tbp,tol_relA,tol_relB)
                return max_dist
            else:
                return dist
@wp.kernel
def check_contact_cornerface(partIDA: wp.uint64,
                             partIDB: wp.uint64,
                             tol: wp.float32,
                             contact_normals: wp.array(dtype=wp.vec3, ndim=1),  # type: ignore
                             contact_points: wp.array(dtype=wp.vec3, ndim=1),  # type: ignore
                             istouching: wp.array(dtype=wp.bool, ndim=1),  # type: ignore
):
    tidA = wp.tid()
    istouchingv, contact_normalsv, contact_pointsv,contact_pointsvA,contact_pointsvB = check_contact_cornerface_f(partIDA, tidA, partIDB, tol)
    istouching[tidA] = istouchingv
    contact_normals[tidA] = contact_normalsv
    contact_points[tidA] = contact_pointsv
@wp.func
def check_contact_cornerface_f(partIDA: wp.uint64,
                               pointidA: wp.int32,
                                partIDB: wp.uint64,
                                tol: wp.float32,
):
    #wp.printf("check_contact_cornerface_f: %d %d %d\n", partIDA, pointidA, partIDB)
    pointA = wp.mesh_get(partIDA).points[pointidA]
    #wp.printf("pointA: %f %f %f\n", pointA[0], pointA[1], pointA[2])
    p = wp.mesh_get(partIDB).points.shape[0]
    #wp.printf("partIDB: %d\n", p)
    query = wp.mesh_query_point_sign_normal(partIDB,point=pointA,max_dist= tol,epsilon=tol/100.)
    #wp.printf("query: %f\n", query.result)
    c0B=wp.mesh_eval_position(partIDB, query.face, 0., 0.)
    c1B=wp.mesh_eval_position(partIDB, query.face, 1., 0.)
    c2B=wp.mesh_eval_position(partIDB, query.face, 0., 1.)
    tolrelu = 0.#tol/wp.length(c0B-c1B)/1.1
    tolrelv = 0.#tol/wp.length(c0B-c2B)/1.1
    #wp.printf("query.result: %d, query.u: %f, query.v: %f, query.face: %d\n", query.result, query.u, query.v, query.face)
    if query.result and query.u > tolrelu and query.v > tolrelv and (query.v/(1.-tolrelv) + query.u/(1.-tolrelu)) < 1.0:
        pointB = wp.mesh_eval_position(partIDB, query.face, query.u, query.v)

        delta = pointA - pointB
        dist = wp.length(delta) * query.sign
        if query.sign < 0:
            wp.printf("Warning: pointA is inside partB, pointA: %f %f %f, pointB: %f %f %f\n",pointA[0], pointA[1], pointA[2], pointB[0], pointB[1], pointB[2])
        if dist < tol:
            #wp.printf("dist: %f\n", dist)
            #wp.printf("query.v, query.u, query.face: %f %f %d\n",query.v, query.u, query.face)
            istouching=True
            contact_normals = wp.mesh_eval_face_normal(partIDB,query.face)
            #contact_normals = wp.normalize(-delta)
            #if wp.length(contact_normals-contact_normals_old) > tol and query.u > tolrelu and query.v > tolrelv and (query.v/(1.-tolrelv) + query.u/(1.-tolrelu)) < 1.0:
                #wp.printf("Warning: assumption incorrect in the new calculations\n")
            if wp.length(contact_normals) < tol:
                wp.printf("Warning: contact normal is zero length: query.face: %d", query.face)
            contact_points = (pointB + pointA)/2.0
            contact_pointA = pointA
            contact_pointB = pointB
        else:
            istouching = False
            contact_normals = wp.mesh_eval_face_normal(partIDB,query.face)
            if wp.length(contact_normals) < tol:
                wp.printf("Warning: contact normal is zero length: query.face: %d", query.face)
            contact_points = (pointB + pointA)/2.0
            contact_pointA = pointA
            contact_pointB = pointB
    else:
        istouching = False
        contact_normals = wp.vec3(0.0, 0.0, 0.0)
        contact_points = wp.vec3(0.0, 0.0, 0.0)
        contact_pointA = pointA
        contact_pointB = wp.vec3(0.0, 0.0, 0.0)
    return istouching, contact_normals, contact_points,contact_pointA,contact_pointB
@wp.kernel
def check_contact_edge(partIDA: wp.uint64,
                       partIDB: wp.uint64,
                       tol:wp.float32,
                       angle_tol:wp.float32,
                       contact_normals: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                       contact_points: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                       istouching: wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                       isdegen: wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                       ):
    edgeidA,edgeidB = wp.tid()
    istouchingv,contact_normalsv, contact_pointsv,contact_pointsvA,contact_pointsvB, isdegenv = check_contact_edge_f(partIDA, partIDB, edgeidA, edgeidB, tol,angle_tol)
    istouching[edgeidA,edgeidB] = istouchingv
    contact_normals[edgeidA,edgeidB] = contact_normalsv
    contact_points[edgeidA,edgeidB] = contact_pointsv
    isdegen[edgeidA,edgeidB] = isdegenv
@wp.func
def check_contact_edge_f(partIDA: wp.uint64,
                       partIDB: wp.uint64,
                       edgeidA:wp.int32,
                       edgeidB:wp.int32,
                       tol:wp.float32,
                       angle_tol:wp.float32
                       ):#-> wp.tuple(wp.bool,wp.vec3,wp.vec3,bool): # type: ignore
    """returns istouching, contact_normals, contact_points, isdegen"""
    p0A,p1A,p2A,p0Aidx,p1Aidx,p2Aidx = get_triangle_vertices(partIDA,edgeidA)
    p0B,p1B,p2B,p0Bidx,p1Bidx,p2Bidx = get_triangle_vertices(partIDB,edgeidB)
    
    eA = p1A - p0A
    eB = p1B - p0B
    tolrelA = tol#/wp.length(eA)
    tolrelB = tol#/wp.length(eB)
    nA = wp.mesh_eval_face_normal(partIDA,edgeidA/3)
    nB = wp.mesh_eval_face_normal(partIDB,edgeidB/3)

    query = wp.closest_point_edge_edge(p0A, p1A,p0B,p1B,1e-5)
    
    contact_points = (query[0]*p1A+(1.0-query[0])*p0A+ query[1]*p1B+(1.0-query[1])*p0B)/2.0
    contact_pointA = query[0]*p1A+(1.0-query[0])*p0A
    contact_pointB = query[1]*p1B+(1.0-query[1])*p0B
    contact_normals_old =  wp.normalize(wp.cross(eB, eA))
    contact_normals = wp.normalize(contact_pointB-contact_pointA)

    if wp.length(contact_normals_old) < tol:
        pass
        #wp.printf("Warning: parallel sides: %d\n", edgeidA, edgeidB)
        #return False, wp.vec3(0.0,0.0,0.0), wp.vec3(0.0, 0.0, 0.0),wp.vec3(0.0, 0.0, 0.0),wp.vec3(0.0, 0.0, 0.0), False
    if query[2] > tol:
        #if query[2] < 1.3*tol:
            #wp.printf("close contact point: %f %f %f, d:%f,  edgeidA: %d, edgeidB: %d, query0 %f / %f, query1 %f / %f \n",
            #         contact_points[0], contact_points[1], contact_points[2], query[2], edgeidA, edgeidB,query[0], tolrelA,query[1],tolrelB)
        #wp.printf("Warning: contact point is too far from the edge: contact_pointA: %f %f %f, contact_pointB: %f %f %f, edgeidA: %d, edgeidB: %d\n",
        #         contact_pointA[0], contact_pointA[1], contact_pointA[2],
        #         contact_pointB[0], contact_pointB[1], contact_pointB[2],
        #         edgeidA, edgeidB)
        return False, contact_normals_old, contact_points,contact_pointA,contact_pointB, False
    #wp.printf("contact point: %f %f %f, edgeidA: %d, edgeidB: %d, query0 %f / %f, query1 %f / %f \n",
    #           contact_points[0], contact_points[1], contact_points[2], edgeidA, edgeidB,query[0], tolrelA,query[1],tolrelB)

    if query[0] > 0.+tolrelA and query[0] < 1.-tolrelA and query[1] > 0.+tolrelB and query[1] < 1.0-tolrelB:
        if wp.length(contact_normals) < tol:
            wp.printf("Warning: exact contact point, blocks might be pinned\n")
            if wp.dot(contact_normals_old,nB) < wp.dot(contact_normals_old,nA)-tol and  wp.dot(contact_normals_old,nB) < -tol:
                return True,contact_normals_old,contact_points,contact_pointA,contact_pointB,False
            else:
                return False, contact_normals_old, contact_points,contact_pointA,contact_pointB, True 
        closestA = check_dist_face_from_edge(partIDA, edgeidA, contact_pointB,contact_pointA,angle_tol)
        #assert closestA == check_dist_face_from_edge(partIDA,get_rev_edgeid(partIDA,edgeidA),contact_pointB,contact_pointA,tol)
        closestB = check_dist_face_from_edge(partIDB, edgeidB, contact_pointA,contact_pointB,angle_tol)
        #assert closestB == check_dist_face_from_edge(partIDB,get_rev_edgeid(partIDB,edgeidB),contact_pointA,contact_pointB,tol)
        if closestA and closestB:
            pass
            #wp.printf("faces are not parallel, consider decreasing the tolerance\n")
        #if True:#closestA and closestB:
            #wp.printf("edge to edge case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
        else:
            #wp.printf("edge to edge case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
        #return True,contact_normals,contact_points,contact_pointA,contact_pointB,False
    #edge cases
    if (query[0]>=(1.-tolrelA) or query[0]<=0.+tolrelA) and (query[1]>=(1.-tolrelB) or query[1]<=(0.+tolrelB)):
        #wp.printf("corner to corner case\n")
        #check if one of the corners can be improved
        if query[0] <= tolrelA and query[1] <= tolrelB:
            #wp.printf("corner to corner case: p0Aidx: %d, p0Bidx: %d\n", p0Aidx, p0Bidx)
            closestA = check_dist_edges_from_corner(partIDA, p0Aidx, p0B,exclude = edgeidA,tol=angle_tol)
            #wp.printf("closestA: %s\n", closestA)
            closestB = check_dist_edges_from_corner(partIDB, p0Bidx, p0A,exclude = edgeidB,tol=angle_tol)
            #wp.printf("closestB: %s\n", closestB)
            if closestA and closestB:
                #wp.printf("corner to corner case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
                return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
            #wp.printf("corner to corner case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
        else:
            #wp.printf("cc nA: %f %f %f, n_edges: %f %f %f\n",nA[0],nA[1],nA[2],contact_normals[0],contact_normals[1],contact_normals[2])
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, False
    elif (query[0]<1.0-tolrelA and query[0]>0.+tolrelA and (query[1]<=0.+tolrelB)): #(query[0]<=tolrelA and query[1]<1.0-tolrelB) or (query[0]<1.0-tolrelA and query[1]<=tolrelB):
        closestB = check_dist_edges_from_corner(partIDB, p0Bidx, contact_pointA,exclude = edgeidB,tol=angle_tol)
        closestA = check_dist_face_from_edge(partIDA, edgeidA, contact_pointB,contact_pointA,angle_tol)
        if closestB and closestA:
            #wp.printf("edge to corner case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
        else:
            #wp.printf("corner to edge case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
    elif query[0]<=0.+tolrelA and query[1]<1.-tolrelB and query[1]>0.+tolrelB:
        closestA = check_dist_edges_from_corner(partIDA, p0Aidx, contact_pointB,exclude = edgeidA,tol=angle_tol)
        closestB = check_dist_face_from_edge(partIDB, edgeidB, contact_pointA,contact_pointB,angle_tol)
        if closestA and closestB:
            #wp.printf("edge to corner case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
        else:
            #wp.printf("edge to corner case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
    else:
        #duplicated edges take care of the last cases
        return False, contact_normals, contact_points,contact_pointA,contact_pointB, False
def check_contact_batch_wrapper(partA_wp,partA_wpid,partB:List[trimesh.Trimesh],n_blocks_t,
                                 tol=1e-5,identical_tol =None,angle_tol = 1e-2,n_points_max=8, n_edges_max=36): 
    n_old = wp.array(n_blocks_t, dtype=wp.int32)
    n_contact_max = (n_edges_max*n_edges_max + n_points_max*2)*n_blocks_t.max()
    #print(f"partA_shape: {partA_wp.shape}, partB_shape: {len(partB)}, n_blocks_t: {n_blocks_t}, n_contact_max: {n_contact_max}")
    if identical_tol is None:
        identical_tol = tol/10
    n_batch = len(partB)
    contact_normals_out = wp.zeros(dtype=wp.vec3, shape=(n_batch, n_contact_max))
    contact_points_out = wp.zeros(dtype=wp.vec3, shape=(n_batch, n_contact_max))
    contact_points_outA = wp.zeros(dtype=wp.vec3, shape=(n_batch, n_contact_max))
    contact_points_outB = wp.zeros(dtype=wp.vec3, shape=(n_batch, n_contact_max))
    istouching_out = wp.zeros(dtype=wp.bool, shape=(n_batch, n_contact_max))
    isdegen_out = wp.zeros(dtype=wp.bool, shape=(n_batch, n_contact_max))
    contact_partA = wp.zeros(dtype=wp.int32, shape=(n_batch, n_contact_max))

    partB_wp = [wp.Mesh(wp.array(partB[i].vertices,dtype=wp.vec3),
                wp.array(partB[i].faces.flatten(),dtype=wp.int32)) for i in range(len(partB))]
    partB_wpid = wp.array([m.id for m in partB_wp], dtype=wp.uint64)
    wp.launch(check_contact_ker, dim=(n_batch, n_contact_max), inputs=(wp.array(partA_wpid,dtype=wp.uint64),partB_wpid, tol, angle_tol,n_old, n_edges_max, n_points_max),
                outputs=(contact_normals_out, contact_points_out,contact_points_outA,contact_points_outB, istouching_out, isdegen_out, contact_partA))
    
    contacts = [[] for i in range(n_batch)]
    contacts_degen = [[] for i in range(n_batch)]
    contact_partA = contact_partA.numpy()
    contact_normals_out = contact_normals_out.numpy()
    contact_points_out = contact_points_out.numpy()
    contact_points_outA = contact_points_outA.numpy()
    contact_points_outB = contact_points_outB.numpy()
    isdegen_out = isdegen_out.numpy()
    istouching_out = istouching_out.numpy()#| isdegen_out #combine touching and degenerate contacts
    dir_AB = contact_points_outB[istouching_out] - contact_points_outA[istouching_out]
    valid_normals = np.einsum("jk,jk->j",contact_normals_out[istouching_out],dir_AB)>0
    istouching_out[istouching_out] = valid_normals
    idx_b,idx_c = np.nonzero(istouching_out)
    idx_b_degen,idx_c_degen = np.nonzero(isdegen_out)
    logtol_pos = -int(np.log10(identical_tol))
    logtol_angle = -int(np.log10(angle_tol))
    for i in range(n_batch):
        if any(istouching_out[i]):
            contact_a = contact_partA[i,idx_c[idx_b==i]]
            contact_normals = contact_normals_out[i,idx_c[idx_b==i]]
            contact_normals = np.round(contact_normals,logtol_angle)
            contact_normals = contact_normals/ np.linalg.norm(contact_normals,axis=1)[:,None] #renormalize normals
            contact_points = contact_points_out[i,idx_c[idx_b==i]]
            contact_points = np.round(contact_points,logtol_pos)
            contact_pointsA = contact_points_outA[i,idx_c[idx_b==i]]
            contact_pointsB = contact_points_outB[i,idx_c[idx_b==i]]
            nu,nidx = np.unique(contact_normals,return_inverse=True,axis = 0)
            nu = nu/ np.linalg.norm(nu,axis=1)[:,None] #normalize normals
            for k, n in enumerate(nu):
                ida = contact_a[nidx==k]
                uida,uidaidx = np.unique(ida,return_inverse=True)
                for j, uida_id in enumerate(uida):
                    points = contact_points[nidx==k][uidaidx==j]
                    pointsA = contact_pointsA[nidx==k][uidaidx==j]
                    pointsB = contact_pointsB[nidx==k][uidaidx==j]
                    points,idx = np.unique(points,return_index=True,axis=0)
                    contact = {'partIDA':uida_id,
                               'partIDB':n_blocks_t[i],
                                'normal':n,
                                'points':points,
                                'pointsA':pointsA[idx],
                                'pointsB':pointsB[idx]}
                    contacts[i].append(contact)
    #degenerate contacts
    for i in range(n_batch):
        if any(isdegen_out[i]):
            contact_a = contact_partA[i,idx_c_degen[idx_b_degen==i]]
            contact_normals = contact_normals_out[i,idx_c_degen[idx_b_degen==i]]
            contact_normals = np.round(contact_normals,logtol_angle)
            contact_points = contact_points_out[i,idx_c_degen[idx_b_degen==i]]
            contact_points = np.round(contact_points,logtol_pos)
            contact_pointsA = contact_points_outA[i,idx_c_degen[idx_b_degen==i]]
            contact_pointsB = contact_points_outB[i,idx_c_degen[idx_b_degen==i]]
            nu,nidx = np.unique(contact_normals,return_inverse=True,axis = 0)
            nu = nu/ np.linalg.norm(nu,axis=1)[:,None] #normalize normals
            for k, n in enumerate(nu):
                ida = contact_a[nidx==k]
                uida,uidaidx = np.unique(ida,return_inverse=True)
                for j, uida_id in enumerate(uida):
                    points = contact_points[nidx==k][uidaidx==j]
                    pointsA = contact_pointsA[nidx==k][uidaidx==j]
                    pointsB = contact_pointsB[nidx==k][uidaidx==j]
                    points,idx = np.unique(points,return_index=True,axis=0)
                    contact_degen = {'partIDA':uida_id,
                               'partIDB':n_blocks_t[i],
                                'normal':n,
                                'points':points,
                                'pointsA':pointsA[idx],
                                'pointsB':pointsB[idx]}
                    contacts_degen[i].append(contact_degen)
    return contacts,contacts_degen
@wp.kernel
def check_contact_ker(partA:wp.array(dtype=wp.uint64,ndim=2),# type: ignore
                      partB:wp.array(dtype=wp.uint64,ndim=1),# type: ignore
                      tol:wp.float32,
                      angle_tol:wp.float32,
                      n_old:wp.array(dtype=wp.int32, ndim=1),  # type: ignore
                      n_edges_max:wp.int32, 
                      n_points_max:wp.int32, 
                      contact_normals_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                      contact_points_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                      contact_pointsA_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                      contact_pointsB_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                      istouching_out:wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                      isdegen_out:wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                      contact_partA:wp.array(dtype=wp.int32, ndim=2),  # type: ignore
                      ):
    batchid,idx_contact = wp.tid()
    partAid, contact_type, idxA,idxB = get_inv_contact_index(idx_contact, n_edges_max, n_points_max)
    if partAid > n_old[batchid]:
        return 
    if contact_type == 0:  # edge-edge contact
        #wp.printf("edge-edge contact: %d %d %d %d %d\n", batchid, partAid, idxA, idxB,idx_contact)
        edgeidA = idxA
        edgeidB = idxB
        if edgeidA >= wp.mesh_get(partA[batchid,partAid]).indices.shape[0] or edgeidB >= wp.mesh_get(partB[batchid]).indices.shape[0]:
            return
        is_touching, n_ee, p_ee ,p_eeA,p_eeB, is_degen = check_contact_edge_f(partA[batchid,partAid], partB[batchid],edgeidA,edgeidB,tol,angle_tol)
        istouching_out[batchid,idx_contact] = is_touching
        isdegen_out[batchid,idx_contact] = is_degen
        contact_partA[batchid,idx_contact] = partAid
        contact_points_out[batchid,idx_contact] = p_ee
        contact_pointsA_out[batchid,idx_contact] = p_eeA
        contact_pointsB_out[batchid,idx_contact] = p_eeB
        contact_normals_out[batchid,idx_contact] = n_ee
    if contact_type == 1:  # corner-face contact
        #wp.printf("corner-face contact: %d %d %d %d %d\n", batchid, partAid, idxA, idxB,idx_contact)
        pointidA = idxA
        if pointidA >= wp.mesh_get(partA[batchid,partAid]).points.shape[0]:
            wp.printf("pointidA out of bounds: %d >= %d", pointidA, wp.mesh_get(partA[batchid,partAid]).points.shape[0])
            return
        is_touching, n_cf, p_cf,p_cfA,p_cfB = check_contact_cornerface_f(partA[batchid,partAid], pointidA, partB[batchid], tol)
        if is_touching and wp.length(n_cf) < tol:
            wp.printf("corner-face contact: %d %d %d %d %d\n", batchid, partAid, idxA, idxB,idx_contact)
            wp.printf("contact normal is too small: %f %f %f\n", n_cf[0], n_cf[1], n_cf[2])
        istouching_out[batchid,idx_contact] = is_touching
        #isdegen_out[batchid,idx_contact] = False #save up some memory
        contact_partA[batchid,idx_contact] = partAid
        contact_points_out[batchid,idx_contact] = p_cf
        contact_pointsA_out[batchid,idx_contact] = p_cfA
        contact_pointsB_out[batchid,idx_contact] = p_cfB
        contact_normals_out[batchid,idx_contact] = -n_cf
    if contact_type == 2:  # face-corner contact
        pointidB = idxB
        if pointidB >= wp.mesh_get(partB[batchid]).points.shape[0]:
            return
        is_touching, n_fc, p_fc,p_fcB,p_fcA = check_contact_cornerface_f(partB[batchid], pointidB, partA[batchid,partAid], tol)
        contact_partA[batchid,idx_contact] = partAid
        istouching_out[batchid,idx_contact] = is_touching
        #isdegen_out[batchid,idx_contact] = False #save up some memory
        contact_points_out[batchid,idx_contact] = p_fc
        contact_pointsA_out[batchid,idx_contact] = p_fcA
        contact_pointsB_out[batchid,idx_contact] = p_fcB
        contact_normals_out[batchid,idx_contact] = n_fc

def check_contact_wrapper(partA:trimesh.Trimesh,partB:trimesh.Trimesh,
                          tol=1e-5):
    """
    Check if two parts are in contact.
    """
    partAwp = wp.Mesh(wp.array(partA.vertices,dtype=wp.vec3),wp.array(partA.faces.flatten(),dtype=wp.int32))
    partBwp = wp.Mesh(wp.array(partB.vertices,dtype=wp.vec3),wp.array(partB.faces.flatten(),dtype=wp.int32))

    contact_normals_ee = wp.zeros(dtype=wp.vec3, shape=(partAwp.indices.shape[0],partBwp.indices.shape[0]))
    contact_points_ee = wp.zeros(dtype=wp.vec3, shape=(partAwp.indices.shape[0], partBwp.indices.shape[0]))
    istouching_ee = wp.zeros(dtype=wp.bool, shape=(partAwp.indices.shape[0], partBwp.indices.shape[0]))
    isdegen_ee = wp.zeros(dtype=wp.bool, shape=(partAwp.indices.shape[0], partBwp.indices.shape[0]))
    wp.launch(check_contact_edge,dim = (partAwp.indices.shape[0], partBwp.indices.shape[0]), inputs=(partAwp.id, partBwp.id,tol), outputs=(contact_normals_ee, contact_points_ee,istouching_ee,isdegen_ee))
    
    contact_normals_cf = wp.zeros(dtype=wp.vec3, shape=(partAwp.points.shape[0]))
    contact_points_cf = wp.zeros(dtype=wp.vec3, shape=(partAwp.points.shape[0]))
    istouching_cf = wp.zeros(dtype=wp.bool, shape=(partAwp.points.shape[0]))
    wp.launch(check_contact_cornerface,dim = partAwp.points.shape[0], inputs=(partAwp.id, partBwp.id,tol*2), outputs=(contact_normals_cf, contact_points_cf,istouching_cf))
    
    contact_normals_fc = wp.zeros(dtype=wp.vec3, shape=(partBwp.points.shape[0]))
    contact_points_fc = wp.zeros(dtype=wp.vec3, shape=(partBwp.points.shape[0]))
    istouching_fc = wp.zeros(dtype=wp.bool, shape=(partBwp.points.shape[0]))
    wp.launch(check_contact_cornerface,dim = partBwp.points.shape[0], inputs=(partBwp.id, partAwp.id,tol*2), outputs=(contact_normals_fc, contact_points_fc,istouching_fc))
    
    contact_points = np.vstack([contact_points_ee.numpy()[istouching_ee.numpy()], contact_points_cf.numpy()[istouching_cf.numpy()], contact_points_fc.numpy()[istouching_fc.numpy()]])
    contact_normals = np.vstack([contact_normals_ee.numpy()[istouching_ee.numpy()], -contact_normals_cf.numpy()[istouching_cf.numpy()], contact_normals_fc.numpy()[istouching_fc.numpy()]])
    normals_degen = contact_normals_ee.numpy()[isdegen_ee.numpy()]

    nu,nidx = np.unique(contact_normals,return_inverse=True,axis = 0)
    contacts = []
    for k,n in enumerate(nu):
        contact_points_n = np.unique(contact_points[nidx==k],axis=0)
        #contact_points_n = np.unique(np.round(contact_points_n,int(-np.log10(tol)+1)),axis=0)
        contacts.append({"points":contact_points_n,
                        "partIDA":None,  
                        "partIDB":None, 
                        "normal":n})
    return False,contacts,normals_degen

@wp.func
def get_edge_vertices(partID:wp.uint64, edgeid:wp.int32):
    """
    Get the vertices of an edge in a mesh.
    """
    p0 = wp.mesh_get_point(partID,edgeid)
    p0idx = wp.mesh_get_index(partID, edgeid)
    if (edgeid) % 3 == 2:
        p1 = wp.mesh_get_point(partID,edgeid-2)
        p1idx = wp.mesh_get_index(partID, edgeid-2)
    else:
        p1 = wp.mesh_get_point(partID,edgeid+1)
        p1idx = wp.mesh_get_index(partID, edgeid+1)
    return p0, p1, p0idx, p1idx
@wp.func
def get_triangle_vertices(partID:wp.uint64, pointid:wp.int32):
    """
    Get the vertices of a triangle in a mesh.
    """
    p0 = wp.mesh_get_point(partID,pointid)
    p0idx = wp.mesh_get_index(partID, pointid)
    if (pointid) % 3 == 2:
        p1 = wp.mesh_get_point(partID,pointid-2)
        p1idx = wp.mesh_get_index(partID, pointid-2)
    else:
        p1 = wp.mesh_get_point(partID,pointid+1)
        p1idx = wp.mesh_get_index(partID, pointid+1)

    if (pointid) % 3 == 0:
        p2 = wp.mesh_get_point(partID,pointid+2)
        p2idx = wp.mesh_get_index(partID, pointid+2)
    else:
        p2 = wp.mesh_get_point(partID,pointid-1)
        p2idx = wp.mesh_get_index(partID, pointid-1)
    return p0, p1, p2, p0idx, p1idx, p2idx
@wp.func
def get_contact_index(partidx:wp.int32, typeidx:wp.int32, idxA:wp.int32,idxB:wp.int32, n_edges_max:wp.int32,n_points_max:wp.int32):
    """
    Flatten the contact indices into a single index.
    type: 0 for edge-edge, 1 for corner-face, 2 for face-corner
    """
    sizepart = (n_edges_max*n_edges_max)+2*(n_points_max)
    if typeidx == 0:
        offset_type = 0
        sizeA = n_edges_max
        sizeB = 1
    elif typeidx == 1:
        offset_type = n_edges_max*n_edges_max
        sizeA = 1
        sizeB = 0
    elif typeidx == 2:
        offset_type = n_edges_max*n_edges_max + (n_points_max)
        sizeA = 0
        sizeB = 1
    return partidx*sizepart + offset_type + idxA*sizeA + idxB*sizeB
@wp.func
def get_inv_contact_index(idx:wp.int32, n_edges_max:wp.int32, n_points_max:wp.int32):
    """
    Inverse of get_contact_index.
    """
    sizepart = (n_edges_max*n_edges_max)+2*(n_points_max)
    partidx = idx // sizepart
    idx = idx % sizepart
    if idx < n_edges_max*n_edges_max:
        typeidx = 0
        idxA = idx // n_edges_max
        idxB = idx % n_edges_max
    elif idx < n_edges_max*n_edges_max + n_points_max:
        typeidx = 1
        idxA = idx - n_edges_max*n_edges_max
        idxB = 0
    else:
        typeidx = 2
        idxA = 0
        idxB = idx - (n_edges_max*n_edges_max + n_points_max)
    return partidx, typeidx, idxA, idxB
@wp.func
def get_edge_from_vertices(partID:wp.uint64, p0idx:wp.int32, p1idx:wp.int32):
    """
    Get the edge id from the vertices in a mesh.
    """
    nidx = wp.mesh_get(partID).indices.shape[0]
    for i in range(0,nidx,3):
        for j in range(3):
            if (wp.mesh_get_index(partID,i+j) == p0idx and wp.mesh_get_index(partID,i+(j+1)%3) == p1idx):
                return wp.int32(i+j)
    return wp.int32(-1)  # Edge not found
@wp.func
def get_adj_normal(partID:wp.uint64, edgeid:wp.int32):
    """
    Get the adjacent normal of an edge in a mesh.
    """
    p0, p1, p0idx, p1idx = get_edge_vertices(partID, edgeid)
    eidx = get_edge_from_vertices(partID, p1idx, p0idx)
    if eidx == -1:
        wp.printf("Error: edge not found for vertices %d and %d\n", p1idx, p0idx)
        return wp.vec3(0.0, 0.0, 0.0)
    n = wp.mesh_eval_face_normal(partID, eidx/3)
    return wp.normalize(n)
@wp.func
def get_rev_edgeid(partID:wp.uint64, edgeid:wp.int32):
    """
    Get the reverse edge id of an edge in a mesh.
    """
    _, _, p0idx, p1idx = get_edge_vertices(partID, edgeid)
    return get_edge_from_vertices(partID, p1idx, p0idx)
@wp.func
def mesh_eval_edges_length(partID:wp.uint64, faceid:wp.int32):
    """
    Evaluate the length of an edge in a mesh.
    """
    p0, p1,p2,_, _, _ = get_triangle_vertices(partID,faceid)
    return wp.length(p1 - p0),wp.length(p2 - p0)
@wp.func
def check_dist_edges_from_corner(partID:wp.uint64, pointid:wp.int32, point:wp.vec3,exclude:wp.int32 = wp.int32(-1),tol:wp.float32 = 1e-2,
                                 n_edges:wp.int32 = 36):
    """
    Check if all edges go away from a corner in a mesh.
    """
    nidx = wp.mesh_get(partID).indices.shape[0]
    if exclude >= 0:
        p0,p1,p0idx,p1idx = get_edge_vertices(partID, exclude)
        e = wp.normalize(p1 - p0)
        d_min = wp.dot(wp.normalize(point - p0), e)
        dir_ref = wp.normalize(point - p0)
        e_ref = wp.normalize(p1 - p0)
    else:
        d_min = 0.0
    #wp.printf("check_dist_edges_from_corner: pointid: %d, exclude: %d, d_min: %f\n", pointid, exclude, d_min)
    for i in range(0,nidx,3):
        for j in range(3):
            if (i+j != exclude) and (wp.mesh_get_index(partID,i+j) == pointid):
                p0idx = i+j
                p1idx = i+(j+1)%3
                p0 = wp.mesh_get_point(partID, p0idx)
                p1 = wp.mesh_get_point(partID, p1idx)
                e = wp.normalize(p1 - p0)
                d = wp.dot(wp.normalize(point - p0), e)
                dir_e = wp.normalize(e - wp.dot(e, e_ref) * e_ref)
                #wp.printf("check_dist_edges_from_corner: p0: %f %f %f, p1: %f %f %f, point: %f %f %f, d: %f\n",
                #         p0[0], p0[1], p0[2], p1[0], p1[1], p1[2],
                #         point[0], point[1], point[2], d)
                if wp.dot(dir_ref, dir_e) > tol:
                    #wp.printf("check_dist_edges_from_corner: edge %d goes away from the corner: %f\n", i+j, wp.dot(dir_ref, dir_e))
                    return  False
                #if d > d_min:
                #    return False
    #wp.printf("check_dist_edges_from_corner: all edges go away from the corner\n")
    return True
@wp.func
def check_dist_face_from_edge(partID:wp.uint64, edgeid:wp.int32, point:wp.vec3,pointonedge:wp.vec3,tol:wp.float32):
    """
    Check if all edges go away from a corner in a mesh.
    """
    
    d = wp.normalize(point - pointonedge)
    p0,p1,p2,p0idx, p1idx, p2idx = get_triangle_vertices(partID, edgeid)
    #wp.printf("check_dist_face_from_edge: edgeid: %d, p0idx: %d, p1idx: %d, p2idx: %d\n", edgeid, p0idx, p1idx, p2idx)
    dir1 = wp.normalize(p1 - p0)
    otherdir = wp.normalize(p2-p0-wp.dot(p2-p0, dir1)*dir1)
    closer = wp.dot(d,otherdir)>= tol
    #wp.printf("check_dist_face_from_edge: d: %f %f %f, p2-p0: %f %f %f, closer: %f\n",
    #         d[0], d[1], d[2], p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2], wp.dot(d,p2-p0))
    if closer:
        #wp.printf("check_dist_face_from_edge: face goes away from the corner: %f\n",wp.dot(d,wp.normalize(p2-p0)))
        #wp.printf("check_dist_face_from_edge: face goes away from the corner: %f pe-p0 %f\n",wp.dot(d,wp.normalize(p2-p0)),  wp.length(pointonedge-p0))
        return False
    rev_edgeid = get_edge_from_vertices(partID, p1idx, p0idx)
    
    p0,p1,p2,p0idx, p1idx, p2idx = get_triangle_vertices(partID, rev_edgeid)
    #wp.printf("check_dist_face_from_edge: revedgeid: %d, p0idx: %d, p1idx: %d, p2idx: %d\n", rev_edgeid, p0idx, p1idx, p2idx)
    dir1 = wp.normalize(p1 - p0)
    otherdir = wp.normalize(p2-p0-wp.dot(p2-p0, dir1)*dir1)
    closer = wp.dot(d,otherdir)>= tol
    #wp.printf("check_dist_face_from_edge: rev: d: %f %f %f, p2-p0: %f %f %f, closer: %f\n",
    #         d[0], d[1], d[2], p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2], wp.dot(d,p2-p0))
    if closer:
        #wp.printf("check_dist_face_from_edge: rev face goes away from the corner: %f pe-p0 %f\n",wp.dot(d,wp.normalize(p2-p0)),  wp.length(pointonedge-p0))
        return False
    #wp.printf("check_dist_face_from_edge: all faces go away from the corner\n")
    return True
if __name__ == "__main__":
    import os
    import polyscope as ps

    obj_folder_path = './gym/static_envs/simulator/cont3Dsimulator/data/arch'
    objs = []
    for f in os.listdir(obj_folder_path):
        file_path = os.path.join(obj_folder_path, f)
        if os.path.isfile(file_path) and f[: -4].isdigit() and f[-3: ] == 'obj': 
            trimesh_obj = trimesh.load(file_path, force = 'mesh', )
            partID = int(f[:-4])
            assert trimesh_obj.is_watertight, "Sanitize the blocks first"
            trimesh_obj.fix_normals()
            objs.append(trimesh_obj)
    wp.set_device("cpu")
    contact_normals = wp.array(dtype=wp.vec3, shape=(0))
    contact_points = wp.array(dtype=wp.vec3, shape=(0))
    contacts_degen = wp.array(dtype=wp.vec3, shape=(0))
    p,c,d = check_contact_wrapper(objs[0], objs[1], tol=1e-5)
    print(f"n_contacts:{len(c)}")
    for ci in c:
        print(f"Contact points: {ci['points']}, normal: {ci['normal']}")
    partA = wp.Mesh(wp.array(objs[0].vertices,dtype=wp.vec3),wp.array(objs[0].faces.flatten(),dtype=wp.int32))
    partB = wp.Mesh(wp.array(objs[1].vertices,dtype=wp.vec3),wp.array(objs[1].faces.flatten(),dtype=wp.int32))
    #wp.launch(check_contact_edge,dim = [partA.indices.shape[0],partB.indices.shape[0]], inputs=(partA.id, partB.id,1e-5,2.0), outputs=(contact_normals, contacts_degen,contact_points))
    #wp.launch(check_contact,dim = partB.points.shape[0], inputs=(partA.id, partB.id), outputs=(contact_normals, contacts_degen,contact_points))
    