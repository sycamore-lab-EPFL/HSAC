from typing import List
import scipy
import trimesh
import numpy as np
import warp as wp
# The init() function prints the directory of the kernel cache which contains the .cpp files
# generated from Warp kernels. You can put breakpoints in these C++ files through Visual Studio Code,
# but it's generally more convenient to use wp.breakpoint(). See the example below.
global graph_contact
global graph_translation
graph_contact= None
graph_translation = None
def init_kernels(n_batchs,n_blocks_max,n_points_max,n_edges_max,mode="release",verbose=True):
    wp.config.mode = mode
    wp.config.verbose = verbose
    wp.init()
    n_batchs_s = wp.static(wp.int32(n_batchs))
    n_blocks_max_s = wp.static(wp.int32(n_blocks_max))
    n_points_max_s = wp.static(wp.int32(n_points_max))
    n_edges_max_s = wp.static(wp.int32(n_edges_max))
    n_contact_max_s =  wp.static((n_edges_max*n_edges_max + n_points_max*2)*n_blocks_max)
    @wp.kernel
    def min_dist_ker(partA_wpid:wp.array2d(dtype=wp.uint64), # type: ignore
                     partB_wpid:wp.array1d(dtype=wp.uint64), # type: ignore
                     direction: wp.array1d(dtype=wp.vec3), # type: ignore
                     ignore_batch: wp.array1d(dtype=wp.bool), # type: ignore
                     n_old: wp.array1d(dtype=wp.int32), # type: ignore
                     max_dist: wp.float32,
                     tol:wp.float32,
                     distances_out:wp.array2d(dtype=wp.float32) # type: ignore
                        ):
        bid, did = wp.tid()
        if ignore_batch[bid]:
            return
        partid, distance_type, idxA,idxB = get_inv_distance_index(did)
        if partid >= n_old[bid]:
            return
        #wp.printf("min_dist_ker: %d %d %d %d %d %d\n", bid, did, partid, distance_type, idxA, idxB)
        if distance_type == 0:
            distances_out[bid, did] = ray_edge_f(partA_wpid[bid,partid], partB_wpid[bid],idxA,idxB, direction[bid], tol, max_dist, True)
        elif distance_type == 1:
            distances_out[bid, did] = ray_edge_f(partA_wpid[bid,partid], partB_wpid[bid],idxA,idxB, direction[bid], tol, max_dist, False)
        elif distance_type == 2:
            distances_out[bid, did] = ray_corner_face_f(wp.mesh_get(partA_wpid[bid,partid]).points[idxA], partB_wpid[bid], direction[bid], max_dist)
        elif distance_type == 3:
            distances_out[bid, did] = ray_corner_face_f(wp.mesh_get(partB_wpid[bid]).points[idxB], partA_wpid[bid,partid], -direction[bid], max_dist)
        else:
            wp.printf("Invalid distance type: %d\n", distance_type)
            return
    @wp.kernel
    def check_contact_ker(partA:wp.array(dtype=wp.uint64,ndim=2),# type: ignore
                        partB:wp.array(dtype=wp.uint64,ndim=1),# type: ignore
                        ignore_batch:wp.array(dtype=wp.bool,ndim=1),# type: ignore
                        tol:wp.float32,
                        angle_tol:wp.float32,
                        n_old:wp.array(dtype=wp.int32, ndim=1),  # type: ignore
                        contact_normals_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                        contact_points_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                        contact_pointsA_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                        contact_pointsB_out:wp.array(dtype=wp.vec3, ndim=2),  # type: ignore
                        istouching_out:wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                        isdegen_out:wp.array(dtype=wp.bool, ndim=2),  # type: ignore
                        contact_partA:wp.array(dtype=wp.int32, ndim=2),  # type: ignore
                        ):
        batchid,idx_contact = wp.tid()
        
        if ignore_batch[batchid]:
            return
        
        partAid, contact_type, idxA,idxB = get_inv_contact_index(idx_contact) 
        #wp.breakpoint()
        if partAid >= n_old[batchid]:
            return
        #wp.printf("check_contact_ker: %d %d %d %d %d %d\n", batchid, idx_contact, partAid, contact_type, idxA, idxB)
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
        elif contact_type == 1:  # corner-face contact
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
        elif contact_type == 2:  # face-corner contact
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
        else:
            wp.printf("Invalid contact type: %d\n", contact_type)
            return
    @wp.func
    def ray_corner_face_f(cornerA:wp.vec3f,# type: ignore
                    partIDB:wp.uint64,
                    d:wp.vec3, # type: ignore
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
    @wp.func
    def ray_edge_f(partIDA:wp.uint64,
                partIDB:wp.uint64,
                edgeidA:wp.int32,
                    edgeidB:wp.int32,
                direction:wp.vec3, # type: ignore
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
            #remove points that are already in contact or going away
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
                tol_relB = tol/wp.length(p0B-p1B)/2.#tol*dist/wp.length(p0B-p1B)*factor
                tol_relA = tol/wp.length(p0A-p1A)/2.#tol*dist/wp.length(p0A-p1A)*factor
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
        query = wp.mesh_query_point_sign_normal(partIDB,point=pointA,max_dist= tol*10.,epsilon=tol/1000.)
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
    @wp.func
    def check_dist_edges_from_corner(partID:wp.uint64, pointid:wp.int32, point:wp.vec3,
                                     exclude:wp.int32 = wp.int32(-1),tol:wp.float32=1e-2):
        """
        Check if all edges go away from a corner in a mesh.
        """
        if exclude >= 0:
            p0,p1,p0idx,p1idx = get_edge_vertices(partID, exclude)
            e = wp.normalize(p1 - p0)
            d_min = wp.dot(wp.normalize(point - p0), e)
            dir_ref = wp.normalize(point - p0)
            e_ref = wp.normalize(p1 - p0)
        else:
            d_min = 0.0
        nidx = wp.mesh_get(partID).indices.shape[0]
        for i in range(0,n_edges_max_s,3):
            for j in range(3):
                if (i+j != exclude) and (wp.mesh_get_index(partID,i+j) == pointid):
                    p0idx = i+j
                    p1idx = i+(j+1)%3
                    #if p0idx < 0 or p1idx < 0 or p0idx >= nidx or p1idx >= nidx:
                    #    continue
                    if not (p0idx < 0 or p1idx < 0 or p0idx >= nidx or p1idx >= nidx):
                        p0 = wp.mesh_get_point(partID, p0idx)
                        p1 = wp.mesh_get_point(partID, p1idx)
                        e = wp.normalize(p1 - p0)
                        d = wp.dot(wp.normalize(point - p0), e)
                        dir_e = wp.normalize(e - wp.dot(e, e_ref) * e_ref)
                        if wp.dot(dir_ref, dir_e) > tol:
                            return False
        return True
    @wp.func
    def check_contact_edge_f(partIDA: wp.uint64,
                        partIDB: wp.uint64,
                        edgeidA:wp.int32,
                        edgeidB:wp.int32,
                        tol:wp.float32,
                        angle_tol:wp.float32,
                        #ret_arr:wp.array(dtype=wp.bool, ndim=2)# type: ignore
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
                #closestA = (return_array_edge_dir<angle_tol).all()
                #wp.printf("closestA: %s\n", closestA)
                closestB = check_dist_edges_from_corner(partIDB, p0Bidx, p0A,exclude = edgeidB,tol=angle_tol)
                #closestB = (return_array_edge_dir<angle_tol).all()
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
            #closestB = (return_array_edge_dir<angle_tol).all()
            closestA = check_dist_face_from_edge(partIDA, edgeidA, contact_pointB,contact_pointA,tol=angle_tol)
            #closestA = (return_array_edge_dir<angle_tol).all()
            if closestB and closestA:
                #wp.printf("edge to corner case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
                return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
            else:
                #wp.printf("corner to edge case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
                return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
        elif query[0]<=0.+tolrelA and query[1]<1.-tolrelB and query[1]>0.+tolrelB:
            closestA = check_dist_edges_from_corner(partIDA, p0Aidx, contact_pointB,exclude = edgeidA,tol=angle_tol)
            #closestA = (return_array_edge_dir<angle_tol).all()
            closestB = check_dist_face_from_edge(partIDB, edgeidB, contact_pointA,contact_pointB,tol=angle_tol)
            #closestB = (return_array_edge_dir<angle_tol).all()
            if closestA and closestB:
                #wp.printf("edge to corner case accepted: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
                return True, contact_normals, contact_points,contact_pointA,contact_pointB, False
            else:
                #wp.printf("edge to corner case refused: %f %f %f\n",contact_points[0], contact_points[1], contact_points[2])
                return False, contact_normals, contact_points,contact_pointA,contact_pointB, True
        else:
            #duplicated edges take care of the last cases
            return False, contact_normals, contact_points,contact_pointA,contact_pointB, False
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
    def get_inv_contact_index(idx:wp.int32):
        """
        Inverse of get_contact_index.
        """
        sizepart = (n_edges_max_s*n_edges_max_s)+2*(n_points_max_s)
        partidx = idx // sizepart
        idx = idx % sizepart
        if idx < n_edges_max_s*n_edges_max_s:
            typeidx = 0
            idxA = idx // n_edges_max_s
            idxB = idx % n_edges_max_s
        elif idx < n_edges_max_s*n_edges_max_s + n_points_max_s:
            typeidx = 1
            idxA = idx - n_edges_max_s*n_edges_max_s
            idxB = 0
        else:
            typeidx = 2
            idxA = 0
            idxB = idx - (n_edges_max_s*n_edges_max_s + n_points_max_s)
        return partidx, typeidx, idxA, idxB
    @wp.func
    def get_inv_distance_index(idx:wp.int32):
        """
        Inverse of get_distance_index.
        """
        sizepart = 2*(n_edges_max_s*n_edges_max_s)+2*(n_points_max_s)
        partidx = idx // sizepart
        idx = idx % sizepart
        if idx < n_edges_max_s*n_edges_max_s:
            #non-degen
            typeidx = 0
            idxA = idx // n_edges_max_s
            idxB = idx % n_edges_max_s
        elif idx < 2*n_edges_max_s*n_edges_max_s:
            #degen
            typeidx = 1
            idxA =(idx-n_edges_max_s*n_edges_max_s)// n_edges_max_s
            idxB =(idx-n_edges_max_s*n_edges_max_s)% n_edges_max_s
        elif idx < 2*n_edges_max_s*n_edges_max_s + n_points_max_s:
            #corner-face
            typeidx = 2
            idxA = idx - 2*n_edges_max_s*n_edges_max_s
            idxB = 0
        else:
            #face-corner
            typeidx = 3
            idxA = 0
            idxB = idx - (2*n_edges_max_s*n_edges_max_s + n_points_max_s)
        return partidx, typeidx, idxA, idxB
    @wp.func
    def get_edge_from_vertices(partID:wp.uint64, p0idx:wp.int32, p1idx:wp.int32):
        """
        Get the edge id from the vertices in a mesh.
        """
        nidx = wp.mesh_get(partID).indices.shape[0]
        for i in range(0,n_edges_max_s,3):
            for j in range(3):
                if i+(j+1)%3 < nidx and (wp.mesh_get_index(partID,i+j) == p0idx and wp.mesh_get_index(partID,i+(j+1)%3) == p1idx):
                    return wp.int32(i+j)
        return wp.int32(-1)  # Edge not found
    def translate_toAll_wrapper(old_blocks_ids,n_old_blocks,partBs,direction,tol,max_dist=np.inf,env_ids = None,use_graph_capture=False):
        #print(f"translate_toAll_wrapper: n_old_blocks: {n_old_blocks}, env_ids: {env_ids}")
        global graph_translation
        #with wp.ScopedTimer("init translate",color='red',use_nvtx=True):
        n_dists = n_blocks_max_s*(2*(n_edges_max_s*n_edges_max_s)+2*(n_points_max_s))
        distances = wp.full(value=max_dist,dtype=wp.float32, shape=(n_batchs_s, n_dists))
        n_old = wp.array(n_old_blocks, dtype=wp.int32)
        if env_ids is None:
            ignore_batch = wp.zeros((n_batchs_s),dtype=wp.bool)
            partB_wp = [wp.Mesh(wp.array(partBs[i].vertices,dtype=wp.vec3),
                                wp.array(partBs[i].faces.flatten(),dtype=wp.int32)) for i in range(n_batchs_s)]
            partB_wpid = wp.array([m.id for m in partB_wp], dtype=wp.uint64)
            direction = wp.array(direction,dtype=wp.vec3)
        else:
            ignore_batch = np.ones((n_batchs_s),dtype=bool)
            ignore_batch[env_ids] = False
            ignore_batch = wp.array(ignore_batch,dtype=wp.bool)
            partB_wp = [wp.Mesh(wp.array(partBs[i].vertices,dtype=wp.vec3),
                                wp.array(partBs[i].faces.flatten(),dtype=wp.int32)) for i in range(len(env_ids))]
            partB_wpid = np.zeros((n_batchs_s),dtype=np.uint64)
            partB_wpid[env_ids] = np.array([m.id for m in partB_wp], dtype=np.uint64)
            partB_wpid = wp.array(partB_wpid,dtype=wp.uint64)
            direction_full = np.zeros((n_batchs_s,3),dtype=np.float32)
            direction_full[env_ids] = direction
            direction = wp.array(direction_full,dtype=wp.vec3)
    
        #with wp.ScopedTimer("translate ker",color='green',use_nvtx=True):
        wp.launch(min_dist_ker,
                dim = (n_batchs_s,n_dists),
                inputs=(wp.array(old_blocks_ids,dtype=wp.uint64),
                        partB_wpid,
                        direction,
                        ignore_batch,
                        n_old,
                        wp.float32(max_dist),
                        tol),
                        outputs=(distances,))
        #with wp.ScopedTimer("translate numpy",color='magenta',use_nvtx=True):
        wp.synchronize()
        distances = distances.numpy()-tol/2.
        min_dist = np.min(distances,axis=1)
        size_part = 2*(n_edges_max_s*n_edges_max_s)+2*(n_points_max_s)
        min_dist_over = np.min(distances.reshape(n_batchs_s,-1,size_part)[:,:,n_edges_max_s*n_edges_max_s:],axis=(1,2))
        return min_dist[env_ids], min_dist_over[env_ids]
    def check_contact_batch_wrapper(partA_wp,partA_wpid,partB:List[trimesh.Trimesh],n_blocks_t,
                                 tol=1e-5,identical_tol =None,angle_tol = 1e-2, use_graph_capture=False,env_ids = None): 
        #print(f"check_contact_batch_wrapper: n_blocks_t: {n_blocks_t}, env_ids: {env_ids}")
        global graph_contact
        #with wp.ScopedTimer("init contact",color='blue',use_nvtx=True):
        n_old = wp.array(n_blocks_t, dtype=wp.int32)
        
        #print(f"partA_shape: {partA_wp.shape}, partB_shape: {len(partB)}, n_blocks_t: {n_blocks_t}, n_contact_max: {n_contact_max}")
        if identical_tol is None:
            identical_tol = tol/10
        if env_ids is None:
            ignore_batch = wp.zeros((n_batchs_s),dtype=wp.bool)
            partB_wp = [wp.Mesh(wp.array(partB[i].vertices,dtype=wp.vec3),
                                wp.array(partB[i].faces.flatten(),dtype=wp.int32)) for i in range(n_batchs_s)]
            partB_wpid = wp.array([m.id for m in partB_wp], dtype=wp.uint64)
        else:
            ignore_batch = np.ones((n_batchs_s),dtype=bool)
            ignore_batch[env_ids] = False
            ignore_batch = wp.array(ignore_batch,dtype=wp.bool)
            # FIX: Create dummy mesh for ignored batches to avoid mesh_get(0) segfault
            partB_wp = []
            partB_wpid_list = np.zeros((n_batchs_s),dtype=np.uint64)
            # Create meshes for active batches
            for i, env_id in enumerate(env_ids):
                partB_wp.append(wp.Mesh(wp.array(partB[i].vertices,dtype=wp.vec3),
                                        wp.array(partB[i].faces.flatten(),dtype=wp.int32)))
                partB_wpid_list[env_id] = partB_wp[i].id
            # Create dummy empty meshes for ignored batches (prevents null mesh_get calls)
            dummy_vertices = wp.array(np.array([[0, 0, 0]], dtype=np.float32), dtype=wp.vec3)
            dummy_faces = wp.array(np.array([0, 0, 0], dtype=np.int32), dtype=wp.int32)
            for i in range(n_batchs_s):
                if i not in env_ids:
                    dummy_mesh = wp.Mesh(dummy_vertices, dummy_faces)
                    partB_wpid_list[i] = dummy_mesh.id
            partB_wpid = wp.array(partB_wpid_list, dtype=wp.uint64)
        contact_normals_out = wp.zeros(dtype=wp.vec3, shape=(n_batchs_s, n_contact_max_s))
        contact_points_out = wp.zeros(dtype=wp.vec3, shape=(n_batchs_s, n_contact_max_s))
        contact_points_outA = wp.zeros(dtype=wp.vec3, shape=(n_batchs_s, n_contact_max_s))
        contact_points_outB = wp.zeros(dtype=wp.vec3, shape=(n_batchs_s, n_contact_max_s))
        istouching_out = wp.zeros(dtype=wp.bool, shape=(n_batchs_s, n_contact_max_s))
        isdegen_out = wp.zeros(dtype=wp.bool, shape=(n_batchs_s, n_contact_max_s))
        contact_partA = wp.zeros(dtype=wp.int32, shape=(n_batchs_s, n_contact_max_s))
        
        
        #with wp.ScopedTimer("contact ker",color='magenta',use_nvtx=True):
        wp.launch(check_contact_ker, dim=(n_batchs_s, n_contact_max_s),
                            inputs=(wp.array(partA_wpid,dtype=wp.uint64),partB_wpid,ignore_batch, tol, angle_tol,n_old),
                            outputs=(contact_normals_out, contact_points_out,contact_points_outA,contact_points_outB, istouching_out, isdegen_out, contact_partA))
        
        # Ensure GPU operations are complete before accessing data        
        contacts = [[] for i in range(n_batchs_s)]
        contacts_degen = [[] for i in range(n_batchs_s)]
        #with wp.ScopedTimer("contact ker numpy",color='green',use_nvtx=True):
        wp.synchronize()
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
        for i in env_ids:
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
                        if points.shape[0]>=3:
                            try:
                                hull = scipy.spatial.ConvexHull(points[:,:2])
                            except scipy.spatial.qhull.QhullError:
                                pass
                            else:
                                points = points[hull.vertices]
                                pointsA = pointsA[hull.vertices]
                                pointsB = pointsB[hull.vertices]
                        contact = {'partIDA':uida_id,
                                'partIDB':n_blocks_t[i],
                                    'normal':n,
                                    'points':points,
                                    'pointsA':pointsA,
                                    'pointsB':pointsB,
                                    'batchID':i}
                        contacts[i].append(contact)
        #degenerate contacts
        for i in env_ids:
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
    return translate_toAll_wrapper, check_contact_batch_wrapper