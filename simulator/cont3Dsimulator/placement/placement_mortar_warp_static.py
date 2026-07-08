from typing import List
import scipy
import trimesh
import numpy as np
import warp as wp

global graph_contact
global graph_translation
graph_contact = None
graph_translation = None

# Minimum number of points for a contact to be kept. A load-bearing contact is
# an area patch (>= 3 points); single-point and edge (1-2 point) touches are
# grazing corner/edge contacts -- e.g. a block's bottom corner falling within
# mortar_thickness of a neighbouring ground brick -- whose lateral normals are
# not load paths. Set to 1 to keep every contact (legacy behaviour).
MIN_LOADBEARING_POINTS = 3


def _build_mortar_block(cloud, min_volume=1e-9, min_thickness=1e-4):
    """Build the mortar mesh filling an A-B joint as the convex hull of the
    joint's interface point cloud.

    ``cloud`` is the set of contact points found on *both* blocks' surfaces
    (``pointsA`` and ``pointsB`` pooled across every contact normal of the
    joint). A coplanar face-to-face interface almost never occurs in practice;
    the realistic interface is a sparse vertex/edge contact -- e.g. the six
    points of an apex-against-face joint -- whose convex hull is the thin solid
    that fills the gap between the two blocks. Returns ``None`` when the points
    are coplanar/collinear (a degenerate joint with no fillable volume).

    A (near-)coplanar cloud must be rejected: its convex hull is a flat sliver
    whose degenerate triangulation builds an invalid warp mesh BVH and then
    segfaults the next slide/contact query. ``min_volume`` alone does not catch
    these (a wide flat sliver can clear it), so the cloud's thickness along its
    thinnest principal axis is checked explicitly against ``min_thickness``.

    ``min_thickness`` should be the simulator's contact tolerance ``tol``: two
    surfaces within ``tol`` are treated as touching, so a joint thinner than
    ``tol`` is a contact (handled by the direct block-block fallback), not a
    mortar gap. The caller passes ``tol`` in; the default is only a backstop.
    """
    pts = np.unique(cloud, axis=0)
    if pts.shape[0] < 4:
        return None
    centred = pts - pts.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=True)
    if s.shape[0] < 3 or np.ptp(centred @ vt[-1]) < min_thickness:
        return None  # flat/degenerate joint: keep the direct block-block contact instead
    try:
        mesh = trimesh.convex.convex_hull(pts)
    except Exception:
        return None
    # A non-watertight hull is degenerate (near-coplanar/collinear cloud): a valid
    # convex hull is always watertight. Such a mesh builds an invalid warp BVH that
    # intermittently hangs or segfaults the slide/contact queries, so drop it and
    # keep the direct block-block contact instead.
    if mesh.is_empty or mesh.volume < min_volume or not mesh.is_watertight:
        return None
    return mesh

def _interface_normals(ptsA, ptsB):
    """Contact normals of the two mortar interfaces, fitted to the contact
    points lying on each block's surface.

    The points sampled on block A are coplanar on A's contact face, so their
    best-fit plane normal is A's own surface normal -- and likewise for B. This
    keeps the two interfaces' normals independent (A's face may be tilted
    relative to B's, as for an apex resting against a flat face) and well
    defined even where a block meets the mortar at a single vertex/edge. Both
    are oriented along the joint axis A->B to match the partIDA->partIDB
    convention: ``nA`` points from A into the mortar, ``nB`` from the mortar
    into B.
    """
    g = ptsB.mean(0) - ptsA.mean(0)          # joint axis, A -> B
    glen = np.linalg.norm(g)
    g = g / glen if glen > 1e-12 else np.array([0., 0., 1.])

    def _fit(points):
        pts = np.unique(points, axis=0)
        if pts.shape[0] >= 3:
            _, s, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
            if s.shape[0] >= 2 and s[1] > 1e-6:   # a real plane, not a line
                n = vt[-1]
                nn = np.linalg.norm(n)
                if nn > 1e-9:
                    n = n / nn
                    return n if n @ g >= 0 else -n
        return g                                   # point/line contact: gap axis
    return _fit(ptsA), _fit(ptsB)


def _interface_patches(mortar, ptsA, ptsB, cos_tol=0.999):
    """Segment the mortar hull into the contact patches it shares with A and B.

    A hull facet whose three vertices all originate from one block lies on that
    block's contact face; a facet with mixed origin bridges the gap and is
    dropped. Co-oriented facets are clustered by normal, so a single flat face
    yields one patch while a block that meets the mortar across several faces
    yields one patch per face -- this is what lifts the "one normal per
    interface" limitation of a single plane fit. Returns ``(patchesA, patchesB)``
    where each patch is ``(normal, points)`` with the normal oriented along the
    joint axis A->B (the partIDA->partIDB convention). A side with no
    pure-origin facet (a true vertex/edge contact) returns an empty list, and
    the caller falls back to :func:`_interface_normals`.
    """
    g = ptsB.mean(0) - ptsA.mean(0)
    glen = np.linalg.norm(g)
    g = g / glen if glen > 1e-12 else np.array([0., 0., 1.])

    V = mortar.vertices
    uA = np.unique(ptsA, axis=0)
    uB = np.unique(ptsB, axis=0)
    dA = np.linalg.norm(V[:, None] - uA[None], axis=2).min(1) if len(uA) else np.full(len(V), np.inf)
    dB = np.linalg.norm(V[:, None] - uB[None], axis=2).min(1) if len(uB) else np.full(len(V), np.inf)
    is_A = dA <= dB                       # origin of each hull vertex

    faces = mortar.faces
    # Orient every facet normal along +g so co-planar facets cluster regardless
    # of winding and the result already follows the contact convention.
    fn = np.where((mortar.face_normals @ g)[:, None] >= 0,
                  mortar.face_normals, -mortar.face_normals)
    area = mortar.area_faces
    centroid = mortar.center_mass
    tcen = mortar.triangles_center

    def _patches(vertex_on_side, into_mortar):
        # A facet is a genuine contact face only if the mortar body lies on the
        # side its (g-oriented) normal points to: into the mortar for A, out
        # toward B for B. This rejects the gap's side walls and the hull's
        # bridging facets, which cap the cloud rather than touch a block.
        signed = np.einsum('ij,ij->i', fn, centroid - tcen)
        keep_side = signed > 1e-9 if into_mortar else signed < -1e-9
        on_side = vertex_on_side[faces].all(axis=1) & keep_side
        fidx = np.where(on_side)[0]
        out, used = [], np.zeros(len(fidx), dtype=bool)
        for a in range(len(fidx)):
            if used[a]:
                continue
            grp = [fidx[a]]
            used[a] = True
            for b in range(a + 1, len(fidx)):
                if not used[b] and fn[fidx[b]] @ fn[fidx[a]] > cos_tol:
                    grp.append(fidx[b])
                    used[b] = True
            grp = np.array(grp)
            w = area[grp]
            n = (fn[grp] * w[:, None]).sum(0)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-12 else fn[fidx[a]]
            pts = V[np.unique(faces[grp].ravel())]
            out.append((n, pts))
        return out

    return _patches(is_A, into_mortar=True), _patches(~is_A, into_mortar=False)


def _emit_face_contact(out, normal, surf_pts, partIDA, partIDB, batch_id, mortar_piece):
    """Append a single mortar-to-block face contact.

    ``surf_pts`` lie on the shared block/mortar face and are ordered into a
    polygon in the plane of ``normal``. Because the mortar face is coincident
    with the block face, ``pointsA``/``pointsB``/``points`` all hold the same
    surface points.
    """
    p_sub = np.unique(surf_pts, axis=0)
    if p_sub.shape[0] == 0:
        return
    if p_sub.shape[0] >= 3 and np.linalg.norm(normal) >= 1e-3:
        # Order the contact polygon in its plane (perpendicular to normal).
        ref = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 1., 0.])
        u = np.cross(normal, ref); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        pts_2d = np.column_stack([p_sub @ u, p_sub @ v])
        try:
            hull = scipy.spatial.ConvexHull(pts_2d)
        except scipy.spatial.qhull.QhullError:
            pass
        else:
            p_sub = p_sub[hull.vertices]
    out.append({
        'partIDA':      partIDA,
        'partIDB':      partIDB,
        'normal':       normal,
        'points':       p_sub,
        'pointsA':      p_sub,
        'pointsB':      p_sub,
        'batchID':      batch_id,
        'mortar_block': mortar_piece,
    })


def init_kernels(n_batchs, n_blocks_max, n_points_max, n_edges_max, mode="release", verbose=True):
    wp.config.mode = mode
    wp.config.verbose = verbose
    wp.init()
    n_batchs_s        = wp.static(wp.int32(n_batchs))
    n_blocks_max_s    = wp.static(wp.int32(n_blocks_max))
    n_points_max_s    = wp.static(wp.int32(n_points_max))
    n_points_max_2_s  = wp.static(wp.int32(n_points_max * 2))
    n_edges_max_s     = wp.static(wp.int32(n_edges_max))
    n_k_b_edge_end_s  = wp.static(wp.int32(n_points_max * 2 + n_edges_max))

    @wp.kernel
    def check_contact_mortar_ker(
        partA:               wp.array(dtype=wp.uint64, ndim=2),  # type: ignore
        partB:               wp.array(dtype=wp.uint64, ndim=1),  # type: ignore
        ignore_batch:        wp.array(dtype=wp.bool,   ndim=1),  # type: ignore
        mortar_thickness:    wp.float32,
        tol:                 wp.float32,
        n_old:               wp.array(dtype=wp.int32,  ndim=1),  # type: ignore
        contact_normals_out: wp.array(dtype=wp.vec3,   ndim=3),  # type: ignore
        contact_points_out:  wp.array(dtype=wp.vec3,   ndim=3),  # type: ignore
        contact_pointsA_out: wp.array(dtype=wp.vec3,   ndim=3),  # type: ignore
        contact_pointsB_out: wp.array(dtype=wp.vec3,   ndim=3),  # type: ignore
        istouching_out:      wp.array(dtype=wp.bool,   ndim=3),  # type: ignore
        contact_partA_out:   wp.array(dtype=wp.int32,  ndim=3),  # type: ignore
    ):
        batchid, partAid, k = wp.tid()

        if ignore_batch[batchid]:
            return
        if partAid >= n_old[batchid]:
            return

        if k < n_points_max_s:
            # B-vertex pass: sdf(A, p) <= mortar_thickness  =>  p is in offset(A, r)
            vid = k
            if vid >= wp.mesh_get(partB[batchid]).points.shape[0]:
                return
            p = wp.mesh_get(partB[batchid]).points[vid]
            query = wp.mesh_query_point_sign_normal(
                partA[batchid, partAid], p, mortar_thickness + tol, tol * 0.01
            )
            if not query.result:
                return
            closest_A   = wp.mesh_eval_position(partA[batchid, partAid], query.face, query.u, query.v)
            signed_dist = wp.length(p - closest_A) * query.sign
            if signed_dist <= mortar_thickness:
                d = p - closest_A
                if wp.length(d) > tol * wp.float32(0.01):
                    normal = wp.normalize(d)
                else:
                    normal = wp.mesh_eval_face_normal(partA[batchid, partAid], query.face)
                contact_normals_out[batchid, partAid, k] = normal
                contact_pointsA_out[batchid, partAid, k] = closest_A
                contact_pointsB_out[batchid, partAid, k] = p
                contact_points_out[batchid,  partAid, k] = (closest_A + p) * wp.float32(0.5)
                istouching_out[batchid,      partAid, k] = True
                contact_partA_out[batchid,   partAid, k] = partAid
        elif k < n_points_max_2_s:
            # A-vertex pass: sdf(B, p) <= 0  =>  A vertex is inside B (A-interface of joint)
            vid = k - n_points_max_s
            if vid >= wp.mesh_get(partA[batchid, partAid]).points.shape[0]:
                return
            p = wp.mesh_get(partA[batchid, partAid]).points[vid]
            query = wp.mesh_query_point_sign_normal(
                partB[batchid], p, mortar_thickness + tol, tol * 0.01
            )
            if not query.result:
                return
            closest_B   = wp.mesh_eval_position(partB[batchid], query.face, query.u, query.v)
            signed_dist = wp.length(p - closest_B) * query.sign
            if signed_dist <= mortar_thickness:
                d = closest_B - p
                if wp.length(d) > tol * wp.float32(0.01):
                    normal = wp.normalize(d)
                else:
                    normal = wp.mesh_eval_face_normal(partB[batchid], query.face)
                contact_normals_out[batchid, partAid, k] = normal
                contact_pointsA_out[batchid, partAid, k] = p
                contact_pointsB_out[batchid, partAid, k] = closest_B
                contact_points_out[batchid,  partAid, k] = (p + closest_B) * wp.float32(0.5)
                istouching_out[batchid,      partAid, k] = True
                contact_partA_out[batchid,   partAid, k] = partAid
        elif k < n_k_b_edge_end_s:
            # B-edge crossing pass: find where edge e of B crosses offset(A)'s boundary.
            # One endpoint inside offset(A), the other outside => interpolate the exact crossing.
            e = k - n_points_max_2_s
            n_idx = wp.mesh_get(partB[batchid]).indices.shape[0]
            if e >= n_idx:
                return
            v0 = wp.mesh_get_point(partB[batchid], e)
            if e % 3 == 2:
                e1 = e - 2
            else:
                e1 = e + 1
            v1 = wp.mesh_get_point(partB[batchid], e1)

            # First check v0 with tight radius — cheap early-exit if v0 is outside.
            q0 = wp.mesh_query_point_sign_normal(
                partA[batchid, partAid], v0, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q0.result:
                return
            c0   = wp.mesh_eval_position(partA[batchid, partAid], q0.face, q0.u, q0.v)
            sdf0 = wp.length(v0 - c0) * q0.sign
            if sdf0 > mortar_thickness:
                return  # v0 outside; the reverse direction (v1→v0) will handle this edge

            # v0 is inside the mortar zone; query v1 with a large radius so distant
            # "outside" vertices are still reachable.
            q1 = wp.mesh_query_point_sign_normal(
                partA[batchid, partAid], v1, mortar_thickness * wp.float32(50.0), tol * wp.float32(0.01)
            )
            if not q1.result:
                return  # v1 unreachably far — crossing is essentially at v0, already detected
            c1   = wp.mesh_eval_position(partA[batchid, partAid], q1.face, q1.u, q1.v)
            sdf1 = wp.length(v1 - c1) * q1.sign
            if sdf1 <= mortar_thickness:
                return  # both inside — no boundary crossing on this edge

            # Linear interpolation to the boundary crossing
            delta = sdf1 - sdf0
            if wp.abs(delta) < tol * wp.float32(0.001):
                return
            t = (mortar_thickness - sdf0) / delta
            if t <= wp.float32(0.0) or t >= wp.float32(1.0):
                return
            p_cross = v0 + t * (v1 - v0)

            # Query A at the crossing point to get the A-surface contact point.
            q_c = wp.mesh_query_point_sign_normal(
                partA[batchid, partAid], p_cross, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q_c.result:
                return
            closest_A = wp.mesh_eval_position(partA[batchid, partAid], q_c.face, q_c.u, q_c.v)
            # Query B from the A contact point to get the B-surface contact point
            # (the face of B actually in contact with the mortar, not B's lateral edge).
            q_b = wp.mesh_query_point_sign_normal(
                partB[batchid], closest_A, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q_b.result:
                return
            closest_B = wp.mesh_eval_position(partB[batchid], q_b.face, q_b.u, q_b.v)
            d = closest_B - closest_A
            if wp.length(d) > tol * wp.float32(0.01):
                normal = wp.normalize(d)
            else:
                normal = wp.mesh_eval_face_normal(partB[batchid], q_b.face)
            contact_normals_out[batchid, partAid, k] = normal
            contact_pointsA_out[batchid, partAid, k] = closest_A
            contact_pointsB_out[batchid, partAid, k] = closest_B
            contact_points_out[batchid,  partAid, k] = (closest_A + closest_B) * wp.float32(0.5)
            istouching_out[batchid,      partAid, k] = True
            contact_partA_out[batchid,   partAid, k] = partAid
        else:
            # A-edge crossing pass: find where edge e of A crosses B's mortar zone boundary.
            # Symmetric to the B-edge pass with roles of A and B swapped.
            e = k - n_k_b_edge_end_s
            n_idx = wp.mesh_get(partA[batchid, partAid]).indices.shape[0]
            if e >= n_idx:
                return
            v0 = wp.mesh_get_point(partA[batchid, partAid], e)
            if e % 3 == 2:
                e1 = e - 2
            else:
                e1 = e + 1
            v1 = wp.mesh_get_point(partA[batchid, partAid], e1)

            # First check v0 against B with tight radius
            q0 = wp.mesh_query_point_sign_normal(
                partB[batchid], v0, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q0.result:
                return
            c0   = wp.mesh_eval_position(partB[batchid], q0.face, q0.u, q0.v)
            sdf0 = wp.length(v0 - c0) * q0.sign
            if sdf0 > mortar_thickness:
                return  # v0 outside; the reverse direction (v1→v0) will handle this edge

            # v0 is inside B's mortar zone; query v1 with large radius
            q1 = wp.mesh_query_point_sign_normal(
                partB[batchid], v1, mortar_thickness * wp.float32(50.0), tol * wp.float32(0.01)
            )
            if not q1.result:
                return
            c1   = wp.mesh_eval_position(partB[batchid], q1.face, q1.u, q1.v)
            sdf1 = wp.length(v1 - c1) * q1.sign
            if sdf1 <= mortar_thickness:
                return  # both inside — no boundary crossing

            delta = sdf1 - sdf0
            if wp.abs(delta) < tol * wp.float32(0.001):
                return
            t = (mortar_thickness - sdf0) / delta
            if t <= wp.float32(0.0) or t >= wp.float32(1.0):
                return
            p_cross = v0 + t * (v1 - v0)

            # Query B at the crossing point to find the B contact face point.
            q_c = wp.mesh_query_point_sign_normal(
                partB[batchid], p_cross, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q_c.result:
                return
            closest_B = wp.mesh_eval_position(partB[batchid], q_c.face, q_c.u, q_c.v)
            # Query A from the B contact point to get the A contact face point
            # (the face of A actually in contact with the mortar, not A's lateral edge).
            q_a = wp.mesh_query_point_sign_normal(
                partA[batchid, partAid], closest_B, mortar_thickness + tol, tol * wp.float32(0.01)
            )
            if not q_a.result:
                return
            closest_A = wp.mesh_eval_position(partA[batchid, partAid], q_a.face, q_a.u, q_a.v)
            d = closest_B - closest_A
            if wp.length(d) > tol * wp.float32(0.01):
                normal = wp.normalize(d)
            else:
                normal = wp.mesh_eval_face_normal(partB[batchid], q_c.face)
            contact_normals_out[batchid, partAid, k] = normal
            contact_pointsA_out[batchid, partAid, k] = closest_A
            contact_pointsB_out[batchid, partAid, k] = closest_B
            contact_points_out[batchid,  partAid, k] = (closest_A + closest_B) * wp.float32(0.5)
            istouching_out[batchid,      partAid, k] = True
            contact_partA_out[batchid,   partAid, k] = partAid

    def check_contact_mortar_batch_wrapper(
        partA_wp, partA_wpid, partB: List[trimesh.Trimesh], n_blocks_t,
        tol=1e-5, identical_tol=None, angle_tol=1e-2,
        mortar_thickness=0.1, use_graph_capture=False, env_ids=None,
    ):
        n_old = wp.array(n_blocks_t, dtype=wp.int32)

        if identical_tol is None:
            identical_tol = tol / 10

        if env_ids is None:
            ignore_batch = wp.zeros(n_batchs_s, dtype=wp.bool)
            partB_wp     = [wp.Mesh(wp.array(partB[i].vertices,       dtype=wp.vec3),
                                    wp.array(partB[i].faces.flatten(), dtype=wp.int32))
                            for i in range(n_batchs_s)]
            partB_wpid   = wp.array([m.id for m in partB_wp], dtype=wp.uint64)
        else:
            ignore_batch_np          = np.ones(n_batchs_s, dtype=bool)
            ignore_batch_np[env_ids] = False
            ignore_batch             = wp.array(ignore_batch_np, dtype=wp.bool)
            partB_wp        = []
            partB_wpid_list = np.zeros(n_batchs_s, dtype=np.uint64)
            dummy_verts = wp.array(np.array([[0, 0, 0]], dtype=np.float32), dtype=wp.vec3)
            dummy_faces = wp.array(np.array([0, 0, 0],  dtype=np.int32),   dtype=wp.int32)
            for i, env_id in enumerate(env_ids):
                m = wp.Mesh(wp.array(partB[i].vertices,       dtype=wp.vec3),
                            wp.array(partB[i].faces.flatten(), dtype=wp.int32))
                partB_wp.append(m)
                partB_wpid_list[env_id] = m.id
            for i in range(n_batchs_s):
                if i not in env_ids:
                    partB_wpid_list[i] = wp.Mesh(dummy_verts, dummy_faces).id
            partB_wpid = wp.array(partB_wpid_list, dtype=wp.uint64)

        k_max = 2 * n_points_max + 2 * n_edges_max
        contact_normals_out = wp.zeros(dtype=wp.vec3,  shape=(n_batchs_s, n_blocks_max_s, k_max))
        contact_points_out  = wp.zeros(dtype=wp.vec3,  shape=(n_batchs_s, n_blocks_max_s, k_max))
        contact_pointsA_out = wp.zeros(dtype=wp.vec3,  shape=(n_batchs_s, n_blocks_max_s, k_max))
        contact_pointsB_out = wp.zeros(dtype=wp.vec3,  shape=(n_batchs_s, n_blocks_max_s, k_max))
        istouching_out      = wp.zeros(dtype=wp.bool,  shape=(n_batchs_s, n_blocks_max_s, k_max))
        contact_partA_out   = wp.zeros(dtype=wp.int32, shape=(n_batchs_s, n_blocks_max_s, k_max))

        wp.launch(
            check_contact_mortar_ker,
            dim=(n_batchs_s, n_blocks_max_s, k_max),
            inputs=(
                wp.array(partA_wpid, dtype=wp.uint64),
                partB_wpid,
                ignore_batch,
                wp.float32(mortar_thickness),
                wp.float32(tol),
                n_old,
            ),
            outputs=(
                contact_normals_out,
                contact_points_out,
                contact_pointsA_out,
                contact_pointsB_out,
                istouching_out,
                contact_partA_out,
            ),
        )
        wp.synchronize()

        normals   = contact_normals_out.numpy()
        points    = contact_points_out.numpy()
        pointsA   = contact_pointsA_out.numpy()
        pointsB   = contact_pointsB_out.numpy()
        touching  = istouching_out.numpy()
        partA_ids = contact_partA_out.numpy()

        # Discard contacts where the normal does not point from A toward B
        dir_AB = pointsB[touching] - pointsA[touching]
        valid  = np.einsum("ij,ij->i", normals[touching], dir_AB) > 0
        touching[touching] = valid

        logtol_pos   = -int(np.log10(identical_tol))
        logtol_angle = -int(np.log10(angle_tol))

        active_ids     = env_ids if env_ids is not None else np.arange(n_batchs_s)
        contacts       = [[] for _ in range(n_batchs_s)]
        contacts_degen = [[] for _ in range(n_batchs_s)]
        mortar_blocks  = [[] for _ in range(n_batchs_s)]

        ib, ia, ik = np.nonzero(touching)

        for i in active_ids:
            mask = ib == i
            if not np.any(mask):
                continue
            blocks_i  = ia[mask]
            k_i       = ik[mask]
            normals_i = np.round(normals[i, blocks_i, k_i], logtol_angle)
            nrm       = np.linalg.norm(normals_i, axis=1, keepdims=True)
            normals_i = normals_i / np.where(nrm > 0, nrm, 1.0)
            pts_i     = np.round(points[i, blocks_i, k_i], logtol_pos)
            ptsA_i    = pointsA[i, blocks_i, k_i]
            ptsB_i    = pointsB[i, blocks_i, k_i]
            ida       = partA_ids[i, blocks_i, k_i]

            # The new block (partB) is placed with id n_blocks_t[i] and the mortar
            # blocks are appended to the assembly right after it, so the m-th
            # mortar block of this batch receives id n_blocks_t[i] + 1 + m.
            new_block_id = n_blocks_t[i]

            # Group every contact entry of this batch by the partA block it
            # touches (partB is always the freshly placed block). The mortar of
            # an A-B joint is built from *all* of that joint's interface points
            # pooled across normals: a real joint is a sparse vertex/edge contact
            # (e.g. 6 points for an apex-against-face joint), so no single
            # per-normal patch ever holds enough points to form a volume on its
            # own -- but their convex hull does.
            uida, uidaidx = np.unique(ida, return_inverse=True)
            for j, uida_id in enumerate(uida):
                joint = uidaidx == j

                # Gap between the two parts at this joint: ptsA and ptsB are the
                # matched closest points on each block's surface, so their distance
                # is the local separation. When every pair is within the contact
                # tolerance ``tol`` the parts are directly in contact (no gap to
                # fill), so no mortar block is created -- the direct block-block
                # contact below carries the joint. This also avoids the degenerate
                # near-coplanar mortar hull whose warp BVH hangs the slide kernel.
                gap = np.linalg.norm(ptsA_i[joint] - ptsB_i[joint], axis=1)
                if gap.max() < tol:
                    mortar_piece = None
                else:
                    # Mortar block = convex hull of the pooled interface point cloud.
                    cloud = np.vstack([ptsA_i[joint], ptsB_i[joint]])
                    try:
                        mortar_piece = _build_mortar_block(cloud, min_thickness=tol)
                    except Exception as e:
                        print(f"Error creating mortar piece mesh: {e}")
                        mortar_piece = None

                if mortar_piece is None:
                    # Degenerate joint (no fillable volume): keep the direct
                    # block-block contact so the connection is not lost. Group by
                    # normal over every entry of the joint.
                    nu, nidx = np.unique(normals_i[joint], return_inverse=True, axis=0)
                    nlen = np.linalg.norm(nu, axis=1, keepdims=True)
                    nu = nu / np.where(nlen > 0, nlen, 1.0)
                    p_j, pA_j, pB_j = pts_i[joint], ptsA_i[joint], ptsB_i[joint]
                    for n_idx, n in enumerate(nu):
                        same_n = nidx == n_idx
                        p_sub, idx = np.unique(p_j[same_n], return_index=True, axis=0)
                        contacts[i].append({
                            'partIDA':      uida_id,
                            'partIDB':      new_block_id,
                            'normal':       n,
                            'points':       p_sub,
                            'pointsA':      pA_j[same_n][idx],
                            'pointsB':      pB_j[same_n][idx],
                            'batchID':      i,
                            'mortar_block': None,
                        })
                    continue

                mortar_id = new_block_id + 1 + len(mortar_blocks[i])
                mortar_blocks[i].append(mortar_piece)

                # Segment the mortar's faces into the patches it shares with A
                # and with B, one contact per distinct face per side. Each
                # patch's normal is that block's own surface normal, so a block
                # touching the mortar across several faces yields several
                # contacts rather than one averaged normal.
                pA = ptsA_i[joint]
                pB = ptsB_i[joint]
                patchesA, patchesB = _interface_patches(mortar_piece, pA, pB)
                if not patchesA or not patchesB:
                    # A true vertex/edge contact has no pure-origin facet on that
                    # side: fall back to a single plane-fit / gap-axis normal.
                    nA_fb, nB_fb = _interface_normals(pA, pB)
                    if not patchesA:
                        patchesA = [(nA_fb, np.unique(pA, axis=0))]
                    if not patchesB:
                        patchesB = [(nB_fb, np.unique(pB, axis=0))]

                # block A <-> mortar, contact points on A's surface.
                for n, pts in patchesA:
                    _emit_face_contact(contacts[i], n, pts,
                                       uida_id, mortar_id, i, mortar_piece)
                # mortar <-> new block, contact points on B's surface.
                for n, pts in patchesB:
                    _emit_face_contact(contacts[i], n, pts,
                                       mortar_id, new_block_id, i, mortar_piece)
        # Keep only load-bearing contacts (area patches): drop the grazing
        # single-point / edge corner contacts whose lateral normals are spurious.
        contacts = [[c for c in cl if c['points'].shape[0] >= MIN_LOADBEARING_POINTS]
                    for cl in contacts]
        return contacts, contacts_degen, mortar_blocks

    return check_contact_mortar_batch_wrapper
