def check_contact_ziq(partA,partB,tol):
    triangles = compute_contacted_triangles_ziq(partA, partB,tol)
    if triangles != []:
        contacts = compute_intersection_between_triangles_ziq(triangles)
        return compute_contacts_convexhull_ziq(contacts)
    else:
        return []
def compute_contacted_triangles_ziq(partA: trimesh.Trimesh,
                                    partB: trimesh.Trimesh,
                                    tol=1e-5,
                                    device = 'cpu'):
        triangles = []
        normalA = torch.tensor(partA.face_normals, device=device)
        normalB = torch.tensor(partB.face_normals, device=device)
        centerA = torch.tensor(partA.triangles_center, device=device)
        centerB = torch.tensor(partB.triangles_center, device=device)
        partIDAs, partIDBs = torch.meshgrid(torch.arange(normalA.shape[0]), torch.arange(normalB.shape[0]), indexing="ij")
        angle_ij = torch.arccos(torch.clip(torch.einsum("ijk, ijk->ij", normalA[partIDAs, :], -normalB[partIDBs, :]), -1, 1))
        dist_ij = torch.abs(torch.einsum("ijk, ijk -> ij", (centerA[partIDAs, :] - centerB[partIDBs, :]), normalA[partIDAs, :]))
        flag = torch.logical_and(angle_ij < tol, dist_ij < tol)
        contact_pairs = flag.nonzero().cpu().numpy()
        for faceA, faceB in contact_pairs:
            nA = normalA[faceA].cpu().numpy()
            cA = centerA[faceA].cpu().numpy()
            plane_transform = trimesh.geometry.plane_transform(cA, nA)
            vA = trimesh.transformations.transform_points(partA.triangles[faceA], plane_transform)
            vB = trimesh.transformations.transform_points(partB.triangles[faceB], plane_transform)
            triangles.append([np.copy(vA), np.copy(vB), np.copy(nA), np.copy(plane_transform)])
        return triangles
def compute_intersection_between_triangles_ziq(triangles,tol_area=1e-6,tol=1e-5):
    contacts = []
    for vA, vB, nA, plane_transform in triangles:
        polyA = Polygon(vA)
        polyA = shapely.geometry.polygon.orient(polyA, sign=1.0)
        polyB = Polygon(vB)
        polyB = shapely.geometry.polygon.orient(polyB, sign=1.0)
        data = polyA.intersection(polyB)
        if data.area < tol_area:
            continue
        contact_points = from_shapely_polygon_to_np_points(data)
        if contact_points is not None:
            inv = np.linalg.inv(plane_transform)
            contact_points = trimesh.transformations.transform_points(contact_points, inv)
            duplicate = False
            for contact in contacts:
                angle_rad = np.arccos(np.clip(np.dot(nA, contact[0]), -1.0, 1.0))
                dist = abs(np.dot((contact_points[0] - contact[1][0]), nA))
                if angle_rad < tol and dist < tol:
                    contact[1] = np.vstack([contact[1], contact_points])
                    duplicate = True
                    break
            if not duplicate:
                contacts.append([np.copy(nA), copy.deepcopy(contact_points), np.copy(plane_transform)])
    return contacts

def from_shapely_polygon_to_np_points(data,tol=1e-5):
    if isinstance(data, Polygon):
        data = data.simplify(tol, preserve_topology=False)
        xx, yy = data.exterior.coords.xy
        xx = xx.tolist()[:-1]
        yy = yy.tolist()[:-1]
        if len(xx) > 0 and len(yy) > 0:
            contact_points = np.zeros((len(xx), 3), dtype=np.float64)
            contact_points[:, 0] = xx
            contact_points[:, 1] = yy
            return contact_points
    return None
def compute_contacts_convexhull_ziq(contacts):
    result = []
    for contact in contacts:
        # remove duplicate points
        rounded_points = np.round(contact[1], decimals=5)
        unique_points, indices = np.unique(rounded_points, axis=0, return_index=True)
        contact[1] = contact[1][indices]
        projection = contact[2]
        plane = np.linalg.inv(projection)
        data = {"points": contact[1], "normal": contact[0], "plane": plane}
        if contact[1].shape[0] >= 3:
            # convexhull
            contact_points_2d = trimesh.transformations.transform_points(contact[1], projection)[:, :2]
            mpt = MultiPoint(contact_points_2d)
            cvxhull_points = from_shapely_polygon_to_np_points(mpt.convex_hull)
            if cvxhull_points is not None:
                cvxhull_points = trimesh.transformations.transform_points(cvxhull_points, plane)
                faces = []
                for fid in range(0, len(cvxhull_points) - 2):
                    faces.append([0, fid + 1, fid + 2])
                cvxhull_contact_mesh = trimesh.Trimesh(vertices=cvxhull_points, faces=faces)
                data["points"] = cvxhull_points
                data["mesh"] = cvxhull_contact_mesh
        result.append(data)
    return result