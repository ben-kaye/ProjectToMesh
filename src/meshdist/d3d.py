import torch
import matplotlib.pyplot as plt
import einops
import plotly.graph_objects as go


def dist_to_tri_single_args(tri, point):
    assert tri.shape == (3, 3)
    assert point.shape == (3,)

    a, b, c = (tri[k, :] for k in range(3))
    ab = b - a
    bc = c - b
    ca = a - c

    plane_normal = torch.linalg.cross(ab, -ca)
    plane_normal = plane_normal/plane_normal.norm()

    ap_3d = point - a
    dist_ = dot(plane_normal, ap_3d)

    p_in_plane = point - dist_*plane_normal
    # ap_plane = p_in_plane - a

    in_tri = check_in_tri(tri, p_in_plane)

    # len_sq_a = dot(ca, ca)
    # len_sq_b = dot(ab, ab)
    # coord_a = dot(ap_plane, -ca)/len_sq_a
    # coord_b = dot(ap_plane, ab)/len_sq_b

    # def in_coord(c):
    #     return (c >= 0.0) & (c <= 1.0)
    # in_a, in_b = (in_coord(c) for c in (coord_a, coord_b))
    # in_tri = False
    # if in_a and in_b:
    #     # do check for tri
    #     angle = torch.arccos(dot(ab, -ca)/(len_sq_a * len_sq_b).sqrt())
    #     angle_2 =
    #     # <a,b> = |a||b|cos(phi)

    if in_tri:
        # distance is orthogonal distance to plane
        return dist_, p_in_plane
    else:
        # else compute distance with loci of edges,
        dist_point_pairs = (dist_to_line_single(a0, b0, point)
                            for a0, b0 in zip((a, b, c), (b, c, a)))
        dists_, points = (torch.stack(d, dim=-1)
                          for d in zip(*dist_point_pairs))
        indices = dists_.argmin(dim=-1)
        return dists_[indices], points[:, indices]


def project_to_mesh(vertices, faces, query, TOP_K_VERTICES=3):
    from scipy.spatial import KDTree

    naive_dists, top_k_vertices = KDTree(
        data=vertices).query(query, k=TOP_K_VERTICES)
    top_k_vertices = torch.tensor(top_k_vertices)

    min_points = []

    # get the corresponding faces
    def min_point(top_k_vertices, q):
        k_k, f_k = top_k_vertices[None, :].eq(
            faces.flatten()[:, None]).nonzero(as_tuple=True)
        unique_matches = faces[torch.unique(k_k//3, sorted=False)]
        dist_, point_min = project_to_tri(vertices[unique_matches], q[None])
        return point_min

    min_points = tuple(min_point(matches, q)
                       for matches, q in zip(top_k_vertices, query))

    min_points = torch.cat(min_points, dim=0)
    
    return min_points


def project_to_tri(tris, points):
    '''
    get perpendicular distance to faces, and if not in any face tubes, get distance to the edges
    '''

    assert tris.shape[-2:] == (3, 3)
    assert points.shape[-1:] == (3,)
    NUM_QUERIES = points.shape[0]
    NUM_TRIS = tris.shape[0]

    a, b, c = (tris[:, k, :] for k in range(3))
    ab = b - a
    bc = c - b
    ca = a - c

    # get the face normal
    plane_normal = torch.linalg.cross(ab, -ca)
    plane_normal = plane_normal/plane_normal.norm(dim=1)[:, None]

    # project a vector from plane to query on to the normal unit vector
    ap_3d = points - a[:, None, :]
    dist_ = tri_dot(plane_normal, ap_3d)

    p_in_plane = points[None, :, :].expand(
        NUM_TRIS, -1, -1) - dist_[:, :, None] * plane_normal[:, None, :]

    in_triangle_tube = check_in_tri(tris, p_in_plane)

    min_dists = torch.zeros(NUM_TRIS, NUM_QUERIES)
    min_dists[in_triangle_tube] = dist_[in_triangle_tube]

    # if the point lies in one tube, set all tris not in the tube to max dist
    min_dists[~in_triangle_tube & in_triangle_tube.any(dim=0)] = torch.inf

    min_points = torch.zeros(NUM_TRIS, NUM_QUERIES, 3)
    min_points[in_triangle_tube] = p_in_plane[in_triangle_tube]

    out_of_tri_points = ~in_triangle_tube.any(dim=0)

    # does a query point not lie in the tube of a triangle, eg in the set (not UNION of tubes)
    # FIXME ignore duplicate edges
    if out_of_tri_points.any():

        # a, b, c = (t for t in (a, b, c))

        # find the smallest dist edge
        dist_point_pairs = (squaredist_to_edge(a0, b0, points[out_of_tri_points])
                            for a0, b0 in zip((a, b, c), (b, c, a)))

        line_dists, line_points = (torch.stack(d, dim=2)
                                   for d in zip(*dist_point_pairs))

        # get smallest distance of the 3
        indices = line_dists.argmin(dim=-1)

        # FIXME TODO, do the sqrt at the end of the procedure

        min_dists[:, out_of_tri_points] = line_dists.gather(
            index=indices[..., None], dim=2)[:, :, 0].sqrt()

        min_points[:, out_of_tri_points] = line_points.gather(
            dim=2, index=indices[:, :, None, None].expand(-1, -1, 1, 3))[:, :, 0, :]

    min_indices = min_dists.abs().argmin(dim=0)
    min_points[min_indices, torch.arange(NUM_QUERIES), :].shape

    return min_dists[min_indices, torch.arange(NUM_QUERIES)], min_points[min_indices, torch.arange(NUM_QUERIES), :]


def check_in_tri(tri, in_plane_query):
    '''check if a point is in the 3D triangle by solving the linear combination of vertices and checking if weights are all >= 0.'''
    A = torch.ones((*tri.shape[:-2], 4, 3))
    A[..., :3, :] = transpose(tri)

    if len(tri.shape) > 2:
        A = A[:, None, :, :].expand(-1, in_plane_query.shape[1], -1, -1)

    p = torch.ones((*in_plane_query.shape[:-1], 4, 1))
    p[..., :3, 0] = in_plane_query

    coords = torch.linalg.lstsq(
        A, p
    ).solution

    in_tri = (coords >= 0).all(dim=-2)

    return in_tri[..., 0]


def squaredist_to_edge(verts_a, verts_b, points):
    '''
    return the square distance to an edge

    '''
    ap = points[None, :, :] - verts_a[:, None, :]
    ab = verts_b - verts_a

    ab_len_squared = dot(ab, ab)

    dist_ratio = torch.clamp(tri_dot(ab, ap)/ab_len_squared[:, None], torch.tensor(
        0., device=points.device), torch.tensor(1., device=points.device))
    # get min dist to point
    point_on_line = dist_ratio[:, :, None] * ab[:, None, :]

    square_dist = (ap - point_on_line).pow(2).sum(dim=-1).sqrt()

    min_point = verts_a + point_on_line
    return square_dist, min_point


def dist_to_line_single(vert_a, vert_b, point):
    ap = point - vert_a
    ab = vert_b - vert_a

    ab_len_squared = dot(ab, ab)

    # get min dist to point
    point_on_line = torch.clamp(dot(ap, ab)/ab_len_squared, torch.tensor(
        0., device=point.device), torch.tensor(1., device=point.device)) * ab

    dist_ = (ap - point_on_line).pow(2).sum(dim=-1).sqrt()

    # perp_dir = torch.linalg.cross(torch.linalg.cross(ab, ap), ab)
    # perp_dir = perp_dir.norm()
    # dist_ = dot(ap, perp_dir)
    min_point = vert_a + point_on_line
    return dist_, min_point


def outer(a, b):
    return einops.einsum(a, b, '... i, ... j -> ... i j')


def transpose(matrix):
    return einops.rearrange(matrix, '... i j -> ... j i')


def dot(a, b):
    return einops.einsum(a, b, '... i, ... i -> ...')


def shaped_dot(a, b):
    return einops.einsum(a, b, 'n k, q k -> n q')


def tri_dot(a, b):
    return einops.einsum(a, b, 'n k, n q k -> n q')
