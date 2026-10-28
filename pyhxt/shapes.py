"""Simple closed surfaces to mesh: boxes and spheres, and putting several together.

Each returns (points, triangles, markers), ready for pyhxt.tetrahedralize.
"""
import numpy as np


def box(lo=(0., 0., 0.), hi=(1., 1., 1.), n=1, marker=1):
    """The surface of the box from `lo` to `hi`, each face split into n x n squares of two
    triangles."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    t = np.linspace(0., 1., n + 1)
    points, triangles = [], []
    for axis in range(3):
        for side in (0., 1.):
            u, v = np.meshgrid(t, t, indexing='ij')
            p = np.zeros((n + 1, n + 1, 3))
            p[..., axis] = side
            p[..., (axis + 1) % 3], p[..., (axis + 2) % 3] = u, v
            k = len(points) and sum(len(q) for q in points)
            idx = k + np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
            a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
            quads = np.stack([a, b, c, d], -1).reshape(-1, 4)
            if side == 0.:
                quads = quads[:, ::-1]
            triangles.append(np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
            points.append(p.reshape(-1, 3))
    points = lo + np.concatenate(points) * (hi - lo)
    triangles = np.concatenate(triangles)
    # the faces share their edges' points: merge them
    points, inverse = np.unique(np.round(points, 12), axis=0, return_inverse=True)
    triangles = inverse.reshape(-1)[triangles]
    return points, triangles, np.full(len(triangles), marker)


def sphere(centre=(0., 0., 0.), radius=1., subdivisions=3, marker=1):
    """An icosphere: the icosahedron's faces split `subdivisions` times, every point on
    the sphere (20 * 4**subdivisions triangles)."""
    g = (1. + 5. ** .5) / 2.
    points = np.array([[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g],
                       [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1],
                       [-g, 0, 1]], dtype=float)
    triangles = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                          [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                          [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                          [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    for _ in range(subdivisions):
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                        triangles[:, [2, 0]]]), axis=1)
        edges, index = np.unique(edges, axis=0, return_inverse=True)
        index = index.reshape(3, -1).T + len(points)
        mid = points[edges].sum(axis=1)
        points = np.concatenate([points, mid / np.linalg.norm(mid, axis=1)[:, None]])
        a, b, c = triangles.T
        ab, bc, ca = index.T
        triangles = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                                    np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    return np.asarray(centre, float) + radius * points, triangles, \
        np.full(len(triangles), marker)


def combine(*surfaces):
    """Several (points, triangles, markers) as one, the points of each after the last's."""
    points, triangles, markers, n = [], [], [], 0
    for p, t, m in surfaces:
        points.append(p)
        triangles.append(np.asarray(t) + n)
        markers.append(m)
        n += len(p)
    return np.concatenate(points), np.concatenate(triangles), np.concatenate(markers)
