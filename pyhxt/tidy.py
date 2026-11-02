"""Two passes over a finished mesh for the tets HXT's optimiser may not touch, because
improving them would change the surface (pyhxt_tidy.c). Mesh.peel_flat and
Mesh.relax_slivers apply them to a mesh tetrahedralize made; these take plain arrays.

The measure of a tet here is the angle (degrees) by which it falls short of flat: the
smaller of its smallest dihedral angle and 180 less its largest. A sliver or a cap has a
dihedral near 0, a tet with a face collapsing onto a line one near 180, and the smallest
angle alone misses the second kind.
"""
import numpy as np

from pyhxt import _lib


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def flatness(points, tets):
    """Each tet's distance from flat (degrees) and its signed volume, on every core.

    :return: two (m,) float64 arrays.
    """
    X, T = _f64(points).reshape(-1, 3), _i64(tets).reshape(-1, 4)
    q, v = np.empty(len(T)), np.empty(len(T))
    _lib.library().pyhxt_quality(len(T), X.ctypes.data, T.ctypes.data, q.ctypes.data,
                                 v.ctypes.data)
    return q, v


def peel_flat(points, tets, triangles, markers, protect=(), flat=1e-9):
    """Removes the tets of no volume HXT's boundary recovery leaves spanning two adjacent
    boundary triangles that lie in one plane: four boundary points, two of the tet's faces
    boundary triangles of one marker, 6|V| below `flat` times its longest edge cubed. No
    vertex of such a tet is free, so nothing can move it, and its one interior edge holds an
    unknown in a cell of no volume. The two triangles' shared diagonal is swapped for the
    tet's other two faces: the boundary is the same set of points, its triangulation not.
    Triangles whose marker is in `protect` are never changed (a periodic face, whose
    triangles must match its twin's).

    :return: (tets, triangles, removed): the tets left, in their order; the triangles,
        the changed rows rewritten (same marker, oriented as the triangle they replace); and
        an (m,) bool mask of the tets removed.
    """
    X, T = _f64(points).reshape(-1, 3), _i64(tets).reshape(-1, 4).copy()
    tri = _i64(triangles).reshape(-1, 3).copy()
    col, prot = _i64(markers).reshape(-1), _i64(list(protect))
    if len(col) != len(tri):
        raise ValueError('one marker per triangle')
    removed = np.zeros(len(T), dtype=np.uint8)
    n = _lib.library().pyhxt_peel_flat(len(X), X.ctypes.data, len(T), T.ctypes.data,
                                       len(tri), tri.ctypes.data, col.ctypes.data,
                                       float(flat), len(prot), prot.ctypes.data,
                                       removed.ctypes.data)
    if n < 0:
        raise MemoryError('pyhxt_peel_flat')
    return T[:len(T) - n], tri, removed.view(bool)


def relax_slivers(points, tets, angle, fixed, slide=None, sweeps=3, maxfev=400):
    """Moves the free vertices of the tets within `angle` degrees of flat to where the
    worst tet of their star is furthest from flat, keeping every tet of the star the same
    way round. Gauss-Seidel over the vertices, worst first, `sweeps` times; a move is kept
    only if it improves its star. The search is Nelder-Mead exactly as scipy.optimize's
    (from a simplex 0.2 times the star's size, at most `maxfev` evaluations a vertex).

    fixed: (n,) bool, the vertices that may not move (the boundary's).
    slide: (mask, centre, radius), or None: fixed vertices that may move along a sphere, the
        (n,) mask, (n, 3) centres and (n,) radii. A tet whose four vertices are all fixed
        can only be helped by these; they move in the sphere's tangent plane and are put
        back on it. Only a vertex every boundary triangle around which lies on the one
        sphere should be in it, or the boundary moves.

    :return: (points, (before, after, slid)): the points (a float64 array: `points` itself,
        changed in place, if it already was one), the number of tets within `angle` of flat
        before and after, and of vertices slid.
    """
    X = np.asarray(points)
    if not (X.dtype == np.float64 and X.flags.c_contiguous):
        X = _f64(X)
    X = X.reshape(-1, 3)
    T = _i64(tets).reshape(-1, 4)
    fx = np.ascontiguousarray(fixed, dtype=np.uint8).reshape(-1)
    if len(fx) != len(X):
        raise ValueError('one fixed flag per point')
    sl = cen = rad = None
    if slide is not None:
        sl = np.ascontiguousarray(slide[0], dtype=np.uint8).reshape(-1)
        cen, rad = _f64(slide[1]).reshape(-1, 3), _f64(slide[2]).reshape(-1)
        if not (len(sl) == len(cen) == len(rad) == len(X)):
            raise ValueError('slide: one mask entry, centre and radius per point')
    q = flatness(X, T)[0]
    stats = np.zeros(3, dtype=np.int64)
    rc = _lib.library().pyhxt_relax_slivers(
        len(X), X.ctypes.data, len(T), T.ctypes.data, q.ctypes.data, fx.ctypes.data,
        None if sl is None else sl.ctypes.data, None if cen is None else cen.ctypes.data,
        None if rad is None else rad.ctypes.data, float(angle), int(sweeps), int(maxfev),
        stats.ctypes.data)
    if rc != 0:
        raise MemoryError('pyhxt_relax_slivers')
    return X, tuple(int(s) for s in stats)
