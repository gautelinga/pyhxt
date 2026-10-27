"""pyhxt: HXT, the parallel tetrahedral mesher of Marot, Pellerin & Remacle, on numpy arrays.

    >>> mesh = pyhxt.tetrahedralize(points, triangles, size=0.05, threads=8)
    >>> mesh.points, mesh.tets, mesh.volume

The surface is kept exactly: every input point and triangle is in the output, unmoved and
unsplit (HXT's boundary recovery inserts no point on the surface; it fails rather than do
so). See README.md for what differs from TetGen.
"""
import ctypes
import os
from collections import namedtuple

import numpy as np

__all__ = ['tetrahedralize', 'Mesh', 'library']

# pyhxt_flags.h
PREBOND, NO_RECOVER_DELAUNAY, SEEDS_FIRST, NONREPRODUCIBLE = 1, 2, 4, 8
FAST = PREBOND | NO_RECOVER_DELAUNAY | SEEDS_FIRST

Mesh = namedtuple('Mesh', 'points tets volume triangles triangle_colors')
Mesh.__doc__ = """HXT's mesh. points: (n, 3), the input points first, then the seeds, then
HXT's own; tets: (m, 4) int64, positively oriented as HXT leaves them; volume: (m,) the
volume each tet is in (an index into `volumes`, or HXT's own numbering without them);
triangles, triangle_colors: the surface as HXT returns it (the input's)."""

_lib = None


def library():
    """libpyhxt.so: $PYHXT_LIB, else beside this package or in the repository's root
    (where build.sh puts it)."""
    global _lib
    if _lib is None:
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.environ.get('PYHXT_LIB'), os.path.join(here, 'libpyhxt.so'),
                     os.path.join(here, os.pardir, 'libpyhxt.so')):
            if path and os.path.exists(path):
                break
        else:
            raise OSError('libpyhxt.so not found: run build.sh, or set PYHXT_LIB')
        lib = ctypes.CDLL(path)
        u32, f64 = (np.ctypeslib.ndpointer(t, flags='C_CONTIGUOUS')
                    for t in (np.uint32, np.float64))
        lib.pyhxt_tetmesh.argtypes = [
            ctypes.c_uint32, f64, ctypes.c_uint32, f64, ctypes.c_uint64, u32, u32,
            ctypes.c_uint32, u32, u32, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.pyhxt_counts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_uint64),
                                     ctypes.POINTER(ctypes.c_uint64)]
        lib.pyhxt_copy.argtypes = [ctypes.c_void_p, f64, u32, u32, u32]
        lib.pyhxt_copy_triangle_colors.argtypes = [ctypes.c_void_p, u32]
        lib.pyhxt_free.argtypes = [ctypes.c_void_p]
        _lib = lib
    return _lib


def tetrahedralize(points, triangles, colors=None, volumes=None, holes=None, seeds=None,
                   size=None, size_min=0., threads=0, quality=0.3, optimize=True,
                   fast=True, reproducible=True, log=None):
    """Fills the volumes a closed triangle surface bounds with tetrahedra, keeping the
    surface exactly.

    :param points: (n, 3) surface points.
    :param triangles: (m, 3) triangles indexing `points`; each closed surface (shell) in
        any orientation.
    :param colors: (m,) a positive surface number per triangle (default: all 1).
    :param volumes: the volumes to mesh, each a list of the surface numbers bounding it
        (the inner surfaces of holes included). Default: HXT finds the volumes itself and
        meshes all of them, the inside of holes too (see `holes`).
    :param holes: (k, 3) points inside regions not to keep: the volumes holding them are
        dropped from the output. Without `volumes` they are meshed first, so prefer
        `volumes` for speed. They also serve as seeds.
    :param seeds: (k, 3) extra points inside the solid (or anywhere), inserted first. A
        point inside each hole whose surface points are co-spherical (a sphere, a bead)
        keeps HXT's initial Delaunay from degenerating there: 10x faster on a sphere
        packing. They are not in any output tet outside the volumes they are in.
    :param size: the largest edge length HXT refines to (default: no bound; sizes are
        interpolated from the surface's edges).
    :param size_min: the smallest.
    :param threads: 0 for OpenMP's default.
    :param quality: HXT's optimisation threshold (its quality measure, 1 for the regular
        tet); gmsh's default is 0.3.
    :param optimize: run HXT's mesh improvement.
    :param fast: pyhxt's faster boundary recovery and seeds first (default); False runs
        HXT exactly as gmsh 4.13.1 does.
    :param reproducible: the same mesh for the same input and threads; False is faster.
    :param log: a file HXT's messages are appended to.
    :rtype: Mesh
    """
    lib = library()
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1, 3)
    colors = np.ones(len(triangles), dtype=np.uint32) if colors is None else \
        np.ascontiguousarray(colors, dtype=np.uint32)
    if len(colors) != len(triangles) or (colors == 0).any():
        raise ValueError('one positive surface number per triangle')
    extra = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in (holes, seeds)
             if a is not None]
    seed = np.ascontiguousarray(np.vstack(extra) if extra else np.zeros((0, 3)))
    nholes = 0 if holes is None else len(extra[0])
    if volumes is None:
        offsets, surfaces = np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32)
    else:
        offsets = np.cumsum([0] + [len(v) for v in volumes]).astype(np.uint32)
        surfaces = np.array([s for v in volumes for s in v], dtype=np.uint32)
    flags = (FAST if fast else 0) | (0 if reproducible else NONREPRODUCIBLE)
    lc = float(np.linalg.norm(points.max(0) - points.min(0)))

    mesh = ctypes.c_void_p()
    status = lib.pyhxt_tetmesh(
        len(points), points, len(seed), seed, len(triangles), triangles, colors,
        len(offsets) - 1, offsets, surfaces, int(threads), float(size_min),
        float(size) if size else 0., lc, 1e-12, float(quality), int(bool(optimize)), flags,
        None if log is None else os.fsencode(log), ctypes.byref(mesh))
    if status != 0:
        raise RuntimeError('HXT failed with status {}{}'.format(
            status, '' if log is None else ' (see {})'.format(log)))
    try:
        nv, nt, ntri = ctypes.c_uint32(), ctypes.c_uint64(), ctypes.c_uint64()
        lib.pyhxt_counts(mesh, nv, nt, ntri)
        X = np.empty((nv.value, 3))
        T = np.empty((nt.value, 4), dtype=np.uint32)
        col = np.empty(nt.value, dtype=np.uint32)
        tri = np.empty((ntri.value, 3), dtype=np.uint32)
        lib.pyhxt_copy(mesh, X, T, col, tri)
        tcol = np.empty(ntri.value, dtype=np.uint32)
        lib.pyhxt_copy_triangle_colors(mesh, tcol)
    finally:
        lib.pyhxt_free(mesh)

    nvol = len(offsets) - 1 if volumes is not None else None
    keep = (T[:, 3] != np.iinfo(np.uint32).max) & (col != np.iinfo(np.uint32).max)
    if nvol is not None:
        keep &= col < nvol
    if nholes:
        hole_vertices = len(points) + np.arange(nholes)
        dropped = np.unique(col[keep & np.isin(T, hole_vertices).any(axis=1)])
        keep &= ~np.isin(col, dropped)
    return Mesh(X, T[keep].astype(np.int64), col[keep].astype(np.int64),
                tri.astype(np.int64), tcol.astype(np.int64))
