"""pyhxt: fast, threaded tetrahedral meshing of closed surfaces, with HXT.

    import pyhxt
    mesh = pyhxt.tetrahedralize(points, triangles, size=0.05)

See README.md, and tetrahedralize's documentation.
"""
import logging
import re
import time

import numpy as np

from pyhxt import _lib
from pyhxt._surface import surface
from pyhxt.mesh import Mesh

__version__ = '0.2.0'
__all__ = ['tetrahedralize', 'Mesh', 'HXTError', 'library']

logger = logging.getLogger('pyhxt')
library = _lib.library

_LEVELS = {0: logging.INFO, 1: logging.DEBUG, 2: logging.WARNING, 3: logging.ERROR,
           4: logging.DEBUG}
_TIMERS = {'tEmptyMesh': 'empty_mesh', 'tVerifyBnd': 'verify_boundary',
           'tBndRecovery': 'boundary_recovery', 'tConvertMesh': 'convert',
           'tRefine': 'refine', 'tOptimize': 'optimize'}


class HXTError(RuntimeError):
    """HXT could not mesh the surface; the message is HXT's."""


def tetrahedralize(points, triangles=None, markers=None, *, volumes=None, holes=None,
                   regions=None, seeds=None, size=None, size_min=0., point_sizes=None,
                   size_field=None, threads=0, quality=0.3, optimize=True, fast=True,
                   reproducible=True, log=None):
    """Fills the volumes a closed triangle surface bounds with tetrahedra, keeping the
    surface exactly: every input point and triangle is in the mesh, unmoved and unsplit.

    The surface
        points, triangles: (n, 3) points and (m, 3) triangles indexing them, each closed
            surface in either orientation. Or `points` alone: a meshio, pyvista or
            trimesh surface (its triangle data "markers" as markers).
        markers: (m,) positive integers, one per triangle (default: all 1). They are kept
            on the boundary triangles of the result, and `volumes` refers to them.

    Which volumes
        Without `volumes`, every closed volume is meshed but those holding a point of
        `holes`. `volumes`: a list of the volumes to mesh, each the list of markers of
        the surfaces that bound it (its holes' surfaces included).
        holes: (k, 3) points inside regions to leave empty, as TetGen's holes.
        regions: {label: point}: name the volumes. A tet's `volume` is then the label of
            the region point in its volume, or -1 where there is none.

    Sizes (edge lengths)
        By default the sizes are interpolated from the surface's edges into the volume.
        size, size_min: the largest and smallest.
        point_sizes: (n,) sizes at the input points, where the surface's edges should not
            set them (0 to keep the edges').
        size_field: a function of an (N, 3) array of points returning N sizes, called in
            batches while the mesh is refined; a point's size is the smaller of the field's
            and the interpolated one.

    Speed
        threads: 0 for OpenMP's default (every core).
        seeds: (k, 3) points inserted before the surface. One inside each closed surface
            whose points lie on a sphere (a bead, a ball) makes HXT's first Delaunay
            mesh 10x faster there. Holes are seeds too.
        fast: pyhxt's faster boundary recovery and seeds first (default). False runs HXT
            exactly as gmsh 4.13.1 does.
        reproducible: the same mesh for the same input and number of threads; False is
            faster (~15% at 50 M tets) but the mesh changes from run to run.

    Quality
        quality: HXT improves tets below this on its quality measure (1 for the regular
            tet); gmsh's default is 0.3. optimize: False skips the improvement.

    log: a file to append HXT's messages to. They also go to the logger "pyhxt".

    :rtype: Mesh
    :raises HXTError: with HXT's message if it cannot mesh the surface.
    """
    lib = _lib.library()
    if triangles is None:
        points, triangles, found = surface(points)
        markers = found if markers is None else markers
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.uint32).reshape(-1, 3)
    markers = np.ones(len(triangles), dtype=np.uint32) if markers is None else \
        np.ascontiguousarray(markers, dtype=np.uint32).reshape(-1)
    if len(markers) != len(triangles) or (markers == 0).any():
        raise ValueError('one positive marker per triangle')
    if len(triangles) and triangles.max() >= len(points):
        raise ValueError('a triangle indexes a point that is not there')
    extra = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in (holes, seeds)
             if a is not None]
    seed = np.ascontiguousarray(np.vstack(extra) if extra else np.zeros((0, 3)))
    nholes = 0 if holes is None else len(extra[0])
    if volumes is None:
        offsets, surfaces = np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32)
    else:
        offsets = np.cumsum([0] + [len(v) for v in volumes]).astype(np.uint32)
        surfaces = np.array([s for v in volumes for s in v], dtype=np.uint32)
    sizes = None
    if point_sizes is not None:
        sizes = np.ascontiguousarray(point_sizes, dtype=np.float64).reshape(-1)
        if len(sizes) != len(points):
            raise ValueError('one point size per point')
    flags = (_lib.FAST if fast else 0) | (0 if reproducible else _lib.NONREPRODUCIBLE)
    flags |= _lib.CHECK_CENTRES   # always: a guard against HXT crashing (README)
    if size and len(triangles):
        edges = np.linalg.norm(points[triangles] - points[np.roll(triangles, 1, axis=1)],
                               axis=2)
        coarse = np.quantile(edges, .9)
        if coarse > 2 * size:
            logger.warning(
                f'the surface is much coarser than the size asked for (its longer edges '
                f'are {coarse:.3g}, the size {size:.3g}): HXT does not split surface '
                f'triangles, so the tets next to them are shaped by them. A surface made '
                f'at about the size gives better tets there.')
    lc = float(np.linalg.norm(points.max(0) - points.min(0)))

    messages, failure = [], []

    def on_message(level, text):
        text = text.decode(errors='replace')
        messages.append((level, text))
        logger.log(_LEVELS.get(level, logging.INFO), text)

    def on_field(pts, n):
        try:
            a = np.ctypeslib.as_array(pts, shape=(n, 4))
            s = np.asarray(size_field(a[:, :3].copy()), dtype=np.float64).reshape(-1)
            if len(s) != n:
                raise ValueError(f'size_field returned {len(s)} sizes for {n} points')
            a[:, 3] = np.where(s > 0, np.minimum(a[:, 3], s), a[:, 3])
            return 0
        except BaseException as e:  # noqa: BLE001 - raised again after HXT stops
            failure.append(e)
            return 1

    message_cb = _lib.MESSAGE(on_message)
    field_cb = _lib.FIELD(on_field) if size_field is not None else None
    mesh = _lib.ctypes.c_void_p()
    start = time.time()
    status = lib.pyhxt_tetmesh(
        len(points), points, None if sizes is None else sizes.ctypes.data, len(seed), seed,
        nholes, len(triangles), triangles, markers, len(offsets) - 1, offsets, surfaces,
        int(threads), float(size_min), float(size) if size else 0., lc, 1e-12,
        float(quality), int(bool(optimize)), flags,
        None if field_cb is None else _lib.ctypes.cast(field_cb, _lib.ctypes.c_void_p),
        _lib.ctypes.cast(message_cb, _lib.ctypes.c_void_p),
        None if log is None else str(log).encode(), _lib.ctypes.byref(mesh))
    if failure:
        if mesh:
            lib.pyhxt_free(mesh)
        raise failure[0]
    if status != 0:
        errors = [t for lvl, t in messages if lvl == 3] or [t for _, t in messages[-3:]]
        raise HXTError(f'HXT failed (status {status}): ' + ' | '.join(errors))
    try:
        nv, nt, ntri = (_lib.ctypes.c_uint32(), _lib.ctypes.c_uint64(),
                        _lib.ctypes.c_uint64())
        lib.pyhxt_counts(mesh, nv, nt, ntri)
        X = np.empty((nv.value, 3))
        T = np.empty((nt.value, 4), dtype=np.uint32)
        col = np.empty(nt.value, dtype=np.uint32)
        tri = np.empty((ntri.value, 3), dtype=np.uint32)
        lib.pyhxt_copy(mesh, X, T, col, tri)
        tcol = np.empty(ntri.value, dtype=np.uint32)
        lib.pyhxt_copy_triangle_colors(mesh, tcol)
        located = None
        if regions:
            where = np.ascontiguousarray([regions[k] for k in regions], dtype=np.float64)
            located = np.empty(len(where), dtype=np.uint32)
            lib.pyhxt_locate(mesh, len(where), where, located)
    finally:
        lib.pyhxt_free(mesh)

    # HXT's exterior: its ghost tets, and the colour of the outside and of the holes
    out = np.iinfo(np.uint32).max
    keep = (T[:, 3] != out) & (col != out)
    if volumes is not None:
        keep &= col < len(volumes)
    T, col = T[keep].astype(np.int64), col[keep].astype(np.int64)
    if regions:
        label = np.full(int(col.max(initial=0)) + 1, -1, dtype=np.int64)
        for name, c in zip(regions, located):
            if c == out:
                raise ValueError(f'region {name!r}: its point is in no meshed volume')
            label[c] = name
        col = label[col]
    return Mesh(X, T, col, tri.astype(np.int64), tcol.astype(np.int64),
                _info(messages, time.time() - start, len(T), nv.value, threads, flags))


def _info(messages, seconds, ntets, nvertices, threads, flags):
    """What HXT's messages say, as numbers."""
    info = {'tets': ntets, 'vertices': nvertices, 'seconds': seconds,
            'threads': threads, 'flags': flags, 'missing_facets': 0, 'timings': {}}
    for _, text in messages:
        m = re.match(r'(t\w+)\s*=\s*([\d.]+)', text)
        if m and m[1] in _TIMERS:
            info['timings'][_TIMERS[m[1]]] = info['timings'].get(_TIMERS[m[1]], 0.) + float(m[2])
        m = re.match(r'Recovering (\d+) missing facet', text)
        if m:
            info['missing_facets'] += int(m[1])
    info['warnings'] = [t for lvl, t in messages if lvl == 2]
    return info
