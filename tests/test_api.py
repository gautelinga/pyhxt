"""pyhxt.tetrahedralize on small closed surfaces: the volume is filled exactly, the surface
kept, holes dropped, in pyhxt's fast mode as in gmsh's."""
import numpy as np
import pytest

import pyhxt


def cube(lo=0., hi=1.):
    x = np.array([[(i & 1), (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return lo + (hi - lo) * x, tris


def volume(points, tets):
    a, b, c, d = (points[tets[:, k]] for k in range(4))
    return np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))).sum() / 6


def boundary(tets):
    faces = np.sort(np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]],
                                    tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]), axis=1)
    f, n = np.unique(faces, axis=0, return_counts=True)
    return f[n == 1]


@pytest.mark.parametrize('fast', [True, False])
def test_cube(fast):
    x, t = cube()
    m = pyhxt.tetrahedralize(x, t, size=0.2, threads=2, fast=fast)
    assert volume(m.points, m.tets) == pytest.approx(1, rel=1e-12)
    assert np.array_equal(m.points[:8], x)
    assert np.array_equal(boundary(m.tets), np.unique(np.sort(t, axis=1), axis=0))


@pytest.mark.parametrize('fast', [True, False])
@pytest.mark.parametrize('how', ['holes', 'volumes'])
def test_cube_with_a_hole(fast, how):
    xo, to = cube()
    xi, ti = cube(0.4, 0.6)
    x = np.vstack([xo, xi])
    t = np.vstack([to, ti + 8])
    colors = np.repeat([1, 2], 12)
    kw = {'holes': [[0.5, 0.5, 0.5]]} if how == 'holes' else {'volumes': [[1, 2]]}
    m = pyhxt.tetrahedralize(x, t, colors, size=0.1, threads=2, fast=fast, **kw)
    assert volume(m.points, m.tets) == pytest.approx(1 - 0.2 ** 3, rel=1e-12)
    assert np.array_equal(m.points[:16], x)
    assert np.array_equal(boundary(m.tets), np.unique(np.sort(t, axis=1), axis=0))
    assert len(np.unique(m.volume)) == 1


def test_reproducible():
    x, t = cube()
    a, b = (pyhxt.tetrahedralize(x, t, size=0.1, threads=2) for _ in range(2))
    assert np.array_equal(a.points, b.points) and np.array_equal(a.tets, b.tets)
