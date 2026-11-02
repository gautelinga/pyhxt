"""The passes after meshing: flatness, peeling flat tets, relaxing slivers (pyhxt_tidy.c)."""
import numpy as np
import pytest

import pyhxt
from pyhxt import shapes


def boundary_faces(tets):
    f = np.sort(np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]],
                                tets[:, [0, 1, 2]]]), axis=1)
    f, n = np.unique(f, axis=0, return_counts=True)
    return {tuple(x) for x in f[n == 1]}


def ball_in_box(subdivisions=3):
    return shapes.combine(shapes.box(n=4),
                          shapes.sphere((.5, .5, .5), .3, subdivisions=subdivisions, marker=2))


def test_flatness_is_the_dihedral_angles_distance_from_flat():
    m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.08, threads=2)
    q, vol = pyhxt.flatness(m.points, m.tets)
    a = m.dihedral_angles()
    assert q == pytest.approx(np.minimum(a.min(axis=1), 180 - a.max(axis=1)), abs=1e-9)
    assert np.abs(vol) == pytest.approx(m.cell_volumes(), rel=1e-12)
    assert np.array_equal(m.flatness(), q)


@pytest.mark.parametrize('protect,peeled', [((), 1), ((7,), 0)])
def test_a_flat_tet_on_two_coplanar_triangles_is_peeled(protect, peeled):
    """The square ABCD split along AC, a tet of no volume on it, two tets on BD above: the
    diagonal is swapped, the boundary is the triangles again; with its marker protected the
    tet stays."""
    A, B, C, D, E = range(5)
    X = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [.5, .5, 1.]])
    T = np.array([[A, B, C, D], [A, B, D, E], [B, C, D, E]])
    tri = np.array([[A, C, B], [A, D, C], [A, B, E], [B, C, E], [C, D, E], [D, A, E]])
    markers = np.array([7, 7, 8, 8, 8, 8])
    tets, tri2, removed = pyhxt.peel_flat(X, T, tri, markers, protect=protect)
    assert removed.sum() == peeled and len(tri2) == len(tri)
    if peeled:
        assert list(removed) == [True, False, False]
        assert np.array_equal(tets, T[1:])
        assert {tuple(sorted(t)) for t in tri2} == boundary_faces(tets)
        assert {tuple(sorted(t)) for t in tri2[:2]} == {(A, B, D), (B, C, D)}
        normals = np.cross(X[tri2[:2, 1]] - X[tri2[:2, 0]], X[tri2[:2, 2]] - X[tri2[:2, 0]])
        assert (normals[:, 2] < 0).all(), 'oriented as the triangles they replace'
    else:
        assert np.array_equal(tets, T) and np.array_equal(tri2, tri)


def test_a_mesh_without_flat_tets_is_left_alone():
    m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.08, threads=2)
    tets, tri = m.tets.copy(), m.triangles.copy()
    assert m.peel_flat() == 0
    assert np.array_equal(m.tets, tets) and np.array_equal(m.triangles, tri)


def test_relaxing_slivers_moves_only_interior_vertices_and_helps():
    m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.06, threads=2)
    X0, total = m.points.copy(), m.cell_volumes().sum()
    before, after, slid = m.relax_slivers(angle=20.)
    on = np.unique(m.triangles)
    assert before > 0 and after < before and slid == 0
    assert np.array_equal(m.points[on], X0[on])
    assert not np.array_equal(m.points, X0)
    assert (pyhxt.flatness(m.points, m.tets)[1] > 0).all(), 'no tet turned inside out'
    assert m.cell_volumes().sum() == pytest.approx(total, rel=1e-12)


def test_sliding_vertices_stay_on_their_sphere():
    """A cap: four points of the unit sphere, close together, one flat tet. No vertex is
    free, so only sliding along the sphere can help it; the points stay on the sphere."""
    a = np.radians([0., 120., 240.])
    X = np.vstack([[0., 0., 1.], np.c_[.2 * np.cos(a), .2 * np.sin(a), np.full(3, np.sqrt(.96))]])
    T = np.array([[0, 1, 2, 3]])
    if pyhxt.flatness(X, T)[1][0] < 0:
        T = T[:, [0, 2, 1, 3]]
    before_q = pyhxt.flatness(X, T)[0][0]
    slide = (np.ones(4, dtype=bool), np.zeros((4, 3)), np.ones(4))
    X2, (before, _, slid) = pyhxt.relax_slivers(X, T, 30., np.ones(4, dtype=bool), slide)
    q, vol = pyhxt.flatness(X2, T)
    assert before == 1 and slid > 0 and q[0] > before_q and vol[0] > 0
    assert np.linalg.norm(X2, axis=1) == pytest.approx(1., abs=1e-12)


def test_fixed_vertices_without_slide_do_not_move():
    a = np.radians([0., 120., 240.])
    X = np.vstack([[0., 0., 1.], np.c_[.2 * np.cos(a), .2 * np.sin(a), np.full(3, np.sqrt(.96))]])
    X0 = X.copy()
    _, (before, after, slid) = pyhxt.relax_slivers(X, [[0, 1, 2, 3]], 30., np.ones(4, bool))
    assert (before, after, slid) == (1, 1, 0) and np.array_equal(X, X0)


def test_relaxing_is_deterministic():
    m1 = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.06, threads=2)
    m2 = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.06, threads=2)
    assert m1.relax_slivers(20.) == m2.relax_slivers(20.)
    assert np.array_equal(m1.points, m2.points)
