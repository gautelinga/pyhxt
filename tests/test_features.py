"""Sizes, region labels, diagnostics and other libraries' meshes."""
import logging

import numpy as np
import pytest

import pyhxt
from pyhxt import shapes


def volume(points, tets):
    a, b, c, d = (points[tets[:, k]] for k in range(4))
    return np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))) / 6


def ball_in_box():
    """A unit box (marker 1) with a ball of radius 0.3 (marker 2) at its centre."""
    return shapes.combine(shapes.box(n=2),
                          shapes.sphere((.5, .5, .5), .3, subdivisions=3, marker=2))


def test_size_field_refines_where_it_asks():
    calls = []

    def field(x):
        calls.append(len(x))
        return np.where(x[:, 0] < .5, .03, 1.)

    m = pyhxt.tetrahedralize(*shapes.box(), size_field=field, threads=2)
    centre_x = m.points[m.tets].mean(axis=1)[:, 0]
    assert calls
    assert (centre_x < .5).sum() > 20 * (centre_x >= .5).sum()
    assert volume(m.points, m.tets).sum() == pytest.approx(1, rel=1e-12)


def test_point_sizes_set_the_size_at_the_surface():
    p, t, _ = shapes.box()
    coarse = pyhxt.tetrahedralize(p, t, threads=2)
    fine = pyhxt.tetrahedralize(p, t, point_sizes=np.full(len(p), .1), threads=2)
    assert len(fine.tets) > 50 * len(coarse.tets)


def test_a_failing_size_field_raises_its_error():
    def field(x):
        raise KeyError('no size here')

    with pytest.raises(KeyError, match='no size here'):
        pyhxt.tetrahedralize(*shapes.box(), size_field=field, threads=2)


def test_regions_name_the_volumes():
    m = pyhxt.tetrahedralize(*ball_in_box(), regions={7: (.05, .05, .05), 3: (.5, .5, .5)},
                             size=.1, threads=2)
    assert set(np.unique(m.volume)) == {3, 7}
    ball = volume(m.points, m.tets[m.volume == 3]).sum()
    assert ball == pytest.approx(4 / 3 * np.pi * .3 ** 3, rel=.02)
    assert volume(m.points, m.tets).sum() == pytest.approx(1, rel=1e-12)


def test_holes_leave_the_ball_out():
    m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.1, threads=2)
    p, t, mk = ball_in_box()
    a, b, c = (p[t[mk == 2][:, k]] for k in range(3))
    ball = abs(np.einsum('ij,ij->', a, np.cross(b, c))) / 6   # divergence theorem
    assert volume(m.points, m.tets).sum() == pytest.approx(1 - ball, rel=1e-12)
    assert set(np.unique(m.markers)) == {1, 2}


def test_info_and_logging(caplog):
    with caplog.at_level(logging.INFO, logger='pyhxt'):
        m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.1, threads=2)
    assert m.info['tets'] == len(m.tets) and m.info['vertices'] == len(m.points)
    assert {'empty_mesh', 'refine', 'optimize'} <= set(m.info['timings'])
    assert any('Delaunay' in r.message for r in caplog.records)


def test_an_open_surface_raises_hxts_error():
    p, t, _ = shapes.box()
    with pytest.raises(pyhxt.HXTError, match='HXT failed'):
        pyhxt.tetrahedralize(p, t[:-1], threads=2)


def test_meshio_in_and_out(tmp_path):
    meshio = pytest.importorskip('meshio')
    p, t, mk = ball_in_box()
    surface = meshio.Mesh(p, [('triangle', t)], cell_data={'markers': [mk]})
    m = pyhxt.tetrahedralize(surface, holes=[(.5, .5, .5)], size=.1, threads=2)
    assert set(np.unique(m.markers)) == {1, 2}
    m.write(tmp_path / 'mesh.vtu')
    back = meshio.read(tmp_path / 'mesh.vtu')
    assert np.array_equal(back.cells_dict['tetra'], m.tets)
    assert len(m.to_meshio(boundary=True).cells_dict['triangle']) == len(m.triangles)


def test_pyvista_in_and_out():
    pv = pytest.importorskip('pyvista')
    surface = pv.Sphere(radius=.5, theta_resolution=24, phi_resolution=24)
    m = pyhxt.tetrahedralize(surface, threads=2)
    grid = m.to_pyvista()
    assert grid.n_cells == len(m.tets)
    assert volume(m.points, m.tets).sum() == pytest.approx(surface.volume, rel=1e-9)


def test_trimesh_in():
    trimesh = pytest.importorskip('trimesh')
    m = pyhxt.tetrahedralize(trimesh.creation.icosphere(subdivisions=2), threads=2)
    assert len(m.tets) > 0
