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


def test_an_open_surface_is_named():
    p, t, _ = shapes.box()
    with pytest.raises(ValueError, match='not closed: 3 edges'):
        pyhxt.tetrahedralize(p, t[:-1], threads=2)


def test_repeated_points_are_named():
    p, t, _ = shapes.box()
    p, t = p[t].reshape(-1, 3), np.arange(3 * len(t)).reshape(-1, 3)   # as an unmerged STL
    with pytest.raises(ValueError, match='repeat another point: merge them'):
        pyhxt.tetrahedralize(p, t, threads=2)


def test_a_degenerate_triangle_is_named():
    p, t, _ = shapes.box()
    t[3, 2] = t[3, 0]
    with pytest.raises(ValueError, match='1 triangles use the same point twice'):
        pyhxt.tetrahedralize(p, t, threads=2)


def test_two_regions_sharing_a_wall_are_closed():
    """A box split by an inner wall: edges where the wall meets the box have three triangles."""
    left, right = shapes.box((0, 0, 0), (.5, 1, 1), n=2), shapes.box((.5, 0, 0), (1, 1, 1), n=2)
    p, t, _ = shapes.combine(left, right)
    p, inverse = np.unique(np.round(p, 12), axis=0, return_inverse=True)
    t = inverse.reshape(-1)[t]
    wall = (p[t][:, :, 0] == .5).all(axis=1)
    t = t[~wall | (np.arange(len(t)) < len(left[1]))]   # the wall once: the left box's
    m = pyhxt.tetrahedralize(p, t, regions={1: (.25, .5, .5), 2: (.75, .5, .5)}, threads=2)
    assert set(np.unique(m.volume)) == {1, 2}
    assert m.cell_volumes()[m.volume == 1].sum() == pytest.approx(.5, rel=1e-12)


def test_cell_volumes_and_angles():
    from pyhxt.mesh import Mesh
    regular = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.]])
    corner = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    m = Mesh(np.vstack([regular, corner]), np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
             np.array([1, 1]), np.zeros((0, 3), int), np.zeros(0, int))
    assert m.cell_volumes() == pytest.approx([8 / 3, 1 / 6])
    a = m.dihedral_angles()
    assert a[0] == pytest.approx(np.full(6, np.degrees(np.arccos(1 / 3))))
    assert a[1] == pytest.approx([90, 90, 90] + 3 * [np.degrees(np.arccos(1 / 3 ** .5))])


def test_report():
    m = pyhxt.tetrahedralize(*ball_in_box(), holes=[(.5, .5, .5)], size=.1, threads=2)
    r = m.report()
    assert 'dihedral angles' in r and 'within 5 / 10 / 15' in r
    assert m.cell_volumes().sum() == pytest.approx(volume(m.points, m.tets).sum())


def test_command_line(tmp_path, capsys):
    meshio = pytest.importorskip('meshio')
    from pyhxt.__main__ import main
    p, t, mk = ball_in_box()
    meshio.Mesh(p, [('triangle', t)], cell_data={'markers': [mk]}).write(tmp_path / 'in.vtu')
    main([str(tmp_path / 'in.vtu'), str(tmp_path / 'out.vtu'), '--size', '.1',
          '--hole', '.5', '.5', '.5', '--threads', '2', '--boundary', str(tmp_path / 'b.vtu')])
    assert 'wrote' in capsys.readouterr().out
    assert len(meshio.read(tmp_path / 'out.vtu').cells_dict['tetra']) > 0
    assert set(np.unique(meshio.read(tmp_path / 'b.vtu').cell_data['markers'][0])) == {1, 2}
    open_surface = tmp_path / 'open.vtu'
    meshio.Mesh(p, [('triangle', t[:-1])]).write(open_surface)
    with pytest.raises(SystemExit, match='not closed'):
        main([str(open_surface), str(tmp_path / 'x.vtu')])


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


@pytest.mark.parametrize('fast', [True, False])
def test_a_grid_surface_does_not_abort(fast):
    """A box whose faces are a 10 x 10 grid, meshed at 0.05: a refinement point got an empty
    cavity, and HXT (gmsh 4.13.1's too) aborted. pyhxt skips such a point."""
    m = pyhxt.tetrahedralize(*shapes.box(n=10), size=.05, fast=fast, threads=1)
    assert volume(m.points, m.tets).sum() == pytest.approx(1, rel=1e-12)


def test_many_threads_on_a_small_mesh():
    """The same failure, seen as a crash in the optimiser on 16 threads."""
    m = pyhxt.tetrahedralize(*shapes.combine(shapes.box(n=4), shapes.sphere(
        (.5, .5, .5), .3, subdivisions=4, marker=2)), holes=[(.5, .5, .5)], size=.05,
        threads=16)
    p, t, _ = shapes.sphere((.5, .5, .5), .3, subdivisions=4)
    a, b, c = (p[t[:, k]] for k in range(3))
    ball = abs(np.einsum('ij,ij->', a, np.cross(b, c))) / 6
    assert volume(m.points, m.tets).sum() == pytest.approx(1 - ball, rel=1e-12)
