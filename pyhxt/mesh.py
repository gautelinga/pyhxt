"""The mesh tetrahedralize returns, and its way out to other libraries."""
import numpy as np


class Mesh:
    """A tetrahedral mesh.

    points: (n, 3) float64. The input points come first, unmoved and in their order,
        then any seeds, then the points HXT added.
    tets: (m, 4) int64, indexing points.
    volume: (m,) int64, the volume each tet is in: the label of its region point with
        `regions`, else the index in `volumes`, else HXT's own numbering.
    triangles: (k, 3) int64, the boundary triangles (the input's), and
    markers: (k,) int64, their markers (the input's, or 1).
    info: dict, what HXT reported (counts, timings).
    """

    def __init__(self, points, tets, volume, triangles, markers, info=None):
        self.points, self.tets, self.volume = points, tets, volume
        self.triangles, self.markers = triangles, markers
        self.info = info if info is not None else {}

    def __repr__(self):
        return (f'Mesh({len(self.points)} points, {len(self.tets)} tets in '
                f'{len(np.unique(self.volume))} volumes, {len(self.triangles)} boundary '
                f'triangles)')

    def to_meshio(self, boundary=False):
        """A meshio.Mesh: the tets with their volume as cell data ("volume"), or with
        `boundary`, the boundary triangles with their markers ("markers")."""
        import meshio
        if boundary:
            return meshio.Mesh(self.points, [('triangle', self.triangles)],
                               cell_data={'markers': [self.markers]})
        return meshio.Mesh(self.points, [('tetra', self.tets)],
                           cell_data={'volume': [self.volume]})

    def to_pyvista(self):
        """A pyvista.UnstructuredGrid of the tets, with "volume" as cell data."""
        import pyvista as pv
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: self.tets}, self.points)
        grid.cell_data['volume'] = self.volume
        return grid

    def write(self, path, boundary=False, **kwargs):
        """Writes the mesh (or with `boundary`, its boundary triangles) through meshio, in
        the format its suffix says (.vtu, .xdmf, .msh, ...)."""
        self.to_meshio(boundary).write(path, **kwargs)
