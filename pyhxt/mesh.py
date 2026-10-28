"""The mesh tetrahedralize returns, and its way out to other libraries."""
import numpy as np


def _chunks(n, size=1 << 20):
    """Slices of at most `size` tets: the per-tet arrays of a large mesh stay small."""
    return (slice(i, min(i + size, n)) for i in range(0, n, size))


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

    def cell_volumes(self):
        """(m,) the volume of each tet."""
        out = np.empty(len(self.tets))
        for s in _chunks(len(self.tets)):
            a, b, c, d = (self.points[self.tets[s, k]] for k in range(4))
            out[s] = np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))) / 6
        return out

    def dihedral_angles(self):
        """(m, 6) each tet's six dihedral angles in degrees, at its edges 01 02 03 12 13 23.
        A regular tet's are 70.5; a flat tet (a sliver) has angles near 0 and 180."""
        out = np.empty((len(self.tets), 6))
        for s in _chunks(len(self.tets)):
            p = self.points[self.tets[s]]
            normals = []
            for i, (a, b, c) in enumerate(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))):
                n = np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a])
                n *= np.sign(np.einsum('ij,ij->i', n, p[:, a] - p[:, i]))[:, None]   # outward
                normals.append(n / np.linalg.norm(n, axis=1)[:, None])
            for e, (f, g) in enumerate(((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))):
                # an edge lies on the faces opposite its tet's other two vertices, f and g
                cos = np.clip(np.einsum('ij,ij->i', normals[f], normals[g]), -1, 1)
                out[s, e] = 180 - np.degrees(np.arccos(cos))
        return out

    def report(self):
        """A few lines on the mesh: its size, each volume's tets and extent, and the
        dihedral angles (the smallest and largest, and how many tets are near flat)."""
        v, a = self.cell_volumes(), self.dihedral_angles()
        lo, hi = a.min(axis=1), a.max(axis=1)
        lines = [repr(self)]
        for label in np.unique(self.volume):
            inside = self.volume == label
            lines.append(f'  volume {label}: {inside.sum()} tets, volume {v[inside].sum():.6g}')
        lines.append(f'  dihedral angles: {lo.min():.2f} to {hi.max():.2f} degrees')
        within = [int(((lo < t) | (hi > 180 - t)).sum()) for t in (5, 10, 15)]
        lines.append(f'  tets within 5 / 10 / 15 degrees of flat: {within[0]} / {within[1]} '
                     f'/ {within[2]}')
        if 'seconds' in self.info:
            lines.append(f'  meshed in {self.info["seconds"]:.2f} s')
        return '\n'.join(lines)

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
