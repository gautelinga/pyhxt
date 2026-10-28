"""Surfaces from other libraries: meshio, pyvista and trimesh objects."""
import numpy as np


def surface(obj, markers='markers'):
    """(points, triangles, markers or None) from a meshio.Mesh, a pyvista.PolyData or a
    trimesh.Trimesh; `markers` is the name of their triangle data to take as markers."""
    if hasattr(obj, 'cells_dict'):                        # meshio
        if 'triangle' not in obj.cells_dict:
            raise ValueError('the meshio mesh has no triangles')
        tri = obj.cells_dict['triangle']
        data = obj.cell_data_dict.get(markers, {}).get('triangle')
        return np.asarray(obj.points)[:, :3], tri, data
    if hasattr(obj, 'is_all_triangles') and hasattr(obj, 'faces'):   # pyvista
        if not obj.is_all_triangles:
            obj = obj.triangulate()
        tri = np.asarray(obj.faces).reshape(-1, 4)[:, 1:]
        data = obj.cell_data.get(markers)
        return np.asarray(obj.points), tri, data
    if hasattr(obj, 'vertices') and hasattr(obj, 'faces'):          # trimesh
        return np.asarray(obj.vertices), np.asarray(obj.faces), None
    raise TypeError(f'not a surface pyhxt reads: {type(obj).__name__} (give points and '
                    'triangles, or a meshio, pyvista or trimesh mesh)')
