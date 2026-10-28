"""Meshing a surface read from a file (STL, OBJ, PLY, ... through meshio), and writing the
result back out.

    python 04_from_a_file.py [surface file]

Without a file, it makes one (a sphere, as an STL) to read.
"""
import sys

import meshio

import pyhxt
from pyhxt import shapes

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = 'sphere.stl'
    points, triangles, _ = shapes.sphere(radius=1., subdivisions=4)
    meshio.Mesh(points, [('triangle', triangles)]).write(path)
    print('wrote', path)

surface = meshio.read(path)          # pyvista and trimesh surfaces work the same way
mesh = pyhxt.tetrahedralize(surface)  # sizes follow the surface's own edges
print(mesh)

mesh.write('from_a_file.vtu')
print('wrote from_a_file.vtu')
