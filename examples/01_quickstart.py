"""The smallest useful example: a unit box with a spherical hole in it.

    python 01_quickstart.py
"""
import pyhxt
from pyhxt import shapes

# a closed surface: the box's six faces and a sphere inside it
box = shapes.box((0, 0, 0), (1, 1, 1), n=20, marker=1)   # triangles about as big as the tets
ball = shapes.sphere(centre=(.5, .5, .5), radius=.3, subdivisions=4, marker=2)
points, triangles, markers = shapes.combine(box, ball)

# fill the space between them; the point inside the ball says "leave this empty"
mesh = pyhxt.tetrahedralize(points, triangles, markers,
                            holes=[(.5, .5, .5)], size=.05)

print(mesh.report())   # size, volumes, and the dihedral angles: how good the tets are

# the input surface is in the mesh unchanged: same points first, same triangles
assert (mesh.points[:len(points)] == points).all()

try:
    mesh.write('quickstart.vtu')                       # needs meshio; open it in ParaView
    mesh.write('quickstart_boundary.vtu', boundary=True)
    print('wrote quickstart.vtu and quickstart_boundary.vtu')
except ImportError:
    print('pip install meshio to write the mesh to a file')
