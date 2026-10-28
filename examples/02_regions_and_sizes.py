"""Two materials, labelled, with a mesh that is finer near where they meet.

A ball inside a box: both volumes are meshed (no holes), and `regions` names them. A size
field asks for small tets near the ball's surface and lets them grow away from it.

    python 02_regions_and_sizes.py
"""
import numpy as np

import pyhxt
from pyhxt import shapes

CENTRE, RADIUS = np.array([.5, .5, .5]), .3
points, triangles, markers = shapes.combine(
    shapes.box(n=12, marker=1),
    shapes.sphere(CENTRE, RADIUS, subdivisions=5, marker=2))


def size(x):
    """Target edge length at the points x (an (N, 3) array): 0.01 at the ball's surface,
    growing linearly to 0.08 away from it."""
    distance = np.abs(np.linalg.norm(x - CENTRE, axis=1) - RADIUS)
    return np.clip(.01 + .5 * distance, .01, .08)


WATER, STEEL = 1, 2
mesh = pyhxt.tetrahedralize(points, triangles, markers, size_field=size,
                            regions={WATER: (.05, .05, .05), STEEL: tuple(CENTRE)})
print(mesh)

v = mesh.cell_volumes()
for name, label in (('water', WATER), ('steel', STEEL)):
    inside = mesh.volume == label
    print(f'{name}: {inside.sum()} tets, volume {v[inside].sum():.5f}')
print(f'the ball should be about {4 / 3 * np.pi * RADIUS ** 3:.5f} (its surface is faceted)')

try:
    mesh.write('two_materials.vtu')   # colour by the cell data "volume" in ParaView
    print('wrote two_materials.vtu')
except ImportError:
    pass
