"""A box of non-overlapping spheres, the fluid between them meshed.

Every sphere is a hole. Its points all lie on one sphere, which makes the first Delaunay
mesh inside it degenerate and slow; the hole point inside each one, inserted first, avoids
that, and pyhxt's boundary recovery is faster too (fast=True, the default; fast=False runs
HXT as gmsh does). The stages printed show where they differ; on sphere packings of 50 M
tets and more, the fast mode halved the time.

    python 03_sphere_packing.py [number of spheres]
"""
import sys
import time

import numpy as np

import pyhxt
from pyhxt import shapes

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(1)

# spheres placed at random, kept inside the box and apart from each other
centres, radius = [], .07
while len(centres) < n:
    c = rng.uniform(radius + .01, 1 - radius - .01, 3)
    if all(np.linalg.norm(c - o) > 2 * radius + .01 for o in centres):
        centres.append(c)

surface = shapes.combine(shapes.box(n=50),
                         *(shapes.sphere(c, radius, subdivisions=3, marker=2) for c in centres))
print(f'{n} spheres, {len(surface[1])} surface triangles')

for fast in (True, False):
    t = time.time()
    mesh = pyhxt.tetrahedralize(*surface, holes=centres, size=.02, fast=fast)
    timings = mesh.info['timings']
    print(f'fast={fast!s:5}: {len(mesh.tets):,} tets in {time.time() - t:.1f} s '
          f'(first Delaunay {timings["empty_mesh"]:.2f} s, boundary recovery '
          f'{timings["boundary_recovery"]:.2f} s, refinement {timings["refine"]:.2f} s)')
