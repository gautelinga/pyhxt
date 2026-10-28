# pyhxt

**Fast, multi-threaded tetrahedral meshing of closed surfaces, from Python.**

You give pyhxt a closed triangle surface (a box, a part, a pack of spheres, anything
watertight) and it fills the inside with tetrahedra. The surface comes back exactly as you
gave it: no point moved, no triangle split. Under the hood is HXT, the parallel mesher
from the gmsh project, used here on its own and made faster still for large meshes.

```python
import pyhxt
from pyhxt import shapes

points, triangles, markers = shapes.combine(
    shapes.box(n=20),                                        # a unit box ...
    shapes.sphere(centre=(.5, .5, .5), radius=.3, subdivisions=4, marker=2))  # ... with a ball

mesh = pyhxt.tetrahedralize(points, triangles, markers, holes=[(.5, .5, .5)], size=.05)
print(mesh)              # Mesh(11573 points, 56256 tets in 1 volumes, 9920 boundary triangles)
mesh.write('box.vtu')    # open it in ParaView (needs meshio)
```

It is a young library (0.2), Linux only for now.

## Install

    pip install git+https://github.com/gautelinga/pyhxt

This compiles the mesher (about 10 seconds; it needs gcc, g++ and OpenMP, which a normal
Linux setup has). numpy is the only dependency. For reading and writing mesh files, also
install meshio (`pip install "pyhxt[meshio]"`); pyvista and trimesh surfaces work too if
you have them.

## What it does

**Keeps your surface.** The first points of `mesh.points` are your points, in your order;
the boundary triangles are yours, with your markers on them (`mesh.markers`). Nothing is
added on the surface, so two meshes made on matching surfaces fit together, and periodic
faces stay periodic.

**Holes and regions.** A closed surface inside another one is either a hole or a second
material:

```python
mesh = pyhxt.tetrahedralize(points, triangles, markers, holes=[(.5, .5, .5)])   # leave it empty
mesh = pyhxt.tetrahedralize(points, triangles, markers,                        # or label both
                            regions={1: (.05, .05, .05), 2: (.5, .5, .5)})
mesh.volume              # 1 or 2 for every tet
```

**Mesh size.** By default the tets take their size from the surface's triangles. You can
bound it everywhere (`size=`), set it at the surface points (`point_sizes=`), or give a
function of position, which is called in batches as the mesh is refined:

```python
def size(x):                       # x: an (N, 3) array of points
    return .01 + .5 * abs(np.linalg.norm(x - centre, axis=1) - radius)   # finer near the ball

mesh = pyhxt.tetrahedralize(points, triangles, markers, size_field=size)
```

**Other libraries.** Pass a meshio, pyvista or trimesh surface instead of arrays, and get
the result back as one of them:

```python
mesh = pyhxt.tetrahedralize(meshio.read('part.stl'))
mesh.to_meshio(), mesh.to_pyvista(), mesh.write('part.vtu')
```

**What happened.** `mesh.info` has the counts and how long each stage took; HXT's own
messages go to the Python logger `"pyhxt"`; if HXT cannot mesh the surface, you get a
`pyhxt.HXTError` with its message.

## Examples

In [`examples/`](examples):

1. [`01_quickstart.py`](examples/01_quickstart.py): a box with a spherical hole, written
   to a `.vtu` file.
2. [`02_regions_and_sizes.py`](examples/02_regions_and_sizes.py): two labelled materials
   and a size field that refines near the interface.
3. [`03_sphere_packing.py`](examples/03_sphere_packing.py): the space between 60 spheres,
   with the time each stage takes.
4. [`04_from_a_file.py`](examples/04_from_a_file.py): an STL file in, a `.vtu` out.

## How fast

pyhxt was built for very large meshes of sphere packings (the pore space of a random close
packing of 100 beads, for flow simulations). On one node with 56 threads (Intel Xeon Gold
6254), in mesh-sphere-packing:

| tets | pyhxt's meshing | the whole build (surfaces, checks, file) | memory |
|---|---|---|---|
| 53 million | 30 s | 1 min | 5 GB |
| 424 million | 3 min | 6 min | 37 GB |
| 1.19 billion | 7 min | 14 min | 101 GB |

Small meshes take well under a second (`examples/01_quickstart.py`: 56 thousand tets in
0.5 s on a laptop).

## Compared with other meshers

- **TetGen** (in Python: `tetgen`, MeshPy) also keeps the surface, and bounds tet volumes
  and shapes. It runs on one thread, so large meshes take much longer.
- **gmsh** contains HXT, and pyhxt's `fast=False` runs it the same way, but gmsh's model
  costs time of its own: for 56 million tets, half of gmsh's meshing time was spent
  outside HXT.
- **fTetWild** (`pytetwild`) meshes broken, self-intersecting surfaces robustly, but
  moves the surface within a tolerance instead of keeping it.

## Good to know

- **The surface must be closed and should not be much coarser than the size you ask for.**
  HXT never splits a surface triangle; if you ask for tets far smaller than the surface's
  triangles, it can fail, and on some such surfaces it crashes (gmsh 4.13.1 crashes on the
  same input). pyhxt warns when the surface is much coarser than `size`. Make the surface
  at about the size you want.
- **Repeatability.** The same input on the same number of threads gives the same mesh
  (`reproducible=True`). Turning that off is faster for large meshes (about 15% at 50
  million tets), but every run then gives a different mesh.
- **Threads.** `threads=0` (the default) uses every core.
- **Quality.** HXT improves tets below a quality threshold (`quality=0.3` by default);
  unlike TetGen it has no hard bound on angles. In the sphere-packing meshes above, after
  mesh-sphere-packing's own pass over the flattest tets, the smallest dihedral angles were
  5-9 degrees.

## How it works

pyhxt compiles HXT's tetrahedral mesher from gmsh 4.13.1's sources (the subset in
[`gmsh-4.13.1/`](gmsh-4.13.1), unchanged apart from the switches below) into
`libpyhxt.so`, and calls it with ctypes on numpy arrays.

With `fast=True` (the default) three changes make large meshes faster:

1. **Seeds first.** The points of a sphere all lie on one sphere, which makes the first
   Delaunay mesh inside it degenerate, and every test there needs exact arithmetic. A point
   inside each hole, inserted first, avoids it: on a welded sphere packing, the first mesh
   took 1.3 s instead of 15.6 s.
2. **Faster boundary recovery.** The surface edges and triangles that the first mesh already
   has are attached directly, using the map HXT keeps, so TetGen's recovery code only
   searches for the missing ones.
3. **No global Delaunay pass** after the boundary is recovered (the mesh quality after
   optimisation was the same).

At 53 million tets these took HXT from 108 to 57 seconds. pyhxt also skips refinement
points that fall outside their tetrahedron, and limits the optimiser to one thread per
100,000 tets, both against HXT failures on small or awkward meshes.

With `fast=False`, HXT runs as gmsh runs it. Through the C interface with no switches at
all, the mesh is gmsh's to the bit: mesh-sphere-packing checks this against gmsh 4.13.1 on
meshes of up to 53 million tets. The switches are in [`pyhxt_flags.h`](pyhxt_flags.h), and
every change to gmsh's files is marked `pyhxt:` in them.

## Building from a clone

    ./build.sh                  # libpyhxt.so in the clone; $CC and $CXX if not gcc/g++
    PYTHONPATH=. python -m pytest tests
    python test.py              # the C library alone, standard library only

The build uses `-ffp-contract=off`, so that compilers that fuse multiply-adds (gcc on
ARM) give the same numbers, and so the same meshes, as on x86-64.

Wheels for x86-64 and aarch64 Linux are built by the `wheels` workflow (on a `v*` tag).

## Licence and credits

pyhxt is under the GNU AGPL v3 (`LICENSE`). HXT is by Célestin Marot, Jeanne Pellerin and
Jean-François Remacle, under the GPL v2 or later; its boundary recovery is TetGen's code
(Hang Si), under the AGPL v3 outside gmsh. If you use pyhxt in research, please cite HXT:

- C. Marot, J. Pellerin, J.-F. Remacle, *One machine, one minute, three billion
  tetrahedra*, Int. J. Numer. Methods Eng. 117 (2019) 967-990.
- C. Marot, J.-F. Remacle, *Quality tetrahedral mesh generation with HXT*, arXiv:2008.08508.
