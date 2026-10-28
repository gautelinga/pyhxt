# pyhxt

Fast, multi-threaded tetrahedral meshing of closed surfaces, from Python.

pyhxt fills a closed triangle surface with tetrahedra using HXT, the parallel Delaunay
mesher from [gmsh](https://gmsh.info), built on its own. Your surface is kept exactly: no
point is moved and no triangle is split.

```python
import pyhxt
from pyhxt import shapes

# a unit box with a ball inside it
points, triangles, markers = shapes.combine(
    shapes.box(n=20),
    shapes.sphere(centre=(.5, .5, .5), radius=.3, subdivisions=4, marker=2))

mesh = pyhxt.tetrahedralize(points, triangles, markers, holes=[(.5, .5, .5)], size=.05)
print(mesh.report())     # points, tets, volumes and dihedral angles
mesh.write('box.vtu')    # needs meshio
```

## Install

    pip install git+https://github.com/gautelinga/pyhxt

This compiles the mesher, which takes about 10 seconds and needs gcc, g++ and OpenMP. pyhxt
runs on Linux only. numpy is its only dependency. To read and write mesh files, install
meshio too: `pip install "pyhxt[meshio]"`.

## Command line

    pyhxt part.stl part.vtu --size 0.05
    pyhxt shell.stl shell.vtu --hole 0 0 0 --boundary shell_boundary.vtu --threads 8

Any surface format meshio reads goes in, and any volume format it writes comes out. The
command prints a short report on the mesh. `pyhxt --help` lists the options, which match
the ones below.

## Usage

`pyhxt.tetrahedralize(points, triangles, markers=None, **options)` returns a `Mesh`.

**Input:** `points` (n, 3) and `triangles` (m, 3), or a meshio, pyvista or trimesh surface
in place of both.
- Each closed surface may face either way.
- `markers` (m,) are optional positive integers, one per triangle. They stay on the
  boundary triangles of the result.

| option | |
|---|---|
| `size`, `size_min` | largest and smallest edge length (default: taken from the surface) |
| `point_sizes` | the target size at each input point |
| `size_field` | a function of an (N, 3) array of points that returns N sizes; it is called during refinement |
| `holes` | points inside regions to leave empty |
| `regions` | `{label: point}`: each tet gets the label of the region it lies in, in `mesh.volume` |
| `volumes` | which volumes to mesh, each given as the markers of the surfaces around it |
| `seeds` | points inserted before the surface. One inside each closed surface whose points all lie on a sphere makes the first step much faster |
| `threads` | 0 (the default) uses every core |
| `quality` | tets below this quality are improved (default 0.3, as in gmsh); `optimize=False` skips the improvement |
| `reproducible` | by default, the same input on the same number of threads gives the same mesh; `False` is about 15% faster on large meshes |
| `fast` | pyhxt's faster boundary recovery (the default); `False` runs HXT as gmsh does |
| `log` | a file for HXT's messages |

**The result:**
- `mesh.points`: the vertices, with your points first and in your order;
- `mesh.tets`: the tetrahedra;
- `mesh.volume`: the volume or region label of each tet;
- `mesh.triangles` and `mesh.markers`: the boundary;
- `mesh.info`: counts and the time each stage took.

To check the result, `mesh.report()` gives a few lines: the size, each volume, and the
dihedral angles, with how many tets are close to flat. `mesh.cell_volumes()` and
`mesh.dihedral_angles()` give the numbers per tet.

To use another library, call `mesh.to_meshio()` or `mesh.to_pyvista()`, or `mesh.write(path)` to
save a file. If HXT can't mesh the surface, you get a `pyhxt.HXTError`; HXT's messages also go to
the `pyhxt` logger.

Examples, in [`examples/`](examples):
- a box with a hole;
- two labelled regions with a size field;
- the space between many spheres;
- an STL file in and a `.vtu` file out.

## Good to know

- **The surface is checked first.** A gap in the surface, repeated points and degenerate
  triangles are reported as such, rather than as an HXT error. A surface may have inner walls
  (two regions sharing a face): give the shared triangles once.

- **Make the surface at about the size you want.** HXT never splits surface triangles, so a
  large one shapes the tets next to it. pyhxt warns when the surface is much coarser than `size`.
- **Angles.** HXT improves poor tets but, unlike TetGen, guarantees no angle bound.
- **Speed.** Small meshes take well under a second. A 1.2-billion-tet mesh took HXT 7
  minutes on 56 cores and about 100 GB of memory.
- **Other meshers.**
  - TetGen also keeps the surface, but runs on one thread.
  - fTetWild copes with broken surfaces, but moves them.
  - gmsh contains HXT; pyhxt skips gmsh's geometry model and its overhead.

## How it works

pyhxt compiles HXT from gmsh 4.13.1's sources (in [`gmsh-4.13.1/`](gmsh-4.13.1); every change is
marked `pyhxt:`) into `libpyhxt.so`, and calls it with ctypes.

- **`fast=True`** makes three changes:
  - surface triangles already in the first Delaunay mesh are attached directly rather than
    searched for;
  - seeds go in first;
  - the global Delaunay pass after boundary recovery is skipped.
- **Crash fixes.** pyhxt fixes two crashes that HXT has in gmsh 4.13.1: a refinement point
  with an empty cavity, and refinement points outside the tet that made them.
- **Exact mode.** With no switches, the C library gives gmsh's mesh, bit for bit.

To develop:

    ./build.sh                          # libpyhxt.so in the clone
    PYTHONPATH=. python -m pytest tests

Wheels for x86-64 and aarch64 Linux are built by the `wheels` workflow on `v*` tags.

## Licence and citation

pyhxt is under the GNU AGPL v3. HXT is by Célestin Marot, Jeanne Pellerin and Jean-François
Remacle (GPL v2 or later). Its boundary recovery is Hang Si's TetGen code, which is under the
AGPL v3 outside gmsh. If you use pyhxt, please cite HXT:

- C. Marot, J. Pellerin, J.-F. Remacle, *One machine, one minute, three billion tetrahedra*,
  Int. J. Numer. Methods Eng. 117 (2019) 967–990.
- C. Marot, J.-F. Remacle, *Quality tetrahedral mesh generation with HXT*, arXiv:2008.08508.
