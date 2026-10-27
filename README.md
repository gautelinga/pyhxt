# pyhxt

HXT, the parallel tetrahedral mesher of Marot, Pellerin & Remacle, built on its own as
`libpyhxt.so`, outside gmsh. The aim is the same basic functionality as TetGen's
tetrahedralisation of a closed surface (the surface kept exactly, volumes, holes, a size
bound), but threaded, fast and for large meshes. It is not a TetGen drop-in.

**State: a minimal library (0.1.0).** `pip install` builds it; one Python function, and the C
interface under it. Linux with gcc and OpenMP only, no wheels yet. It is used by
mesh-sphere-packing's `mesher: hxt` (`hxt_lib`).

## Install and use

    pip install git+https://github.com/gautelinga/pyhxt     # builds libpyhxt.so, ~10 s
    # or, in a clone: pip install .   (./build.sh alone for the bare library)

```python
import pyhxt
mesh = pyhxt.tetrahedralize(points, triangles, holes=hole_points, size=0.05, threads=8)
mesh.points      # (n, 3): the input points first, unmoved, then the seeds, then HXT's
mesh.tets        # (m, 4)
mesh.volume      # (m,): the volume each tet is in
```

- **The surface is kept exactly.** The input triangles are the mesh's boundary; HXT's
  boundary recovery adds no point on the surface (it fails instead).
- **Volumes.** Without `volumes`, HXT finds every closed volume and meshes all but those
  holding a hole point, which are left out before refinement, as TetGen's holes. With
  `volumes` (each a list of the surface numbers, `colors`, that bound it) only those are
  meshed.
- **Seeds.** `seeds` (and `holes`) are inserted before the surface. Where a hole's surface
  points all lie on one sphere, as beads do, a point inside it keeps HXT's initial Delaunay
  from degenerating (10x faster on a sphere packing, below).
- **Size.** `size` bounds the edge length; sizes are otherwise interpolated from the
  surface's edges. TetGen bounds volumes instead (`-a`) and radius-edge ratios (`-q`);
  HXT has no such bound, but optimises tets below `quality` (its own measure, 1 for the
  regular tet).
- **Repeatability.** The same input on the same number of threads gives the same mesh
  (`reproducible=True`, the default; off is faster).
- **`fast=False`** runs HXT exactly as gmsh 4.13.1 does (and, given the input in gmsh's
  order, gives gmsh's mesh to the bit).

Tests: `pytest tests` (the API) and `python3 test.py [libpyhxt.so]` (the C library, standard
library only). The C interface is in `pyhxt.c`; `pyhxt_abi()` says which version of it a
library has (2: hole points).

## What pyhxt changes in HXT (`pyhxt_flags.h`)

With no flag set (`fast=False`), HXT runs exactly as in gmsh 4.13.1 and gives gmsh's mesh.
The default (`fast=True`) sets the first three; the mesh is then another one of the same
surface and quality.

| flag | what | h = 0.025 welded sphere packing, 8 threads (laptop) |
|---|---|---|
| `PYHXT_PREBOND` | boundary recovery bonds the surface edges and triangles the mesh already has before TetGen's random-order passes, which then search only for the missing ones | recover boundary 5.8 → 2.3 s |
| `PYHXT_NO_RECOVER_DELAUNAY` | no global Delaunay recovery (a Lawson pass over every face) after the boundary is recovered; the quality after optimisation was the same to three digits | 9.9 s → 0 |
| `PYHXT_SEEDS_FIRST` | the embedded points (seeds, holes) go into the empty mesh before the surface | empty mesh 15.6 → 1.3 s, with a seed inside each bead |
| `PYHXT_NONREPRODUCIBLE` | HXT's reproducible mode off | refinement 9.6 → 7.4 s; a new mesh each run |

pyhxt also leaves the volumes holding hole points out right after HXT colours its volumes
(`hxt_tetMesh.c`), whatever the flags.

Why seeds: points that all lie on one sphere have a degenerate Delaunay inside it, and every
in-sphere test there falls back to exact arithmetic. In a welded packing there is nothing
inside the beads; at h = 0.025 that was 9.9 M adaptive in-sphere tests (0.24 M when contact
discs put points inside the beads), and 0.2 M with one seed per bead inserted first.

On the 53 M tet build (h = 0.0175 welded, 8 threads, laptop) the three default flags took
HXT from 108 s to 57 s (empty mesh 25.7 → 2.5 s, boundary recovery 36.0 → 7.4 s), with the
same quality: smallest dihedral 6.51°, 0 / 19 / 1087 tets within 5 / 10 / 15° of flat
against 0 / 19 / 1122.

## Why

HXT is the fastest of the open tetrahedral meshers ("One machine, one minute, three billion
tetrahedra", IJNME 2019; "Quality tetrahedral mesh generation with HXT", arXiv 2008.08508),
but it is only reachable through gmsh, whose model costs time of its own: at 56 M tets on 8
threads, HXT's phases took 87 s of gmsh's 195 s (2026-10-27). The rest is gmsh copying the
surface into HXT and building an object per node and tet from its result (`Gmsh2Hxt`,
`Hxt2Gmsh` in gmsh's `meshGRegionHxt.cpp`, the latter serial), and the caller copying them out
again. Called directly, HXT's arrays are copied once.

Other Python meshers do other jobs: `tetgen` (pyvista) and MeshPy wrap TetGen, which is
serial; `pytetwild` wraps fTetWild, which meshes messy input robustly but approximates the
surface within an envelope rather than keeping it.

## Same mesh as gmsh

`pyhxt.c` calls `hxtTetMesh` with what gmsh 4.13.1's `meshGRegionHxt.cpp` gives it:
reproducible, refine and optimise on, quality threshold 0.3, tolerance 1e-12, sizes
interpolated from the surface and capped by the size bound, gmsh's size callback (with no
size field it caps a size by the model's characteristic length), and the brep. With the
input in gmsh's order and the output numbered as gmsh numbers it (mesh-sphere-packing's
`hxtmesh._direct_hxt`), the mesh is gmsh's, by the sha256 of the final mesh: h = 0.035 welded
and flattened (1 and 47 shells) on 4 threads, and h = 0.0175 welded (two fluid regions, 53 M
tets) on 8 threads against gmsh 4.13.1's PyPI wheel on x86-64. HXT's input hash, which it
logs, matches gmsh's too. As through gmsh, the mesh depends on the number of threads.

## Build

`build.sh [output]` uses `$CC` and `$CXX` (gcc and g++ by default; both need OpenMP). `-ffp-contract=off` throughout, as gmsh compiles `predicates.c`: x86-64's baseline has no
fused multiply-add, so this changes nothing there, but gcc on aarch64 fuses by default,
which would change the numbers and so the mesh.

## Sources

`gmsh-4.13.1/` is the subset of gmsh-4.13.1-source.tgz (https://gmsh.info/src/, sha256
77972145f431726026d50596a6a44fb3c1c95c21255218d66955806b86edbe8d) that HXT's tet mesher
needs, unchanged: `contrib/hxt/{core,predicates,tetMesh,tetBR}`, tetMesh's command-line
programs left out. Boundary recovery is HXT's own copy of TetGen's (`tetBR/src/tetgenBR`,
built with `HXT_WITHOUT_GMSH`), not the copy gmsh builds with; the meshes above are the same
with either. pyhxt's changes to these files are marked `pyhxt:` (`tetBR/src/tetgenBR.cxx`,
`tetBR/src/hxt_boundary_recovery.cxx`, `tetMesh/src/hxt_tetRefine.c`,
`tetMesh/src/hxt_tetMesh.c`); with no flag and no hole point they change nothing.

## Licence

AGPL-3.0 (`LICENSE`). HXT is under the GPL, version 2 or later
(`gmsh-4.13.1/contrib/hxt/LICENSE.txt`). Its boundary recovery is TetGen's code, which is
relicensed under gmsh's licence only for use in gmsh (`contrib/hxt/CREDITS.txt`); outside
gmsh it is under TetGen's own licence, the AGPL v3 for TetGen 1.5 and 1.6 (the file header
still carries older wording). `pyhxt.c`, `build.sh` and `test.py` are under the AGPL v3 or
later.
