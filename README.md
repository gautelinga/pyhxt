# pyhxt

HXT, the parallel tetrahedral mesher of Marot, Pellerin & Remacle, built on its own as
`libpyhxt.so`, outside gmsh. The aim is the same basic functionality as TetGen's
tetrahedralisation of a closed surface (the surface kept exactly, volumes, holes, a size
bound), but threaded, fast and for large meshes. It is not a TetGen drop-in.

**State: a working prototype.** The library and a ctypes call of it work and give gmsh's mesh
to the bit (below). The Python API (`pyhxt.tetrahedralize(...)` on numpy arrays, holes and
regions, sizes), packaging and wheels are still to come. Today it is used by
mesh-sphere-packing's `mesher: hxt` (`hxt_lib`).

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

    ./build.sh            # gcc and g++ with OpenMP; about 10 s
    python3 test.py       # the unit cube

`-ffp-contract=off` throughout, as gmsh compiles `predicates.c`: x86-64's baseline has no
fused multiply-add, so this changes nothing there, but gcc on aarch64 fuses by default,
which would change the numbers and so the mesh.

## Sources

`gmsh-4.13.1/` is the subset of gmsh-4.13.1-source.tgz (https://gmsh.info/src/, sha256
77972145f431726026d50596a6a44fb3c1c95c21255218d66955806b86edbe8d) that HXT's tet mesher
needs, unchanged: `contrib/hxt/{core,predicates,tetMesh,tetBR}`, tetMesh's command-line
programs left out. Boundary recovery is HXT's own copy of TetGen's (`tetBR/src/tetgenBR`,
built with `HXT_WITHOUT_GMSH`), not the copy gmsh builds with; the meshes above are the same
with either.

## Licence

AGPL-3.0 (`LICENSE`). HXT is under the GPL, version 2 or later
(`gmsh-4.13.1/contrib/hxt/LICENSE.txt`). Its boundary recovery is TetGen's code, which is
relicensed under gmsh's licence only for use in gmsh (`contrib/hxt/CREDITS.txt`); outside
gmsh it is under TetGen's own licence, the AGPL v3 for TetGen 1.5 and 1.6 (the file header
still carries older wording). `pyhxt.c`, `build.sh` and `test.py` are under the AGPL v3 or
later.
