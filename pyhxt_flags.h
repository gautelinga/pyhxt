/* pyhxt: the switches pyhxt adds to HXT (gmsh 4.13.1), set per call by pyhxt_tetmesh.
 * With none set, HXT runs exactly as in gmsh. AGPL-3.0-or-later. */
#ifndef PYHXT_FLAGS_H
#define PYHXT_FLAGS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
extern int pyhxt_flags;
/* the hole points: vertices pyhxt_hole_first ... + pyhxt_num_holes - 1; the volumes holding
 * them are made exterior right after HXT colours its volumes, so not refined or output */
extern uint32_t pyhxt_hole_first, pyhxt_num_holes;
/* HXT's triangle-to-tet map (4 tet + facet, or HXT_NO_ADJACENT for a missing triangle),
 * set while boundary recovery runs, so that PYHXT_PREBOND finds each surface triangle's
 * tet without a search */
extern uint64_t* pyhxt_tri2tet;
#ifdef __cplusplus
}
#endif
/* boundary recovery: bond the surface edges and triangles the empty mesh already has
 * before TetGen's random-order passes, which then only search for the missing ones */
#define PYHXT_PREBOND 1
/* boundary recovery: skip TetGen's global Delaunay recovery (a Lawson pass over every
 * face) after the boundary is recovered */
#define PYHXT_NO_RECOVER_DELAUNAY 2
/* empty mesh: insert the embedded (seed) points before the surface points; a point
 * inside each hole bounded by co-spherical points keeps its Delaunay from degenerating */
#define PYHXT_SEEDS_FIRST 4
/* HXT's reproducible option off: faster, but the mesh changes from run to run */
#define PYHXT_NONREPRODUCIBLE 8
/* refinement: do not insert a point that is not inside the tet that made it; with sizes
 * far below a nearly flat tet's edges, HXT's point could land outside the mesh and break
 * the insertion. Set with the caller's sizes. */
#define PYHXT_CHECK_CENTRES 16
/* optimisation: at most a thread per 100 000 tets (HXT's parallel edge removal crashed on
 * a small mesh with many threads) */
#define PYHXT_SMALL_MESH_THREADS 32
#endif
